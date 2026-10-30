# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import contextlib
import hashlib
import importlib
import importlib.util
import os
import time
import re
from collections.abc import Callable
from typing import Any

import torch
import torch._inductor.compile_fx
import torch._ops
import torch.fx as fx


from vllm.compilation.compiler_interface import CompilerInterface
from vllm.logger import init_logger

logger = init_logger(__name__)

# py_code cache artifact suffix. Files are content-addressed as
# "{key}.{graph_digest}.ngc" — see _npugraph_digest for why the digest is
# part of the name.
_NPUGRAPH_CACHE_SUFFIX = ".ngc"


def _load_static_kernel_cache_patch():
    """Resolve omni's static_kernel_cache_patch helper regardless of how this file's
    package is named in the current process.

    The omni tree may be imported under a different top-level name
    (``omni_npu``), and forked dynamo workers can fail to import the
    package entirely even where the parent process imported it fine.
    This file and static_kernel_cache_patch.py always ship in the same directory,
    so the last resort is loading it by path from next to this file.
    """
    for modname in (
        "omni.compilation.static_kernel_cache_patch",
        "omni_npu.compilation.static_kernel_cache_patch",
    ):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        logger.info("static_kernel_cache patch resolver: level=package modname=%s", modname)
        return module.apply_static_kernel_cache_patch
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "static_kernel_cache_patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "omni_static_kernel_cache_patch", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info("static_kernel_cache patch resolver: level=file path=%s", path)
    return module.apply_static_kernel_cache_patch


_config_unreachable_warned = False


def _get_npugraph_ex_config() -> dict[str, Any] | None:
    """Read the npugraph_ex_config sub-dict.

    The physical source directory is ``omni/`` but setup.py installs it as
    ``omni_npu`` (OMNI_NPU_PACKAGE_DIR), so the engine always imports this
    tree under the ``omni_npu`` name — no alias probing needed.
    """
    global _config_unreachable_warned
    try:
        module = importlib.import_module(
            "omni_npu.compilation.npugraph_ex_config"
        )
    except ImportError:
        if not _config_unreachable_warned:
            _config_unreachable_warned = True
            logger.warning(
                "npugraph_ex_config unreachable under omni_npu; "
                "compilation config defaults apply"
            )
        return None
    return module.get_aclgraph_config().npugraph_ex_config


# ---- [npugraph-canon] canonicalize the graph before digesting ----
# Gate-2 experiment: the same logical graph can reach the backend in two
# equivalent synchronization writings (torch core `streams.*` vs npugraph_ex
# `tagged_event_*`). Hashing the raw graph therefore yields two digests for
# one graph (false-negative miss -> 5-boot ladder + duplicate artifacts).
# The fix under test: run npugraph_ex's OWN replace_stream_event_pass on a
# copy inside the backend call window and digest the translated form, so
# both writings converge to the tagged vocabulary the backend itself emits.
# Gate 1 (vcjob fxgate1, 12/12 PASS) validated the isolation below:
#  - pass runs on a deepcopy; caller graph untouched;
#  - GraphCounter.COUNT restored (real pass must keep its numbering);
#  - _create_event_by_name patched in EVERY module that holds its own
#    imported reference (graph_pass + replace_stream_event; `from ... import`
#    binds per module -- gate1 v4/v5 lesson), so no real NPU event is created;
#  - the pass MUST run inside the backend window: outside it the dynamo
#    external-object table is gone ("User object is no longer alive", v2).
def _canonicalize_for_digest(graph: fx.GraphModule) -> fx.GraphModule:
    import copy as _copy

    import npugraph_ex._acl_concrete_graph.graph_pass as _gp
    import npugraph_ex._acl_concrete_graph.replace_stream_event as _rse

    def _no_event(_name):
        """Stand-in for the backend's event factory: creates nothing."""

    pass_fn = _rse.replace_stream_event_pass
    counter = _rse.GraphCounter
    saved = counter.COUNT
    sites = []
    for mod in (_gp, _rse):
        if hasattr(mod, "_create_event_by_name"):
            sites.append((mod, mod._create_event_by_name))
            mod._create_event_by_name = _no_event
    try:
        cp = _copy.deepcopy(graph)
        pass_fn(cp)
        return cp
    finally:
        for mod, orig in sites:
            mod._create_event_by_name = orig
        counter.COUNT = saved


def _graph_flavor(graph: fx.GraphModule) -> str:
    streams = 0
    tagged = 0
    for node in graph.graph.nodes:
        t = str(node.target)
        if "streams." in t:
            streams += 1
        elif (
            "npugraph_ex.tagged_event" in t
            or "npugraph_ex.scope_" in t
            or "npugraph_ex.record_tagged" in t
        ):
            tagged += 1
    if streams and tagged:
        return "mixed"
    if streams:
        return "streams"
    if tagged:
        return "tagged"
    return "none"


def _shapes_sig(example_inputs: list[Any]) -> str:
    parts: list[str] = []
    for item in example_inputs:
        if isinstance(item, torch.Tensor):
            parts.append(f"{tuple(item.shape)}:{item.dtype}")
        elif isinstance(item, (int, float, bool)) or item is None:
            parts.append(repr(item))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _canon_digest(
    graph: fx.GraphModule, example_inputs: list[Any], key: str
) -> tuple[str | None, Any]:
    """Digest of the pass-translated copy. Returns (digest, ms) or (None, exc)."""
    try:
        t0 = time.perf_counter()
        canonical = _canonicalize_for_digest(graph)
        ms = (time.perf_counter() - t0) * 1000.0
        return _npugraph_digest(canonical, example_inputs, key), ms
    except Exception as exc:  # noqa: BLE001 -- experiment never blocks the boot
        return None, exc


def _log_canon(event: str, key: str, graph, example_inputs, raw: str) -> str | None:
    """Compute + log the canonical digest; return it for use as the cache key."""
    canon, info = _canon_digest(graph, example_inputs, key)
    logger.info(
        "[npugraph-canon] event=%s key=%s flavor=%s raw=%s canon=%s shapes=%s cost_ms=%s pid=%s",
        event,
        key,
        _graph_flavor(graph),
        raw,
        canon if canon else f"FAIL:{info}",
        _shapes_sig(example_inputs),
        ("%.1f" % info) if canon else "-",
        os.getpid(),
    )
    return canon


_npugraph_cache_pool_registered = False
_npugraph_cache_pool_unreachable_warned = False


def _npugraph_cache_pool():
    """Resolve the optional cache bridge (fusion package, npugraph_cache).

    Returns None when the bridge is unavailable (not installed); every
    caller treats it as a pure optional layer.
    """
    global _npugraph_cache_pool_registered, _npugraph_cache_pool_unreachable_warned
    import atexit

    try:
        module = importlib.import_module("npugraph_cache.npugraph_cache_store")
    except ImportError:
        if not _npugraph_cache_pool_unreachable_warned:
            _npugraph_cache_pool_unreachable_warned = True
            logger.warning("npugraph_cache.npugraph_cache_store unreachable; pooling off")
        return None
    if not _npugraph_cache_pool_registered:
        atexit.register(module.note_exit)
        _npugraph_cache_pool_registered = True
    return module


_engine_build_hash: str | None = None


def _engine_build_fingerprint() -> str:
    """Hash the sources that decide py_code bytes for one graph digest.

    The digest pins WHICH graph an artifact belongs to; this factor pins
    WHICH codegen build produced it: this adaptor (its save-hook tag
    namespacing rewrites the emitted source) and the whole torch_npu
    npugraph_ex package (the codegen itself). Version strings alone miss
    SFS-patched files — a hot patch never moves a version, and a patched
    build reused stale pool objects under unchanged keys (field: 13/16 md5
    mismatch). The pool bridge is deliberately NOT hashed: it only
    transports artifacts and cannot change their bytes, so bridge-only
    patches keep the pool bundle — and its reuse — stable.
    """
    global _engine_build_hash
    if _engine_build_hash is not None:
        return _engine_build_hash
    hasher = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "npugraph_ex.py"), "rb") as f:
            hasher.update(b"npugraph_ex.py")
            hasher.update(f.read())
    except OSError as exc:
        logger.warning("engine build fingerprint: adaptor unreadable: %s", exc)
        hasher.update(f"adaptor-unreadable:{exc}".encode())
    backend = None
    for modname in ("torch_npu.dynamo.npugraph_ex", "npugraph_ex"):
        try:
            backend = importlib.import_module(modname)
            break
        except ImportError:
            continue
    if backend is not None and getattr(backend, "__file__", None):
        path = backend.__file__
        try:
            if os.path.basename(path) == "__init__.py":
                package = os.path.dirname(path)
                for entry in sorted(os.listdir(package)):
                    if entry.endswith(".py"):
                        with open(os.path.join(package, entry), "rb") as f:
                            hasher.update(entry.encode())
                            hasher.update(f.read())
            else:
                with open(path, "rb") as f:
                    hasher.update(os.path.basename(path).encode())
                    hasher.update(f.read())
        except OSError as exc:
            logger.warning("engine build fingerprint: backend unreadable: %s", exc)
            hasher.update(f"backend-unreadable:{exc}".encode())
    else:
        hasher.update(b"npugraph_ex-backend-absent")
    _engine_build_hash = hasher.hexdigest()[:12]
    return _engine_build_hash


def _ensure_static_kernel_runtime_setup(
    npugraph_ex_config: dict[str, Any],
) -> None:
    """Once-per-process runtime setup that must precede the first aclgraph
    capture. Shared by compile() and the py_code load() path: on a full
    py_code cache hit the vLLM framework returns early from
    CompilerManager.load and compile() is never entered, but the
    static-kernel machinery below is still required at capture time —
    skipping it would silently leave every captured graph on dynamic
    kernels. Both operations are idempotent.
    """
    if not (
        npugraph_ex_config.get("static_kernel_compile", False)
        or npugraph_ex_config.get("super_kernel_optimize", False)
    ):
        return
    # Static-kernel compile cache (static_kernel_cache_enable): torch_npu's
    # CompilerConfig cannot express the four cache keys
    # (static_kernel_compile is a plain OptionValue bool that
    # rejects unknown attributes), so we shadow
    # compile_static_kernel with a vendored entry at runtime.
    # force_eager stays True — the vendored entry bypasses the gate
    # itself and the backend keeps returning the eager callable that
    # omni's ACLGraphWrapper captures.
    if npugraph_ex_config.get("static_kernel_cache_enable", False):
        apply_static_kernel_cache_patch = _load_static_kernel_cache_patch()
        apply_static_kernel_cache_patch(npugraph_ex_config)
    # !2903 compile-concurrency tuning. Bare import:
    # static_kernel_compile.py ships in this same tree, so a
    # missing module means the workspace itself is inconsistent.
    from omni_npu import envs as omni_envs
    from omni_npu.compilation.static_kernel_compile import (
        configure_static_kernel_compile_jobs,
    )
    configure_static_kernel_compile_jobs(
        omni_envs.OMNI_STATIC_KERNEL_COMPILE_JOBS
    )


def _install_py_code_save_hook(
    cache_path: str,
    write_path: str | None,
) -> tuple[Callable[[], None], dict[str, Any]]:
    """Capture the codegen source of the next npugraph_ex compilation.

    Patches _NpuFxCompiler._get_compiled_gm for the duration of one
    backend call. torch_npu 2.12.0 ships the package both as top-level
    ``npugraph_ex`` (the instance the backend built by
    npugraph_ex.get_npu_backend actually uses) and as
    ``torch_npu.dynamo.npugraph_ex``; the two share files but not module
    identity (SK_T2 dual-instance landmine), so patch every instance.

    ``write_path`` is where the artifact is written; ``None`` means
    disk-free pool mode (capture for the pool upload only, no local
    write). Returns (restore, state); state["saved"] is True after the
    backend call iff the artifact was captured, state["skip_reason"]
    explains why not otherwise.
    """
    state: dict[str, Any] = {"saved": False, "skip_reason": None}
    patched: list[tuple[Any, Any]] = []

    def _make_patched(orig_get_compiled_gm: Callable) -> Callable:
        def patched_get_compiled_gm(self, gm, example_inputs):
            compiled = orig_get_compiled_gm(self, gm, example_inputs)
            if state["saved"] or not cache_path:
                return compiled
            try:
                py_code = compiled.get_code()
            except Exception as exc:  # noqa: BLE001
                state["skip_reason"] = f"get_code failed: {exc}"
                return compiled
            # get_code() returns the runner object itself (not a str) when
            # codegen cannot express the configuration — such graphs are
            # not serializable and must not be cached
            # (torch_npu 2.12.0 npu_fx_compiler.py, fx1 probe 2026-10-15).
            if not isinstance(py_code, str) or not py_code:
                state["skip_reason"] = "codegen unsupported for this graph"
                return compiled
            # Triton kernel indices (kernel_side_table) are registered
            # in-process at compile time and are not serializable across
            # process boundaries; loading such py_code in a new process
            # would hit an AssertionError in kernel_side_table.get_kernel().
            if "triton_kernel_wrapper" in py_code:
                state["skip_reason"] = "graph contains Triton kernels"
                return compiled
            # A loaded artifact re-creates its tagged events at exec time
            # under the tag strings baked into the code. Those strings carry
            # the writing boot's build counter ("graph_0_..."), so two
            # loaded units share event tags and _create_event_by_name's
            # get-or-create hands both the SAME event object — interleaved
            # execution can then satisfy one unit's wait with the other
            # unit's stale record. Give each artifact's events a
            # digest-unique namespace. Bare stream tags
            # ("graph_N_get_external_object_by_index", no _K suffix) stay
            # shared on purpose: one stream object across units is the
            # intended get-or-create semantics and adds no NPU streams.
            artifact_digest = os.path.basename(cache_path).split(".")[1]
            py_code, namespaced = re.subn(
                r"'(graph_\d+_[^']+)'",
                lambda m: m.group(0)
                if re.fullmatch(
                    r"graph_\d+_get_external_object_by_index", m.group(1)
                )
                else "'gfx"
                + artifact_digest[:12]
                + "_"
                + m.group(1).split("_", 2)[2]
                + "'",
                py_code,
            )
            if write_path is None:
                # Disk-free pool mode: capture for the pool upload only.
                state["saved"] = True
                # Keep the exact bytes for the optional pool upload: uploading
                # from memory means a torn SFS direct-write can never be
                # published to the pool (review M2).
                state["py_code"] = py_code
                return compiled
            try:
                os.makedirs(os.path.dirname(write_path), exist_ok=True)
                tmp_path = write_path + ".tmp"
                with open(tmp_path, "w") as f:
                    f.write(py_code)
                try:
                    os.replace(tmp_path, write_path)
                except OSError:
                    # sfs_turbo mounts reject rename(2) (92B startup line,
                    # 2026-10-13), and VLLM_CACHE_ROOT points at SFS so the
                    # cache survives container recycling. Writers of one
                    # key are serialized within a rank, so a direct write
                    # is a safe fallback where atomic replace is not.
                    with open(write_path, "w") as f:
                        f.write(py_code)
                    os.remove(tmp_path)
                state["saved"] = True
                # Keep the exact bytes for the optional pool upload: uploading
                # from memory means a torn SFS direct-write can never be
                # published to the pool (review M2).
                state["py_code"] = py_code
                logger.info(
                    "Saved npugraph_ex compiled graph code to %s "
                    "(%d event tags namespaced)",
                    write_path,
                    namespaced,
                )
            except OSError as exc:
                # A cache write failure must never break compilation.
                state["skip_reason"] = f"artifact write failed: {exc}"
            return compiled

        return patched_get_compiled_gm

    for modname in (
        "npugraph_ex.npu_fx_compiler",
        "torch_npu.dynamo.npugraph_ex.npu_fx_compiler",
    ):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue
        cls = getattr(module, "_NpuFxCompiler", None)
        if cls is None or not callable(
            getattr(cls, "_get_compiled_gm", None)
        ):
            continue
        orig = cls._get_compiled_gm
        if getattr(orig, "_omni_npugraph_cache_patched", False):
            continue
        wrapped = _make_patched(orig)
        wrapped._omni_npugraph_cache_patched = True  # type: ignore[attr-defined]
        cls._get_compiled_gm = wrapped
        patched.append((cls, orig))

    if not patched:
        state["skip_reason"] = "no npu_fx_compiler instance patchable"

    def restore() -> None:
        for cls, orig in patched:
            cls._get_compiled_gm = orig

    return restore, state


def _pool_blocks_of(example_inputs):
    """(args index, block count) of the KV pool tensor among the inputs.

    The pool tensor dominates every other input by numel (~N*5*1024 vs the
    4-8k of hidden states); its largest dim is the KV block count N. Units
    that never touch the pool return (None, None): their digests contain no
    N anywhere, so nothing needs the treatment.
    """
    best_idx, best_numel = None, 0
    for idx, item in enumerate(example_inputs):
        numel_fn = getattr(item, "numel", None)
        if numel_fn is None or not callable(numel_fn):
            continue
        try:
            numel = int(numel_fn())
        except Exception:  # noqa: BLE001
            continue
        if numel >= (1 << 20) and numel > best_numel:
            best_idx, best_numel = idx, numel
    if best_idx is None:
        return None, None
    try:
        blocks = max(int(d) for d in example_inputs[best_idx].shape)
    except Exception:  # noqa: BLE001
        return best_idx, None
    return best_idx, blocks


def _pool_input_indices(example_inputs):
    """Indices of every input whose shape carries the KV block count N.

    N comes from the dominant pool tensor (K/V, ~N*5*1024 elements); every
    input whose shape CONTAINS that value participates -- including 1-D
    block-table arrays of length N whose numel falls far below any size
    heuristic. v45b evidence: a 920-site rewrite over the pool-sized
    inputs still left one such small guard stale (length-N array, numel
    6633 << 2^20) and the engine asserted at capture -- "expected size
    6632==6633". Units with no pool tensor return an empty list.
    """
    _, blocks = _pool_blocks_of(example_inputs)
    if blocks is None:
        return []
    indices = []
    for idx, item in enumerate(example_inputs):
        shape = getattr(item, "shape", None)
        if shape is None:
            continue
        try:
            if any(int(d) == blocks for d in shape):
                indices.append(idx)
        except Exception:  # noqa: BLE001 — symbolic dims
            continue
    return indices


def _rewrite_pool_guards(py_code, example_inputs, indices):
    """Rewrite every pool guard's shape/stride literals to the live input's.

    N reaches the artifact only through the first-run assert_size_stride
    guards (the op sequence is N-free); cross-N reuse rewrites them from
    the CURRENT tensors -- exact for any layout. Every occurrence for
    every pool index is rewritten, and two fail-open checks keep a stale
    guard from ever loading: v44's count=1 rewrite left the second pool
    tensor's guard stale and the engine asserted at capture time
    ("expected size 6633==6632") -- loud, but boot-breaking.
    """
    for pos, idx in enumerate(indices):
        tensor = example_inputs[idx]
        try:
            shape = tuple(int(d) for d in tensor.shape)
            stride = tuple(int(d) for d in tensor.stride())
        except Exception:  # noqa: BLE001 — symbolic dims: cannot pin literals
            return None
        literal = f"{shape}, {stride}"
        pattern = re.compile(
            r"(assert_size_stride\(args\[" + str(idx) + r"\], )\([^)]*\), \([^)]*\)"
        )
        first_match = pattern.search(py_code)
        py_code, count = pattern.subn(r"\g<1>" + literal, py_code)
        if not count and pos == 0:
            return None  # the primary pool input has no guard: unexpected shape
        # Cross-check with a whitespace-tolerant count: a guard whose format
        # drifted out of the strict pattern is invisible to BOTH the rewrite
        # and the residual check below -- the counts must agree or the whole
        # artifact is a miss (fail-open, never a stale guard).
        tolerant = re.findall(
            r"assert_size_stride\s*\(\s*args\[" + str(idx) + r"\]", py_code
        )
        if len(tolerant) != count:
            return None
        if first_match is not None:
            old_shape = first_match.group(0).split("(", 2)[2].split(")")[0]
            if old_shape != ", ".join(str(d) for d in shape):
                logger.info(
                    "npugraph_cache: pool guard rewritten args[%d] (%s) -> %s"
                    " (%d site(s))",
                    idx,
                    old_shape,
                    shape,
                    count,
                )
    # Residual: every pool guard must now carry the live shape literal --
    # anything left stale would assert at capture time.
    for idx in indices:
        shape = tuple(int(d) for d in example_inputs[idx].shape)
        for match in re.finditer(
            r"assert_size_stride\(args\[" + str(idx) + r"\], \(([^)]*)\)", py_code
        ):
            try:
                found = tuple(
                    int(x) for x in match.group(1).split(",") if x.strip()
                )
            except ValueError:
                return None
            if found != shape:
                return None
    return py_code


def _load_pooled_unit(pool, cache_dir, name, example_inputs):
    """Serve one unit from the pool, or None for a miss.

    The single pool-hit path shared by compile() and load(). The local file
    is never consulted (the pool is sole-source). Every failure degrades to
    a miss the caller turns into a recompile: a fetch error, a pool guard
    this boot cannot adapt (never load a stale guard -- it would assert at
    capture time) and bytes that do not rebuild. Only the last one schedules
    a heal, because only proven-bad bytes justify overwriting the object;
    an un-adaptable guard is a generation mismatch, not a poison marker.
    """
    try:
        py_code = pool.fetch(cache_dir, name)
    except Exception as exc:  # noqa: BLE001 - pool never blocks compile
        logger.warning("npugraph_cache: fetch path failed for %s: %s", name, exc)
        return None
    if py_code is None:
        return None
    try:
        pool_indices = _pool_input_indices(example_inputs)
        if pool_indices:
            rewritten = _rewrite_pool_guards(py_code, example_inputs, pool_indices)
            if rewritten is None:
                logger.warning(
                    "npugraph_cache: pool guard not adaptable for %s "
                    "(pool args=%s); recompiling",
                    name,
                    pool_indices,
                )
                return None
            py_code = rewritten
        elif _pool_blocks_of(example_inputs)[0] is not None:
            # A pool-sized input exists but its N is underivable (symbolic
            # dims): the guards cannot be adapted, and a stale guard asserts
            # at capture time -- never load one (v47 round B).
            logger.warning(
                "npugraph_cache: pool tensor present but N is symbolic for "
                "%s; recompiling",
                name,
            )
            return None
        from npugraph_ex.npu_fx_compiler import (
            _CompiledFxArtifacts,
            _CompiledFxGraph,
        )

        artifacts = _CompiledFxArtifacts()
        artifacts.py_code = py_code
        try:
            compiled_fn = _CompiledFxGraph.load_artifacts(artifacts)
        except Exception as exc:  # noqa: BLE001 - heal channel
            logger.warning(
                "npugraph_cache: pooled artifact unloadable (%s): %s", name, exc
            )
            pool.note_load_failed(cache_dir, name)
            return None
        logger.info("Loaded npugraph_ex compiled graph %s (pool)", name)
        pool.note_resolved()
        return compiled_fn
    except Exception as exc:  # noqa: BLE001 - pool never blocks compile
        logger.warning("npugraph_cache: pooled unit failed for %s: %s", name, exc)
        return None


#: npugraph_ex sync ops whose second argument is a runtime-assigned stream
#: ordinal; only their relative identity is semantic (see _ordinals_normalized).
_TAGGED_STREAM_OPS = (
    "tagged_event_record_on_stream",
    "tagged_event_wait_on_stream",
    "record_tagged_on_stream",
)


def _npugraph_digest(
    graph: fx.GraphModule, example_inputs: list[Any], key: str | None = None
) -> str:
    """Content digest of one compilation unit.

    The vLLM cache key ("artifact_compile_range_..._subgraph_N") names the
    (compile range, subgraph index) slot, not its content: the MTP draft
    and the target model can compile through the same CompilerManager and
    collide on that slot, and runtime-derived shapes (e.g. the KV block
    count from memory profiling, which the framework's config hash never
    sees) are baked into the codegen source. Digesting the graph structure
    plus the example-input shapes makes the artifact content-addressed: a
    collision or a shape drift turns into a clean cache miss instead of a
    wrong-graph load. Computed on the pre-mutation graph on both the save
    and the load path (compile() tuple-ifies the graph in place).

    The structure is canonicalized: nodes are identified by topological
    position, not by dynamo's variable names. Raw graph.code is NOT
    boot-stable — fx5/C1r3-vs-C2r3 produced byte-identical codegen output
    for the same slot yet different raw digests, i.e. the input graphs
    were isomorphic under different names (the framework itself patches
    autograd_cache_normalize_inputs for exactly this reason). Placeholder
    names are dropped: their position binds them to the example inputs.
    get_attr/call_module targets are kept — they identify which parameter
    or submodule a node touches and derive from the model tree, not from
    tracer counters.

    Leaf rendering is process-stable (fx_c2r5 lesson): the run-6 vs run-7
    backbone artifacts proved byte-identical across boots while every
    digest changed, so the residual variance lived in repr(), not in the
    graphs — CPython embeds memory addresses in function/object reprs and
    torch renders symbolic values under runtime-assigned names ("s0",
    "s3", ...). Every leaf therefore goes through _stable_repr: backed
    syms by their hint, unbacked syms by a canonical letter assigned in
    first-occurrence order (rename noise cannot move the digest while
    tied-vs-untied structure stays visible), callables by module.qualname
    (op overloads keep their identity-rich repr), address-bearing reprs
    by type.

    With NPUGRAPH_DIGEST_DUMP set and VLLM_CACHE_ROOT on a persistent volume,
    the exact hashed text is written to npugraph_digest_dump/{key}.{digest}.
    {pid}.txt — a cross-boot diff of these files pinpoints any remaining
    nondeterminism line by line instead of hiding it inside the hash.
    """
    # KV block count N is capacity, not computation: it reaches the digest
    # only through the pool tensor's shape. Normalizing that one value lets
    # artifacts be shared across N generations; the load side rewrites the
    # first-run guards to the current shapes (_rewrite_pool_guards).
    _kv_blocks = _pool_blocks_of(example_inputs)[1]
    sym_letters: dict[str, str] = {}

    def _sym_letter(sym: Any) -> str:
        name = str(sym)
        if name not in sym_letters:
            sym_letters[name] = chr(ord("a") + len(sym_letters))
        return sym_letters[name]

    def _stable_repr(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            dims = []
            for dim in value.shape:
                try:
                    dim_int = int(dim)
                    dims.append(
                        "N" if _kv_blocks is not None and dim_int == _kv_blocks
                        else str(dim_int)
                    )
                except Exception:  # noqa: BLE001 — unbacked symint dim
                    dims.append(_sym_letter(dim))
            return f"tensor:{value.dtype}:{','.join(dims)}"
        if isinstance(value, torch.SymInt):
            try:
                return f"symint:{int(value)}"
            except Exception:  # noqa: BLE001 — unbacked
                return f"symint:${_sym_letter(value)}"
        if isinstance(value, torch.SymFloat):
            try:
                return f"symfloat:{float(value)}"
            except Exception:  # noqa: BLE001 — unbacked
                return f"symfloat:${_sym_letter(value)}"
        if isinstance(value, torch.SymBool):
            try:
                return f"symbool:{bool(value)}"
            except Exception:  # noqa: BLE001 — unbacked
                return f"symbool:${_sym_letter(value)}"
        if isinstance(value, str):
            # npugraph_ex tag strings ("graph_4_get_external_object_by_
            # index_2", emitted by replace_stream_event/graph_pass) embed a
            # per-boot build counter: it counts the graphs BUILT before this
            # one, so a boot that loads cached units instead of building
            # them shifts every later tag (fx seven-run: eagle tags moved
            # graph_4/8/12 -> graph_1/4/7, by exactly the loaded-unit
            # count). Only the suffix identifies the object across boots.
            return repr(re.sub(r"^graph_\d+_", "graph*_", value))
        if isinstance(
            value, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
        ):
            # An OpOverload repr spells out both the operator and its
            # overload, so it is stable across processes and keeps operator
            # identity — do not trade it for a name.
            return repr(value)
        if callable(value):
            module = getattr(value, "__module__", None) or ""
            qualname = getattr(value, "__qualname__", None)
            if qualname:
                # Plain functions repr as "<function f at 0x...>" — the
                # address is per-process; source identity is the stable
                # counterpart.
                return f"callable:{module}.{qualname}"
        rendered = repr(value)
        if " at 0x" in rendered:
            # Any other address-bearing repr is process-scoped by
            # definition; the type is the stable proxy.
            cls = type(value)
            return f"obj:{cls.__module__}.{cls.__qualname__}"
        return rendered

    node_index = {node.name: i for i, node in enumerate(graph.graph.nodes)}

    def _canon(value: Any) -> str:
        if isinstance(value, fx.Node):
            return f"n{node_index[value.name]}"
        if isinstance(value, (tuple, list)):
            open_tok, close_tok = (
                ("t(", ")") if isinstance(value, tuple) else ("l(", ")")
            )
            return open_tok + ",".join(_canon(v) for v in value) + close_tok
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: _stable_repr(kv[0]))
            return "d{" + ",".join(
                f"{_stable_repr(k)}:{_canon(v)}" for k, v in items
            ) + "}"
        return _stable_repr(value)

    lines: list[str] = []
    # Runtime-assigned ordinals (NPU stream ids) leak into the args of the
    # npugraph_ex sync ops, and only their RELATIVE identity is semantic
    # (which ops share a stream): the absolute number depends on how many
    # streams were created before the model's — a co-resident transfer
    # engine shifts it by one and churned the whole digest generation
    # (v32 forensics: 258 tagged-op args '112'->'113' between the weight
    # cold-load and pool-hit boots, nothing else). Normalize the known
    # ordinal slots by first occurrence, the same relative-identity trick
    # as _sym_letter. Fail-open by construction: a signature drift simply
    # skips normalization, which costs a clean miss — never a wrong hit.
    stream_ords: dict[str, str] = {}

    def _stream_token(raw: str) -> str:
        if raw not in stream_ords:
            stream_ords[raw] = f"s{len(stream_ords)}"
        return stream_ords[raw]

    def _ordinals_normalized(target: str, args: tuple) -> tuple:
        if "npugraph_ex.scope_enter" in target:
            if len(args) > 1 and isinstance(args[1], (list, tuple)) and len(args[1]) > 2:
                raw = args[1][2]  # (label, priority, STREAM, device, type)
                if isinstance(raw, str) and raw.isdigit():
                    values = list(args[1])
                    values[2] = _stream_token(raw)
                    return (*args[:1], type(args[1])(values), *args[2:])
            return args
        if "npugraph_ex." in target and any(
            op in target for op in _TAGGED_STREAM_OPS
        ):
            if len(args) > 1:
                raw = args[1]  # (tag|node, STREAM, device, type[, flag])
                if isinstance(raw, str) and raw.isdigit():
                    return (*args[:1], _stream_token(raw), *args[2:])
        return args

    for node in graph.graph.nodes:
        target = (
            ""
            if node.op in ("placeholder", "output")
            else _stable_repr(node.target)
        )
        args = _ordinals_normalized(target, node.args)
        lines.append(f"{node.op}|{target}|{_canon(args)}|{_canon(node.kwargs)}")
    for item in example_inputs:
        lines.append(_stable_repr(item))
    text = "\n".join(lines) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]

    if key is not None and os.environ.get("NPUGRAPH_DIGEST_DUMP"):
        try:
            cache_root = os.environ.get("VLLM_CACHE_ROOT", "")
            if cache_root:
                dump_dir = os.path.join(cache_root, "npugraph_digest_dump")
                os.makedirs(dump_dir, exist_ok=True)
                with open(
                    os.path.join(dump_dir, f"{key}.{digest}.{os.getpid()}.txt"),
                    "w",
                ) as f:
                    f.write(text)
        except Exception:  # noqa: BLE001 — diagnostics never break compile
            pass
    return digest


def graph_output_is_tuple(graph: fx.GraphModule) -> bool:
    """Return whether the FX graph output is already a tuple."""
    output_node = graph.graph.output_node()
    if not output_node.args:
        return False

    return_value = output_node.args[0]
    if isinstance(return_value, tuple):
        return True
    return (
        isinstance(return_value, fx.Node)
        and return_value.op == "call_function"
        and return_value.target is tuple
    )


@contextlib.contextmanager
def _tracing_context_matching_inputs(example_inputs: list[Any]):
    """Align the ambient TracingContext with the fake mode carried by the inputs.

    For a single-size compile range, vLLM's piecewise backend builds the example
    inputs under a freshly created FakeTensorMode (create_concrete_args), while
    Dynamo's TracingContext still holds the mode it traced the graph with. Any
    detect_fake_mode() call inside the npugraph_ex backend then returns the traced
    mode, which the example inputs were never allocated under, and the mismatch
    surfaces as a failure deeper in the backend. vLLM solves the same problem on
    the Inductor path by making standalone_compile reuse the mode attached to the
    inputs; do the equivalent here by swapping in a TracingContext built on that
    mode for the duration of the backend call.
    """
    from torch._guards import TracingContext, tracing
    from torch._subclasses.fake_tensor import FakeTensor

    input_fake_mode = None
    for x in example_inputs:
        if isinstance(x, FakeTensor):
            input_fake_mode = x.fake_mode
            break

    current = TracingContext.try_get()
    if input_fake_mode is None or (
        current is not None and current.fake_mode is input_fake_mode
    ):
        yield
        return

    with tracing(TracingContext(input_fake_mode)):
        yield


class NpuGraphExAdaptor(CompilerInterface):
    name = "npugraph_ex"

    def __init__(self) -> None:
        # Set by initialize_cache(); disable_cache defaults to True so a
        # boot that never initializes the cache (or keeps
        # VLLM_DISABLE_COMPILE_CACHE=1) behaves exactly as before this
        # cache existed.
        self.cache_dir: str | None = None
        self.disable_cache: bool = True

    def compute_hash(self, vllm_config: Any) -> str:
        """Compiler-specific factors for the framework's cache-dir hash.

        The framework already hashes the vllm config (model, parallelism,
        compilation/speculative configs) and the content of every traced
        file; add only what changes the npugraph_ex artifact itself. The
        build fingerprint folds the codegen sources in, so a patched build
        gets a new cache-dir snapshot — isolating every artifact store
        keyed on this directory (the pool bundle reuses the snapshot).
        """
        factors: dict[str, Any] = {"backend": "npugraph_ex"}
        try:
            import torch_npu

            factors["torch_npu_version"] = torch_npu.__version__
            factors["cann_version"] = str(
                torch_npu.npu.utils.get_cann_version()
            )
        except Exception as exc:  # noqa: BLE001
            factors["torch_npu_version"] = f"unavailable:{exc}"
        factors["build"] = _engine_build_fingerprint()
        npugraph_ex_config = _get_npugraph_ex_config()
        if npugraph_ex_config is not None:
            # Every knob forwarded to CompilerConfig changes the compiled
            # artifact (capture_limit, clone/inplace passes, static/super
            # kernel switches, and the fx cache gate itself).
            factors["npugraph_ex_config"] = dict(
                sorted(npugraph_ex_config.items())
            )
        logger.debug("NpuGraphExAdaptor hash factors: %s", factors)
        return hashlib.sha256(
            str(sorted(factors.items())).encode()
        ).hexdigest()[:10]

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Any = None,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:

        from omni_npu.compilation.npugraph_ex_config import get_aclgraph_config
        npugraph_ex_config = get_aclgraph_config().npugraph_ex_config
        if not npugraph_ex_config.get("enable", False):
            return graph, None

        # Must run before the first capture even when every graph is
        # served from the py_code cache (a full hit never re-enters
        # compile(); load() runs the same setup).
        _ensure_static_kernel_runtime_setup(npugraph_ex_config)
        pool = _npugraph_cache_pool()

        # py_code cache: digest the graph BEFORE the in-place tuple fix
        # below rewrites its output, so save and load see the same
        # pre-mutation content. No config key gates this cache: the switch
        # is the framework's own VLLM_DISABLE_COMPILE_CACHE
        # (initialize_cache -> disable_cache), and adding a key to
        # npugraph_ex_config would shift the static-kernel pool fingerprint
        # (it hashes the whole additional_config).
        cache_path: str | None = None
        if (
            key is not None
            and getattr(self, "cache_dir", None)
            and not getattr(self, "disable_cache", True)
        ):
            digest = _npugraph_digest(graph, example_inputs, key)
            # Unconditional canon: an exception inside the pass logs FAIL and
            # returns None, which keeps the raw digest (automatic fail-safe).
            canon = _log_canon("compile", key, graph, example_inputs, digest)
            if canon:
                digest = canon
            cache_path = os.path.join(
                self.cache_dir, f"{key}.{digest}{_NPUGRAPH_CACHE_SUFFIX}"
            )
            # The framework only calls load() when the slot index already
            # exists, so a brand-new node (empty cache root) reaches compile()
            # for every unit -- the pool read must live here as well as in
            # load() (review H1). The pool is always on and sole-source:
            # the local file is never consulted, every unit comes from the
            # pool or a fresh compile.
            if pool is not None:
                name = os.path.basename(cache_path)
                compiled_fn = _load_pooled_unit(
                    pool, self.cache_dir, name, example_inputs
                )
                if compiled_fn is not None:
                    return compiled_fn, (key, cache_path)

        from torch._inductor.compile_fx import graph_returns_tuple
        fx_graph = graph.graph
        if not graph_returns_tuple(graph) and not graph_output_is_tuple(graph):
            output_node = fx_graph.output_node()
            return_value = output_node.args[0]
            with fx_graph.inserting_before(output_node):
                tuple_node = fx_graph.create_node("call_function", tuple, args=([return_value], ))
            output_node.args = (tuple_node, )
            graph.recompile()

        import npugraph_ex
        config = npugraph_ex.CompilerConfig()
        config.mode = "npugraph_ex"
        # execute FX graph in eager mode before graph mode to optimize FX graph.
        config.force_eager = True
        # static kernel switch, suitable for static shapes or scenes with less shape changes.
        if npugraph_ex_config.get(
            "static_kernel_compile", False
        ) or npugraph_ex_config.get("super_kernel_optimize", False):
            # Static-kernel runtime setup (patch application + compile
            # concurrency tuning) runs in _ensure_static_kernel_runtime_setup
            # above, shared with the py_code load() path.
            config.static_kernel_compile = True
            # Control whether to enable super kernel optimize
            if npugraph_ex_config.get("super_kernel_optimize", False):
                config.super_kernel_optimize = True
                config.super_kernel_optimize_options = {
                    "dcci_before_kernel_start": [
                        ".*GroupedMatmul.*",
                        ".*AiInfraSparseFlashAttentionPioneer.*",
                        ".*AiInfraKvRmsNormRopeCache.*",
                        ".*AiInfraScatterBlockUpdate.*",
                        ".*AiInfraFusedInferAttentionSink.*"
                    ],
                    "dcci_disable_on_kernel": [".*"]
                }
        # Acquisition count limit configuration
        if npugraph_ex_config.get("capture_limit", 64) != 64:
            config.debug.aclgraph.static_capture_size_limit = npugraph_ex_config.get("capture_limit")
        # Control whether to clone inputs
        if not npugraph_ex_config.get("clone_input", True):
            config.clone_input = False
        # Control whether to clone outputs
        if npugraph_ex_config.get("clone_output", False):
            config.clone_output = True
        # Redundant operator elimination configuration
        if not npugraph_ex_config.get("remove_noop_ops", True):
            config.remove_noop_ops = False
        # Replace non-in-place operators in the middle of FX graph with in-place operators
        if not npugraph_ex_config.get("inplace_pass", True):
            config.inplace_pass = False
        # Convert operators that were converted to non-in-place operations by dynamo Functionalize in inputs back to
        # in-place operators
        if not npugraph_ex_config.get("input_inplace_pass", True):
            config.input_inplace_pass = False
        # Configuration for enabling operator fusion Pass in FX graph
        if not npugraph_ex_config.get("pattern_fusion_pass", True):
            config.pattern_fusion_pass = False
        # Freeze model weights and computation graph to prevent dynamic changes
        if npugraph_ex_config.get("frozen_parameter", False):
            config.frozen_parameter = True
        # insert the GraphPassManager into the npugraph_ex  config
        if "post_grad_custom_post_pass" in compiler_config:
            config.post_grad_custom_post_pass = compiler_config["post_grad_custom_post_pass"]
        if "post_grad_custom_pre_pass" in compiler_config:
            config.post_grad_custom_pre_pass = compiler_config["post_grad_custom_pre_pass"]

        logger.debug(f"static_kernel_compile: {config.static_kernel_compile}")
        logger.debug(f"super_kernel_optimize: {config.super_kernel_optimize}")
        logger.debug(f"capture_limit: {config.capture_limit}")
        logger.debug(f"clone_input: {config.clone_input}")
        logger.debug(f"clone_output: {config.clone_output}")
        logger.debug(f"remove_noop_ops: {config.remove_noop_ops}")
        logger.debug(f"inplace_pass: {config.inplace_pass}")
        logger.debug(f"input_inplace_pass: {config.input_inplace_pass}")
        logger.debug(f"pattern_fusion_pass: {config.pattern_fusion_pass}")
        logger.debug(f"frozen_parameter: {config.frozen_parameter}")
        npugraph_ex_compile = npugraph_ex.get_npu_backend(compiler_config=config)

        restore_save_hook = None
        save_state = None
        if cache_path is not None:
            # Disk-free pool mode: nothing is written locally — the save
            # hook only captures the text for the pool upload. The file
            # fallback only covers a deployment whose bridge module failed
            # to import (a broken install), not a supported mode.
            write_path: str | None = None if pool is not None else cache_path
            restore_save_hook, save_state = _install_py_code_save_hook(
                cache_path, write_path
            )
        try:
            with _tracing_context_matching_inputs(example_inputs):
                # torch's AOT precompile flow (aot_compile_fullgraph, engaged
                # when the vLLM compile-cache framework is on) patches
                # bundled_autograd_cache=True / force_autograd_cache=True
                # around this call; under those flags AOTAutogradCache
                # .make_entry asserts on this backend's artifact (expected
                # OutputCode, got _CompiledFxGraph). The vendored artifact is
                # not AOT-cache-serializable, so opt out for the duration of
                # the call — the same remedy the framework applies for its
                # own InductorAdaptor (compiler_interface patches
                # enable_autograd_cache=False with that rationale). Our
                # persistence layer is the py_code cache, not torch's AOT
                # cache.
                with torch._functorch.config.patch(
                    enable_autograd_cache=False,
                    enable_remote_autograd_cache=False,
                    bundled_autograd_cache=False,
                    force_autograd_cache=False,
                ):
                    compile_graph = npugraph_ex_compile(graph, example_inputs)
        finally:
            if restore_save_hook is not None:
                restore_save_hook()
        if save_state is not None and save_state.get("skip_reason"):
            logger.info(
                "npugraph_ex py_code cache skipped for %s: %s",
                key,
                save_state["skip_reason"],
            )
        # Optional pool upload: the exact bytes that were captured in
        # memory (review M2), so a torn SFS file can never be published.
        if pool is not None and cache_path is not None:
            name = os.path.basename(cache_path)
            saved_code = None if save_state is None else save_state.get("py_code")
            if save_state is not None and save_state.get("saved") and saved_code:
                try:
                    pool.schedule_upsert(self.cache_dir, name, saved_code.encode())
                except Exception as exc:  # noqa: BLE001 - never blocks compile
                    logger.warning(
                        "npugraph_cache: schedule upload failed for %s: %s", name, exc
                    )
            pool.note_resolved()
        # Return the handle even when the artifact was skipped: load()
        # verifies file existence, so an absent artifact is a clean
        # per-graph miss that recompiles — never a wrong load.
        if cache_path is not None:
            return compile_graph, (key, cache_path)
        return compile_graph, None

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Any,
    ) -> Callable[..., Any] | None:
        """Serve one compilation unit from the py_code cache.

        Never raises: any miss or failure returns None, letting the
        framework fall through to compile() (which re-saves the artifact
        under the current digest, keeping the handle table self-healing).
        """
        npugraph_ex_config = _get_npugraph_ex_config() or {}
        if not (
            npugraph_ex_config.get("enable", False)
            # No fx-level key: the framework cache (VLLM_DISABLE_COMPILE_CACHE
            # -> disable_cache) is the gate — see the matching gate in
            # compile(); an extra config key would shift the static-kernel
            # pool fingerprint for sk-enabled boots.
            and getattr(self, "cache_dir", None)
            and not getattr(self, "disable_cache", True)
        ):
            return None
        # A full cache hit never enters compile(); the static-kernel
        # setup must still run before the first capture (idempotent with
        # the compile()-side call on partial hits).
        _ensure_static_kernel_runtime_setup(npugraph_ex_config)

        key = handle[0] if isinstance(handle, tuple) and handle else None
        if not isinstance(key, str) or not key:
            return None
        # The stored handle points at the digest of the boot that wrote
        # it; recompute from the CURRENT graph so a shape drift or a
        # target/draft slot collision resolves to the right artifact or
        # a clean miss.
        digest = _npugraph_digest(graph, example_inputs, key)
        canon = _log_canon("load", key, graph, example_inputs, digest)
        if canon:
            digest = canon
        path = os.path.join(
            self.cache_dir, f"{key}.{digest}{_NPUGRAPH_CACHE_SUFFIX}"
        )
        pool = _npugraph_cache_pool()
        if pool is not None:
            # Sole source: the local file is never consulted; a pool miss
            # falls straight through to compile().
            name = os.path.basename(path)
            compiled_fn = _load_pooled_unit(
                pool, self.cache_dir, name, example_inputs
            )
            if compiled_fn is None:
                # A miss (or a fetch failure) still resolves this unit: the
                # summary counts units, not hits.
                pool.note_resolved()
            return compiled_fn
        if not os.path.exists(path):
            logger.info(
                "npugraph_ex py_code cache miss for %s (digest %s)",
                key,
                digest,
            )
            return None
        try:
            from npugraph_ex.npu_fx_compiler import (
                _CompiledFxArtifacts,
                _CompiledFxGraph,
            )

            with open(path) as f:
                py_code = f.read()
            artifacts = _CompiledFxArtifacts()
            artifacts.py_code = py_code
            compiled_fn = _CompiledFxGraph.load_artifacts(artifacts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "npugraph_ex py_code cache load failed for %s (%s); "
                "falling back to compile",
                path,
                exc,
            )
            return None
        logger.info("Loaded npugraph_ex compiled graph from %s", path)
        return compiled_fn
