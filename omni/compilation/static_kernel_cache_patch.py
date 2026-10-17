# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Vendored runtime replacement of torch_npu's static-kernel cache gate.

Why this exists: torch_npu 9.1.0's ``compile_static_kernel`` reads the four
cache keys only under ``not (force_eager and static_kernel_compile)``, and the
dotted keys (``static_kernel_compile.use_cache_compile`` etc.) are produced
exclusively by ``ModelCacheSaver.extend_config`` at the CacheBackend API layer
(inference/_cache_compiler.py:419-424) — the CompilerConfig surface cannot
express them: ``static_kernel_compile`` is a plain OptionValue bool
(configs/compiler_config.py:32) and its ``__setattr__`` rejects unknown
attributes (_option_base.py:220).

We therefore shadow ``compile_static_kernel`` with a vendored entry that
routes straight into ``_install_or_compile_static_kernel`` with the four cache
parameters — exactly what the official cache_compile path would have done —
while keeping ``force_eager=True`` so the backend still returns the eager
callable that omni's ACLGraphWrapper captures.

Two structural hazards of runtime-patching this module, both handled here:

1. The static_kernel file is loaded as TWO separate module instances from
   the same file: ``npugraph_ex._acl_concrete_graph.static_kernel`` (the one
   the runtime actually uses — ``_acl_concrete_graph/acl_graph.py`` does
   ``from npugraph_ex._acl_concrete_graph.static_kernel import
   compile_static_kernel`` at import time and calls the bound name at
   lines 937/1634) and ``torch_npu.dynamo.npugraph_ex._acl_concrete_graph
   .static_kernel`` (the one a naive patch imports). Patching one instance
   never reaches the other (``is``-identity differs despite identical
   ``__file__``). We patch EVERY loaded instance and give each a closure
   routed to its OWN helper functions so module-local state (Gloo groups,
   dump state) stays consistent.
2. from-import copies: replacing the module attribute does not rebind names
   already imported into importer namespaces. After patching the instances,
   we rescan ``sys.modules`` and rebind every ``compile_static_kernel``
   binding whose function was defined in one of those instances.

Safety: applied at most once per process, only when ``static_kernel_cache_enable`` is
set; every torch_npu symbol is checked before use — any mismatch logs a
warning and leaves the original function in place (recompile every boot,
never wrong results).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_MARKER = "_omni_static_kernel_cache_patched"
# Seed imports, not name matching: the anchor
# (npugraph_ex._acl_concrete_graph.static_kernel) is the copy the runtime
# executes and must be patched; the second seed MATERIALIZES the
# torch_npu.dynamo copy at patch time, so it cannot slip in later as an
# unpatched instance. Actual discovery is file-identity based
# (_find_instances) and needs none of these names.
_SK_NAMES = (
    "npugraph_ex._acl_concrete_graph.static_kernel",
    "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.static_kernel",
)

# Boot-static reuse decision, shared by every patched instance in this
# process. Decided once on the first compile unit and frozen for the boot:
#
#   cache present at first call -> every unit takes the coarse install path
#   cache absent  at first call -> every unit delegates to the ORIGINAL
#       compile entry (full compile; torch_npu's compile branch still does
#       its per-key json bookkeeping).
#
# Without freezing, a genuine miss boot self-short-circuits: unit 1 writes
# its .run, unit 2's coarse scan finds it, installs 1 package and skips
# the remaining units' compilation.
#
# "cache present" additionally requires the pool-restore marker
# (_POOL_MARKER below): only THIS boot's gate-passed pool pull writes it
# (fingerprint-checked manifest, staged restore). Ungated local-disk reuse
# is thereby disabled — stale products from a previous boot or hand-staged
# files never satisfy the decision; every non-pool boot is a full compile.
_POOL_MARKER = ".static_kernel_cache_pool_restored"

_boot_mode: dict = {"decided": False, "use_cache": False}


def _cache_present(cache_dir: str) -> bool:
    import glob

    outputs = os.path.join(cache_dir, "static_kernel_compile_outputs")
    if not os.path.exists(os.path.join(outputs, _POOL_MARKER)):
        return False
    return bool(
        glob.glob(os.path.join(outputs, "*_outputs", "*.run"))
    )


def _make_vendored(sk: Any, cache_dir: str):
    """Build a cache-enabled replacement bound to one module instance.

    ``original`` is captured before the module attribute is overwritten so
    the disable-cache escape hatch still reaches the real implementation
    (calling ``sk.compile_static_kernel`` here would recurse into the
    vendored closure itself).
    """
    original = sk.compile_static_kernel

    def compile_static_kernel(fx_func, *args, config=None, build_dir=None, **kwargs):
        cfg = config or {}
        if not sk._is_multicard_env_valid():
            return
        if cfg.get("disable_static_kernel_compile_cache", False):
            original(fx_func, *args, config=config, build_dir=build_dir, **kwargs)
            return
        if not _boot_mode["decided"]:
            _boot_mode["use_cache"] = _cache_present(cache_dir)
            _boot_mode["decided"] = True
            logger.info(
                "static_kernel_cache boot mode: %s (cache_dir=%s)",
                "cache-reuse" if _boot_mode["use_cache"] else "full-compile",
                cache_dir,
            )
        if not _boot_mode["use_cache"]:
            original(fx_func, *args, config=config, build_dir=build_dir, **kwargs)
            return
        import torch
        import torch_npu

        sk._install_or_compile_static_kernel(
            fx_func,
            *args,
            cached_cann_version=torch_npu.npu.utils.get_cann_version(),
            compile_cache_dir=cache_dir,
            cached_deterministic=torch.are_deterministic_algorithms_enabled(),
            config=config,
            **kwargs,
        )

    setattr(compile_static_kernel, _MARKER, True)
    return compile_static_kernel


def _modules_sharing_file(target: str) -> list[Any]:
    """Return every loaded module whose __file__ resolves to ``target``.

    Pure function (CPU-testable on ut-runner). The same physical file can be
    loaded under several module names (npugraph_ex ships at least twice: the
    runtime copy and a torch_npu.dynamo copy, same ``__file__``, distinct
    module objects). Name-based enumeration couples us to those
    package-internal paths; file identity does not — every loaded module
    born from the anchor's file is collected, whatever it is called.
    Builtins and namespace packages carry no ``__file__`` and are skipped by
    the getattr guard.
    """
    resolved = os.path.realpath(target)
    found: list[Any] = []
    seen: set[int] = set()
    for mod in list(sys.modules.values()):
        path = getattr(mod, "__file__", None)
        if not path or os.path.realpath(path) != resolved:
            continue
        if id(mod) not in seen:
            seen.add(id(mod))
            found.append(mod)
    return found


def _find_instances() -> list[Any]:
    """Collect every loaded static_kernel module instance.

    Seeds are IMPORTED first (materialization semantics preserved — a copy
    that only loads after the patch ran would otherwise come up unpatched);
    discovery then collects everything sharing the anchor's file, so
    package-rename or third-copy layouts no longer need list maintenance.
    """
    import importlib

    preloaded = {name for name in _SK_NAMES if name in sys.modules}
    for name in _SK_NAMES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
    anchor = sys.modules.get(_SK_NAMES[0])
    if anchor is None or getattr(anchor, "__file__", None) is None:
        return []
    instances = _modules_sharing_file(anchor.__file__)
    logger.info(
        "static_kernel_cache instances by file identity: %s (pre-loaded seeds: %s)",
        sorted(m.__name__ for m in instances),
        sorted(preloaded) or "none",
    )
    return instances


def apply_static_kernel_cache_patch(npugraph_ex_config: dict[str, Any]) -> bool:
    """Shadow torch_npu's static-kernel cache gate for this process.

    Returns True when the vendored entry is (or already was) in place on
    every discovered instance.
    """
    cache_dir = npugraph_ex_config.get("static_kernel_cache_dir") or os.getcwd()
    instances = _find_instances()
    if not instances:
        logger.warning(
            "static_kernel_cache_enable: cannot import any of %s; falling back to "
            "recompile-every-boot",
            ", ".join(_SK_NAMES),
        )
        return False

    def _has_expected_layout(sk: Any) -> bool:
        expected_symbols = (
            "compile_static_kernel",
            "_install_or_compile_static_kernel",
            "_is_multicard_env_valid",
        )
        return all(callable(getattr(sk, name, None)) for name in expected_symbols)

    valid = [sk for sk in instances if _has_expected_layout(sk)]
    if not valid:
        logger.warning(
            "static_kernel_cache_enable: torch_npu static_kernel module does not match "
            "the expected layout (compile_static_kernel / "
            "_install_or_compile_static_kernel / _is_multicard_env_valid); "
            "falling back to recompile-every-boot"
        )
        return False

    # Idempotency: every valid instance already carries a vendored entry.
    if all(
        getattr(getattr(sk, "compile_static_kernel"), _MARKER, False)
        for sk in valid
    ):
        return True

    instance_globals = {id(sk.__dict__): sk for sk in valid}
    rebound = 0
    for sk in valid:
        vendored = _make_vendored(sk, cache_dir)
        sk.compile_static_kernel = vendored
        # Rebind from-import copies in importer namespaces (e.g.
        # npugraph_ex._acl_concrete_graph.acl_graph line 26). A binding is
        # ours when the bound function was defined inside that instance's
        # file — checked via its __globals__ identity, which the vendored
        # closure never matches (it is defined in this file).
        for mod in list(sys.modules.values()):
            ns = getattr(mod, "__dict__", None)
            if not ns:
                continue
            cur = ns.get("compile_static_kernel")
            if not callable(cur) or getattr(cur, _MARKER, False):
                continue
            if id(getattr(cur, "__globals__", None)) in instance_globals:
                ns["compile_static_kernel"] = _make_vendored(sk, cache_dir)
                rebound += 1

    logger.info(
        "static kernel compile cache enabled via vendored patch "
        "(compile_cache_dir=%s, instances=%d, rebinds=%d)",
        cache_dir,
        len(valid),
        rebound,
    )
    return True
