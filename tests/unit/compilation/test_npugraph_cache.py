# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for the npugraph_ex py_code cache layer.

The cache is addressed by a content digest of the FX graph plus the example
inputs, the compiled source is persisted as ``{key}.{digest}.ngc`` and the
optional ``npugraph_cache`` bridge is the sole-source store for those bytes.
These tests lock the invariants that make that safe:

* the digest identifies the graph, not the tracer's incidental naming: a
  shape drift, a dtype drift or a different op is a clean miss, while
  placeholder names, tag counters and runtime stream ordinals -- which move
  every boot -- are normalized away;
* the canonicalization pass used for digesting is isolated: it runs on a
  copy, restores ``GraphCounter`` and the event-creation hooks even when it
  raises, and a failure degrades to the raw digest instead of failing the
  boot;
* artifact bytes are captured from the codegen in memory, get a
  digest-unique event-tag namespace, and are written atomically (with the
  SFS direct-write fallback); no cache accident may break a compile;
* the framework cache switch is the only gate: an uninitialized adaptor
  behaves exactly as before, and both ``compile()`` and ``load()`` keep the
  pool optional, never consult a local file when the bridge exists, and
  turn every pool failure into a plain miss;
* once-per-process static-kernel setup runs on the ``load()`` hit path too,
  because a full hit never re-enters ``compile()``.

The npu-facing modules (``npugraph_ex``, its ``_acl_concrete_graph`` pass
modules and ``npu_fx_compiler``) and the pool bridge are injected through
``sys.modules`` so the file runs on CPU with no NPU and no pool installed.
"""
import hashlib
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.fx as fx

import omni_npu.compilation.npugraph_ex as nx
from omni_npu.compilation import npugraph_ex_config as cfg_module

_SUFFIX = ".ngc"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _graph_module(build: "callable") -> fx.GraphModule:
    """Build a real GraphModule from a graph-construction callback."""
    graph = fx.Graph()
    build(graph)
    return fx.GraphModule(torch.nn.Module(), graph)


class _GraphHolder:
    """Stand-in for GraphModule when node targets are synthetic.

    Both ``_npugraph_digest`` and ``_graph_flavor`` only read ``graph.graph``,
    and synthetic targets cannot survive ``GraphModule.recompile``.
    """

    def __init__(self, graph: fx.Graph) -> None:
        self.graph = graph


class _SyncOp:
    """Callable whose repr matches the qualified name npugraph_ex emits."""

    def __init__(self, qualified_name: str) -> None:
        self.__name__ = qualified_name.rsplit(".", 1)[-1]
        self._qualified_name = qualified_name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self._qualified_name

    def __call__(self, *args, **kwargs):  # pragma: no cover - never run
        raise AssertionError("synthetic graph node must not be executed")


def _sym_env():
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    return ShapeEnv()


def _backed_sym(value, hint):
    """A backed SymInt/SymFloat/SymBool leaf (torch renders these by hint)."""
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    env = ShapeEnv()
    sym = env.create_symbol(value, ConstantSource("test"))
    if isinstance(value, bool):
        return env.create_symboolnode(sym)
    if isinstance(value, float):
        return env.create_symfloatnode(sym, hint=hint)
    return env.create_symintnode(sym, hint=hint)


def _add_graph(placeholder_names=("x", "y"), op=torch.ops.aten.add.Tensor):
    """(x op y) -> tuple, with configurable placeholder names."""

    def _build(graph):
        left = graph.placeholder(placeholder_names[0])
        right = graph.placeholder(placeholder_names[1])
        out = graph.call_function(op, (left, right))
        graph.output((out,))

    return _graph_module(_build)


def _add_graph_single_output():
    """(x + y) returned bare, so compile() has to tuple-ify it."""

    def _build(graph):
        left = graph.placeholder("x")
        right = graph.placeholder("y")
        out = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
        graph.output(out)

    return _graph_module(_build)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Reset process-wide latches and never reach a real pool bridge."""
    monkeypatch.setattr(nx, "_config_unreachable_warned", False)
    monkeypatch.setattr(nx, "_npugraph_cache_pool_registered", False)
    monkeypatch.setattr(nx, "_npugraph_cache_pool_unreachable_warned", False)
    monkeypatch.setattr(nx, "_engine_build_hash", None)
    monkeypatch.setitem(sys.modules, "npugraph_cache", None)
    monkeypatch.setitem(sys.modules, "npugraph_cache.npugraph_cache_store", None)


@pytest.fixture
def logger(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(nx, "logger", fake)
    return fake


def _set_config(monkeypatch, cfg):
    monkeypatch.setattr(
        cfg_module,
        "get_aclgraph_config",
        lambda: types.SimpleNamespace(npugraph_ex_config=cfg),
    )


class _Pool:
    """Programmable double for the npugraph_cache bridge."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fetches: list[tuple[str, str]] = []
        self.load_failed: list[tuple[str, str]] = []
        self.upserts: list[tuple[str, str, bytes]] = []
        self.resolved = 0
        self.exits = 0
        self.fetch_error: BaseException | None = None
        self.upsert_error: BaseException | None = None

    def fetch(self, cache_dir, name):
        self.fetches.append((cache_dir, name))
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.objects.get(name)

    def note_load_failed(self, cache_dir, name):
        self.load_failed.append((cache_dir, name))

    def note_resolved(self):
        self.resolved += 1

    def schedule_upsert(self, cache_dir, name, data):
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserts.append((cache_dir, name, data))

    def note_exit(self):
        self.exits += 1


@pytest.fixture
def pool(monkeypatch):
    state = _Pool()
    module = types.ModuleType("npugraph_cache.npugraph_cache_store")
    for attr in (
        "fetch",
        "note_load_failed",
        "note_resolved",
        "schedule_upsert",
        "note_exit",
    ):
        setattr(module, attr, getattr(state, attr))
    package = types.ModuleType("npugraph_cache")
    package.npugraph_cache_store = module
    monkeypatch.setitem(sys.modules, "npugraph_cache", package)
    monkeypatch.setitem(sys.modules, "npugraph_cache.npugraph_cache_store", module)
    return state


class _FakeCompilerConfig:
    """Mirrors every attribute compile() reads back for its debug log."""

    def __init__(self) -> None:
        self.mode = None
        self.force_eager = None
        self.static_kernel_compile = False
        self.super_kernel_optimize = False
        self.capture_limit = 64
        self.clone_input = True
        self.clone_output = False
        self.remove_noop_ops = True
        self.inplace_pass = True
        self.input_inplace_pass = True
        self.pattern_fusion_pass = True
        self.frozen_parameter = False
        self.debug = types.SimpleNamespace(
            aclgraph=types.SimpleNamespace(static_capture_size_limit=None)
        )


class _CompiledUnit:
    """The object ``_get_compiled_gm`` returns; get_code() is the payload."""

    def __init__(self, code):
        self._code = code

    def get_code(self):
        if isinstance(self._code, BaseException):
            raise self._code
        return self._code


class _NpuEnv:
    def __init__(self) -> None:
        self.compiled_code = "code = 1\n"
        self.backend_calls: list[tuple] = []
        self.config: _FakeCompilerConfig | None = None
        self.compiler_classes: list[type] = []
        self.use_compiler = False
        self.loaded = MagicMock(name="loaded_callable")
        self.load_error: BaseException | None = None
        self.loaded_py_codes: list = []
        self.aot_flag_seen: list[dict] = []


@pytest.fixture
def npu_env(monkeypatch):
    """Fake npugraph_ex backend + both npu_fx_compiler instances."""
    env = _NpuEnv()

    class _Artifacts:
        def __init__(self) -> None:
            self.py_code = None
            self.version = None

    class _CompiledGraph:
        @staticmethod
        def load_artifacts(artifacts):
            env.loaded_py_codes.append(artifacts.py_code)
            if env.load_error is not None:
                raise env.load_error
            return env.loaded

    def _backend_compile(gm, example_inputs):
        env.backend_calls.append((gm, example_inputs))
        import torch._functorch.config as functorch_config

        env.aot_flag_seen.append(
            {
                "enable_autograd_cache": functorch_config.enable_autograd_cache,
                "enable_remote_autograd_cache": (
                    functorch_config.enable_remote_autograd_cache
                ),
                "bundled_autograd_cache": functorch_config.bundled_autograd_cache,
                "force_autograd_cache": functorch_config.force_autograd_cache,
            }
        )
        if env.use_compiler:
            return env.compiler_classes[0]()._get_compiled_gm(gm, example_inputs)
        return "compiled"

    backend = types.ModuleType("npugraph_ex")

    def _get_npu_backend(compiler_config=None):
        env.config = compiler_config
        return _backend_compile

    backend.CompilerConfig = _FakeCompilerConfig
    backend.get_npu_backend = _get_npu_backend

    modules = {}
    for modname in (
        "npugraph_ex.npu_fx_compiler",
        "torch_npu.dynamo.npugraph_ex.npu_fx_compiler",
    ):
        compiler_cls = type("_NpuFxCompiler", (), {})

        def _get_compiled_gm(self, gm, example_inputs, _env=env):
            return _CompiledUnit(_env.compiled_code)

        compiler_cls._get_compiled_gm = _get_compiled_gm
        module = types.ModuleType(modname)
        module._NpuFxCompiler = compiler_cls
        module._CompiledFxArtifacts = _Artifacts
        module._CompiledFxGraph = _CompiledGraph
        env.compiler_classes.append(compiler_cls)
        modules[modname] = module

    # a plain reproducer of the backend module, plus the dotted parents
    monkeypatch.setitem(sys.modules, "npugraph_ex", backend)
    for modname, module in modules.items():
        monkeypatch.setitem(sys.modules, modname, module)
    for parent in (
        "torch_npu",
        "torch_npu.dynamo",
        "torch_npu.dynamo.npugraph_ex",
    ):
        monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    return env


@pytest.fixture
def cache_ready(monkeypatch, tmp_path):
    """Adaptor with the framework cache initialized and static setup isolated."""
    cfg = {"enable": True}
    _set_config(monkeypatch, cfg)
    monkeypatch.setattr(nx, "_get_npugraph_ex_config", lambda: cfg)
    setup = MagicMock()
    monkeypatch.setattr(nx, "_ensure_static_kernel_runtime_setup", setup)
    monkeypatch.setattr(nx, "_log_canon", lambda *a, **k: None)
    adaptor = nx.NpuGraphExAdaptor()
    adaptor.initialize_cache(str(tmp_path), disable_cache=False)
    return adaptor, str(tmp_path), setup


def _inputs(shape=(2, 3), dtype=torch.float32):
    return [torch.empty(*shape, dtype=dtype)]


# --------------------------------------------------------------------------
# _get_npugraph_ex_config
# --------------------------------------------------------------------------
class TestGetNpugraphExConfig:
    def test_returns_the_npugraph_ex_sub_dict(self, monkeypatch):
        cfg = {"enable": True, "capture_limit": 32}
        _set_config(monkeypatch, cfg)

        assert nx._get_npugraph_ex_config() is cfg

    def test_unreachable_package_returns_none_and_warns_once(self, monkeypatch, logger):
        # the tree is installed as omni_npu in production; a tree imported
        # under the physical `omni` name must not spam the log every call
        monkeypatch.setitem(
            sys.modules, "omni_npu.compilation.npugraph_ex_config", None
        )

        assert nx._get_npugraph_ex_config() is None
        assert nx._get_npugraph_ex_config() is None

        assert logger.warning.call_count == 1
        assert nx._config_unreachable_warned is True


# --------------------------------------------------------------------------
# _graph_flavor
# --------------------------------------------------------------------------
class TestGraphFlavor:
    def test_classifies_streams_tagged_mixed_and_none(self):
        streams_op = torch.ops.streams.record_event
        tagged_op = _SyncOp("npugraph_ex.tagged_event_record_on_stream")

        def _flavor(*targets):
            graph = fx.Graph()
            arg = graph.placeholder("x")
            for target in targets:
                graph.call_function(target, (arg,))
            graph.output((arg,))
            return nx._graph_flavor(_GraphHolder(graph))

        assert _flavor(torch.ops.aten.add.Tensor) == "none"
        assert _flavor(streams_op) == "streams"
        assert _flavor(tagged_op) == "tagged"
        assert _flavor(streams_op, tagged_op) == "mixed"

    def test_real_npugraph_ex_scope_op_counts_as_tagged(self):
        # the classifier's premise: npugraph_ex scope ops render as
        # "npugraph_ex.scope_*" in str(node.target)
        try:
            import torch_npu  # noqa: F401
            import torch_npu.dynamo.npugraph_ex  # noqa: F401

            scope_enter = torch.ops.npugraph_ex.scope_enter
        except Exception:  # pragma: no cover - image without npugraph_ex
            pytest.skip("torch_npu.dynamo.npugraph_ex unavailable")

        graph = fx.Graph()
        arg = graph.placeholder("x")
        graph.call_function(scope_enter, (arg,))
        graph.output((arg,))

        assert nx._graph_flavor(_GraphHolder(graph)) == "tagged"


# --------------------------------------------------------------------------
# _shapes_sig
# --------------------------------------------------------------------------
class TestShapesSig:
    def test_equal_inputs_have_equal_signatures(self):
        assert nx._shapes_sig(_inputs()) == nx._shapes_sig(_inputs())

    def test_shape_dtype_and_scalar_args_change_the_signature(self):
        base = nx._shapes_sig(_inputs())

        assert nx._shapes_sig(_inputs(shape=(3, 3))) != base
        assert nx._shapes_sig(_inputs(dtype=torch.float16)) != base
        assert nx._shapes_sig(_inputs() + [None]) != base
        assert nx._shapes_sig(_inputs() + [1]) != nx._shapes_sig(_inputs() + [2])

    def test_opaque_inputs_are_skipped(self):
        assert nx._shapes_sig([object()]) == nx._shapes_sig([])


# --------------------------------------------------------------------------
# canonicalization for digesting
# --------------------------------------------------------------------------
def _install_fake_pass(monkeypatch, *, raise_exc=None, add_node=False):
    """npugraph_ex pass doubles, including the two hook-holding modules."""
    state = types.SimpleNamespace(
        calls=0,
        stubbed_names=[],
        count_seen=None,
        raise_exc=raise_exc,
        add_node=add_node,
    )
    counter = type("GraphCounter", (), {"COUNT": -1})

    def _real_create_event(name):  # pragma: no cover - must be replaced
        raise AssertionError("a real NPU event must never be created")

    graph_pass = types.ModuleType("npugraph_ex._acl_concrete_graph.graph_pass")
    replace_mod = types.ModuleType(
        "npugraph_ex._acl_concrete_graph.replace_stream_event"
    )
    graph_pass._create_event_by_name = _real_create_event
    replace_mod._create_event_by_name = _real_create_event
    replace_mod.GraphCounter = counter

    def _replace_stream_event_pass(gm):
        state.calls += 1
        state.count_seen = counter.COUNT
        # the real pass creates an event per tagged op and bumps the counter
        replace_mod._create_event_by_name("graph_0_event")
        counter.COUNT += 1
        if state.add_node:
            with gm.graph.inserting_before(gm.graph.output_node()):
                node = gm.graph.create_node(
                    "call_function", _SyncOp("npugraph_ex.scope_enter"), args=()
                )
            gm.graph.output_node().args = ((node,),)
        if state.raise_exc is not None:
            raise state.raise_exc

    replace_mod.replace_stream_event_pass = _replace_stream_event_pass

    pkg = types.ModuleType("npugraph_ex")
    sub = types.ModuleType("npugraph_ex._acl_concrete_graph")
    monkeypatch.setitem(sys.modules, "npugraph_ex", pkg)
    monkeypatch.setitem(sys.modules, "npugraph_ex._acl_concrete_graph", sub)
    monkeypatch.setitem(
        sys.modules, "npugraph_ex._acl_concrete_graph.graph_pass", graph_pass
    )
    monkeypatch.setitem(
        sys.modules,
        "npugraph_ex._acl_concrete_graph.replace_stream_event",
        replace_mod,
    )
    state.modules = (graph_pass, replace_mod)
    state.counter = counter
    return state


class TestCanonicalizeForDigest:
    def test_pass_runs_on_a_copy_and_returns_the_translated_graph(self, monkeypatch):
        state = _install_fake_pass(monkeypatch, add_node=True)
        graph = _add_graph()
        original_nodes = [node.op for node in graph.graph.nodes]

        canonical = nx._canonicalize_for_digest(graph)

        assert state.calls == 1
        assert canonical is not graph
        assert [node.op for node in graph.graph.nodes] == original_nodes
        assert len(list(canonical.graph.nodes)) == len(original_nodes) + 1

    def test_counter_and_event_hooks_are_restored_and_stubbed_meanwhile(
        self, monkeypatch
    ):
        state = _install_fake_pass(monkeypatch)
        graph_pass, replace_mod = state.modules
        graph_pass_orig = graph_pass._create_event_by_name
        replace_orig = replace_mod._create_event_by_name

        nx._canonicalize_for_digest(_add_graph())

        # the stub swallowed the event creation (the real hook raises)
        assert state.count_seen == -1
        assert graph_pass._create_event_by_name is graph_pass_orig
        assert replace_mod._create_event_by_name is replace_orig
        assert state.counter.COUNT == -1

    def test_failure_restores_state_and_propagates(self, monkeypatch):
        state = _install_fake_pass(monkeypatch, raise_exc=RuntimeError("pass boom"))
        graph_pass, replace_mod = state.modules
        graph_pass_orig = graph_pass._create_event_by_name

        with pytest.raises(RuntimeError, match="pass boom"):
            nx._canonicalize_for_digest(_add_graph())

        assert state.counter.COUNT == -1
        assert graph_pass._create_event_by_name is graph_pass_orig
        assert replace_mod._create_event_by_name is not None


class TestCanonDigest:
    def test_digests_the_translated_copy(self, monkeypatch):
        _install_fake_pass(monkeypatch, add_node=True)
        graph = _add_graph()
        inputs = _inputs()

        canon, cost_ms = nx._canon_digest(graph, inputs, "key0")

        assert isinstance(canon, str) and len(canon) == 16
        assert isinstance(cost_ms, float) and cost_ms >= 0.0
        assert canon != nx._npugraph_digest(graph, inputs)

    def test_pass_failure_degrades_to_none_and_the_exception(self, monkeypatch):
        _install_fake_pass(monkeypatch, raise_exc=ValueError("no table"))

        result, info = nx._canon_digest(_add_graph(), _inputs(), "key0")

        assert result is None
        assert isinstance(info, ValueError)


class TestLogCanon:
    def test_returns_the_canonical_digest_and_logs_the_forensic_line(
        self, monkeypatch, logger
    ):
        monkeypatch.setattr(nx, "_canon_digest", lambda *a: ("abc123def456", 1.5))
        graph = _add_graph()

        assert nx._log_canon("compile", "k1", graph, _inputs(), "rawdigest") == (
            "abc123def456"
        )

        args = logger.info.call_args[0]
        assert args[0].startswith("[npugraph-canon] event=%s")
        event, key, flavor, raw, canon, shapes, cost_ms, pid = args[1:]
        assert (event, key, raw, canon) == ("compile", "k1", "rawdigest", "abc123def456")
        assert flavor == "none"
        assert cost_ms == "1.5"
        assert isinstance(shapes, str) and len(shapes) == 8
        assert pid == os.getpid()

    def test_failure_is_logged_and_returns_none_so_the_raw_digest_survives(
        self, monkeypatch, logger
    ):
        monkeypatch.setattr(nx, "_canon_digest", lambda *a: (None, RuntimeError("x")))

        assert nx._log_canon("load", "k1", _add_graph(), _inputs(), "rawdigest") is None

        args = logger.info.call_args[0]
        assert args[5] == "FAIL:x"
        assert args[7] == "-"


# --------------------------------------------------------------------------
# optional pool bridge
# --------------------------------------------------------------------------
class TestNpugraphCachePool:
    def test_missing_bridge_is_a_silent_opt_out(self, logger):
        assert nx._npugraph_cache_pool() is None
        assert nx._npugraph_cache_pool() is None

        assert logger.warning.call_count == 1
        assert nx._npugraph_cache_pool_unreachable_warned is True

    def test_bridge_is_returned_and_note_exit_registered_once(self, monkeypatch, pool):
        import atexit

        register = MagicMock()
        monkeypatch.setattr(atexit, "register", register)

        assert nx._npugraph_cache_pool() is sys.modules[
            "npugraph_cache.npugraph_cache_store"
        ]
        assert nx._npugraph_cache_pool() is not None

        register.assert_called_once_with(
            sys.modules["npugraph_cache.npugraph_cache_store"].note_exit
        )


# --------------------------------------------------------------------------
# engine build fingerprint
# --------------------------------------------------------------------------
def _hide_backend(monkeypatch):
    for modname in ("torch_npu.dynamo.npugraph_ex", "npugraph_ex"):
        monkeypatch.setitem(sys.modules, modname, None)


def _point_adaptor_at(monkeypatch, tmp_path, source):
    path = tmp_path / "npugraph_ex.py"
    path.write_text(source)
    monkeypatch.setattr(nx, "__file__", str(path))
    return path


class TestEngineBuildFingerprint:
    def test_memoized_per_process(self, monkeypatch):
        monkeypatch.setattr(nx, "_engine_build_hash", "deadbeefcafe")

        assert nx._engine_build_fingerprint() == "deadbeefcafe"

    def test_adaptor_source_changes_the_fingerprint(self, monkeypatch, tmp_path):
        _hide_backend(monkeypatch)
        _point_adaptor_at(monkeypatch, tmp_path, "# v1\n")

        first = nx._engine_build_fingerprint()
        monkeypatch.setattr(nx, "_engine_build_hash", None)
        second = nx._engine_build_fingerprint()
        _point_adaptor_at(monkeypatch, tmp_path, "# v2 hot-patched\n")
        monkeypatch.setattr(nx, "_engine_build_hash", None)
        patched = nx._engine_build_fingerprint()

        assert first == second
        assert patched != first
        assert len(patched) == 12

    def test_backend_package_hashes_every_python_file(self, monkeypatch, tmp_path):
        _point_adaptor_at(monkeypatch, tmp_path, "# adaptor\n")
        package = tmp_path / "bepkg"
        package.mkdir()
        (package / "__init__.py").write_text("BACKEND = 1\n")
        (package / "core.py").write_text("CORE = 1\n")
        (package / "notes.txt").write_text("not code\n")
        fake = types.SimpleNamespace(__file__=str(package / "__init__.py"))
        monkeypatch.setitem(sys.modules, "torch_npu.dynamo.npugraph_ex", None)
        monkeypatch.setitem(sys.modules, "npugraph_ex", fake)

        before = nx._engine_build_fingerprint()

        (package / "notes.txt").write_text("still not code\n")
        monkeypatch.setattr(nx, "_engine_build_hash", None)
        assert nx._engine_build_fingerprint() == before  # non-.py ignored

        (package / "core.py").write_text("CORE = 2\n")
        monkeypatch.setattr(nx, "_engine_build_hash", None)
        assert nx._engine_build_fingerprint() != before

    def test_single_file_backend_hashes_that_file(self, monkeypatch, tmp_path):
        _point_adaptor_at(monkeypatch, tmp_path, "# adaptor\n")
        module_path = tmp_path / "single.py"
        module_path.write_text("X = 1\n")
        monkeypatch.setitem(sys.modules, "torch_npu.dynamo.npugraph_ex", None)
        monkeypatch.setitem(
            sys.modules,
            "npugraph_ex",
            types.SimpleNamespace(__file__=str(module_path)),
        )

        before = nx._engine_build_fingerprint()
        module_path.write_text("X = 2\n")
        monkeypatch.setattr(nx, "_engine_build_hash", None)

        assert nx._engine_build_fingerprint() != before

    def test_absent_backend_and_unreadable_sources_never_raise(
        self, monkeypatch, tmp_path, logger
    ):
        _hide_backend(monkeypatch)
        adaptor_path = _point_adaptor_at(monkeypatch, tmp_path, "# adaptor\n")
        real_open = open

        def _open(path, *args, **kwargs):
            if os.path.abspath(str(path)) == os.path.abspath(str(adaptor_path)):
                raise OSError("SFS hiccup")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)

        fingerprint = nx._engine_build_fingerprint()

        assert len(fingerprint) == 12
        assert logger.warning.called
        assert "adaptor unreadable" in str(logger.warning.call_args)

    def test_unreadable_backend_package_is_a_token_too(
        self, monkeypatch, tmp_path, logger
    ):
        _point_adaptor_at(monkeypatch, tmp_path, "# adaptor\n")
        module_path = tmp_path / "single.py"
        module_path.write_text("X = 1\n")
        monkeypatch.setitem(sys.modules, "torch_npu.dynamo.npugraph_ex", None)
        monkeypatch.setitem(
            sys.modules,
            "npugraph_ex",
            types.SimpleNamespace(__file__=str(module_path)),
        )
        real_open = open

        def _open(path, *args, **kwargs):
            if os.path.abspath(str(path)) == os.path.abspath(str(module_path)):
                raise OSError("backend on a broken mount")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", _open)

        assert len(nx._engine_build_fingerprint()) == 12
        assert "backend unreadable" in str(logger.warning.call_args)


# --------------------------------------------------------------------------
# static-kernel runtime setup
# --------------------------------------------------------------------------
class TestEnsureStaticKernelRuntimeSetup:
    def _patch_helpers(self, monkeypatch):
        apply_patch = MagicMock()
        monkeypatch.setattr(
            nx, "_load_static_kernel_cache_patch", lambda: apply_patch
        )
        from omni_npu import envs as omni_envs
        from omni_npu.compilation import static_kernel_compile

        configure = MagicMock()
        monkeypatch.setattr(
            static_kernel_compile,
            "configure_static_kernel_compile_jobs",
            configure,
        )
        # read the live value instead of pinning one: ``omni_npu.envs`` is a
        # lazy module, and setting an attribute on it would shadow that
        # lookup for every later test in the session
        return apply_patch, configure, omni_envs.OMNI_STATIC_KERNEL_COMPILE_JOBS

    def test_no_static_kernel_switch_means_no_side_effects(self, monkeypatch):
        apply_patch, configure, expected_jobs = self._patch_helpers(monkeypatch)

        nx._ensure_static_kernel_runtime_setup({"enable": True})

        apply_patch.assert_not_called()
        configure.assert_not_called()

    def test_static_kernel_compile_tunes_jobs_but_leaves_the_patch_alone(
        self, monkeypatch
    ):
        apply_patch, configure, expected_jobs = self._patch_helpers(monkeypatch)

        nx._ensure_static_kernel_runtime_setup(
            {"static_kernel_compile": True, "static_kernel_cache_enable": False}
        )

        apply_patch.assert_not_called()
        configure.assert_called_once_with(expected_jobs)

    def test_static_kernel_cache_applies_the_vendored_entry_then_tunes_jobs(
        self, monkeypatch
    ):
        apply_patch, configure, expected_jobs = self._patch_helpers(monkeypatch)
        cfg = {"static_kernel_compile": True, "static_kernel_cache_enable": True}

        nx._ensure_static_kernel_runtime_setup(cfg)

        apply_patch.assert_called_once_with(cfg)
        configure.assert_called_once_with(expected_jobs)

    def test_super_kernel_only_still_needs_the_runtime_setup(self, monkeypatch):
        apply_patch, configure, expected_jobs = self._patch_helpers(monkeypatch)

        nx._ensure_static_kernel_runtime_setup({"super_kernel_optimize": True})

        apply_patch.assert_not_called()
        configure.assert_called_once_with(expected_jobs)


# --------------------------------------------------------------------------
# py_code save hook
# --------------------------------------------------------------------------
def _artifact_path(cache_dir, key="k", digest="0123456789abcdef"):
    return os.path.join(cache_dir, f"{key}.{digest}{_SUFFIX}")


class TestInstallPyCodeSaveHook:
    def _install(self, cache_path, write_path):
        return nx._install_py_code_save_hook(cache_path, write_path)

    def test_both_compiler_instances_are_patched_and_restored(self, npu_env, tmp_path):
        originals = [cls._get_compiled_gm for cls in npu_env.compiler_classes]

        restore, state = self._install(_artifact_path(str(tmp_path)), None)

        for cls, orig in zip(npu_env.compiler_classes, originals):
            assert cls._get_compiled_gm is not orig
        assert state["skip_reason"] is None

        restore()

        for cls, orig in zip(npu_env.compiler_classes, originals):
            assert cls._get_compiled_gm is orig

    def test_a_second_install_does_not_stack_wrappers(self, npu_env, tmp_path):
        restore, _ = self._install(_artifact_path(str(tmp_path)), None)
        wrapped = [cls._get_compiled_gm for cls in npu_env.compiler_classes]

        restore_again, state = self._install(_artifact_path(str(tmp_path)), None)

        assert [cls._get_compiled_gm for cls in npu_env.compiler_classes] == wrapped
        assert state["skip_reason"] == "no npu_fx_compiler instance patchable"
        restore_again()
        restore()

    def test_no_patchable_compiler_is_reported_not_raised(self, monkeypatch, tmp_path):
        for modname in (
            "npugraph_ex.npu_fx_compiler",
            "torch_npu.dynamo.npugraph_ex.npu_fx_compiler",
        ):
            monkeypatch.setitem(sys.modules, modname, None)

        restore, state = nx._install_py_code_save_hook(
            _artifact_path(str(tmp_path)), None
        )

        assert state == {
            "saved": False,
            "skip_reason": "no npu_fx_compiler instance patchable",
        }
        restore()

    def test_a_compiler_module_without_the_hook_is_skipped(self, monkeypatch, tmp_path):
        # an older/renamed torch_npu build: the module exists but the class
        # or the method does not, so there is nothing to wrap
        class _NoHook:
            pass

        module = types.ModuleType("npugraph_ex.npu_fx_compiler")
        module._NpuFxCompiler = _NoHook
        monkeypatch.setitem(sys.modules, "npugraph_ex", types.ModuleType("npugraph_ex"))
        monkeypatch.setitem(sys.modules, "npugraph_ex.npu_fx_compiler", module)
        monkeypatch.setitem(
            sys.modules, "torch_npu.dynamo.npugraph_ex.npu_fx_compiler", None
        )

        restore, state = nx._install_py_code_save_hook(
            _artifact_path(str(tmp_path)), None
        )

        assert state["skip_reason"] == "no npu_fx_compiler instance patchable"
        restore()

    def test_writes_the_artifact_and_namespaces_event_tags(self, npu_env, tmp_path):
        cache_path = _artifact_path(str(tmp_path), digest="0123456789abcdef")
        npu_env.compiled_code = (
            "stream = _get('graph_3_get_external_object_by_index')\n"
            "ev1 = _create_event_by_name('graph_0_event_1')\n"
            "ev2 = _create_event_by_name('graph_1_event_2')\n"
        )
        restore, state = self._install(cache_path, cache_path)

        compiled = npu_env.compiler_classes[0]()._get_compiled_gm(None, None)
        restore()

        assert compiled is not None
        assert state["saved"] is True
        written = open(cache_path).read()
        # bare stream tags stay shared; event tags get the artifact namespace
        assert "graph_3_get_external_object_by_index" in written
        assert "'gfx0123456789ab_event_1'" in written
        assert "'gfx0123456789ab_event_2'" in written
        assert not os.path.exists(cache_path + ".tmp")
        assert state["py_code"] == written

    def test_pool_mode_keeps_the_bytes_in_memory_and_writes_nothing(
        self, npu_env, tmp_path
    ):
        cache_path = _artifact_path(str(tmp_path))
        restore, state = self._install(cache_path, None)

        npu_env.compiler_classes[0]()._get_compiled_gm(None, None)
        restore()

        assert state["saved"] is True
        assert state["py_code"] == "code = 1\n"
        assert os.listdir(tmp_path) == []

    def test_sfs_rename_rejection_falls_back_to_a_direct_write(
        self, npu_env, tmp_path, monkeypatch
    ):
        cache_path = _artifact_path(str(tmp_path))

        def _reject(*args, **kwargs):
            raise OSError("rename(2) not permitted")

        monkeypatch.setattr(nx.os, "replace", _reject)
        restore, state = self._install(cache_path, cache_path)

        npu_env.compiler_classes[0]()._get_compiled_gm(None, None)
        restore()

        assert state["saved"] is True
        assert open(cache_path).read() == "code = 1\n"
        assert not os.path.exists(cache_path + ".tmp")

    @pytest.mark.parametrize(
        "code,reason",
        [
            (RuntimeError("no source"), "get_code failed: no source"),
            (object(), "codegen unsupported for this graph"),
            ("", "codegen unsupported for this graph"),
            (
                "k = triton_kernel_wrapper(a)\n",
                "graph contains Triton kernels",
            ),
        ],
    )
    def test_unsupported_codegen_is_a_skip_not_a_failure(
        self, npu_env, tmp_path, code, reason
    ):
        cache_path = _artifact_path(str(tmp_path))
        npu_env.compiled_code = code
        restore, state = self._install(cache_path, cache_path)

        compiled = npu_env.compiler_classes[0]()._get_compiled_gm(None, None)
        restore()

        assert state["saved"] is False
        assert state["skip_reason"] == reason
        assert compiled is not None  # the caller keeps the compiled unit
        assert not os.path.exists(cache_path)

    def test_write_failure_is_a_skip_and_never_breaks_the_compilation(
        self, npu_env, tmp_path, monkeypatch
    ):
        cache_path = _artifact_path(str(tmp_path / "missing"))
        monkeypatch.setattr(
            nx.os, "makedirs", MagicMock(side_effect=OSError("read-only mount"))
        )
        restore, state = self._install(cache_path, cache_path)

        compiled = npu_env.compiler_classes[0]()._get_compiled_gm(None, None)
        restore()

        assert compiled is not None
        assert state["saved"] is False
        assert state["skip_reason"].startswith("artifact write failed:")

    def test_only_the_first_artifact_of_one_backend_call_is_captured(
        self, npu_env, tmp_path, monkeypatch
    ):
        cache_path = _artifact_path(str(tmp_path))
        npu_env.compiled_code = "first = 1\n"
        restore, state = self._install(cache_path, cache_path)
        instance = npu_env.compiler_classes[0]()

        instance._get_compiled_gm(None, None)
        npu_env.compiled_code = "second = 2\n"
        instance._get_compiled_gm(None, None)
        restore()

        assert open(cache_path).read() == "first = 1\n"
        assert state["py_code"] == "first = 1\n"


# --------------------------------------------------------------------------
# content digest
# --------------------------------------------------------------------------
class TestNpugraphDigest:
    def test_same_graph_and_inputs_hash_identically(self):
        assert nx._npugraph_digest(_add_graph(), _inputs()) == nx._npugraph_digest(
            _add_graph(), _inputs()
        )

    def test_placeholder_names_are_not_identity(self):
        assert nx._npugraph_digest(
            _add_graph(("x", "y")), _inputs()
        ) == nx._npugraph_digest(_add_graph(("lhs", "rhs")), _inputs())

    def test_topological_position_is_identity(self):
        def _build(graph):
            left = graph.placeholder("x")
            right = graph.placeholder("y")
            added = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
            out = graph.call_function(torch.ops.aten.mul.Tensor, (added, right))
            graph.output((out,))

        def _build_swapped(graph):
            left = graph.placeholder("x")
            right = graph.placeholder("y")
            muled = graph.call_function(torch.ops.aten.mul.Tensor, (left, right))
            out = graph.call_function(torch.ops.aten.add.Tensor, (muled, right))
            graph.output((out,))

        assert nx._npugraph_digest(
            _graph_module(_build), _inputs()
        ) != nx._npugraph_digest(_graph_module(_build_swapped), _inputs())

    def test_operator_identity_is_kept(self):
        assert nx._npugraph_digest(
            _add_graph(op=torch.ops.aten.add.Tensor), _inputs()
        ) != nx._npugraph_digest(_add_graph(op=torch.ops.aten.sub.Tensor), _inputs())
        assert nx._npugraph_digest(
            _add_graph(op=torch.ops.aten.add.Tensor), _inputs()
        ) != nx._npugraph_digest(_add_graph(op=torch.ops.aten.add.Scalar), _inputs())

    def test_example_input_shape_and_dtype_drift_is_a_miss(self):
        graph = _add_graph()

        base = nx._npugraph_digest(graph, _inputs())
        assert nx._npugraph_digest(graph, _inputs(shape=(3, 3))) != base
        assert nx._npugraph_digest(graph, _inputs(dtype=torch.bfloat16)) != base

    def test_scalar_and_opaque_example_inputs(self):
        graph = _add_graph()
        with_option = nx._npugraph_digest(graph, _inputs() + [None, 4, True])

        assert isinstance(with_option, str)
        assert nx._npugraph_digest(graph, _inputs() + [None, 4, True]) == with_option
        # an opaque input is rendered by type, so it participates without
        # carrying a per-process address into the digest
        opaque_a = nx._npugraph_digest(graph, _inputs() + [object()])
        assert opaque_a == nx._npugraph_digest(graph, _inputs() + [object()])
        assert opaque_a != nx._npugraph_digest(graph, _inputs())
        assert opaque_a != nx._npugraph_digest(graph, _inputs() + [4])

    def test_kwargs_are_canonicalized_and_dict_order_is_irrelevant(self):
        def _build_a(graph):
            arg = graph.placeholder("x")
            graph.call_function(
                torch.ops.aten.add.Tensor,
                (arg, arg),
                {"alpha": 1, "beta": ("t", 2)},
            )
            graph.output((arg,))

        def _build_b(graph):
            arg = graph.placeholder("x")
            graph.call_function(
                torch.ops.aten.add.Tensor,
                (arg, arg),
                {"beta": ("t", 2), "alpha": 1},
            )
            graph.output((arg,))

        def _build_c(graph):
            arg = graph.placeholder("x")
            graph.call_function(
                torch.ops.aten.add.Tensor,
                (arg, arg),
                {"alpha": 2, "beta": ("t", 2)},
            )
            graph.output((arg,))

        same = nx._npugraph_digest(_graph_module(_build_a), _inputs())
        assert nx._npugraph_digest(_graph_module(_build_b), _inputs()) == same
        assert nx._npugraph_digest(_graph_module(_build_c), _inputs()) != same

    def test_tag_build_counters_are_normalized_but_suffixes_are_kept(self):
        def _digest(tag):
            graph = fx.Graph()
            arg = graph.placeholder("x")
            graph.call_function(
                _SyncOp("npugraph_ex.get_external_object_by_index"), (arg, tag)
            )
            graph.output((arg,))
            return nx._npugraph_digest(_GraphHolder(graph), [])

        # the per-boot build counter moves; the object suffix does not
        assert _digest("graph_4_get_external_object_by_index_2") == _digest(
            "graph_1_get_external_object_by_index_2"
        )
        assert _digest("graph_4_get_external_object_by_index_2") != _digest(
            "graph_4_get_external_object_by_index_3"
        )

    def test_absolute_stream_ordinals_are_normalized(self):
        def _digest(scope_stream, event_stream):
            graph = fx.Graph()
            arg = graph.placeholder("x")
            graph.call_function(
                _SyncOp("npugraph_ex.scope_enter"),
                (arg, ["label", 0, scope_stream, 0, "type"]),
            )
            graph.call_function(
                _SyncOp("npugraph_ex.tagged_event_record_on_stream"),
                (arg, event_stream, 0, "type"),
            )
            graph.output((arg,))
            return nx._npugraph_digest(_GraphHolder(graph), [])

        # a co-resident transfer engine shifts every id by one: same graph
        assert _digest("112", "112") == _digest("113", "113")
        # which ops share a stream is semantic
        assert _digest("112", "112") != _digest("112", "113")

    def test_ordinal_normalization_is_fail_open(self):
        def _digest(*args):
            graph = fx.Graph()
            arg = graph.placeholder("x")
            graph.call_function(_SyncOp("npugraph_ex.scope_enter"), (arg,) + args)
            graph.output((arg,))
            return nx._npugraph_digest(_GraphHolder(graph), [])

        # non-numeric id: kept literally
        assert _digest(["label", 0, "default_stream"]) != _digest(
            ["label", 0, "112"]
        )
        # signature drift: short/foreign argument shapes pass through intact
        first = _digest(["label", 0])
        assert first == _digest(["label", 0])
        assert _digest("not-a-list") == _digest("not-a-list")

    def test_backed_symbolic_leaves_use_their_hints(self):
        def _build(graph):
            arg = graph.placeholder("x")
            graph.call_function(
                torch.ops.aten.add.Tensor,
                (arg, _backed_sym(4, 4)),
            )
            graph.output((arg,))

        def _build_other_hint(graph):
            arg = graph.placeholder("x")
            graph.call_function(
                torch.ops.aten.add.Tensor,
                (arg, _backed_sym(5, 5)),
            )
            graph.output((arg,))

        assert nx._npugraph_digest(
            _graph_module(_build), _inputs()
        ) != nx._npugraph_digest(_graph_module(_build_other_hint), _inputs())

    def test_unbacked_symbolic_leaves_are_lettered_by_first_occurrence(self):
        env = _sym_env()
        first = env.create_unbacked_symint()
        second = env.create_unbacked_symint()
        same_as_first = first

        def _digest(leaf_a, leaf_b):
            graph = fx.Graph()
            arg = graph.placeholder("x")
            graph.call_function(torch.ops.aten.add.Tensor, (arg, leaf_a))
            graph.call_function(torch.ops.aten.add.Tensor, (arg, leaf_b))
            graph.output((arg,))
            return nx._npugraph_digest(_GraphHolder(graph), [])

        tied = _digest(first, same_as_first)
        untied = _digest(first, second)

        assert tied != untied  # shared vs distinct unbacked values stay visible
        assert _digest(second, first) == untied  # rename noise is invisible

    def test_symbolic_tensor_dimensions_are_lettered(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        def _fake_with_unbacked_dim():
            mode = FakeTensorMode(shape_env=_sym_env())
            with mode:
                dim = mode.shape_env.create_unbacked_symint()
                return torch.empty(dim)

        graph = _add_graph()
        digest = nx._npugraph_digest(graph, [_fake_with_unbacked_dim()])

        # two independent unbacked dims render identically, so a dynamic-shape
        # model digests the same every boot
        assert digest == nx._npugraph_digest(graph, [_fake_with_unbacked_dim()])
        # while a concrete shape stays a different unit
        assert digest != nx._npugraph_digest(graph, _inputs())

    def test_symbolic_float_and_bool_leaves(self):
        for leaf in (
            _sym_env().create_unbacked_symfloat(),
            _sym_env().create_unbacked_symbool(),
            _backed_sym(1.5, 1.5),
            _backed_sym(True, True),
        ):
            graph = fx.Graph()
            arg = graph.placeholder("x")
            graph.call_function(torch.ops.aten.add.Tensor, (arg, arg, leaf))
            graph.output((arg,))
            first = nx._npugraph_digest(_GraphHolder(graph), [])
            assert first == nx._npugraph_digest(_GraphHolder(graph), [])

    def test_address_bearing_and_callable_leaves_are_source_stable(self):
        def _factory(label):
            def inner(**kwargs):
                return label

            return inner

        def _digest(target):
            graph = fx.Graph()
            graph.call_function(target, ())
            graph.output(())
            return nx._npugraph_digest(_GraphHolder(graph), [])

        # same source identity (module.qualname) despite distinct objects
        assert _digest(_factory("a")) == _digest(_factory("b"))
        # a different source is a different target
        assert _digest(_factory("a")) != _digest(nx._graph_flavor)

        # an address-bearing object collapses to its type: it cannot churn
        # the digest between processes
        assert nx._npugraph_digest(
            _add_graph(), [object()]
        ) == nx._npugraph_digest(_add_graph(), [object()])

    def test_artifact_naming_is_digest_addressed(self):
        """The cache filename carries the digest, not only the slot key."""
        graph = _add_graph()
        digest_a = nx._npugraph_digest(graph, _inputs())
        digest_b = nx._npugraph_digest(graph, _inputs(shape=(4, 4)))

        name_a = f"artifact_compile_range_x_subgraph_0.{digest_a}{_SUFFIX}"
        name_b = f"artifact_compile_range_x_subgraph_0.{digest_b}{_SUFFIX}"
        assert name_a != name_b


class TestNpugraphDigestDump:
    def _dump_files(self, tmp_path):
        dump_dir = tmp_path / "npugraph_digest_dump"
        return dump_dir, sorted(dump_dir.iterdir()) if dump_dir.exists() else []

    def test_dumped_text_reproduces_the_digest(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NPUGRAPH_DIGEST_DUMP", "1")
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))

        digest = nx._npugraph_digest(_add_graph(), _inputs(), "artifact_x")

        dump_dir, files = self._dump_files(tmp_path)
        assert dump_dir.is_dir()
        assert [f.name for f in files] == [
            f"artifact_x.{digest}.{os.getpid()}.txt"
        ]
        text = files[0].read_text()
        assert hashlib.sha256(text.encode()).hexdigest()[:16] == digest

    def test_no_dump_without_the_env_or_a_cache_root(self, monkeypatch, tmp_path):
        digest = nx._npugraph_digest(_add_graph(), _inputs(), "artifact_x")
        assert not (tmp_path / "npugraph_digest_dump").exists()

        monkeypatch.setenv("NPUGRAPH_DIGEST_DUMP", "1")
        monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
        nx._npugraph_digest(_add_graph(), _inputs(), "artifact_x")
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
        # no key: nothing to name the file after
        assert nx._npugraph_digest(_add_graph(), _inputs()) == digest
        assert not (tmp_path / "npugraph_digest_dump").exists()

    def test_a_broken_dump_target_never_breaks_the_digest(
        self, monkeypatch, tmp_path
    ):
        blocker = tmp_path / "npugraph_digest_dump"
        blocker.write_text("not a directory")
        monkeypatch.setenv("NPUGRAPH_DIGEST_DUMP", "1")
        monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))

        digest = nx._npugraph_digest(_add_graph(), _inputs(), "artifact_x")

        assert isinstance(digest, str) and len(digest) == 16


# --------------------------------------------------------------------------
# adaptor lifecycle: cache is off until the framework initializes it
# --------------------------------------------------------------------------
class TestAdaptorCacheLifecycle:
    def test_defaults_behave_exactly_as_before_the_cache_existed(self):
        adaptor = nx.NpuGraphExAdaptor()

        assert adaptor.cache_dir is None
        assert adaptor.disable_cache is True

    def test_initialize_cache_records_the_framework_decision(self, tmp_path):
        adaptor = nx.NpuGraphExAdaptor()

        adaptor.initialize_cache(str(tmp_path), disable_cache=False, prefix="p")

        assert adaptor.cache_dir == str(tmp_path)
        assert adaptor.disable_cache is False

        adaptor.initialize_cache(str(tmp_path / "off"), disable_cache=True)
        assert adaptor.disable_cache is True

    def test_compute_hash_is_a_stable_ten_char_factor(self, monkeypatch, logger):
        monkeypatch.setattr(nx, "_engine_build_fingerprint", lambda: "build0000000")
        _set_config(monkeypatch, {"enable": True})
        adaptor = nx.NpuGraphExAdaptor()

        first = adaptor.compute_hash(MagicMock())
        second = adaptor.compute_hash(MagicMock())

        assert first == second
        assert len(first) == 10 and int(first, 16) >= 0
        assert "torch_npu_version" in str(logger.debug.call_args)
        assert "build0000000" in str(logger.debug.call_args)

    def test_every_forwarded_config_knob_moves_the_hash(self, monkeypatch):
        monkeypatch.setattr(nx, "_engine_build_fingerprint", lambda: "build")
        adaptor = nx.NpuGraphExAdaptor()

        _set_config(monkeypatch, {"enable": True, "capture_limit": 64})
        plain = adaptor.compute_hash(MagicMock())
        _set_config(monkeypatch, {"enable": True, "capture_limit": 32})
        tuned = adaptor.compute_hash(MagicMock())

        assert plain != tuned

    def test_a_new_codegen_build_moves_the_hash(self, monkeypatch):
        adaptor = nx.NpuGraphExAdaptor()
        _set_config(monkeypatch, {"enable": True})

        monkeypatch.setattr(nx, "_engine_build_fingerprint", lambda: "build-a")
        first = adaptor.compute_hash(MagicMock())
        monkeypatch.setattr(nx, "_engine_build_fingerprint", lambda: "build-b")

        assert adaptor.compute_hash(MagicMock()) != first

    def test_missing_torch_npu_is_a_recorded_factor_not_a_crash(
        self, monkeypatch, logger
    ):
        monkeypatch.setitem(sys.modules, "torch_npu", None)
        monkeypatch.setattr(nx, "_engine_build_fingerprint", lambda: "build")
        _set_config(monkeypatch, None)

        digest = nx.NpuGraphExAdaptor().compute_hash(MagicMock())

        assert len(digest) == 10
        factors = str(logger.debug.call_args)
        assert "unavailable:" in factors
        # an unreachable config contributes no config factor at all
        assert "npugraph_ex_config" not in factors


# --------------------------------------------------------------------------
# compile() with the py_code cache
# --------------------------------------------------------------------------
class TestCompileWithPyCodeCache:
    @pytest.mark.parametrize("gate", ["no_cache_dir", "disabled", "no_key"])
    def test_cache_gates_leave_the_original_path_untouched(
        self, npu_env, monkeypatch, tmp_path, gate
    ):
        monkeypatch.setattr(nx, "_ensure_static_kernel_runtime_setup", MagicMock())
        _set_config(monkeypatch, {"enable": True})
        adaptor = nx.NpuGraphExAdaptor()
        if gate != "no_cache_dir":
            adaptor.initialize_cache(str(tmp_path), disable_cache=False)
        if gate == "disabled":
            adaptor.initialize_cache(str(tmp_path), disable_cache=True)
        key = None if gate == "no_key" else "artifact_0"
        digest_spy = MagicMock(return_value="deadbeefdeadbeef")
        monkeypatch.setattr(nx, "_npugraph_digest", digest_spy)

        result, handle = adaptor.compile(_add_graph(), _inputs(), {}, key=key)

        digest_spy.assert_not_called()
        assert handle is None
        assert result == "compiled"

    def test_digest_is_taken_before_the_in_place_tuple_fix(
        self, npu_env, monkeypatch, cache_ready
    ):
        adaptor, cache_dir, _ = cache_ready
        seen_outputs = []

        def _spy(graph, example_inputs, key=None):
            seen_outputs.append(graph.graph.output_node().args[0])
            return "aaaaaaaaaaaaaaaa"

        monkeypatch.setattr(nx, "_npugraph_digest", _spy)
        monkeypatch.setattr(
            "torch._inductor.compile_fx.graph_returns_tuple", lambda gm: False
        )
        graph = _add_graph_single_output()

        _, handle = adaptor.compile(graph, _inputs(), {}, key="artifact_0")

        # the digest saw the pre-mutation output (a plain node, not a tuple())
        assert len(seen_outputs) == 1
        assert isinstance(seen_outputs[0], fx.Node)
        assert seen_outputs[0].target is not tuple
        # and the graph was tuple-ified for the backend afterwards
        output_arg = graph.graph.output_node().args[0]
        assert output_arg.target is tuple
        assert handle == (
            "artifact_0",
            os.path.join(cache_dir, f"artifact_0.aaaaaaaaaaaaaaaa{_SUFFIX}"),
        )

    def test_pooled_unit_is_served_without_entering_the_backend(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, setup = cache_ready
        pool.objects[f"artifact_0.dddddddddddddddd{_SUFFIX}"] = b"pooled = 1\n"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        compiled, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert compiled is npu_env.loaded
        # the adaptor hands the bridge payload over verbatim (bytes as stored)
        assert npu_env.loaded_py_codes == [b"pooled = 1\n"]
        assert npu_env.backend_calls == []  # a hit never recompiles
        assert pool.fetches == [
            (cache_dir, f"artifact_0.dddddddddddddddd{_SUFFIX}")
        ]
        assert pool.resolved == 1
        assert handle == (
            "artifact_0",
            os.path.join(cache_dir, f"artifact_0.dddddddddddddddd{_SUFFIX}"),
        )
        setup.assert_called_once()

    def test_unloadable_pooled_bytes_are_healed_by_recompiling(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        pool.objects[name] = b"torn = 1\n"
        npu_env.load_error = ValueError("cannot execute")
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        compiled, _ = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert pool.load_failed == [(cache_dir, name)]
        assert npu_env.backend_calls  # fell through to a fresh compile
        assert compiled == "compiled"

    def test_pool_fetch_failure_never_blocks_the_compile(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, _, _ = cache_ready
        pool.fetch_error = RuntimeError("RPC down")
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        compiled, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert compiled == "compiled"
        assert handle is not None

    def test_fresh_compile_uploads_the_exact_captured_bytes_and_writes_nothing(
        self, npu_env, monkeypatch, cache_ready, pool, tmp_path
    ):
        adaptor, cache_dir, _ = cache_ready
        npu_env.use_compiler = True
        npu_env.compiled_code = "ev = _create_event_by_name('graph_0_ev_1')\n"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        compiled, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        assert [upsert[1] for upsert in pool.upserts] == [name]
        uploaded = pool.upserts[0][2].decode()
        assert "'gfxdddddddddddd_ev_1'" in uploaded  # namespaced upload bytes
        assert pool.resolved == 1
        assert os.listdir(cache_dir) == []  # disk-free pool mode
        assert handle == ("artifact_0", os.path.join(cache_dir, name))
        assert isinstance(compiled, _CompiledUnit)

    def test_upload_failure_is_only_a_warning(
        self, npu_env, monkeypatch, cache_ready, pool, logger
    ):
        adaptor, _, _ = cache_ready
        npu_env.use_compiler = True
        pool.upsert_error = RuntimeError("queue closed")
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        compiled, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert isinstance(compiled, _CompiledUnit)
        assert handle is not None
        assert pool.resolved == 1
        assert "schedule upload failed" in str(logger.warning.call_args)

    def test_without_the_bridge_the_artifact_lands_in_the_cache_dir(
        self, npu_env, monkeypatch, cache_ready, tmp_path
    ):
        adaptor, cache_dir, _ = cache_ready
        npu_env.use_compiler = True
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        _, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        path = os.path.join(cache_dir, f"artifact_0.dddddddddddddddd{_SUFFIX}")
        assert handle == ("artifact_0", path)
        assert open(path).read() == "code = 1\n"

    def test_config_knobs_are_forwarded_and_canon_names_the_artifact(
        self, npu_env, monkeypatch, cache_ready
    ):
        adaptor, cache_dir, _ = cache_ready
        cfg = {
            "enable": True,
            "static_kernel_compile": True,
            "super_kernel_optimize": True,
            "capture_limit": 100,
            "clone_input": False,
            "clone_output": True,
            "remove_noop_ops": False,
            "inplace_pass": False,
            "input_inplace_pass": False,
            "pattern_fusion_pass": False,
            "frozen_parameter": True,
        }
        _set_config(monkeypatch, cfg)
        monkeypatch.setattr(nx, "_get_npugraph_ex_config", lambda: cfg)
        monkeypatch.setattr(nx, "_npugraph_digest", lambda *a, **k: "rawrawrawrawrawr")
        monkeypatch.setattr(nx, "_log_canon", lambda *a, **k: "canoncanoncanon1")

        _, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        config = npu_env.config
        assert (config.mode, config.force_eager) == ("npugraph_ex", True)
        assert config.static_kernel_compile is True
        assert config.super_kernel_optimize is True
        assert config.debug.aclgraph.static_capture_size_limit == 100
        assert (config.clone_input, config.clone_output) == (False, True)
        assert (config.remove_noop_ops, config.inplace_pass) == (False, False)
        assert (config.input_inplace_pass, config.pattern_fusion_pass) == (
            False,
            False,
        )
        assert config.frozen_parameter is True
        # the canonical digest (the pass-translated form) names the artifact,
        # so both sync writings converge on one file
        assert handle[1] == os.path.join(
            cache_dir, f"artifact_0.canoncanoncanon1{_SUFFIX}"
        )

    def test_aot_autograd_caches_are_off_for_the_backend_call_and_restored(
        self, npu_env, monkeypatch, cache_ready
    ):
        import torch._functorch.config as functorch_config

        adaptor, _, _ = cache_ready
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )
        before = {
            "enable_autograd_cache": functorch_config.enable_autograd_cache,
            "enable_remote_autograd_cache": (
                functorch_config.enable_remote_autograd_cache
            ),
            "bundled_autograd_cache": functorch_config.bundled_autograd_cache,
            "force_autograd_cache": functorch_config.force_autograd_cache,
        }

        adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert npu_env.aot_flag_seen == [
            {
                "enable_autograd_cache": False,
                "enable_remote_autograd_cache": False,
                "bundled_autograd_cache": False,
                "force_autograd_cache": False,
            }
        ]
        after = {
            "enable_autograd_cache": functorch_config.enable_autograd_cache,
            "enable_remote_autograd_cache": (
                functorch_config.enable_remote_autograd_cache
            ),
            "bundled_autograd_cache": functorch_config.bundled_autograd_cache,
            "force_autograd_cache": functorch_config.force_autograd_cache,
        }
        assert after == before

    def test_skipped_artifact_is_reported_and_the_handle_still_returned(
        self, npu_env, monkeypatch, cache_ready, logger
    ):
        adaptor, cache_dir, _ = cache_ready
        npu_env.use_compiler = True
        npu_env.compiled_code = "triton_kernel_wrapper(x)\n"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        compiled, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert isinstance(compiled, _CompiledUnit)
        # the handle still points at the (absent) artifact: an absent file is
        # a clean per-graph miss, never a wrong load
        assert handle == (
            "artifact_0",
            os.path.join(cache_dir, f"artifact_0.dddddddddddddddd{_SUFFIX}"),
        )
        assert not os.path.exists(handle[1])
        assert any(
            "py_code cache skipped" in str(call)
            for call in logger.info.call_args_list
        )

    def test_static_kernel_setup_runs_before_any_cache_lookup(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, _, setup = cache_ready
        pool.objects[f"artifact_0.dddddddddddddddd{_SUFFIX}"] = b"pooled = 1\n"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )
        order = []
        setup.side_effect = lambda cfg: order.append("setup")
        bridge = sys.modules["npugraph_cache.npugraph_cache_store"]
        original_fetch = bridge.fetch

        def _fetch(*args):
            order.append("fetch")
            return original_fetch(*args)

        monkeypatch.setattr(bridge, "fetch", _fetch)

        adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")

        assert order == ["setup", "fetch"]


# --------------------------------------------------------------------------
# load()
# --------------------------------------------------------------------------
class TestLoadFromPyCodeCache:
    def _write_artifact(self, cache_dir, name, text):
        path = os.path.join(cache_dir, name)
        with open(path, "w") as handle:
            handle.write(text)
        return path

    @pytest.mark.parametrize(
        "config,cache_dir,disable_cache,handle",
        [
            ({"enable": False}, True, False, ("artifact_0", "p")),
            ({"enable": True}, False, False, ("artifact_0", "p")),
            ({"enable": True}, True, True, ("artifact_0", "p")),
            ({"enable": True}, True, False, None),
            ({"enable": True}, True, False, ()),
            ({"enable": True}, True, False, ("", "p")),
            ({"enable": True}, True, False, (7, "p")),
        ],
    )
    def test_every_gate_is_a_plain_miss(
        self, monkeypatch, tmp_path, config, cache_dir, disable_cache, handle
    ):
        monkeypatch.setattr(nx, "_get_npugraph_ex_config", lambda: config)
        monkeypatch.setattr(nx, "_ensure_static_kernel_runtime_setup", MagicMock())
        digest_spy = MagicMock(return_value="d")
        monkeypatch.setattr(nx, "_npugraph_digest", digest_spy)
        adaptor = nx.NpuGraphExAdaptor()
        if cache_dir:
            adaptor.initialize_cache(str(tmp_path), disable_cache=disable_cache)

        result = adaptor.load(handle, _add_graph(), _inputs(), 0, None)

        assert result is None
        digest_spy.assert_not_called()

    def test_hit_path_runs_the_static_kernel_setup(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, setup = cache_ready
        monkeypatch.setattr(
            nx, "_get_npugraph_ex_config", lambda: {"enable": True}
        )
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        pool.objects[name] = b"pooled = 1\n"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        assert (
            adaptor.load(("artifact_0", "stale"), _add_graph(), _inputs(), 0, None)
            is not None
        )
        setup.assert_called_once_with({"enable": True})

    def test_pool_hit_returns_the_callable_and_ignores_the_local_file(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        self._write_artifact(cache_dir, name, "local = 1\n")
        pool.objects[name] = b"pooled = 1\n"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        loaded = adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        )

        assert loaded is npu_env.loaded
        # the local file at the same name is never even read
        assert npu_env.loaded_py_codes == [b"pooled = 1\n"]
        assert pool.resolved == 1

    def test_canon_digest_names_the_lookup(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.canoncanoncanon1{_SUFFIX}"
        pool.objects[name] = b"pooled = 1\n"
        monkeypatch.setattr(nx, "_get_npugraph_ex_config", lambda: {"enable": True})
        monkeypatch.setattr(nx, "_npugraph_digest", lambda *a, **k: "rawrawrawrawrawr")
        monkeypatch.setattr(nx, "_log_canon", lambda *a, **k: "canoncanoncanon1")

        loaded = adaptor.load(
            ("artifact_0", "stale"), _add_graph(), _inputs(), 0, None
        )

        assert loaded is npu_env.loaded
        assert pool.fetches == [(cache_dir, name)]

    def test_pool_miss_never_falls_back_to_the_local_file(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        self._write_artifact(cache_dir, name, "local = 1\n")
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        assert adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        ) is None
        assert npu_env.loaded_py_codes == []
        assert pool.resolved == 1

    def test_unloadable_pool_bytes_are_marked_and_become_a_miss(
        self, npu_env, monkeypatch, cache_ready, pool
    ):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        pool.objects[name] = b"torn = 1\n"
        npu_env.load_error = ValueError("cannot execute")
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        assert adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        ) is None
        assert pool.load_failed == [(cache_dir, name)]
        assert pool.resolved == 1

    def test_pool_fetch_failure_is_a_miss_not_an_exception(
        self, npu_env, monkeypatch, cache_ready, pool, logger
    ):
        adaptor, _, _ = cache_ready
        pool.fetch_error = RuntimeError("RPC down")
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        assert adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        ) is None
        assert "fetch path failed" in str(logger.warning.call_args)

    def test_local_file_hit_without_the_bridge(self, npu_env, monkeypatch, cache_ready):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )
        self._write_artifact(cache_dir, name, "local = 1\n")

        loaded = adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        )

        assert loaded is npu_env.loaded
        assert npu_env.loaded_py_codes == ["local = 1\n"]

    def test_local_miss_and_corrupt_artifact_are_misses(
        self, npu_env, monkeypatch, cache_ready
    ):
        adaptor, cache_dir, _ = cache_ready
        name = f"artifact_0.dddddddddddddddd{_SUFFIX}"
        monkeypatch.setattr(
            nx, "_npugraph_digest", lambda *a, **k: "dddddddddddddddd"
        )

        assert adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        ) is None

        self._write_artifact(cache_dir, name, "local = 1\n")
        npu_env.load_error = ValueError("bad source")
        assert adaptor.load(
            ("artifact_0", "unused"), _add_graph(), _inputs(), 0, None
        ) is None

    def test_compile_then_load_round_trip(self, npu_env, monkeypatch, cache_ready):
        """What compile() saved is what load() serves, under the same name."""
        adaptor, cache_dir, _ = cache_ready
        npu_env.use_compiler = True
        npu_env.compiled_code = "ev = _create_event_by_name('graph_0_ev_1')\n"

        _, handle = adaptor.compile(_add_graph(), _inputs(), {}, key="artifact_0")
        loaded = adaptor.load(handle, _add_graph(), _inputs(), 0, None)

        assert loaded is npu_env.loaded
        saved = open(handle[1]).read()
        assert npu_env.loaded_py_codes == [saved]
        assert "'gfx" in saved  # the persisted bytes are the namespaced ones


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
