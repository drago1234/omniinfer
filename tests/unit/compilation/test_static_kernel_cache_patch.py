# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.compilation.static_kernel_cache_patch (vendored static-kernel
cache gate): file-identity instance enumeration, multi-instance patching with
from-import rebinding, the boot-static reuse decision, and the fail-soft
layout gate. Everything runs on pure stdlib fakes — no torch_npu required.
"""
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

import omni_npu.compilation.static_kernel_cache_patch as skp

_FAKE_SK_SRC = """
STATE = {"original": 0, "install": 0, "multicard": True}


def _is_multicard_env_valid():
    return STATE["multicard"]


def compile_static_kernel(fx_func, *args, config=None, build_dir=None, **kwargs):
    STATE["original"] += 1
    return "original"


def _install_or_compile_static_kernel(fx_func, *args, **kwargs):
    STATE["install"] += 1
    return "install"
"""

_NO_INSTALL_SK_SRC = _FAKE_SK_SRC.split("def _install_or_compile")[0]


@pytest.fixture(autouse=True)
def _fresh_boot_mode():
    skp._boot_mode.clear()
    skp._boot_mode.update({"decided": False, "use_cache": False})
    yield


@pytest.fixture
def fake_torch_stack(monkeypatch):
    """Minimal torch/torch_npu stand-ins for the reuse branch of the
    vendored entry (it reads the CANN version and the deterministic flag)."""
    torch_fake = types.ModuleType("torch")
    torch_fake.are_deterministic_algorithms_enabled = lambda: False
    torch_npu_fake = types.ModuleType("torch_npu")
    torch_npu_fake.npu = types.SimpleNamespace(
        utils=types.SimpleNamespace(get_cann_version=lambda: "test-cann")
    )
    monkeypatch.setitem(sys.modules, "torch", torch_fake)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_fake)


def _load_module(monkeypatch, tmp_path, name, source, variant):
    path = tmp_path / f"static_kernel_{variant}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sk_factory(monkeypatch, tmp_path):
    """Load fake static_kernel modules under arbitrary names; the file
    identity (not the name) decides enumeration."""
    loaded = []

    def _make(name, *, with_install=True, variant=None):
        variant = len(loaded) if variant is None else variant
        source = _FAKE_SK_SRC if with_install else _NO_INSTALL_SK_SRC
        mod = _load_module(monkeypatch, tmp_path, name, source, variant)
        loaded.append(name)
        return mod

    return _make


def _make_pool_cache(root: Path, present: bool) -> str:
    outputs = root / "static_kernel_compile_outputs"
    if present:
        (outputs / "ts_pool_outputs").mkdir(parents=True)
        (outputs / "ts_pool_outputs" / "pkg.run").write_text("run")
        (outputs / skp._POOL_MARKER).write_text("fp\n")
    return str(root)


def _use_seed_names(monkeypatch, *names):
    """Point _SK_NAMES at the fakes so _find_instances never touches the
    real npugraph_ex / torch_npu packages."""
    monkeypatch.setattr(skp, "_SK_NAMES", tuple(names))


# ---------------------------------------------------------------------------
# File-identity enumeration (_modules_sharing_file / _find_instances)
# ---------------------------------------------------------------------------

def test_same_file_two_names_is_enumerated_once_each(sk_factory):
    mod_a = sk_factory("fake_a._acl_concrete_graph.static_kernel")
    mod_b = sk_factory("fake_b._acl_concrete_graph.static_kernel", variant=0)
    assert mod_a is not mod_b
    assert mod_a.__file__ == mod_b.__file__

    found = skp._modules_sharing_file(mod_a.__file__)
    assert sorted(m.__name__ for m in found) == sorted(
        ["fake_a._acl_concrete_graph.static_kernel",
         "fake_b._acl_concrete_graph.static_kernel"]
    )


def test_same_suffix_different_file_is_excluded(sk_factory, tmp_path):
    anchor = sk_factory("fake_anchor._acl_concrete_graph.static_kernel")
    decoy = sk_factory("fake_decoy._acl_concrete_graph.static_kernel")

    found = skp._modules_sharing_file(anchor.__file__)
    names = [m.__name__ for m in found]
    assert "fake_anchor._acl_concrete_graph.static_kernel" in names
    assert "fake_decoy._acl_concrete_graph.static_kernel" not in names


def test_fileless_modules_are_skipped(monkeypatch, sk_factory):
    anchor = sk_factory("fake_anchor._acl_concrete_graph.static_kernel")
    bare = types.ModuleType("fake_no_file_module")  # no __file__ at all
    monkeypatch.setitem(sys.modules, bare.__name__, bare)

    found = skp._modules_sharing_file(anchor.__file__)
    assert bare not in found
    assert anchor in found


def test_find_instances_returns_file_sharing_seed_modules(
    monkeypatch, sk_factory
):
    seed0 = sk_factory("fake_seed0._acl_concrete_graph.static_kernel")
    seed1 = sk_factory("fake_seed1._acl_concrete_graph.static_kernel", variant=0)
    _use_seed_names(
        monkeypatch,
        "fake_seed0._acl_concrete_graph.static_kernel",
        "fake_seed1._acl_concrete_graph.static_kernel",
    )

    instances = skp._find_instances()
    assert set(instances) == {seed0, seed1}


# ---------------------------------------------------------------------------
# apply_static_kernel_cache_patch: layout gate, idempotency, routing, rebind
# ---------------------------------------------------------------------------

def test_apply_rejects_wrong_layout_and_leaves_original(monkeypatch, sk_factory, tmp_path):
    broken = sk_factory("fake_seed0._acl_concrete_graph.static_kernel",
                        with_install=False)
    _use_seed_names(monkeypatch, "fake_seed0._acl_concrete_graph.static_kernel")
    original = broken.compile_static_kernel

    assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": str(tmp_path)}) is False
    assert broken.compile_static_kernel is original
    assert not getattr(broken.compile_static_kernel, skp._MARKER, False)


def test_apply_is_idempotent(monkeypatch, sk_factory, tmp_path):
    mod = sk_factory("fake_seed0._acl_concrete_graph.static_kernel")
    _use_seed_names(monkeypatch, "fake_seed0._acl_concrete_graph.static_kernel")

    assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": str(tmp_path)}) is True
    first = mod.compile_static_kernel
    assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": str(tmp_path)}) is True
    assert mod.compile_static_kernel is first


def test_each_instance_routes_to_its_own_helpers(
    monkeypatch, sk_factory, tmp_path, fake_torch_stack, caplog
):
    mod_a = sk_factory("fake_a._acl_concrete_graph.static_kernel")
    # Same physical file, second module name — the multi-instance case.
    mod_b = sk_factory("fake_b._acl_concrete_graph.static_kernel", variant=0)
    _use_seed_names(
        monkeypatch,
        "fake_a._acl_concrete_graph.static_kernel",
        "fake_b._acl_concrete_graph.static_kernel",
    )
    cache_dir = _make_pool_cache(tmp_path, present=True)

    with caplog.at_level(logging.INFO):
        assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": cache_dir})
        mod_a.compile_static_kernel("fx")
        mod_b.compile_static_kernel("fx")

    # Each vendored closure calls into ITS OWN module instance's helpers —
    # module-local state (Gloo groups, dump state) must never mix.
    assert mod_a.STATE["install"] == 1 and mod_a.STATE["original"] == 0
    assert mod_b.STATE["install"] == 1 and mod_b.STATE["original"] == 0
    assert "static_kernel_cache boot mode: cache-reuse" in caplog.text


def test_from_import_copies_are_rebound_but_foreign_bindings_untouched(
    monkeypatch, sk_factory, tmp_path
):
    mod = sk_factory("fake_seed0._acl_concrete_graph.static_kernel")
    _use_seed_names(monkeypatch, "fake_seed0._acl_concrete_graph.static_kernel")

    importer = types.ModuleType("fake_importer")
    importer.compile_static_kernel = mod.compile_static_kernel  # from-import copy
    monkeypatch.setitem(sys.modules, importer.__name__, importer)

    def _unrelated_compile_static_kernel():
        return "unrelated"

    foreign = types.ModuleType("fake_foreign")
    foreign.compile_static_kernel = _unrelated_compile_static_kernel
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)

    assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": str(tmp_path)})
    # The from-import copy of the original is rebound to a vendored entry...
    assert getattr(importer.compile_static_kernel, skp._MARKER, False)
    # ...while a same-named function born outside the instance is untouched.
    assert foreign.compile_static_kernel is _unrelated_compile_static_kernel


# ---------------------------------------------------------------------------
# _cache_present and the frozen boot-static decision
# ---------------------------------------------------------------------------

def test_cache_present_requires_marker_and_run_packages(tmp_path):
    assert skp._cache_present(_make_pool_cache(tmp_path, present=True)) is True

    run_only = tmp_path / "run_only"
    (run_only / "static_kernel_compile_outputs" / "ts_pool_outputs").mkdir(
        parents=True
    )
    (run_only / "static_kernel_compile_outputs" / "ts_pool_outputs" / "p.run").write_text("x")
    assert skp._cache_present(str(run_only)) is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert skp._cache_present(str(empty)) is False


def test_boot_miss_routes_to_original(monkeypatch, sk_factory, tmp_path, caplog):
    mod = sk_factory("fake_seed0._acl_concrete_graph.static_kernel")
    _use_seed_names(monkeypatch, "fake_seed0._acl_concrete_graph.static_kernel")

    with caplog.at_level(logging.INFO):
        assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": str(tmp_path)})
        mod.compile_static_kernel("fx")

    assert mod.STATE["original"] == 1 and mod.STATE["install"] == 0
    assert "static_kernel_cache boot mode: full-compile" in caplog.text


def test_boot_decision_is_frozen_after_first_call(
    monkeypatch, sk_factory, tmp_path, fake_torch_stack, caplog
):
    mod = sk_factory("fake_seed0._acl_concrete_graph.static_kernel")
    _use_seed_names(monkeypatch, "fake_seed0._acl_concrete_graph.static_kernel")
    cache_dir = _make_pool_cache(tmp_path, present=True)

    with caplog.at_level(logging.INFO):
        assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": cache_dir})
        mod.compile_static_kernel("fx")
        # The cache disappears mid-boot: the decision must NOT be re-taken.
        import shutil

        shutil.rmtree(tmp_path / "static_kernel_compile_outputs")
        mod.compile_static_kernel("fx")

    assert mod.STATE["install"] == 2 and mod.STATE["original"] == 0
    assert caplog.text.count("static_kernel_cache boot mode:") == 1


def test_disable_escape_hatch_reaches_original(
    monkeypatch, sk_factory, tmp_path, fake_torch_stack
):
    mod = sk_factory("fake_seed0._acl_concrete_graph.static_kernel")
    _use_seed_names(monkeypatch, "fake_seed0._acl_concrete_graph.static_kernel")
    cache_dir = _make_pool_cache(tmp_path, present=True)

    assert skp.apply_static_kernel_cache_patch({"static_kernel_cache_dir": cache_dir})
    mod.compile_static_kernel(
        "fx", config={"disable_static_kernel_compile_cache": True}
    )
    assert mod.STATE["original"] == 1 and mod.STATE["install"] == 0


def test_no_instances_fails_soft(monkeypatch, tmp_path, caplog):
    _use_seed_names(monkeypatch, "fake_missing._acl_concrete_graph.static_kernel")
    with caplog.at_level(logging.WARNING):
        assert skp.apply_static_kernel_cache_patch(
            {"static_kernel_cache_dir": str(tmp_path)}
        ) is False
    assert "falling back to recompile-every-boot" in caplog.text
