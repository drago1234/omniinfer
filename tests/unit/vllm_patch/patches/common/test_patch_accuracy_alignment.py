# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

PATCH_PATH = (
    next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "omni" / "diagnostics" / "accuracy_alignment").is_dir()
    )
    / "omni/diagnostics/accuracy_alignment/patch_accuracy_alignment.py"
)


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _PatchBase:
    @classmethod
    def apply_bypass_conflict(cls, *names):
        for name in names:
            setattr(cls, f"_upstream_{name.lstrip('_')}", getattr(cls._target, name))
        for name in cls._attr_names_to_apply:
            replacement = cls.__dict__.get(name)
            assert replacement is not None
            setattr(cls._target, name, replacement)


@pytest.fixture(autouse=True)
def _restore_omni_npu_modules():
    """Prevent this patch loader's stub modules from leaking to other UTs."""
    prefix = "omni_npu"
    snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(f"{prefix}.")
    }
    yield
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)
    sys.modules.update(snapshot)


def _load_patch(enabled):
    registered = []
    calls = []

    class Runner:
        def load_model(self, value=None):
            calls.append(("load_model", value))
            return "loaded"

        def _dummy_run(self, value=None):
            calls.append(("dummy", value))
            return "dummy-result"

        def profile_run(self, value=None):
            calls.append(("profile", value))
            return "profile-result"

    def register_patch(name, target):
        def decorate(cls):
            cls._target = target
            registered.append((name, cls))
            return cls

        return decorate

    @contextmanager
    def guard():
        calls.append(("guard_enter",))
        try:
            yield
        finally:
            calls.append(("guard_exit",))

    def install(runner):
        calls.append(("install", runner))

    _module("omni_npu")
    _module("omni_npu.diagnostics.accuracy_alignment")
    _module(
        "omni_npu.diagnostics.accuracy_alignment.config",
        alignment_requested=lambda: enabled,
    )
    _module(
        "omni_npu.diagnostics.accuracy_alignment.installer",
        install_accuracy_alignment=install,
    )
    _module(
        "omni_npu.diagnostics.accuracy_alignment.train_ref",
        dummy_run_guard=guard,
    )
    _module("omni_npu.vllm_patches")
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_PatchBase,
        register_patch=register_patch,
    )
    _module("omni_npu.worker")
    _module("omni_npu.worker.npu_model_runner", NPUModelRunner=Runner)

    module_name = (
        "omni_npu.diagnostics.accuracy_alignment.patch_accuracy_alignment"
    )
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, Runner, registered, calls


def test_disabled_mode_registers_nothing():
    _module_under_test, _runner, registered, _calls = _load_patch(False)
    assert registered == []


def test_patch_chains_load_and_guards_dummy_and_profile_runs():
    module, runner_cls, registered, calls = _load_patch(True)
    assert [name for name, _cls in registered] == ["AccuracyAlignmentTrainRefPatch"]
    module.AccuracyAlignmentPatch.apply()
    runner = runner_cls()
    assert runner.load_model("weights") == "loaded"
    assert runner._dummy_run("shape") == "dummy-result"
    assert runner.profile_run("memory") == "profile-result"
    assert calls == [
        ("load_model", "weights"),
        ("install", runner),
        ("guard_enter",),
        ("dummy", "shape"),
        ("guard_exit",),
        ("guard_enter",),
        ("profile", "memory"),
        ("guard_exit",),
    ]
