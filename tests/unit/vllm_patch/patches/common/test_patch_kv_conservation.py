# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_kv_conservation (vLLM 0.25.1 adaptation).

Every vLLM and omni_npu dependency is stubbed, so the test body itself needs
nothing but the standard library -- no vllm, no torch, no NPU.  It still sits
next to its siblings under tests/unit/, whose conftest imports torch, so a
torch-less environment skips the whole directory rather than this file alone.
Run either way::

    pytest tests/unit/vllm_patch/patches/common/test_patch_kv_conservation.py -v
    python  tests/unit/vllm_patch/patches/common/test_patch_kv_conservation.py
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PATCH_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / "omni" / "vllm_patches").is_dir())
    / "omni/vllm_patches/patches/common/patch_kv_conservation.py"
)

_KV_TESTS_DIR = str(Path(__file__).resolve().parents[4])
if _KV_TESTS_DIR not in sys.path:
    sys.path.insert(0, _KV_TESTS_DIR)
from unit.vllm_patch.patches.patch_test_utils import (  # noqa: E402
    PatchCallRecorder as _Recorder,
    StubVLLMPatch as _VLLMPatch,
    install_named_module as _module,
    preserved_sys_modules,
    run_standalone_tests,
    silent_logger,
)


@pytest.fixture(autouse=True)
def _restore_sys_modules_after_test():
    """Keep the offline vLLM/omni_npu stubs local to each test.

    Without this, the stub ``vllm`` / ``vllm.v1.engine`` /
    ``omni_npu.vllm_patches.core`` modules installed below leak into
    sys.modules and break every later test in the same pytest process that
    imports the real packages (EngineCoreOutput ImportError, missing
    VLLMPatch.apply_bypass_conflict, ...).
    """
    with preserved_sys_modules():
        yield


INTERVAL = 1000


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


def _install_stubs(recorder, interval=INTERVAL):
    """(Re)install the stub tree; returns the fake targets under test."""

    class EngineCoreProc:
        def has_work(self):
            return self._has_work

        def _process_input_queue(self):
            recorder.calls.append(("orig_process_input_queue",))

    def _check(scheduler, trigger):
        recorder.calls.append(("check", scheduler, trigger))

    def _check_idle(scheduler, interval):
        recorder.calls.append(("check_idle", scheduler, interval))
        return True

    def _due(scheduler, interval):
        recorder.calls.append(("due", scheduler, interval))
        return getattr(scheduler, "current_step", 0) % interval == 0

    _module("vllm")
    _module("vllm.logger", init_logger=silent_logger)
    _module("vllm.v1")
    _module("vllm.v1.engine")
    _module("vllm.v1.engine.core", EngineCoreProc=EngineCoreProc)

    envs = _module(
        "omni_npu.envs", OMNI_KV_CONSERVATION_CHECK_INTERVAL=interval
    )

    def _interval():
        # Mirrors the real kv_conservation.interval(): read envs on every
        # call so runtime flips of OMNI_KV_CONSERVATION_CHECK_INTERVAL are
        # observable without re-importing the patch.
        return int(envs.OMNI_KV_CONSERVATION_CHECK_INTERVAL)
    _module("omni_npu", envs=envs)
    _module("omni_npu.diagnostics")
    kv_conservation = _module(
        "omni_npu.diagnostics.kv_conservation",
        check=_check,
        check_idle=_check_idle,
        due=_due,
        interval=_interval,
    )
    _module("omni_npu.vllm_patches")
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=recorder.register_patch,
    )

    # patch_kv_conservation imports the module object and reads attributes
    # off it (kv_conservation.check_idle), so rebind what the module attr
    # lookup sees.
    sys.modules["omni_npu.diagnostics"].kv_conservation = kv_conservation

    return (EngineCoreProc,)


def _load_patch_module(interval=INTERVAL):
    recorder = _Recorder()
    targets = _install_stubs(recorder, interval)
    spec = importlib.util.spec_from_file_location("_patch_kv_conservation", PATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, recorder, targets


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_registers_only_the_idle_mount():
    """The periodic step trigger moved into the forked Scheduler.schedule
    (patch_scheduler_spec_decode_pad); only the idle mount registers here."""
    _mod, rec, (_EngineCoreProc,) = _load_patch_module()

    assert [(n, attrs) for n, _t, attrs in rec.registered] == [
        ("KVConservationIdlePatch", ("_process_input_queue",)),
    ]
    assert not hasattr(_mod, "KVConservationSchedulerPatch")


def test_idle_mount_checks_only_when_engine_has_no_work():
    """has_work() gates the check; upstream polling always runs afterwards."""
    mod, rec, _targets = _load_patch_module()
    mod.KVConservationIdlePatch.apply()

    engine = mod.KVConservationIdlePatch._target()
    engine.scheduler = "the-scheduler"

    engine._has_work = False
    engine._process_input_queue()
    assert rec.calls == [
        ("check_idle", "the-scheduler", INTERVAL),
        ("orig_process_input_queue",),
    ]

    rec.calls.clear()
    engine._has_work = True
    engine._process_input_queue()
    assert rec.calls == [("orig_process_input_queue",)]


def test_idle_mount_disabled_leaves_no_trace():
    mod, rec, _targets = _load_patch_module(interval=0)
    mod.KVConservationIdlePatch.apply()

    engine = mod.KVConservationIdlePatch._target()
    engine.scheduler = "the-scheduler"
    engine._has_work = False
    engine._process_input_queue()

    assert rec.calls == [("orig_process_input_queue",)]


def test_interval_env_is_read_lazily_each_call():
    """_interval() reads envs on every poll, so runtime changes to
    OMNI_KV_CONSERVATION_CHECK_INTERVAL are observable without a restart."""
    mod, rec, _targets = _load_patch_module()
    mod.KVConservationIdlePatch.apply()

    engine = mod.KVConservationIdlePatch._target()
    engine.scheduler = "the-scheduler"

    # Flip the env after the patch module was already imported.
    envs = sys.modules["omni_npu.envs"]
    envs.OMNI_KV_CONSERVATION_CHECK_INTERVAL = 0
    try:
        engine._has_work = False
        engine._process_input_queue()
        assert rec.calls == [("orig_process_input_queue",)]
    finally:
        envs.OMNI_KV_CONSERVATION_CHECK_INTERVAL = INTERVAL


if __name__ == "__main__":
    run_standalone_tests(dict(globals()))
