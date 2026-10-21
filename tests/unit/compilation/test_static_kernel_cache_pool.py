# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.compilation.static_kernel_cache_pool (engine bridge).

Locks the bridge's architectural invariants: the static_kernel_cache_enable
gate runs BEFORE any static_kernel_cache import, pull serialization is
POD-scoped (the local world group's rank_in_group==0 pulls, per-pod
barrier), upload dedup is WORLD-scoped (world rank 0 alone uploads, and
only after the capture-end rendezvous), the group setup is collective
(it runs on every past-the-gate process, non-pullers included), the
reuse marker is cleared only by the pod's rank 0 right before its pull
and rewritten only on a gate-passed HIT (non-pullers never touch it),
hooks never raise, and the upload log protocol (silent on
not-attempted) holds. static_kernel_cache / torch.distributed /
parallel_state_ext / npugraph_ex_config are all injected as sys.modules
fakes.
"""
import enum
import logging
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import omni_npu.compilation.static_kernel_cache_pool as bridge
import omni_npu.compilation.static_kernel_cache_patch as skp

_REAL_SLEEP = time.sleep


def _vllm():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(model="/model/x"),
        additional_config={},
    )


class _FakePullResult(enum.Enum):
    HIT = "hit"
    MISS = "miss"
    CORRUPT = "corrupt"


class _FakeBundleLayout:
    def __init__(self, root):
        self.root = Path(root)


@pytest.fixture(autouse=True)
def _reset_upload_latch():
    """The capture-end rendezvous latch is per-process engine state."""
    bridge._upload_rendezvous_done = False
    yield
    bridge._upload_rendezvous_done = False


@pytest.fixture
def fake_dist(monkeypatch):
    """torch.distributed stand-in; rank/barrier behaviour is programmable.
    ``barrier_groups`` records the group argument of every barrier (None =
    default/world group)."""
    state = {"rank": 0, "initialized": True, "barrier": 0, "barrier_groups": [], "module": None}
    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: state["initialized"]
    dist.get_rank = lambda: state["rank"]

    def _barrier(group=None):
        state["barrier"] += 1
        state["barrier_groups"].append(group)

    dist.barrier = _barrier
    state["module"] = dist
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    return state


@pytest.fixture
def fake_groups(monkeypatch):
    """parallel_state_ext stand-in: local/world GroupCoordinator fakes with
    programmable rank_in_group; the init/get hooks are MagicMocks so tests
    can program initialization failures."""
    local = types.SimpleNamespace(rank_in_group=0, device_group=object())
    world = types.SimpleNamespace(rank_in_group=0, device_group=object())
    pse = types.ModuleType("omni_npu.v1.distributed.parallel_state_ext")
    pse.initialize_local_world_group = MagicMock()
    pse.get_local_world_group = MagicMock(return_value=local)
    pse.get_world_group = MagicMock(return_value=world)
    monkeypatch.setitem(sys.modules, "omni_npu.v1", types.ModuleType("omni_npu.v1"))
    monkeypatch.setitem(
        sys.modules, "omni_npu.v1.distributed", types.ModuleType("omni_npu.v1.distributed")
    )
    monkeypatch.setitem(
        sys.modules, "omni_npu.v1.distributed.parallel_state_ext", pse
    )
    return types.SimpleNamespace(pse=pse, local=local, world=world)


@pytest.fixture
def fake_static_kernel_cache(monkeypatch):
    """static_kernel_cache package fakes: bounded_pull/bounded_upload hooks
    are MagicMocks so tests program return values / exceptions; PullResult
    identity is preserved."""
    static_kernel_cache_store = types.ModuleType("static_kernel_cache.static_kernel_cache_store")
    static_kernel_cache_store.BundleLayout = _FakeBundleLayout
    static_kernel_cache_store.PullResult = _FakePullResult
    static_kernel_cache_store.bounded_pull = MagicMock()
    static_kernel_cache_store.bounded_upload = MagicMock()
    fingerprint = types.ModuleType("static_kernel_cache.fingerprint")
    fingerprint.compute_static_kernel_cache_fingerprint = (
        lambda vllm_config, additional, model_path: ("fp_test", {"f": 1})
    )
    pkg = types.ModuleType("static_kernel_cache")
    pkg.static_kernel_cache_store = static_kernel_cache_store
    pkg.fingerprint = fingerprint
    monkeypatch.setitem(sys.modules, "static_kernel_cache", pkg)
    monkeypatch.setitem(
        sys.modules, "static_kernel_cache.static_kernel_cache_store", static_kernel_cache_store
    )
    monkeypatch.setitem(sys.modules, "static_kernel_cache.fingerprint", fingerprint)
    return pkg


@pytest.fixture
def no_static_kernel_cache(monkeypatch):
    """A None sys.modules entry makes `import static_kernel_cache...` fail fast."""
    monkeypatch.setitem(sys.modules, "static_kernel_cache", None)


@pytest.fixture
def sk_config(monkeypatch, tmp_path):
    """npugraph_ex_config with a writable cache dir; enable is programmable."""
    holder = {"config": {"static_kernel_cache_enable": True, "static_kernel_cache_dir": str(tmp_path)}}
    cfg_module = types.ModuleType("omni_npu.compilation.npugraph_ex_config")
    cfg_module.get_aclgraph_config = lambda: types.SimpleNamespace(
        npugraph_ex_config=holder["config"]
    )
    monkeypatch.setitem(
        sys.modules, "omni_npu.compilation", types.ModuleType("omni_npu.compilation")
    )
    monkeypatch.setitem(
        sys.modules,
        "omni_npu.compilation.npugraph_ex_config",
        cfg_module,
    )
    return holder


@pytest.fixture
def fast_upload_thread(monkeypatch):
    """_wait(mock) blocks until the daemon thread has demonstrably reached
    the pool call, then drains the thread if it is still alive — polling for
    the thread by NAME alone races against a thread that finishes before the
    first enumeration."""
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def _wait(mock, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if mock.call_count >= 1:
                for t in threading.enumerate():
                    if t.name == "static-kernel-cache-upload":
                        t.join(max(0.1, deadline - time.monotonic()))
                return True
            _REAL_SLEEP(0.02)
        return False

    return _wait


def _marker_path(root: Path) -> Path:
    return root / "static_kernel_compile_outputs" / skp._POOL_MARKER


# ---------------------------------------------------------------------------
# Architectural invariant: the config gate precedes any static_kernel_cache import
# ---------------------------------------------------------------------------

def test_disabled_gate_never_imports_static_kernel_cache(
    monkeypatch, sk_config, fake_dist, caplog
):
    sk_config["config"]["static_kernel_cache_enable"] = False
    before = set(sys.modules)

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())
        bridge.schedule_static_kernel_cache_upload(_vllm())

    new = set(sys.modules) - before
    assert not any(name.startswith("static_kernel_cache") for name in new)
    assert "static_kernel_cache" not in caplog.text  # disabled is fully silent
    assert fake_dist["barrier"] == 0  # gate off: no rendezvous at all


# ---------------------------------------------------------------------------
# Pull hook: pod-scoped serialization and marker lifecycle
# ---------------------------------------------------------------------------

def test_pull_miss_clears_stale_marker(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, tmp_path, caplog
):
    _marker_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _marker_path(tmp_path).write_text("stale\n")
    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.return_value = (
        _FakePullResult.MISS
    )

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    assert not _marker_path(tmp_path).exists()
    assert "static_kernel_cache pool pull: miss (fingerprint=fp_test)" in caplog.text
    # Rendezvous is pod-scoped: the local world group's device group.
    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]


def test_pull_hit_rewrites_marker_with_fingerprint(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, tmp_path, caplog
):
    def _land_bundle(*args, **kwargs):
        # Contract of a HIT: the restore already landed the bundle under the
        # product dir, so the marker's parent exists by the time the bridge
        # writes it.
        (tmp_path / "static_kernel_compile_outputs").mkdir(
            parents=True, exist_ok=True
        )
        return _FakePullResult.HIT

    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.side_effect = _land_bundle

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    assert _marker_path(tmp_path).read_text() == "fp_test\n"
    assert "static_kernel_cache pool pull: hit (fingerprint=fp_test)" in caplog.text
    # Marker is on disk before the release barrier: every process's
    # boot-static decision (post-barrier) observes it.
    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]
    layout_arg, fingerprint_arg = fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.call_args[0]
    assert layout_arg.root == tmp_path
    assert fingerprint_arg == "fp_test"


def test_pull_swallows_hook_exceptions(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, tmp_path, caplog
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())  # must not raise

    assert "static_kernel_cache pool pull failed" in caplog.text
    assert not _marker_path(tmp_path).exists()
    # The rendezvous must hold even on failure: the other members are
    # already waiting at theirs.
    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]


def test_pull_local_rank1_barriers_without_touching_the_pool(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, tmp_path, caplog
):
    # The pull decision is POD-scoped: local rank 1 must not pull even at
    # world rank 0 (a single-pod multi-server deployment).
    fake_groups.local.rank_in_group = 1
    fake_dist["rank"] = 0
    # Regression (s10 hot boot 2026-10-20): a slowly spawned sibling's
    # entry-time kill used to delete the marker the pod rank 0 had just
    # written after its pull HIT, sinking the whole pod into full compile.
    # Non-pullers must never touch the marker.
    _marker_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _marker_path(tmp_path).write_text("fp_test\n")

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.assert_not_called()
    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]
    assert "static_kernel_cache pool pull:" not in caplog.text
    assert _marker_path(tmp_path).read_text() == "fp_test\n"
    # Group setup is collective: it ran on the non-puller too, before any
    # rank branching.
    fake_groups.pse.initialize_local_world_group.assert_called_once()


def test_missing_package_degrades_to_full_compile(
    sk_config, fake_dist, fake_groups, no_static_kernel_cache, tmp_path, caplog
):
    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    assert "static_kernel_cache package not importable" in caplog.text
    assert "continuing with full compile" in caplog.text
    assert not _marker_path(tmp_path).exists()
    # Early return must still reach the single rendezvous.
    assert fake_dist["barrier"] == 1


def test_multinode_second_pod_local_rank0_pulls(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, tmp_path, caplog
):
    # Two pods of eight servers each: world rank 8 is its pod's rank 0.
    # Each pod pulls its own copy; the barrier stays pod-local so the pods
    # boot independently.
    fake_dist["rank"] = 8
    fake_groups.local.rank_in_group = 0
    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.return_value = (
        _FakePullResult.MISS
    )

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.assert_called_once()
    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]


def test_multinode_second_pod_local_rank1_does_not_pull(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, caplog
):
    fake_dist["rank"] = 9
    fake_groups.local.rank_in_group = 1

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.assert_not_called()
    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]


def test_pull_abandoned_when_pod_group_unavailable(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, tmp_path, caplog
):
    # No silent fallback: pooling is abandoned entirely with no side
    # effects, and no barrier is owed (every rank fails identically).
    fake_groups.pse.initialize_local_world_group.side_effect = RuntimeError("no local size")
    fake_dist["rank"] = 0
    _marker_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _marker_path(tmp_path).write_text("stale\n")

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())  # must not raise

    assert "local world group unavailable" in caplog.text
    assert "abandoned" in caplog.text
    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.assert_not_called()
    assert fake_dist["barrier"] == 0
    # The kill lives inside the rank-0 pull path, which was never reached:
    # abandonment leaves the product dir untouched. A stale marker riding
    # an abandoned boot is the playbook's per-boot cleanup to prevent, not
    # this path's (the bridge comment covers the fresh-dir argument).
    assert _marker_path(tmp_path).exists()


def test_upload_abandoned_when_pod_group_unavailable(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, caplog
):
    fake_groups.pse.initialize_local_world_group.side_effect = RuntimeError("no local size")
    fake_dist["rank"] = 0

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())  # must not raise

    assert "local world group unavailable" in caplog.text
    assert fake_dist["barrier"] == 0
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.assert_not_called()
    assert "static_kernel_cache pool upload" not in caplog.text


def test_no_distributed_runtime_single_process_pulls_and_uploads(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, fast_upload_thread, caplog
):
    # No distributed runtime at all (tests, CLI-adjacent paths): the single
    # process is its own pod's rank 0 — it pulls, rendezvous are no-ops, and
    # the world-rank-0 upload decision defaults to "this process".
    fake_dist["initialized"] = False
    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.return_value = (
        _FakePullResult.MISS
    )

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())
        assert fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.called
        assert fake_dist["barrier"] == 0

        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert fake_dist["barrier"] == 0  # no rendezvous without a distributed runtime
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.assert_called_once()


# ---------------------------------------------------------------------------
# Upload hook: capture-end rendezvous, world-rank0 dedup, log protocol
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("upload_result", "expected_line"),
    [
        (True, "static_kernel_cache pool upload: committed or already present (fingerprint=fp_test)"),
        (False, "static_kernel_cache pool upload: skipped (bundle unusable) (fingerprint=fp_test)"),
    ],
)
def test_upload_outcome_log_lines(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, fast_upload_thread, caplog,
    upload_result, expected_line,
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.return_value = upload_result

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert expected_line in caplog.text


def test_upload_rendezvous_precedes_the_upload_thread(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, fast_upload_thread, caplog
):
    """The capture-end rendezvous runs synchronously in the boot thread —
    when the hook returns, every process's capture has been accounted for
    and the index is settled, so the upload thread needs no settle delay."""
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.return_value = True

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        # Synchronous, pod-scoped, and exactly once — before any thread work.
        assert fake_dist["barrier"] == 1
        assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert fake_dist["barrier"] == 1  # the upload itself adds no rendezvous


def test_upload_stays_silent_when_pool_not_attempted(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, fast_upload_thread, caplog
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.return_value = None

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert "static_kernel_cache pool upload:" not in caplog.text


def test_upload_swallows_hook_exceptions(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, fast_upload_thread, caplog
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert "static_kernel_cache pool upload failed" in caplog.text


def test_upload_missing_package_is_skipped_silently(
    sk_config, fake_dist, fake_groups, no_static_kernel_cache, caplog
):
    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        # No pool mock exists in this scenario; the thread's own log line is
        # the completion signal.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if "static_kernel_cache package not importable" in caplog.text:
                break
            _REAL_SLEEP(0.02)

    assert "static_kernel_cache package not importable" in caplog.text
    assert "static_kernel_cache pool upload:" not in caplog.text


def test_upload_dedup_is_world_rank0(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, caplog
):
    # A non-zero world rank reaches the rendezvous (the pod must account for
    # every capture) but never uploads: one canonical commit per cluster.
    fake_dist["rank"] = 8
    fake_groups.local.rank_in_group = 0

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())

    assert fake_dist["barrier"] == 1
    assert fake_dist["barrier_groups"] == [fake_groups.local.device_group]
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.assert_not_called()
    assert "static_kernel_cache pool upload" not in caplog.text


def test_upload_reentry_skips_rendezvous_and_upload(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, fast_upload_thread, caplog
):
    # ACLGraph recapture re-runs capture_model on a single process; the
    # re-entry must not wait on a rendezvous the siblings will never join.
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.return_value = True

    with caplog.at_level(logging.DEBUG):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )
        bridge.schedule_static_kernel_cache_upload(_vllm())

    assert fake_dist["barrier"] == 1  # exactly one rendezvous per engine life
    assert fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.call_count == 1
    assert "skipping recapture-time upload" in caplog.text


def test_upload_barrier_failure_never_raises(
    sk_config, fake_dist, fake_groups, fake_static_kernel_cache, monkeypatch, caplog
):
    def _boom(group=None):
        raise RuntimeError("hccl down")

    monkeypatch.setattr(fake_dist["module"], "barrier", _boom)

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())  # must not raise

    assert "static_kernel_cache pool upload scheduling failed" in caplog.text
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.assert_not_called()
