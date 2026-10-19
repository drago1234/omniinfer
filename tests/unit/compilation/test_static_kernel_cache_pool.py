# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.compilation.static_kernel_cache_pool (engine bridge).

Locks the bridge's architectural invariants: the static_kernel_cache_enable gate runs
BEFORE any static_kernel_cache import, the reuse marker is killed up-front and
rewritten only on a gate-passed HIT, hooks never raise, and the upload log
protocol (silent on not-attempted) holds. static_kernel_cache / torch.distributed /
npugraph_ex_config are all injected as sys.modules fakes.
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


@pytest.fixture
def fake_dist(monkeypatch):
    """torch.distributed stand-in; rank/barrier behaviour is programmable."""
    state = {"rank": 0, "barrier": 0}
    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: True
    dist.get_rank = lambda: state["rank"]
    dist.barrier = lambda: state.__setitem__("barrier", state["barrier"] + 1)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    return state


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
    """Drop the 5s settle delay. _wait(mock) blocks until the daemon thread
    has demonstrably reached the pool call, then drains the thread if it is
    still alive — polling for the thread by NAME alone races against a
    thread that finishes before the first enumeration."""
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
# Pull hook: marker lifecycle
# ---------------------------------------------------------------------------

def test_pull_miss_clears_stale_marker(
    sk_config, fake_dist, fake_static_kernel_cache, tmp_path, caplog
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
    assert fake_dist["barrier"] == 1


def test_pull_hit_rewrites_marker_with_fingerprint(
    sk_config, fake_dist, fake_static_kernel_cache, tmp_path, caplog
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
    # Marker is on disk before the release barrier: every rank's boot-static
    # decision (post-barrier) observes it.
    assert fake_dist["barrier"] == 1
    layout_arg, fingerprint_arg = fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.call_args[0]
    assert layout_arg.root == tmp_path
    assert fingerprint_arg == "fp_test"


def test_pull_swallows_hook_exceptions(
    sk_config, fake_dist, fake_static_kernel_cache, tmp_path, caplog
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())  # must not raise

    assert "static_kernel_cache pool pull failed" in caplog.text
    assert not _marker_path(tmp_path).exists()
    # The rendezvous must hold even on failure: non-zero ranks are already
    # waiting at theirs.
    assert fake_dist["barrier"] == 1


def test_nonzero_rank_barriers_without_touching_the_pool(
    sk_config, fake_dist, fake_static_kernel_cache, caplog
):
    fake_dist["rank"] = 1

    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    fake_static_kernel_cache.static_kernel_cache_store.bounded_pull.assert_not_called()
    assert fake_dist["barrier"] == 1
    assert "static_kernel_cache pool pull:" not in caplog.text


def test_missing_package_degrades_to_full_compile(
    sk_config, fake_dist, no_static_kernel_cache, tmp_path, caplog
):
    with caplog.at_level(logging.INFO):
        bridge.maybe_pull_static_kernel_cache(_vllm())

    assert "static_kernel_cache package not importable" in caplog.text
    assert "continuing with full compile" in caplog.text
    assert not _marker_path(tmp_path).exists()
    # Early return must still reach the single rendezvous.
    assert fake_dist["barrier"] == 1


# ---------------------------------------------------------------------------
# Upload hook: log protocol and never-raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("upload_result", "expected_line"),
    [
        (True, "static_kernel_cache pool upload: committed or already present (fingerprint=fp_test)"),
        (False, "static_kernel_cache pool upload: skipped (bundle unusable) (fingerprint=fp_test)"),
    ],
)
def test_upload_outcome_log_lines(
    sk_config, fake_dist, fake_static_kernel_cache, fast_upload_thread, caplog,
    upload_result, expected_line,
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.return_value = upload_result

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert expected_line in caplog.text


def test_upload_stays_silent_when_pool_not_attempted(
    sk_config, fake_dist, fake_static_kernel_cache, fast_upload_thread, caplog
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.return_value = None

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert "static_kernel_cache pool upload:" not in caplog.text


def test_upload_swallows_hook_exceptions(
    sk_config, fake_dist, fake_static_kernel_cache, fast_upload_thread, caplog
):
    fake_static_kernel_cache.static_kernel_cache_store.bounded_upload.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.INFO):
        bridge.schedule_static_kernel_cache_upload(_vllm())
        assert fast_upload_thread(
            fake_static_kernel_cache.static_kernel_cache_store.bounded_upload
        )

    assert "static_kernel_cache pool upload failed" in caplog.text


def test_upload_missing_package_is_skipped_silently(
    sk_config, fake_dist, no_static_kernel_cache, caplog
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
