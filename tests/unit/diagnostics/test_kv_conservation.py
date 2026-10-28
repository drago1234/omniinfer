# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the KV cache block conservation check.

Mirrors the fake plumbing of test_engine_stats.py: stub block / manager /
scheduler objects shaped like the vllm 0.25.1 BlockPool /
SingleTypeKVCacheManager / Scheduler surface the check reads, so no vllm,
torch or NPU is needed and every branch of the reconciliation is reachable
in-process. The check must never raise into the engine loop, so a large
share of these tests assert the degrade-to-error / debug-log paths.
"""

import logging

import pytest

from omni_npu.diagnostics import kv_conservation

pytestmark = pytest.mark.unit

INTERVAL = 25  # arbitrary local default; every test that cares sets it via env


class FakeBlock:
    def __init__(self, block_id, is_null=False):
        self.block_id = block_id
        self.is_null = is_null


class FakeFreeQueue:
    def __init__(self, num_free_blocks=0):
        self.num_free_blocks = num_free_blocks


class FakeBlockPool:
    def __init__(self, num_gpu_blocks, free, null_block_id=0):
        self.num_gpu_blocks = num_gpu_blocks
        self.free_block_queue = FakeFreeQueue(free)
        self.null_block = FakeBlock(null_block_id, is_null=True)


class FakeManager:
    def __init__(self, ledger=None):
        self.req_to_blocks = ledger or {}


class FakeCoordinator:
    def __init__(self, managers):
        self.single_type_managers = managers


class FakeKVCacheManager:
    def __init__(self, pool, managers):
        self.block_pool = pool
        self.coordinator = FakeCoordinator(managers)


class FakeScheduler:
    """Shaped like the Scheduler surface kv_conservation.collect reads."""

    def __init__(self, num_gpu_blocks, free, ledgers=None, requests=None,
                 deferred_frees=None, null_block_id=0):
        ledgers = ledgers or {}
        managers = [FakeManager(ledger) for ledger in ledgers]
        self.kv_cache_manager = FakeKVCacheManager(
            FakeBlockPool(num_gpu_blocks, free, null_block_id), managers
        )
        self.requests = requests if requests is not None else {}
        # ledger keys default to exactly the live requests
        if requests is None:
            for ledger in ledgers:
                for req_id in ledger:
                    self.requests.setdefault(req_id, object())
        if deferred_frees is not None:
            self.deferred_frees = deferred_frees


@pytest.fixture(autouse=True)
def _fresh_module_state():
    kv_conservation.reset()
    yield
    kv_conservation.reset()


@pytest.fixture
def interval_env(monkeypatch):
    """Pin OMNI_KV_CONSERVATION_CHECK_INTERVAL and return a setter."""
    monkeypatch.setenv("OMNI_KV_CONSERVATION_CHECK_INTERVAL", str(INTERVAL))

    def _set(value):
        monkeypatch.setenv("OMNI_KV_CONSERVATION_CHECK_INTERVAL", str(value))

    return _set


def _balanced_scheduler():
    """8-block pool: null + 3 free + 4 held by one live request."""
    ledger = {"r0": [FakeBlock(2), FakeBlock(3), FakeBlock(4), FakeBlock(5)]}
    return FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[ledger])


# ---------------------------------------------------------------------------
# interval(): env read, invalid values, disable
# ---------------------------------------------------------------------------


def test_interval_reads_env():
    assert kv_conservation.interval() > 0  # env or the documented default


def test_interval_env_override_roundtrip(monkeypatch):
    monkeypatch.setenv("OMNI_KV_CONSERVATION_CHECK_INTERVAL", "77")
    assert kv_conservation.interval() == 77


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_interval_non_positive_disables(monkeypatch, raw):
    monkeypatch.setenv("OMNI_KV_CONSERVATION_CHECK_INTERVAL", raw)
    assert kv_conservation.interval() == int(raw)


def test_interval_invalid_falls_back_to_disabled(monkeypatch, caplog):
    monkeypatch.setenv("OMNI_KV_CONSERVATION_CHECK_INTERVAL", "not-a-number")
    with caplog.at_level(logging.WARNING, logger=kv_conservation.logger.name):
        assert kv_conservation.interval() == 0
    assert any("invalid" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# collect(): the reconciliation facts
# ---------------------------------------------------------------------------


def test_collect_balanced_pool_is_ok():
    facts = kv_conservation.collect(_balanced_scheduler())
    assert facts["ok"] is True
    assert facts["total"] == 8
    assert facts["free"] == 3
    assert facts["held"] == 4
    assert facts["deferred"] == 0
    assert facts["orphan"] == {}
    assert facts["holders"] == {"r0": 4}
    assert facts["delta"] == 0


def test_collect_shared_blocks_dedup_across_managers():
    # Two managers (full-attention + SWA) share the prefix blocks 2, 3;
    # block ids are unique across the one pool, so held counts them once.
    ledger_a = {"r0": [FakeBlock(2), FakeBlock(3), FakeBlock(4)]}
    ledger_b = {"r0": [FakeBlock(2), FakeBlock(3), FakeBlock(6)]}
    sched = FakeScheduler(num_gpu_blocks=8, free=2, ledgers=[ledger_a, ledger_b])
    facts = kv_conservation.collect(sched)
    # held = {2, 3, 4, 6} = 4; free 2 + held 4 + null 1 == 7 != 8 -> delta -1
    assert facts["held"] == 4
    assert facts["holders"] == {"r0": 6}  # per-request report sums raw entries
    assert facts["delta"] == -1
    assert facts["ok"] is False


def test_collect_null_blocks_on_ledger_are_ignored():
    # null block appears many times on hybrid/SWA ledgers; it must never be
    # counted as held (its ref_cnt is not maintained).
    ledger = {
        "r0": [FakeBlock(2), FakeBlock(0, is_null=True)],
        "r1": [FakeBlock(0, is_null=True), FakeBlock(3)],
    }
    sched = FakeScheduler(num_gpu_blocks=5, free=2, ledgers=[ledger])
    facts = kv_conservation.collect(sched)
    assert facts["held"] == 2
    assert facts["holders"] == {"r0": 1, "r1": 1}
    assert facts["ok"] is True  # 2 free + 2 held + 1 null == 5


def test_collect_deferred_frees_count_as_transiently_held():
    # Blocks popped from the ledger ahead of their pool return (async sched /
    # PP overlap): neither free nor on any ledger, must join the held side.
    ledger = {"r0": [FakeBlock(4), FakeBlock(5)]}
    deferred = [(3, [FakeBlock(6)])]
    sched = FakeScheduler(
        num_gpu_blocks=8, free=4, ledgers=[ledger], deferred_frees=deferred
    )
    facts = kv_conservation.collect(sched)
    assert facts["deferred"] == 1
    assert facts["held"] == 3  # 2 ledger + 1 deferred
    assert facts["ok"] is True  # 4 free + 3 held + 1 null == 8
    # Without the deferred term the same pool would look short by one block,
    # i.e. every overlapping batch would read as a leak.
    facts_no_deferred = kv_conservation.collect(
        FakeScheduler(num_gpu_blocks=8, free=4, ledgers=[ledger])
    )
    assert facts_no_deferred["delta"] == -1
    assert facts_no_deferred["ok"] is False


def test_collect_deferred_free_without_block_id_is_skipped():
    ledger = {"r0": [FakeBlock(4)]}
    deferred = [(3, [FakeBlock(None), "not-a-block"])]
    sched = FakeScheduler(
        num_gpu_blocks=5, free=3, ledgers=[ledger], deferred_frees=deferred
    )
    facts = kv_conservation.collect(sched)
    assert facts["deferred"] == 0
    assert facts["ok"] is True  # 3 + 1 + 1 == 5


def test_collect_leaked_pool_shortfall_is_delta_negative():
    # A zombie leak: two blocks were freed from the ledger but never returned
    # to the pool -> free is short by 2.
    ledger = {"r0": [FakeBlock(4), FakeBlock(5)]}
    sched = FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[ledger])
    facts = kv_conservation.collect(sched)
    # free 3 + held 2 + 1 = 6 < 8
    assert facts["delta"] == -2
    assert facts["ok"] is False


def test_collect_double_free_shows_as_over_count():
    # A double return drives free above the complement of held blocks.
    ledger = {"r0": [FakeBlock(4)]}
    sched = FakeScheduler(num_gpu_blocks=8, free=7, ledgers=[ledger])
    facts = kv_conservation.collect(sched)
    # free 7 + held 1 + 1 = 9 > 8
    assert facts["delta"] == 1
    assert facts["ok"] is False


def test_collect_orphan_ledger_entries_flagged_even_when_balanced():
    # The request is gone from scheduler.requests but its blocks were never
    # returned: arithmetic can still balance while the blocks are zombies.
    ledger = {"ghost": [FakeBlock(4), FakeBlock(5)]}
    requests = {}
    sched = FakeScheduler(
        num_gpu_blocks=8, free=5, ledgers=[ledger], requests=requests
    )
    facts = kv_conservation.collect(sched)
    assert facts["delta"] == 0
    assert facts["orphan"] == {"ghost": 2}
    assert facts["ok"] is False  # zombie is a violation by itself


def test_collect_all_null_ledger_entry_is_skipped():
    # A request whose ledger holds only null blocks (SWA gap padding) must
    # not appear in holders or held at all.
    ledger = {"r0": [FakeBlock(0, is_null=True)], "r1": [FakeBlock(4)]}
    sched = FakeScheduler(num_gpu_blocks=5, free=3, ledgers=[ledger])
    facts = kv_conservation.collect(sched)
    assert facts["held"] == 1
    assert facts["holders"] == {"r1": 1}
    assert facts["ok"] is True  # 3 free + 1 held + 1 null == 5


def test_collect_missing_kv_plumbing_degrades_to_error():
    class Bare:
        pass

    facts = kv_conservation.collect(Bare())
    assert "error" in facts
    assert "kv cache plumbing" in facts["error"]


def test_collect_scheduler_without_requests_degrades_to_error():
    class NoRequests:
        kv_cache_manager = FakeKVCacheManager(
            FakeBlockPool(4, 3), [FakeManager({})]
        )
        # no .requests attribute at all

    facts = kv_conservation.collect(NoRequests())
    assert "error" in facts
    assert "requests" in facts["error"]


def test_collect_manager_without_ledger_degrades_to_error():
    class NoLedger:
        pass

    sched = FakeScheduler(num_gpu_blocks=4, free=3)
    sched.kv_cache_manager.coordinator.single_type_managers = [NoLedger()]
    facts = kv_conservation.collect(sched)
    assert "error" in facts
    assert "req_to_blocks" in facts["error"]


def test_collect_null_block_held_id_discarded():
    # Even if a ledger mistakenly lists the null block as non-null (upstream
    # regression), collect must not double-count it against the +1 term.
    ledger = {"r0": [FakeBlock(0), FakeBlock(4)]}  # block 0 listed as normal
    sched = FakeScheduler(num_gpu_blocks=5, free=3, ledgers=[ledger])
    facts = kv_conservation.collect(sched)
    assert facts["held"] == 1
    assert facts["ok"] is True  # 3 + 1 + 1 == 5


# ---------------------------------------------------------------------------
# check(): logging contract and return values
# ---------------------------------------------------------------------------


def test_check_passes_silently_and_returns_true(caplog):
    with caplog.at_level(logging.DEBUG, logger=kv_conservation.logger.name):
        assert kv_conservation.check(_balanced_scheduler(), trigger="step") is True
    assert not any(
        r.levelno >= logging.WARNING for r in caplog.records
    )


def test_check_violation_warns_and_returns_false(caplog):
    ledger = {"r0": [FakeBlock(4), FakeBlock(5)]}
    sched = FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[ledger])
    with caplog.at_level(logging.WARNING, logger=kv_conservation.logger.name):
        assert kv_conservation.check(sched, trigger="step") is False
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "block identity broken" in warns[0].message
    assert "leaked blocks" in warns[0].message


def test_check_over_count_violation_message(caplog):
    ledger = {"r0": [FakeBlock(4)]}
    sched = FakeScheduler(num_gpu_blocks=8, free=7, ledgers=[ledger])
    with caplog.at_level(logging.WARNING, logger=kv_conservation.logger.name):
        kv_conservation.check(sched, trigger="step")
    assert any(
        "over-counted" in r.message for r in caplog.records
    )


def test_check_zombie_violation_message(caplog):
    ledger = {"ghost": [FakeBlock(4), FakeBlock(5)]}
    sched = FakeScheduler(
        num_gpu_blocks=8, free=5, ledgers=[ledger], requests={}
    )
    with caplog.at_level(logging.WARNING, logger=kv_conservation.logger.name):
        kv_conservation.check(sched, trigger="idle")
    assert any(
        "zombie ledger entries" in r.message for r in caplog.records
    )


def test_check_collect_crash_degrades_to_none(caplog):
    class Exploding:
        @property
        def kv_cache_manager(self):
            raise RuntimeError("renamed upstream")

    with caplog.at_level(logging.DEBUG, logger=kv_conservation.logger.name):
        assert kv_conservation.check(Exploding(), trigger="step") is None
    assert any("check failed" in r.message for r in caplog.records)


def test_check_unrecognizable_shape_degrades_to_none(caplog):
    with caplog.at_level(logging.DEBUG, logger=kv_conservation.logger.name):
        assert kv_conservation.check(object(), trigger="idle") is None
    assert any("check skipped" in r.message for r in caplog.records)


def test_check_recovery_logged_once_per_trigger(caplog):
    leaking = FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[
        {"r0": [FakeBlock(4), FakeBlock(5)]}
    ])
    with caplog.at_level(logging.INFO, logger=kv_conservation.logger.name):
        # step trigger fails, then recovers: exactly one recovery line.
        kv_conservation.check(leaking, trigger="step")
        assert kv_conservation.check(_balanced_scheduler(), trigger="step") is True
        kv_conservation.check(_balanced_scheduler(), trigger="step")
        infos = [
            r for r in caplog.records
            if r.levelno == logging.INFO and "recovered" in r.message
        ]
        assert len(infos) == 1
        # idle trigger tracks independently: its first pass logs nothing.
        caplog.clear()
        assert kv_conservation.check(_balanced_scheduler(), trigger="idle") is True
        assert not any("recovered" in r.message for r in caplog.records)


def test_check_persistent_violation_never_suppressed(caplog):
    leaking = FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[
        {"r0": [FakeBlock(4), FakeBlock(5)]}
    ])
    with caplog.at_level(logging.WARNING, logger=kv_conservation.logger.name):
        kv_conservation.check(leaking, trigger="step")
        kv_conservation.check(leaking, trigger="step")
        kv_conservation.check(leaking, trigger="step")
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 3


def test_fmt_entries_caps_report_and_marks_extra():
    counts = {f"r{i}": i for i in range(kv_conservation.MAX_REPORT_ENTRIES + 5)}
    rendered = kv_conservation._fmt_entries(counts)
    assert f"+5 more" in rendered
    # Sorted by count descending: the smallest holders are the ones dropped.
    assert "r0:0" not in rendered
    # exactly MAX_REPORT_ENTRIES entries rendered
    assert rendered.count(":") == kv_conservation.MAX_REPORT_ENTRIES


def test_fmt_entries_small_dict_renders_all():
    assert kv_conservation._fmt_entries({"a": 2, "b": 1}) == "a:2, b:1"


# ---------------------------------------------------------------------------
# due(): periodic trigger alignment
# ---------------------------------------------------------------------------


def test_due_on_aligned_step():
    sched = FakeScheduler(num_gpu_blocks=2, free=1)
    sched.current_step = 40
    assert kv_conservation.due(sched, 20) is True


def test_due_not_on_unaligned_step():
    sched = FakeScheduler(num_gpu_blocks=2, free=1)
    sched.current_step = 41
    assert kv_conservation.due(sched, 20) is False


@pytest.mark.parametrize("interval", [0, -3])
def test_due_disabled_for_non_positive_interval(interval):
    sched = FakeScheduler(num_gpu_blocks=2, free=1)
    sched.current_step = 0
    assert kv_conservation.due(sched, interval) is False


def test_due_degrades_when_step_counter_missing():
    # Upstream renamed current_step -> probe must silently disable.
    assert kv_conservation.due(object(), 20) is False


# ---------------------------------------------------------------------------
# check_idle(): wall-clock debounce
# ---------------------------------------------------------------------------


def test_check_idle_disabled_returns_none(monkeypatch):
    sched = _balanced_scheduler()
    assert kv_conservation.check_idle(sched, 0) is None
    assert kv_conservation.check_idle(sched, -1) is None


def test_check_idle_runs_then_debounces(monkeypatch):
    # Freeze the clock so the debounce arithmetic is deterministic.
    clock = {"now": 1000.0}
    monkeypatch.setattr(kv_conservation.time, "monotonic", lambda: clock["now"])

    sched = _balanced_scheduler()
    assert kv_conservation.check_idle(sched, 20) is True
    # Within the debounce window: no second reconciliation.
    clock["now"] += kv_conservation.IDLE_CHECK_MIN_INTERVAL_SEC - 0.1
    assert kv_conservation.check_idle(sched, 20) is None
    # Past the window: runs again.
    clock["now"] += 0.2
    assert kv_conservation.check_idle(sched, 20) is True


def test_check_idle_persistent_leak_still_warns_each_window(caplog, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(kv_conservation.time, "monotonic", lambda: clock["now"])
    leaking = FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[
        {"r0": [FakeBlock(4), FakeBlock(5)]}
    ])
    with caplog.at_level(logging.WARNING, logger=kv_conservation.logger.name):
        kv_conservation.check_idle(leaking, 20)
        clock["now"] += kv_conservation.IDLE_CHECK_MIN_INTERVAL_SEC + 1
        kv_conservation.check_idle(leaking, 20)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 2
    assert all("(idle)" in r.message for r in warns)


def test_reset_clears_debounce_and_recovery_state(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(kv_conservation.time, "monotonic", lambda: clock["now"])
    leaking = FakeScheduler(num_gpu_blocks=8, free=3, ledgers=[
        {"r0": [FakeBlock(4), FakeBlock(5)]}
    ])

    kv_conservation.check_idle(leaking, 20)  # sets ts + last failure
    clock["now"] += 1.0
    kv_conservation.reset()
    assert kv_conservation.check_idle(_balanced_scheduler(), 20) is True
