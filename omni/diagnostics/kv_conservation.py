# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV cache block conservation check.

Validates, during scheduler idle periods and every N scheduling steps, that

    Free Blocks + Active Blocks + Cached Evictable Blocks + 1 (null) == N_total

In the vLLM 0.25.1 block pool, "free" and "cached evictable" share one queue
(``free_block_queue``: ref_cnt==0 blocks, with or without a prefix-cache
hash), so the identity is reconciled as

    free_queue + ledger-held blocks + deferred-free blocks + 1 == num_gpu_blocks

where "ledger-held" is recounted from every single-type manager's
``req_to_blocks`` instead of trusting ``ref_cnt`` alone. The two sides
cross-check: a double free drives ref_cnt negative or desynchronizes the
ledger; a leaked request (freed from ``scheduler.requests`` but its blocks
never returned) shows up as orphan ledger entries plus a pool shortfall.

Concurrency model: the check runs on the engine main thread (inside
schedule() or _process_input_queue), so there is no concurrent mutation to
guard against. Every read is defensive: an upstream rename must degrade to
a debug log, never raise into the engine loop.

"""

import logging
import time
from omni_npu import envs

logger = logging.getLogger(__name__)

# Block-id sets can be large (100k+ blocks on production pools). Cap how many
# orphan / holder entries are rendered into the warning so a bad leak never
# produces a multi-megabyte log line.
MAX_REPORT_ENTRIES = 16

# Minimum wall-clock seconds between two idle checks. The idle branch of the
# engine loop re-fires on every poll when the queue is empty and non-blocking
# (DP variant), so without a floor a drained-but-spinning engine would run the
# reconciliation in a tight loop. The pool cannot mutate while the engine is
# blocked waiting for input, so one check per window loses nothing.
IDLE_CHECK_MIN_INTERVAL_SEC = 10.0

# Step-count lookback for the "recovered" info line: a check that passes again
# after a previous failure is worth one line, but only once.
_last_result: dict[str, bool] = {}
# Wall-clock timestamp of the last idle check.
_last_idle_check_ts: float = 0.0


def interval() -> int:
    """Check interval in steps; <=0 disables both triggers."""
    try:
        return int(envs.OMNI_KV_CONSERVATION_CHECK_INTERVAL)
    except (TypeError, ValueError):
        logger.warning(
            "[kv-conservation] invalid OMNI_KV_CONSERVATION_CHECK_INTERVAL, disabling",
        )
        return 0


def reset() -> None:
    """Clear the recovery-tracking state. Intended for tests only."""
    _last_result.clear()
    global _last_idle_check_ts
    _last_idle_check_ts = 0.0


def _drain_deferred_block_ids(scheduler) -> set[int]:
    """Block ids popped from the ledger but not yet returned to the pool.

    defer_block_free (async scheduling / PP overlap) parks blocks in
    ``scheduler.deferred_frees`` until their fence step is processed. They are
    neither in the free queue nor in any req_to_blocks, so they must be counted
    on the "held" side of the identity or every overlapping batch would look
    like a leak.
    """
    ids: set[int] = set()
    for _fence, blocks in getattr(scheduler, "deferred_frees", ()) or ():
        for blk in blocks:
            block_id = getattr(blk, "block_id", None)
            if block_id is not None:
                ids.add(block_id)
    return ids


def collect(scheduler) -> dict:
    """Reconcile the block pool against the per-manager request ledgers.

    Returns a plain-facts dict (never raises); "error" carries the reason when
    the scheduler shape is not recognizable so callers can degrade quietly.
    """
    try:
        kv_cache_manager = scheduler.kv_cache_manager
        block_pool = kv_cache_manager.block_pool
        managers = kv_cache_manager.coordinator.single_type_managers
    except AttributeError as e:
        return {"error": f"missing kv cache plumbing: {e!r}"}

    total = block_pool.num_gpu_blocks
    null_block_id = block_pool.null_block.block_id
    free = block_pool.free_block_queue.num_free_blocks

    # Held blocks: every non-null block still on any manager's ledger.
    # block ids are unique across managers (one shared pool), so a set both
    # deduplicates and gives the count directly.
    held_ids: set[int] = set()
    # req_id -> number of non-null blocks, for the "top holders" report.
    holders: dict[str, int] = {}
    # Ledger entries whose request is gone from scheduler.requests: blocks
    # that will never be freed through the normal path.
    orphan: dict[str, int] = {}

    try:
        live_requests = scheduler.requests
    except AttributeError as e:
        return {"error": f"scheduler has no requests dict: {e!r}"}

    for manager in managers:
        try:
            ledger = manager.req_to_blocks
        except AttributeError as e:
            return {"error": f"manager has no req_to_blocks: {e!r}"}
        for req_id, blocks in ledger.items():
            block_ids = [
                blk.block_id for blk in blocks if not blk.is_null
            ]
            if not block_ids:
                continue
            held_ids.update(block_ids)
            n_blocks = len(block_ids)
            holders[req_id] = holders.get(req_id, 0) + n_blocks
            if req_id not in live_requests:
                orphan[req_id] = orphan.get(req_id, 0) + n_blocks

    # The null block is popped from the free queue at pool construction and
    # never reallocated; every occurrence on a ledger is the same block_id.
    # It must not be counted as held (its ref_cnt is not maintained).
    held_ids.discard(null_block_id)

    # Blocks in deferred_frees were popped from the ledger ahead of their
    # return to the pool; count them as transiently held.
    deferred_ids = _drain_deferred_block_ids(scheduler)
    held_ids.update(deferred_ids)

    sum_side = free + len(held_ids) + 1  # +1 for the null block
    return {
        "total": total,
        "free": free,
        "held": len(held_ids),
        "deferred": len(deferred_ids),
        "orphan": orphan,
        "holders": holders,
        # Orphan ledger entries are a violation even when the arithmetic
        # still balances: blocks held by a request that no longer exists are
        # unreclaimable (zombie leak) and the ledger desync is itself a bug.
        "ok": sum_side == total and not orphan,
        "delta": sum_side - total,
    }


def _fmt_entries(counts: dict[str, int]) -> str:
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:MAX_REPORT_ENTRIES]
    rendered = ", ".join(f"{rid}:{n}" for rid, n in top)
    extra = len(counts) - len(top)
    return rendered + (f" (+{extra} more)" if extra > 0 else "")


def check(scheduler, *, trigger: str) -> bool | None:
    """Run one conservation check; log a WARNING on violation.

    Args:
        scheduler: the vLLM Scheduler instance.
        trigger: audit label for the log ("step" / "idle"), also keys the
            recovery tracking.

    Returns:
        True when the identity holds, False on violation, None when the check
        could not run (degraded to a debug log).
    """
    try:
        facts = collect(scheduler)
    except Exception as e:  # noqa: BLE001 - never raise into the engine loop
        logger.debug("[kv-conservation] check failed (%s): %r", trigger, e)
        return None
    if "error" in facts:
        logger.debug("[kv-conservation] check skipped (%s): %s", trigger, facts["error"])
        return None

    if facts["ok"]:
        if _last_result.get(trigger) is False:
            logger.info(
                "[kv-conservation] block identity recovered (%s): "
                "free=%d held=%d deferred=%d total=%d",
                trigger, facts["free"], facts["held"], facts["deferred"], facts["total"],
            )
        _last_result[trigger] = True
        return True

    holders = facts["holders"]
    orphan = facts["orphan"]
    delta = facts["delta"]
    if delta < 0:
        kind = "leaked blocks (pool short by %d)" % -delta
    elif delta > 0:
        kind = "over-counted blocks (pool over by %+d)" % delta
    else:
        kind = "zombie ledger entries (arithmetic balanced)"
    logger.warning(
        "[kv-conservation] block identity broken (%s): %s; "
        "free=%d held=%d deferred=%d null=1 total=%d (sum=%d); "
        "orphan ledger entries: %s; top holders: %s",
        trigger, kind,
        facts["free"], facts["held"], facts["deferred"], facts["total"],
        facts["free"] + facts["held"] + 1,
        _fmt_entries(orphan) if orphan else "none",
        _fmt_entries(holders),
    )
    _last_result[trigger] = False
    return False


def due(scheduler, interval_sec: int) -> bool:
    """Whether the periodic (per-step) check is due on this schedule() call.

    Counts steps via the scheduler's own current_step counter, so no extra
    state needs survives patch re-application. interval_sec <= 0 disables.
    """
    if interval_sec <= 0:
        return False
    try:
        step = scheduler.current_step
    except AttributeError:
        return False
    return step % interval_sec == 0


def check_idle(scheduler, interval_sec: int) -> bool | None:
    """Debounced wrapper for the engine-loop idle branch.

    Runs at most one full reconciliation per IDLE_CHECK_MIN_INTERVAL_SEC
    while the engine sits idle; a persistent leak still warns on every window
    (check() itself never suppresses violations).
    """
    if interval_sec <= 0:
        return None
    global _last_idle_check_ts
    now = time.monotonic()
    if now - _last_idle_check_ts < IDLE_CHECK_MIN_INTERVAL_SEC:
        return None
    _last_idle_check_ts = now
    return check(scheduler, trigger="idle")
