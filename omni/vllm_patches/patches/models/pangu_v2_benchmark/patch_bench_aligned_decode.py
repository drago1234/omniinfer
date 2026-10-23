# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""Bench-aligned decode gate for PD-disaggregation decode nodes.

In a data-parallel decode deployment, requests finish receiving their remote KV
at different times on different DP ranks. Without coordination, each rank starts
decoding as soon as its own first request is ready, which produces ragged
batches across ranks. This patch holds every rank back until *all* DP ranks have
accumulated at least ``OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD`` finished KV
receptions, then releases everywhere at once so the first decode step runs with
a fuller, balanced batch — giving aligned, comparable batches for benchmarking.

Mechanism: while the gate is closed the engine-core loop moves newly finished
``finished_recving_kv_req_ids`` into a private holding set, so the scheduler's
``_update_waiting_for_remote_kv`` sees them as not-yet-ready and keeps the
requests in ``WAITING_FOR_REMOTE_KVS``. Once an all-reduce (MIN == logical AND)
confirms every rank is ready, the held ids are handed back to the scheduler and
the gate is released, so the next ``schedule()`` admits them together. This
deliberately does NOT override any Scheduler method, so it composes with other
patches that wrap ``_update_waiting_for_remote_kv``. The gate is one-shot: once
released it stays open for the engine's lifetime.

Tuning:
    OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD - per-rank finished-KV-reception
        count required before release (default 20). ``0`` effectively disables
        the gate (it releases on the first check). The legacy alias
        ``OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD`` is also accepted.

Caveats: this is a benchmarking aid — gating decode start until a batch
accumulates deliberately inflates TTFT and is not intended for production
serving. The client-side concurrency must be at least
``threshold * num_dp_ranks``, otherwise no rank reaches the threshold and the
gate never releases. While the gate is closed, a request that is aborted has its
KV blocks freed one step late (it stays in the holding set).
"""

import os

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger
from vllm.v1.engine.core import DPEngineCoreProc

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

_RELEASED_ATTR = "bench_aligned_decode_released"
_HELD_ATTR = "_bench_aligned_decode_held_req_ids"


_THRESHOLD_FILE_ENV = "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD_FILE"
_DEFAULT_THRESHOLD_FILE = "/tmp/omni_gate_threshold"


def _threshold() -> int:
    """Release threshold for this DP rank, in finished-KV receptions.

    Read on every gate step so it can be re-tuned while the gate is still
    closed: an integer in ``OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD_FILE``
    (default ``/tmp/omni_gate_threshold``) wins over the env var, which stays
    the fallback. This matters because the threshold must never exceed the
    per-rank decode KV capacity, and that capacity is only known once the
    engine has started (~5 at GPU_UTIL 0.8 for 128K+8K, ~8 at 0.95).
    """
    path = os.getenv(_THRESHOLD_FILE_ENV, _DEFAULT_THRESHOLD_FILE)
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                return int(text)
        except (OSError, ValueError):
            pass
    return envs.OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD


def _all_ranks_ready(dp_group, local_ready: bool) -> bool:
    # MIN over ints == logical AND: True only if EVERY rank is ready.
    ready = torch.tensor([int(local_ready)], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(ready, op=ReduceOp.MIN, group=dp_group)
    return bool(ready.item())


@register_patch("BenchAlignedDecodeEngineCorePatch", DPEngineCoreProc)
class BenchAlignedDecodeEngineCorePatch(VLLMPatch):
    _attr_names_to_apply = ["_has_global_unfinished_reqs"]
    # Captured at import time (before any patch is applied), so this is the
    # genuine upstream implementation. We delegate to it for the periodic
    # finish-sync all-reduce instead of duplicating its body.
    _original_has_global_unfinished_reqs = DPEngineCoreProc._has_global_unfinished_reqs

    @classmethod
    def apply(cls):
        explicitly_set = (
            envs.is_set("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD")
            or envs.is_set("OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD")
        )
        if not explicitly_set:
            return
        if envs.OMNI_HYBRID_ALIGNED_DECODE:
            raise ValueError("Hybrid and PD benchmark decode gates cannot be enabled together")
        super().apply()

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Bench-aligned decode gate check. Runs every engine step, in the same
        # loop position as the original (right before the global-unfinished
        # all-reduce), because the upstream 32-step skip lives inside the
        # delegated call below.
        scheduler = self.scheduler
        if not getattr(scheduler, _RELEASED_ATTR, False):
            held = getattr(scheduler, _HELD_ATTR, None)
            if held is None:
                held = set()
                setattr(scheduler, _HELD_ATTR, held)

            # Move newly-finished KV receptions out of the scheduler's view so
            # they accumulate instead of being admitted one at a time. The
            # scheduler's _try_promote_blocked_waiting_request then keeps these
            # requests in WAITING_FOR_REMOTE_KVS (its "not in finished" path).
            prev_held = len(held)
            if scheduler.finished_recving_kv_req_ids:
                held.update(scheduler.finished_recving_kv_req_ids)
                scheduler.finished_recving_kv_req_ids.clear()

            # held only grows until release, so log only when it advances.
            if len(held) > prev_held:
                logger.info(
                    "[pd-bench-aligned-decode] waiting, %d held locally", len(held)
                )

            if _all_ranks_ready(self.dp_group, len(held) >= _threshold()):
                # Release: hand the accumulated ids back so the next schedule()
                # admits them together, then open the gate permanently.
                scheduler.finished_recving_kv_req_ids.update(held)
                held.clear()
                setattr(scheduler, _RELEASED_ATTR, True)
                logger.info("[pd-bench-aligned-decode] gate released")

        return BenchAlignedDecodeEngineCorePatch._original_has_global_unfinished_reqs(
            self, local_unfinished
        )
