# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One-shot, cross-rank decode gate for hybrid DP benchmarks on vLLM 0.25.1.

While closed, schedule prefills but hold requests whose prefill is complete.
Release on every DP rank together once each rank has the configured number of
decode-ready requests. This deliberately increases TTFT and retains their KV
blocks; use it only for controlled benchmarks with enough requests per rank.
"""

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import DPEngineCoreProc

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)
_RELEASED_ATTR = "hybrid_aligned_decode_released"
_logged_engage = False


def _hybrid_gate_enabled() -> bool:
    if not envs.OMNI_HYBRID_ALIGNED_DECODE:
        return False
    if (envs.is_set("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD")
            or envs.is_set("OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD")):
        raise ValueError("Hybrid and PD benchmark decode gates cannot be enabled together")
    return True


def _ready_for_decode(request) -> bool:
    return request.num_computed_tokens >= request.num_prompt_tokens


def _all_ranks_ready(dp_group, local_ready: bool) -> bool:
    # The DP control group uses CPU collectives. MIN over 0/1 is logical AND.
    ready = torch.tensor([int(local_ready)], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(ready, op=ReduceOp.MIN, group=dp_group)
    return bool(ready.item())


@register_patch("HybridAlignedDecodeScheduler", Scheduler)
class HybridAlignedDecodeSchedulerPatch(VLLMPatch):
    # Wrap the scheduler already installed by other model patches.
    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        if not _hybrid_gate_enabled():
            return
        target = cls._target
        applied = target.__dict__.get("_omni_npu_applied_patches")
        if applied is None:
            applied = target._omni_npu_applied_patches = {}
        key = "hybrid_aligned_decode_schedule_wrap"
        if key in applied:
            raise ValueError(f"{target.__name__}.schedule already wrapped by {applied[key]}")
        original_schedule = target.schedule

        def schedule(self, throttle_prefills: bool = False):
            if (getattr(self, _RELEASED_ATTR, False)
                    or not envs.OMNI_HYBRID_ALIGNED_DECODE):
                return original_schedule(self, throttle_prefills=throttle_prefills)

            held = [r for r in self.running if _ready_for_decode(r)]
            if not held:
                return original_schedule(self, throttle_prefills=throttle_prefills)

            global _logged_engage
            if not _logged_engage:
                logger.info("[hybrid-aligned-decode] holding %d decode-ready requests", len(held))
                _logged_engage = True

            self.running = [r for r in self.running if not _ready_for_decode(r)]
            # Held requests still occupy model-runner slots. Reserve their
            # capacity while the underlying scheduler admits more prefills.
            max_running = self.max_num_running_reqs
            self.max_num_running_reqs = max_running - len(held)
            try:
                return original_schedule(self, throttle_prefills=throttle_prefills)
            finally:
                self.max_num_running_reqs = max_running
                # Restore earlier requests first to preserve FCFS ordering.
                self.running = held + self.running

        target.schedule = schedule
        applied[key] = cls.__name__
        logger.info("patch applied: %s => %s.schedule", cls.__name__, target.__name__)


@register_patch("HybridAlignedDecodeEngineCore", DPEngineCoreProc)
class HybridAlignedDecodeEngineCorePatch(VLLMPatch):
    _attr_names_to_apply = ["_has_global_unfinished_reqs"]
    _original_has_global_unfinished_reqs = DPEngineCoreProc._has_global_unfinished_reqs

    @classmethod
    def apply(cls):
        if not _hybrid_gate_enabled():
            return
        super().apply()

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        scheduler = self.scheduler
        if (not getattr(scheduler, _RELEASED_ATTR, False)
                and envs.OMNI_HYBRID_ALIGNED_DECODE):
            ready = sum(_ready_for_decode(r) for r in scheduler.running)
            threshold = envs.OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD
            if _all_ranks_ready(self.dp_group, ready >= threshold):
                setattr(scheduler, _RELEASED_ATTR, True)
                logger.info("[hybrid-aligned-decode] gate released; local ready=%d", ready)

        # Preserve vLLM 0.25.1's finish-sync cadence and pause consensus.
        return HybridAlignedDecodeEngineCorePatch._original_has_global_unfinished_reqs(
            self, local_unfinished
        )
