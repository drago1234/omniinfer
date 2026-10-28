# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache block conservation check mounts.

Upholds, during scheduler idle periods and every N scheduling steps, the block
identity

    Free + Active + Cached-Evictable + 1 (null) == N_total

and logs a WARNING naming the leak candidates when it breaks. A broken identity
means some request's blocks were never returned (zombie leak, e.g. client
disconnect / abort paths) or were returned twice (double free).

The idle mount, engine-process side:
  - ``KVConservationIdlePatch`` wraps ``EngineCoreProc._process_input_queue``:
    the idle branch of the engine busy loop (has_work() False) is the exact
    "queue drained" moment the requirement calls for, and it never runs
    concurrently with schedule()/update_from_output() on the engine thread.
    ``_process_input_queue`` is inherited by DPEngineCoreProc, so one mount
    covers both engine flavors. Idle callbacks could not be used: they are
    one-shot pop-consumed by _notify_idle_state_callbacks.
  - The periodic per-step trigger lives in the forked
    ``patch_scheduler_spec_decode_pad.schedule`` instead (that patch owns the
    schedule attribute, so a second wrapper here could not compose).

The check itself lives in ``omni_npu.diagnostics.kv_conservation`` and never
raises into the engine loop; the wrappers keep that contract for the interval
read too.
"""

from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc

from omni_npu.diagnostics import kv_conservation
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


_orig_process_input_queue = EngineCoreProc._process_input_queue


@register_patch("KVConservationIdlePatch", EngineCoreProc)
class KVConservationIdlePatch(VLLMPatch):
    _attr_names_to_apply = ["_process_input_queue"]

    def _process_input_queue(self):
        # Idle == no work for this engine and nothing already in the local
        # input queue. has_work() covers unfinished + finished-not-yet-flushed
        # requests and the batch queue, so a False verdict here means the
        # scheduler queues are drained and the busy loop is about to block --
        # the safest moment to audit the block pool. Any abort arriving during
        # the check is picked up by the upstream loop afterwards.
        # check_idle debounces: the non-blocking DP loop re-enters this branch
        # in a tight spin while idle.
        if not self.has_work():
            interval = kv_conservation.interval()
            if interval:
                kv_conservation.check_idle(self.scheduler, interval)

        return _orig_process_input_queue(self)
