# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Thin NPU input hooks for realtime KV reuse from omni-models."""
from vllm.v1.outputs import ModelRunnerOutput
from omni_models.models.openpangu_vl.realtime_video.metadata import NORMALIZED_IMAGE_ATTR
from omni_models.models.openpangu_vl.realtime_video.kv_worker import (
    copy_worker as realtime_copy_worker,
    init_mrope_worker as realtime_init_mrope_worker,
    init_pd_decode_mrope_worker,
    position_worker as realtime_position_worker,
)
from omni_models.models.openpangu_vl.realtime_video.metadata import (
    MROPE_DELTA_KEY,
    request_extra_args,
)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu_model_runner import NPUModelRunner


@register_patch("RealtimeKVModelRunnerPatch", NPUModelRunner)
class RealtimeKVModelRunnerPatch(VLLMPatch):
    """Apply P KV reuse and D decode position alignment in separate engines."""

    _attr_names_to_apply = [
        "_prepare_inputs",
        "_init_mrope_positions",
        "sample_tokens",
    ]

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        # NPUModelRunner.execute_model() already owns the device/inference
        # context. Copy the inherited KV immediately before native input
        # preparation, while still guaranteeing completion before model forward.
        realtime_copy_worker(self, scheduler_output)
        result = RealtimeKVModelRunnerPatch._upstream_prepare_inputs(
            self, scheduler_output, num_scheduled_tokens
        )
        realtime_position_worker(self, num_scheduled_tokens)
        return result

    def _init_mrope_positions(self, req_state) -> None:
        transfer = request_extra_args(req_state).get("kv_transfer_params")
        if not isinstance(transfer, dict) or MROPE_DELTA_KEY not in transfer:
            RealtimeKVModelRunnerPatch._upstream_init_mrope_positions(
                self, req_state
            )
        realtime_init_mrope_worker(self, req_state)
        init_pd_decode_mrope_worker(req_state)

    def sample_tokens(self, *args, **kwargs):
        result = RealtimeKVModelRunnerPatch._upstream_sample_tokens(
            self, *args, **kwargs
        )
        return RealtimeKVModelRunnerPatch._attach_pd_mrope_deltas(self, result)

    @staticmethod
    def _attach_pd_mrope_deltas(runner, result):
        output = getattr(result, "_model_runner_output", result)
        deltas = getattr(runner, "_realtime_pd_mrope_deltas", None)
        if isinstance(output, ModelRunnerOutput) and deltas:
            ready = {
                request_id: deltas.pop(request_id)
                for request_id in output.req_ids
                if request_id in deltas
            }
            if ready:
                output._realtime_pd_mrope_deltas = ready
        return result

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict(
            "_prepare_inputs",
            "_init_mrope_positions",
            "sample_tokens",
        )
