# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Optional round-robin routing for the vLLM 0.25.1 internal DP client."""

from vllm.logger import init_logger
from vllm.v1.engine.core_client import (
    DPLBAsyncMPClient,
    get_late_interaction_engine_index,
)

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)
_COUNTER_ATTR = "_omni_rr_counter"
_logged_engage = False


@register_patch("DPRoundRobinClient", DPLBAsyncMPClient)
class DPRoundRobinClientPatch(VLLMPatch):
    _attr_names_to_apply = ["get_core_engine_for_request"]
    _original_get_core_engine_for_request = (
        DPLBAsyncMPClient.get_core_engine_for_request
    )

    @classmethod
    def apply(cls):
        if not envs.OMNI_DP_ROUND_ROBIN:
            return
        super().apply()

    def get_core_engine_for_request(self, request):
        # Keep explicit DP rank and late-interaction pooling affinity intact.
        if (not envs.OMNI_DP_ROUND_ROBIN
                or request.data_parallel_rank is not None
                or get_late_interaction_engine_index(
                    request.pooling_params, len(self.core_engines)
                ) is not None):
            return DPRoundRobinClientPatch._original_get_core_engine_for_request(
                self, request
            )

        num_engines = len(self.core_engines)
        counter = getattr(self, _COUNTER_ATTR, 0)
        eng_index = (self.eng_start_index + counter) % num_engines
        setattr(self, _COUNTER_ATTR, counter + 1)

        global _logged_engage
        if not _logged_engage:
            logger.info("[dp-round-robin] routing across %d DP engines", num_engines)
            _logged_engage = True

        # Match vLLM's local optimistic load estimate and abort bookkeeping.
        self.lb_engines[eng_index][0] += self.client_count
        chosen_engine = self.core_engines[eng_index]
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
