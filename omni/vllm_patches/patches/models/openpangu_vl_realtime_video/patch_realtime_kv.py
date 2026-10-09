# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Thin vLLM core hooks for the realtime implementation in omni-models."""

from contextlib import contextmanager

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput
from PIL import Image
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataParser
from omni_npu.connector.llmdatadist_connector_v1 import LLMDataDistConnector
from omni_models.models.openpangu_vl.realtime_video.kv_session import (
    close_session,
    finish_request,
    get_store,
    realtime_swa_reserve_blocks,
)
from omni_models.models.openpangu_vl.realtime_video.metadata import (
    MROPE_DELTA_KEY,
    POSITION_OFFSET_KEY,
    NORMALIZED_IMAGE_ATTR,
    is_realtime_kv_request,
    kv_reuse_plan,
    realtime_kv_metadata,
    request_extra_args,
)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@contextmanager
def _retain_realtime_kv_boundaries(
    kv_cache_manager,
    total_computed_tokens: int = 0,
    num_prompt_tokens: int = 0,
):
    """Keep realtime replay pages without changing the model configuration."""
    changed = []
    for manager in kv_cache_manager.coordinator.single_type_managers:
        kind = type(manager.kv_cache_spec).__name__
        if kind not in {"MomeSpec", "ShareKVSlidingWindowSpec"}:
            continue
        configured = manager.num_extra_reserved_blocks
        if kind == "MomeSpec":
            reserved = max(configured, 1)
        else:
            reserved = configured + realtime_swa_reserve_blocks(
                total_computed_tokens,
                num_prompt_tokens,
                manager.block_size,
            )
        if reserved != configured:
            manager.num_extra_reserved_blocks = reserved
            changed.append((manager, configured))
    try:
        yield
    finally:
        for manager, configured in changed:
            manager.num_extra_reserved_blocks = configured


@register_patch("RealtimeKVComputedBlocksPatch", KVCacheManager)
class RealtimeKVComputedBlocksPatch(VLLMPatch):
    _attr_names_to_apply = [
        "allocate_slots",
        "remove_skipped_blocks",
    ]

    def allocate_slots(self, request, *args, **kwargs):
        if not is_realtime_kv_request(request):
            return RealtimeKVComputedBlocksPatch._upstream_allocate_slots(
                self, request, *args, **kwargs
            )
        with _retain_realtime_kv_boundaries(
            self,
            int(getattr(request, "num_computed_tokens", 0)),
            int(getattr(request, "num_prompt_tokens", 0)),
        ):
            return RealtimeKVComputedBlocksPatch._upstream_allocate_slots(
                self, request, *args, **kwargs
            )

    def remove_skipped_blocks(
        self,
        request_id: str,
        total_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        store = getattr(self, "_realtime_kv_session_store", None)
        if store is None or request_id not in store.active_requests:
            return RealtimeKVComputedBlocksPatch._upstream_remove_skipped_blocks(
                self,
                request_id,
                total_computed_tokens,
                num_prompt_tokens,
            )
        with _retain_realtime_kv_boundaries(
            self,
            total_computed_tokens,
            num_prompt_tokens or 0,
        ):
            return RealtimeKVComputedBlocksPatch._upstream_remove_skipped_blocks(
                self,
                request_id,
                total_computed_tokens,
                num_prompt_tokens,
            )

    @classmethod
    def apply(cls):
        # Preserve the low-latency profile's existing allocate_slots behavior.
        cls.apply_bypass_conflict(
            "allocate_slots",
            "remove_skipped_blocks",
        )


@register_patch("RealtimeKVHybridCoordinatorPatch", HybridKVCacheCoordinator)
class RealtimeKVHybridCoordinatorPatch(VLLMPatch):
    """Handle the hybrid+connector lookup that bypasses KVCacheManager."""

    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    def find_longest_cache_hit_per_group(self, block_hashes, max_cache_hit_length):
        requests = getattr(self, "_realtime_requests_by_hashes", {})
        request = requests.get(id(block_hashes))
        if request is None or request.block_hashes is not block_hashes:
            return RealtimeKVHybridCoordinatorPatch._upstream_find_longest_cache_hit_per_group(
                self, block_hashes, max_cache_hit_length
            )

        manager = self._realtime_kv_cache_manager
        blocks, count = get_store(manager).computed(request)
        # The generic hybrid lookup's Marconi hint must not leak into this
        # request after the realtime snapshot has supplied its own prefix.
        self.num_uncached_common_prefix_tokens = 0
        return tuple(list(group) for group in blocks.blocks), (count,) * len(
            blocks.blocks
        )

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("find_longest_cache_hit_per_group")


@register_patch("RealtimeKVFinishRequestPatch", Scheduler)
class RealtimeKVFinishRequestPatch(VLLMPatch):
    _attr_names_to_apply = ["add_request", "update_from_output", "_free_request"]

    def add_request(self, request):
        result = RealtimeKVFinishRequestPatch._upstream_add_request(self, request)
        coordinator = self.kv_cache_manager.coordinator
        if is_realtime_kv_request(request) and isinstance(
            coordinator, HybridKVCacheCoordinator
        ):
            requests = getattr(coordinator, "_realtime_requests_by_hashes", None)
            if requests is None:
                requests = {}
                coordinator._realtime_requests_by_hashes = requests
            requests[id(request.block_hashes)] = request
            coordinator._realtime_kv_cache_manager = self.kv_cache_manager
        return result

    def _free_request(self, request, delay_free_blocks=False):
        # Snapshot the prompt KV while the request still owns its block table.
        try:
            finish_request(self, request)
            return RealtimeKVFinishRequestPatch._upstream_free_request(
                self, request, delay_free_blocks
            )
        finally:
            coordinator = self.kv_cache_manager.coordinator
            requests = getattr(coordinator, "_realtime_requests_by_hashes", None)
            if requests is not None:
                requests.pop(id(request.block_hashes), None)

    def update_from_output(self, scheduler_output, model_runner_output: ModelRunnerOutput):
        for request_id, delta in getattr(
            model_runner_output, "_realtime_pd_mrope_deltas", {}
        ).items():
            request = self.requests.get(request_id)
            if request is not None and is_realtime_kv_request(request):
                request.sampling_params.extra_args[MROPE_DELTA_KEY] = int(delta)
        return RealtimeKVFinishRequestPatch._upstream_update_from_output(
            self, scheduler_output, model_runner_output
        )

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict(
            "add_request", "update_from_output", "_free_request"
        )


@register_patch("RealtimeKVSessionControlPatch", EngineCore)
class RealtimeKVSessionControlPatch(VLLMPatch):
    _attr_names_to_apply = ["realtime_kv_session_control"]

    def realtime_kv_session_control(self, action: str, session_key: str):
        store = get_store(self.scheduler.kv_cache_manager)
        store.expire()
        if action == "close":
            return close_session(self.scheduler.kv_cache_manager, session_key)
        if action == "stats":
            return {
                "retained_blocks": store.retained_blocks,
                "free_blocks": store.pool.get_num_free_blocks(),
            }
        raise ValueError(f"unknown realtime KV session action: {action}")


@register_patch("RealtimeKVTransferPatch", LLMDataDistConnector)
class RealtimeKVTransferPatch(VLLMPatch):
    _attr_names_to_apply = ["request_finished"]

    def request_finished(self, request, block_ids):
        window = realtime_kv_metadata(request_extra_args(request))
        if self.is_prefill and window is not None and not window.get("question"):
            # The P-side snapshot already retains the reusable KV. No D
            # request exists for this image-only round, so do not register a
            # transfer or defer freeing the physical request's own blocks.
            return False, None
        result = RealtimeKVTransferPatch._upstream_request_finished(
            self, request, block_ids
        )
        if self.is_prefill and window is not None and window.get("pd_decode"):
            delay_free, params = result
            if params is not None:
                plan = kv_reuse_plan(request) or {}
                params[POSITION_OFFSET_KEY] = int(plan.get("offset", 0))
                params[MROPE_DELTA_KEY] = int(
                    request_extra_args(request)[MROPE_DELTA_KEY]
                )
                return delay_free, params
        return result

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("request_finished")


@register_patch("RealtimeImageParsePatch", MultiModalDataParser)
class RealtimeImageParsePatch(VLLMPatch):
    _attr_names_to_apply = ["_parse_image_data"]

    def _parse_image_data(self, data):
        images = [data] if isinstance(data, Image.Image) else data
        if isinstance(images, (list, tuple)) and images and all(
            getattr(image, NORMALIZED_IMAGE_ATTR, False) for image in images
        ):
            return ImageProcessorItems(images)
        return RealtimeImageParsePatch._upstream_parse_image_data(self, data)

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("_parse_image_data")
