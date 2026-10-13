# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import base64
import zlib
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, Iterator

import numpy as np

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.generate.base.serving import GenerateBaseServing as OpenAIServing
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from vllm.config import VllmConfig

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.plugin_decorators import update_from_output_decorator
from omni_npu.vllm_patches.patches.common.patch_prefilled_token_skip_tokenize import (
    OpenAIServingChatPatch,
)

logger = init_logger(__name__)

# Preserve all native validation when applying the PD compatibility bypass.
_ORIGINAL_VLLM_CONFIG_POST_INIT = VllmConfig.__post_init__
_ORIGINAL_CHAT_STREAM = OpenAIServingChat.chat_completion_stream_generator
_ORIGINAL_CHAT_FULL = OpenAIServingChatPatch.chat_completion_full_generator
_ORIGINAL_COMPLETION_STREAM = OpenAIServingCompletion.completion_stream_generator
_ORIGINAL_COMPLETION_FINAL = (
    OpenAIServingCompletion.request_output_to_completion_response
)
_ORIGINAL_CHAT_CHOICE_INIT = ChatCompletionResponseChoice.__init__
_ORIGINAL_CHAT_STREAM_CHOICE_INIT = ChatCompletionResponseStreamChoice.__init__
_ORIGINAL_COMPLETION_CHOICE_INIT = CompletionResponseChoice.__init__
_ORIGINAL_COMPLETION_STREAM_CHOICE_INIT = CompletionResponseStreamChoice.__init__
_original_add = RequestOutput.add
_ROUTED_EXPERT_KEYS = (
    "routed_experts_shape",
    "routed_experts_dtype",
    "routed_experts_str_len",
    "routed_experts_str",
)
_CURRENT_ROUTED_EXPERTS_PAYLOAD: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_routed_experts_payload",
    default=None,
)
_CHOICE_PATCHES_APPLIED = False


def _request_num_cached_tokens(request: Request) -> int:
    """Return cached-token metadata across vLLM Request schema versions."""
    return getattr(request, "num_cached_tokens", 0)


def _read_slots(mgr: Any, indices: np.ndarray) -> np.ndarray:
    """Copy routing rows for the given physical slots."""
    return mgr.routed_experts_by_slot[indices]


def _set_pd_flags(
    mgr: Any,
    vllm_config: "VllmConfig | None" = None,
) -> None:
    """Record whether routing belongs to a PD prefill or decode node."""
    kv_transfer_config = None
    if vllm_config is not None:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

    if kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance:
        mgr.is_pd_disaggregation = True
        mgr.is_pd_prefill = kv_transfer_config.is_kv_producer
        return

    role = envs.OMNI_PD_ROLE
    if role == "prefill":
        mgr.is_pd_disaggregation = True
        mgr.is_pd_prefill = True
        return
    if role == "decode":
        mgr.is_pd_disaggregation = True
        mgr.is_pd_prefill = False
        return

    mgr.is_pd_disaggregation = False
    mgr.is_pd_prefill = False


def _concatenate_routed_experts(
    current: np.ndarray | None,
    incoming: np.ndarray | None,
) -> np.ndarray | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    if current.ndim != incoming.ndim:
        raise ValueError(
            "Cannot aggregate routed_experts with different ranks: "
            f"{current.ndim} != {incoming.ndim}"
        )
    if current.shape[1:] != incoming.shape[1:]:
        raise ValueError(
            "Cannot aggregate routed_experts with different non-token shapes: "
            f"{current.shape[1:]} != {incoming.shape[1:]}"
        )
    if current.dtype != incoming.dtype:
        raise ValueError(
            "Cannot aggregate routed_experts with different dtypes: "
            f"{current.dtype} != {incoming.dtype}"
        )
    return np.concatenate([current, incoming], axis=0)


@register_patch("RequestOutputRoutedExpertsAggregation", RequestOutput)
class RequestOutputRoutedExpertsAggregationPatch(VLLMPatch):
    _attr_names_to_apply = ["add"]

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        existing_routed_experts = {
            completion.index: completion.routed_experts
            for completion in self.outputs
        }
        incoming_routed_experts = {
            completion.index: completion.routed_experts
            for completion in next_output.outputs
        }

        _original_add(self, next_output, aggregate)

        if not aggregate:
            return

        for completion in self.outputs:
            if completion.index not in existing_routed_experts:
                continue
            if completion.index not in incoming_routed_experts:
                continue

            completion.routed_experts = _concatenate_routed_experts(
                existing_routed_experts[completion.index],
                incoming_routed_experts[completion.index],
            )


def _reject_context_parallel_routed_experts(vllm_config: Any) -> None:
    """Reject incomplete routed-experts export under context parallelism."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not getattr(
        model_config, "enable_return_routed_experts", False
    ):
        return

    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None:
        return

    dcp = getattr(parallel_config, "decode_context_parallel_size", 1)
    pcp = getattr(parallel_config, "prefill_context_parallel_size", 1)
    if dcp > 1 or pcp > 1:
        raise ValueError(
            "enable_return_routed_experts is incompatible with context "
            "parallelism: routed experts are stored per physical KV slot, "
            "which a CP-split request no longer owns end to end. Got "
            f"decode_context_parallel_size={dcp}, "
            f"prefill_context_parallel_size={pcp}. Set both to 1, or drop "
            "--enable-return-routed-experts."
        )


@register_patch("VllmConfigPDCompatPatch", VllmConfig)
class VllmConfigPDCompatPatch(VLLMPatch):
    """Allow routed-experts capture for PD producer and consumer roles.

    Prompt routing is relayed through ``kv_transfer_params`` while decode
    routing uses local slots. Other KV roles and PP/CP combinations remain
    unsupported. The original initializer is rerun with capture temporarily
    disabled so all unrelated validation and platform setup still execute.
    """

    _attr_names_to_apply = ["__post_init__"]

    def __post_init__(self) -> None:
        # Bypass only the PD guard; preserve all other validation.
        kv_cfg = getattr(self, "kv_transfer_config", None)
        role = getattr(kv_cfg, "kv_role", None) if kv_cfg is not None else None
        try:
            _ORIGINAL_VLLM_CONFIG_POST_INIT(self)
        except ValueError as exc:
            if not _is_pd_routed_experts_guard(self, exc):
                raise
            if role not in ("kv_producer", "kv_consumer"):
                raise
            # Keep PP rejected even if configuration changes between passes.
            if (
                self.parallel_config is not None
                and self.parallel_config.pipeline_parallel_size > 1
            ):
                raise ValueError(
                    "--enable-return-routed-experts is incompatible with "
                    "pipeline parallelism (PP > 1)."
                ) from exc
            model_config = self.model_config
            model_config.enable_return_routed_experts = False
            try:
                _ORIGINAL_VLLM_CONFIG_POST_INIT(self)
            finally:
                model_config.enable_return_routed_experts = True
            # Recheck decisions that depend on the restored capture flag.
            if self.use_v2_model_runner:
                self._validate_v2_model_runner()


def _is_pd_routed_experts_guard(vllm_config: Any, exc: ValueError) -> bool:
    """True only for upstream's routed-experts+KV-connector rejection."""
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not getattr(
        model_config, "enable_return_routed_experts", False
    ):
        return False
    return "enable-return-routed-experts is incompatible with KV" in str(exc)


@register_patch("SchedulerInitPatch", Scheduler)
class SchedulerInitPatch(VLLMPatch):
    """Attach PD role and attention-group metadata to the scheduler."""

    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args: Any, **kwargs: Any):
        vllm_config = kwargs["vllm_config"]
        _reject_context_parallel_routed_experts(vllm_config)
        SchedulerInitPatch._upstream__init__(self, *args, **kwargs)

        mgr = getattr(self, "routed_experts_mgr", None)
        if mgr is None:
            return

        # Keep scheduler and manager attention-group selection aligned.
        self.routed_experts_attn_gid = getattr(
            mgr,
            "attn_gid",
            _get_attention_kv_cache_gid(getattr(self, "kv_cache_config", None)),
        )
        _set_pd_flags(mgr, getattr(self, "vllm_config", vllm_config))

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("__init__")


def _get_attention_kv_cache_gid(kv_cache_config) -> int:
    """Find the first attention KV-cache group."""
    from vllm.v1.kv_cache_interface import AttentionSpec

    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if isinstance(group.kv_cache_spec, AttentionSpec):
            return gid
    return 0


def _concatenate_index_ranges(
    cached_indices: np.ndarray,
    fresh_indices: np.ndarray,
) -> np.ndarray:
    if cached_indices.size == 0:
        return fresh_indices
    if fresh_indices.size == 0:
        return cached_indices
    return np.concatenate([cached_indices, fresh_indices], axis=0)


def _get_request_slot_mapping(self: Scheduler, request: Request) -> np.ndarray | None:
    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
    if kv_blocks is None:
        return None

    # Group zero may be non-attention in hybrid models.
    attn_gid = getattr(self, "routed_experts_attn_gid", 0)
    block_ids = kv_blocks.get_block_ids()[attn_gid]
    if not block_ids:
        return None

    kv_cache_config = getattr(self, "kv_cache_config", None)
    if kv_cache_config is not None and attn_gid < len(kv_cache_config.kv_cache_groups):
        block_size = kv_cache_config.kv_cache_groups[attn_gid].kv_cache_spec.block_size
    else:
        block_size = self.kv_cache_manager.coordinator.single_type_managers[0].block_size
    block_ids_array = np.array(block_ids, dtype=np.int32)
    num_blocks = len(block_ids)
    block_offsets = np.arange(block_size, dtype=np.int32)
    slot_mapping = (
        block_offsets.reshape((1, block_size))
        + block_ids_array.reshape((num_blocks, 1)) * block_size
    ).flatten()
    return slot_mapping


def _get_routed_experts_from_slot_mapping(
    mgr: Any,
    request: Request,
    slot_mapping: np.ndarray | None,
    *,
    stopped: bool,
    num_new_tokens: int,
) -> np.ndarray | None:
    if mgr is None or slot_mapping is None:
        return None

    output_kind = (
        RequestOutputKind.FINAL_ONLY
        if request.sampling_params is None
        else request.sampling_params.output_kind
    )
    if not stopped and output_kind == RequestOutputKind.FINAL_ONLY:
        return None

    num_tokens = request.num_tokens - 1
    if num_tokens <= 0:
        return None
    num_cached_tokens = _request_num_cached_tokens(request)
    if num_cached_tokens < 0:
        raise RuntimeError("request.num_cached_tokens is not initialized")
    if num_cached_tokens > num_tokens:
        raise ValueError(
            "request.num_cached_tokens exceeds routed experts token range: "
            f"{num_cached_tokens} > {num_tokens}"
        )

    is_pd_disaggregation = getattr(mgr, "is_pd_disaggregation", False)
    is_prefill = (
        getattr(mgr, "is_pd_prefill", False)
        if is_pd_disaggregation
        else num_tokens == request.num_prompt_tokens
    )
    if is_pd_disaggregation and not is_prefill:
        cached_indices = slot_mapping[0:0]
        fresh_indices = slot_mapping[request.num_prompt_tokens:num_tokens]
    else:
        cached_indices = slot_mapping[:num_cached_tokens]
        fresh_indices = slot_mapping[num_cached_tokens:num_tokens]

    indices = _concatenate_index_ranges(cached_indices, fresh_indices)
    if indices.size == 0:
        return None

    if output_kind == RequestOutputKind.DELTA:
        if num_new_tokens <= 0:
            return None

        if is_pd_disaggregation:
            indices = indices[-num_new_tokens:]
        else:
            prompt_indices = slot_mapping[: min(request.num_prompt_tokens, num_tokens)]
            decode_indices = slot_mapping[request.num_prompt_tokens:num_tokens]
            delta_indices = decode_indices[-num_new_tokens:]
            prompt_sent = getattr(
                request,
                "_omni_stream_prompt_routed_experts_sent",
                False,
            )
            if prompt_sent:
                indices = delta_indices
            else:
                indices = _concatenate_index_ranges(prompt_indices, delta_indices)
                if indices.size > 0:
                    request._omni_stream_prompt_routed_experts_sent = True
        if indices.size == 0:
            return None
    elif output_kind not in (
        RequestOutputKind.FINAL_ONLY,
        RequestOutputKind.CUMULATIVE,
    ):
        raise ValueError(f"Unsupported RequestOutputKind: {output_kind}")

    return _read_slots(mgr, indices)


@register_patch("SchedulerRoutedExpertsPatch", Scheduler)
class SchedulerRoutedExpertsPatch(VLLMPatch):
    """Extend the active scheduler output chain for PD decode nodes.

    Upstream handling remains responsible for buffer writes, block snapshots,
    speculative decoding, PD finish notifications and request cleanup.
    Decode nodes then rebuild their output from local decode slots because
    prompt routing arrives separately through ``kv_transfer_params``.
    """

    _attr_names_to_apply = ["update_from_output"]
    _upstream_update_from_output = Scheduler.update_from_output

    @classmethod
    def apply(cls):
        # Retain earlier wrappers, including the PD finish-notification hook.
        cls.apply_bypass_conflict("update_from_output")

    @update_from_output_decorator
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        mgr = getattr(self, "routed_experts_mgr", None)
        is_pd_decode = (
            mgr is not None
            and getattr(mgr, "is_pd_disaggregation", False)
            and not getattr(mgr, "is_pd_prefill", False)
        )

        # Snapshot context before native handling frees stopped requests.
        pd_decode_context: dict[str, tuple[Request, np.ndarray | None]] = {}
        if is_pd_decode:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request is not None:
                    pd_decode_context[req_id] = (
                        request,
                        _get_request_slot_mapping(self, request),
                    )

        # Preserve the current scheduling, notification and cleanup chain.
        engine_core_outputs = SchedulerRoutedExpertsPatch._upstream_update_from_output(
            self, scheduler_output, model_runner_output
        )

        if not getattr(
            self.vllm_config.model_config, "enable_return_routed_experts", False
        ):
            return engine_core_outputs

        if not is_pd_decode:
            return engine_core_outputs

        # Local prompt slots are invalid on decode nodes; emit decode rows only.
        outputs_by_request_id: dict[str, EngineCoreOutput] = {}
        for engine_core_output in engine_core_outputs.values():
            for request_output in engine_core_output.outputs:
                outputs_by_request_id[request_output.request_id] = request_output

        for req_id, (request, slot_mapping) in pd_decode_context.items():
            existing = outputs_by_request_id.get(req_id)
            if existing is None:
                continue
            existing.routed_experts = _get_routed_experts_from_slot_mapping(
                mgr,
                request,
                slot_mapping,
                stopped=existing.finish_reason is not None,
                num_new_tokens=len(existing.new_token_ids),
            )

        return engine_core_outputs


@register_patch("RoutedExpertsSerializationPatch", OpenAIServing)
class RoutedExpertsSerializationPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_get_routed_experts_serialization_mode",
        "convert_ndarray_to_str",
        "restore_str_to_ndarray",
        "concatenate_dict_and_ndarray",
        "add_ndarray_info_to_dict",
    ]

    def _get_routed_experts_serialization_mode(self) -> str:
        # Fall back for configs created outside the CLI path.
        return getattr(
            self.engine_client.vllm_config,
            "routed_experts_serialization_mode",
            "zip_base64",
        )

    def convert_ndarray_to_str(
        self, data: np.ndarray
    ) -> tuple[str, tuple[int, ...], str]:
        shape = data.shape
        dtype_str = str(data.dtype)
        raw_bytes = data.tobytes()
        mode = self._get_routed_experts_serialization_mode()
        if mode == "zip_base64":
            serialized = zlib.compress(raw_bytes, level=1)
        elif mode == "base64":
            serialized = raw_bytes
        else:
            raise ValueError(f"Unsupported routed experts serialization mode: {mode}")
        b64_data = base64.b64encode(serialized).decode("utf-8")
        return b64_data, shape, dtype_str

    def restore_str_to_ndarray(
        self, b64_data: str, shape: tuple[int, ...], dtype_str: str
    ) -> np.ndarray:
        mode = self._get_routed_experts_serialization_mode()
        serialized = base64.b64decode(b64_data)
        if mode == "zip_base64":
            raw_bytes = zlib.decompress(serialized)
        elif mode == "base64":
            raw_bytes = serialized
        else:
            raise ValueError(f"Unsupported routed experts serialization mode: {mode}")
        return np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape)

    def concatenate_dict_and_ndarray(
        self, routed_experts_dict: dict[str, Any], expert_ids_arr: np.ndarray
    ) -> np.ndarray:
        prefill_expert_ids = routed_experts_dict["routed_experts_str"]
        shape = tuple(routed_experts_dict["routed_experts_shape"])
        dtype = routed_experts_dict["routed_experts_dtype"]
        if len(shape) != expert_ids_arr.ndim:
            raise ValueError(
                "Cannot concatenate routed_experts with different ranks: "
                f"{shape} vs {expert_ids_arr.shape}"
            )
        if tuple(shape[1:]) != tuple(expert_ids_arr.shape[1:]):
            raise ValueError(
                "Cannot concatenate routed_experts with different non-token shapes: "
                f"{shape[1:]} vs {expert_ids_arr.shape[1:]}"
            )
        if dtype != str(expert_ids_arr.dtype):
            raise ValueError(
                "Cannot concatenate routed_experts with different dtypes: "
                f"{dtype} vs {expert_ids_arr.dtype}"
            )
        arr = self.restore_str_to_ndarray(prefill_expert_ids, shape, dtype)
        return np.concatenate([arr, expert_ids_arr], axis=0)

    def add_ndarray_info_to_dict(
        self, data: np.ndarray, routed_experts_dict: dict[str, Any]
    ) -> None:
        b64_data, shape, dtype = self.convert_ndarray_to_str(data)
        routed_experts_dict["routed_experts_shape"] = shape
        routed_experts_dict["routed_experts_dtype"] = dtype
        routed_experts_dict["routed_experts_str_len"] = len(b64_data)
        routed_experts_dict["routed_experts_str"] = b64_data


def _is_prefill_node(serving: Any) -> bool:
    engine_client = getattr(serving, "engine_client", None)
    vllm_config = getattr(engine_client, "vllm_config", None)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(
        kv_transfer_config, "is_kv_transfer_instance", False
    ):
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        if kv_role == "kv_producer" or getattr(
            kv_transfer_config, "is_kv_producer", False
        ):
            return True
        if kv_role == "kv_consumer" or getattr(
            kv_transfer_config, "is_kv_consumer", False
        ):
            return False

    role = envs.OMNI_PD_ROLE
    if role == "prefill":
        return True
    if role == "decode":
        return False
    return False


def _extract_kv_routed_experts(
    kv_transfer_params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not kv_transfer_params:
        return None
    if all(key in kv_transfer_params for key in _ROUTED_EXPERT_KEYS):
        return {key: kv_transfer_params[key] for key in _ROUTED_EXPERT_KEYS}
    return None


def _build_routed_experts_payload(
    serving: Any,
    output: Any,
    kv_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    routed_experts = getattr(output, "routed_experts", None)
    if routed_experts is not None and getattr(routed_experts, "shape", None) is not None:
        if routed_experts.shape[0] == 0:
            routed_experts = None

    if routed_experts is not None:
        if kv_payload is not None:
            routed_experts = serving.concatenate_dict_and_ndarray(
                kv_payload, routed_experts
            )
        payload: dict[str, Any] = {}
        serving.add_ndarray_info_to_dict(routed_experts, payload)
        return payload
    return kv_payload


def _stash_prefill_routed_experts(serving: Any, request_output: Any) -> None:
    if not _is_prefill_node(serving):
        return

    outputs = getattr(request_output, "outputs", None) or []
    for output in outputs:
        routed_experts = getattr(output, "routed_experts", None)
        if routed_experts is None or getattr(routed_experts, "shape", None) is None:
            continue
        if routed_experts.shape[0] == 0:
            continue
        if getattr(request_output, "kv_transfer_params", None) is None:
            request_output.kv_transfer_params = {}
        serving.add_ndarray_info_to_dict(
            routed_experts,
            request_output.kv_transfer_params,
        )
        return


def _attach_context_routed_experts(choice: Any) -> None:
    payload = _CURRENT_ROUTED_EXPERTS_PAYLOAD.get()
    if payload is not None:
        choice.routed_experts = payload


def _ensure_choice_patches_applied() -> None:
    global _CHOICE_PATCHES_APPLIED
    if _CHOICE_PATCHES_APPLIED:
        return

    def _wrap_chat_stream_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_CHAT_STREAM_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    def _wrap_chat_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_CHAT_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    def _wrap_completion_stream_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_COMPLETION_STREAM_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    def _wrap_completion_choice_init(self, *args: Any, **kwargs: Any):
        _ORIGINAL_COMPLETION_CHOICE_INIT(self, *args, **kwargs)
        _attach_context_routed_experts(self)

    ChatCompletionResponseStreamChoice.__init__ = _wrap_chat_stream_choice_init
    ChatCompletionResponseChoice.__init__ = _wrap_chat_choice_init
    CompletionResponseStreamChoice.__init__ = _wrap_completion_stream_choice_init
    CompletionResponseChoice.__init__ = _wrap_completion_choice_init
    _CHOICE_PATCHES_APPLIED = True


class _OutputSequenceProxy:
    def __init__(
        self,
        outputs: list[Any],
        payload_builder: Callable[[Any], dict[str, Any] | None],
    ):
        self._outputs = outputs
        self._payload_builder = payload_builder

    def __iter__(self) -> Iterator[Any]:
        for output in self._outputs:
            token = _CURRENT_ROUTED_EXPERTS_PAYLOAD.set(
                self._payload_builder(output)
            )
            try:
                yield output
            finally:
                _CURRENT_ROUTED_EXPERTS_PAYLOAD.reset(token)

    def __len__(self) -> int:
        return len(self._outputs)

    def __getitem__(self, index: int) -> Any:
        return self._outputs[index]


class _RequestOutputProxy:
    def __init__(
        self,
        request_output: Any,
        payload_builder: Callable[[Any], dict[str, Any] | None],
    ):
        self._request_output = request_output
        self.outputs = _OutputSequenceProxy(request_output.outputs, payload_builder)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._request_output, name)


async def _wrap_request_output_stream(
    serving: Any,
    request: Any,
    result_generator: AsyncIterator[Any],
) -> AsyncIterator[Any]:
    kv_payload = _extract_kv_routed_experts(getattr(request, "kv_transfer_params", None))
    prefill_attached_choices: set[int] = set()

    def _build_streaming_payload(output: Any) -> dict[str, Any] | None:
        output_kv_payload = None
        if bool(getattr(output, "token_ids", None)) and output.index not in prefill_attached_choices:
            prefill_attached_choices.add(output.index)
            output_kv_payload = kv_payload
        return _build_routed_experts_payload(serving, output, output_kv_payload)

    async for result in result_generator:
        _stash_prefill_routed_experts(serving, result)
        yield _RequestOutputProxy(
            result,
            _build_streaming_payload,
        )


async def _wrap_indexed_request_output_stream(
    serving: Any,
    request: Any,
    result_generator: AsyncIterator[tuple[int, Any]],
) -> AsyncIterator[tuple[int, Any]]:
    kv_payload = _extract_kv_routed_experts(getattr(request, "kv_transfer_params", None))
    prefill_attached_outputs: set[tuple[int, int]] = set()

    async for prompt_idx, result in result_generator:
        def _build_streaming_payload(
            output: Any,
            current_prompt_idx: int = prompt_idx,
        ) -> dict[str, Any] | None:
            output_kv_payload = None
            key = (current_prompt_idx, output.index)
            if bool(getattr(output, "token_ids", None)) and key not in prefill_attached_outputs:
                prefill_attached_outputs.add(key)
                output_kv_payload = kv_payload
            return _build_routed_experts_payload(serving, output, output_kv_payload)

        _stash_prefill_routed_experts(serving, result)
        yield prompt_idx, _RequestOutputProxy(
            result,
            _build_streaming_payload,
        )


@register_patch("ExpertIdServingChatStream", OpenAIServingChat)
class ExpertIdServingChatStream(VLLMPatch):
    _attr_names_to_apply = ["chat_completion_stream_generator"]

    async def chat_completion_stream_generator(
        self,
        request: Any,
        result_generator: AsyncIterator[Any],
        request_id: str,
        model_name: str,
        conversation: list[Any],
        tokenizer: Any,
        request_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ):
        _ensure_choice_patches_applied()
        async for chunk in _ORIGINAL_CHAT_STREAM(
            self,
            request,
            _wrap_request_output_stream(self, request, result_generator),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            *args,
            **kwargs,
        ):
            yield chunk


# Reuse the PD registration name so only the outer APC patch is applied.
# The saved method references form APC -> expert export -> PD -> vLLM.
@register_patch("PrefilledTokenSkipOpenAIServingChat", OpenAIServingChat)
class ExpertIdServingChatFinal(VLLMPatch):
    """Attach expert routing without duplicating PD response handling."""

    _attr_names_to_apply = ["chat_completion_full_generator"]

    async def chat_completion_full_generator(
        self,
        request: Any,
        result_generator: AsyncIterator[Any],
        *args: Any,
        **kwargs: Any,
    ):
        final_res = None

        async def final_result_with_routing():
            nonlocal final_res
            async for result in result_generator:
                final_res = result
            if final_res is not None:
                # Stash P routing before the PD method builds the wire response.
                _stash_prefill_routed_experts(self, final_res)
                yield final_res

        # PD owns token reuse, tool parsing and generation-error handling.
        response = await _ORIGINAL_CHAT_FULL(
            self,
            request,
            final_result_with_routing(),
            *args,
            **kwargs,
        )
        if isinstance(response, ErrorResponse) or final_res is None:
            return response

        kv_payload = _extract_kv_routed_experts(
            getattr(request, "kv_transfer_params", None)
        )
        for choice, output in zip(response.choices, final_res.outputs):
            payload = _build_routed_experts_payload(
                self, output, kv_payload
            )
            if payload is not None:
                choice.routed_experts = payload
        return response


@register_patch("ExpertIdServingCompletionStream", OpenAIServingCompletion)
class ExpertIdServingCompletionStream(VLLMPatch):
    _attr_names_to_apply = ["completion_stream_generator"]

    async def completion_stream_generator(
        self,
        request: Any,
        engine_prompts: list[Any],
        result_generator: AsyncIterator[tuple[int, Any]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: Any,
        request_metadata: Any,
    ):
        _ensure_choice_patches_applied()
        async for chunk in _ORIGINAL_COMPLETION_STREAM(
            self,
            request,
            engine_prompts,
            _wrap_indexed_request_output_stream(self, request, result_generator),
            request_id,
            created_time,
            model_name,
            num_prompts,
            tokenizer,
            request_metadata,
        ):
            yield chunk


@register_patch("ExpertIdServingCompletionFinal", OpenAIServingCompletion)
class ExpertIdServingCompletionFinal(VLLMPatch):
    _attr_names_to_apply = ["request_output_to_completion_response"]

    def request_output_to_completion_response(
        self,
        final_res_batch: list[Any],
        request: Any,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: Any,
        request_metadata: Any,
    ):
        # --- skip_tokenize P-side: set prompt_token_ids for D-side relay ---
        skip_decode = envs.OMNI_SKIP_DECODE_TOKENIZE
        if skip_decode and _is_prefill_node(self) and final_res_batch:
            first_res = final_res_batch[0]
            if first_res.kv_transfer_params is None:
                first_res.kv_transfer_params = {}
            raw_pt_ids = getattr(first_res, "prompt_token_ids", None)
            pt_ids = list(raw_pt_ids) if raw_pt_ids is not None else []
            first_res.kv_transfer_params["prompt_token_ids"] = pt_ids
            logger.info(
                "[SKIP_DECODE_CHECK][P-side-completion-nonstream] "
                "request_output_to_completion_response: set kv_transfer_params "
                "prompt_token_ids len=%s",
                len(pt_ids),
            )
        # --- existing ExpertId logic ---
        payloads: list[dict[str, Any] | None] = []
        kv_payload = _extract_kv_routed_experts(getattr(request, "kv_transfer_params", None))
        for result in final_res_batch:
            _stash_prefill_routed_experts(self, result)
            for output in result.outputs:
                payloads.append(
                    _build_routed_experts_payload(
                        self,
                        output,
                        kv_payload,
                    )
                )
        response = _ORIGINAL_COMPLETION_FINAL(
            self,
            final_res_batch,
            request,
            request_id,
            created_time,
            model_name,
            tokenizer,
            request_metadata,
        )
        for choice, payload in zip(response.choices, payloads):
            if payload is not None:
                choice.routed_experts = payload
        return response
