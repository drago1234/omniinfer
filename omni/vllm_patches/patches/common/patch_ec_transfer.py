# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backport the EC transfer-parameter path missing from vLLM 0.25.1.

The connector has EC request/finish hooks, but vLLM 0.25.1 does not propagate
``ec_transfer_params`` end to end. Each patch here fills a required link:
register the EC side-channel environment variables; accept the parameter in
the OpenAI request and forward it through sampling params into ``Request``;
collect ``request_finished()`` data in the scheduler and attach it to
``EngineCoreOutput``; extend the msgspec output schema and aliases for that
field; and forward it through the output processor and ``RequestOutput``.
The final OpenAI response relay remains in ``patch_serving_apc.py`` to compose
with the existing serving patch chain.

There is no vLLM 0.25.1 plugin hook for extending these internal request,
scheduler, and engine-output contracts. Subclassing would require replacing
vLLM's construction/wiring of those components, so the existing OmniInfer
patch registry is the narrower compatibility layer.

This shim targets vLLM 0.25.1. The upstream implementation from vLLM PR
#42433 is first available in vLLM 0.26.0; remove this shim and the EC response
relay from ``patch_serving_apc.py`` when OmniInfer's minimum supported vLLM
version is raised to 0.26.0 or later.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TYPE_CHECKING

import vllm.envs as vllm_envs_module
import vllm.v1.core.sched.scheduler as scheduler_module
import vllm.v1.engine as engine_module
import vllm.v1.engine.coordinator as coordinator_module
import vllm.v1.engine.core as core_module
import vllm.v1.engine.core_client as core_client_module
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput as _EngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import (
    FinishReason,
    OutputProcessor,
    OutputProcessorOutput,
    RequestState,
)
from vllm.v1.request import Request

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

if TYPE_CHECKING:
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

_orig_request_output_init = RequestOutput.__init__
_orig_request_output_add = RequestOutput.add
_orig_request_init = Request.__init__
_orig_scheduler_update_from_output = Scheduler.update_from_output
_orig_engine_core_add_request = EngineCore.add_request
_orig_make_request_output = RequestState.make_request_output
_orig_to_sampling_params = ChatCompletionRequest.to_sampling_params


def _register_ec_transfer_fields() -> None:
    from pydantic.fields import FieldInfo

    changed = False

    if "ec_transfer_params" not in ChatCompletionRequest.model_fields:
        ChatCompletionRequest.model_fields["ec_transfer_params"] = FieldInfo(
            annotation=dict[str, Any] | None,
            default=None,
            description=(
                "ECTransfer parameters used for encoder-cache disaggregated serving."
            ),
        )
        changed = True

    if "ec_transfer_params" not in ChatCompletionResponse.model_fields:
        ChatCompletionResponse.model_fields["ec_transfer_params"] = FieldInfo(
            annotation=dict[str, Any] | None,
            default=None,
            description="ECTransfer parameters.",
        )
        changed = True

    if changed:
        ChatCompletionRequest.model_rebuild(force=True)
        ChatCompletionResponse.model_rebuild(force=True)
        ChatCompletionRequest.field_names = None
        ChatCompletionResponse.field_names = None


@register_patch("ECTransferEnvVarsPatch", vllm_envs_module)
class ECTransferEnvVarsPatch(VLLMPatch):
    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        env_map = getattr(vllm_envs_module, "environment_variables", None)
        if not isinstance(env_map, dict):
            raise TypeError("vllm.envs.environment_variables must be a dict")

        env_map.setdefault(
            "VLLM_EC_SIDE_CHANNEL_HOST",
            lambda: vllm_envs_module.os.getenv("VLLM_EC_SIDE_CHANNEL_HOST", "localhost"),
        )
        env_map.setdefault(
            "VLLM_EC_SIDE_CHANNEL_PORT",
            lambda: int(vllm_envs_module.os.getenv("VLLM_EC_SIDE_CHANNEL_PORT", "5601")),
        )


@register_patch("ECTransferRequestOutputPatch", RequestOutput)
class ECTransferRequestOutputPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__", "add"]

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs,
        outputs,
        finished: bool,
        metrics=None,
        lora_request=None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        _orig_request_output_init(
            self,
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=outputs,
            finished=finished,
            metrics=metrics,
            lora_request=lora_request,
            encoder_prompt=encoder_prompt,
            encoder_prompt_token_ids=encoder_prompt_token_ids,
            num_cached_tokens=num_cached_tokens,
            kv_transfer_params=kv_transfer_params,
            **kwargs,
        )
        self.ec_transfer_params = ec_transfer_params

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        _orig_request_output_add(self, next_output, aggregate)
        self.ec_transfer_params = getattr(next_output, "ec_transfer_params", None)


@register_patch("ECTransferRequestPatch", Request)
class ECTransferRequestPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: Request.SamplingParams | None,
        pooling_params: Request.PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: Request.torch.Tensor | None = None,
        prompt_is_token_ids: list[bool] | None = None,
        mm_features: list[Request.MultiModalFeatureSpec] | None = None,
        lora_request: "Request.LoRARequest | None" = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["Request.BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
        abort_immediately: bool = False,
    ) -> None:
        _orig_request_init(
            self,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            client_index=client_index,
            arrival_time=arrival_time,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            lora_request=lora_request,
            cache_salt=cache_salt,
            priority=priority,
            trace_headers=trace_headers,
            block_hasher=block_hasher,
            resumable=resumable,
            reasoning_ended=reasoning_ended,
            reasoning_parser_kwargs=reasoning_parser_kwargs,
            abort_immediately=abort_immediately,
        )

        self.ec_transfer_params: dict[str, Any] | None = None
        if sampling_params is not None and sampling_params.extra_args is not None:
            self.ec_transfer_params = sampling_params.extra_args.get(
                "ec_transfer_params"
            )


@register_patch("ECTransferSchedulerPatch", Scheduler)
class ECTransferSchedulerPatch(VLLMPatch):
    _attr_names_to_apply = [
        "get_ec_connector",
        "update_from_output",
        "_free_request",
    ]

    def get_ec_connector(self) -> "ECConnectorBase | None":
        return self.ec_connector

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        ec_params_by_req: dict[str, dict[str, Any] | None] = {}

        def _free_request_adapter(
            request: Request, delay_free_blocks: bool = False
        ) -> dict[str, Any] | None:
            kv_xfer_params, ec_xfer_params = ECTransferSchedulerPatch._free_request(
                self,
                request,
                delay_free_blocks,
            )
            ec_params_by_req[request.request_id] = ec_xfer_params
            return kv_xfer_params

        original_free_request = getattr(self, "_free_request")
        self._free_request = _free_request_adapter
        try:
            engine_core_outputs = _orig_scheduler_update_from_output(
                self,
                scheduler_output,
                model_runner_output,
            )
        finally:
            self._free_request = original_free_request

        if not ec_params_by_req:
            return engine_core_outputs

        for outputs in engine_core_outputs.values():
            for output in outputs.outputs:
                req_id = output.request_id
                if req_id in ec_params_by_req:
                    output.ec_transfer_params = ec_params_by_req.pop(req_id)

        return engine_core_outputs

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not request.is_finished():
            raise RuntimeError("Cannot free an unfinished request")

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params


@register_patch("ECTransferEngineCorePatch", EngineCore)
class ECTransferEngineCorePatch(VLLMPatch):
    _attr_names_to_apply = ["add_request"]

    def add_request(self, request: Request, request_wave: int = 0):
        if (
            getattr(request, "ec_transfer_params", None) is not None
            and self.scheduler.get_ec_connector() is None
        ):
            logger.warning(
                "Got ec_transfer_params, but no ECConnector found. "
                "Disabling ECTransfer for this request."
            )

        return _orig_engine_core_add_request(self, request, request_wave)


@register_patch("ECTransferEngineInitPatch", engine_module)
class ECTransferEngineInitPatch(VLLMPatch):
    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        struct_fields = getattr(_EngineCoreOutput, "__struct_fields__", ())
        if "ec_transfer_params" in struct_fields:
            return

        class EngineCoreOutputWithEC(_EngineCoreOutput):
            ec_transfer_params: dict[str, Any] | None = None

        class EngineCoreOutputsWithEC(_EngineCoreOutputs):
            outputs: list[EngineCoreOutputWithEC] = []

        # msgspec resolves postponed annotations in the module namespace, where
        # this function-local class is unavailable. Bind its actual type before
        # any decoder inspects the patched output schema.
        EngineCoreOutputsWithEC.__annotations__["outputs"] = list[EngineCoreOutputWithEC]

        engine_module.EngineCoreOutput = EngineCoreOutputWithEC
        engine_module.EngineCoreOutputs = EngineCoreOutputsWithEC

        core_module.EngineCoreOutput = EngineCoreOutputWithEC
        core_module.EngineCoreOutputs = EngineCoreOutputsWithEC
        core_client_module.EngineCoreOutputs = EngineCoreOutputsWithEC
        coordinator_module.EngineCoreOutputs = EngineCoreOutputsWithEC
        scheduler_module.EngineCoreOutput = EngineCoreOutputWithEC
        scheduler_module.EngineCoreOutputs = EngineCoreOutputsWithEC


@register_patch("ECTransferRequestStatePatch", RequestState)
class ECTransferRequestStatePatch(VLLMPatch):
    _attr_names_to_apply = ["make_request_output"]

    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ):
        request_output = _orig_make_request_output(
            self,
            new_token_ids,
            pooling_output,
            finish_reason,
            stop_reason,
            kv_transfer_params,
        )
        if request_output is not None:
            request_output.ec_transfer_params = ec_transfer_params
        return request_output


@register_patch("ECTransferOutputProcessorPatch", OutputProcessor)
class ECTransferOutputProcessorPatch(VLLMPatch):
    _attr_names_to_apply = ["process_outputs"]

    def process_outputs(
        self,
        engine_core_outputs,
        engine_core_timestamp: float | None = None,
        iteration_stats=None,
    ) -> OutputProcessorOutput:
        request_outputs = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue

            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            pooling_output = engine_core_output.pooling_output
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            ec_transfer_params = getattr(engine_core_output, "ec_transfer_params", None)
            if engine_core_output.routed_experts is not None:
                req_state.routed_experts_chunks.append(
                    engine_core_output.routed_experts
                )

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                req_state.is_prefilling = False

            if pooling_output is None:
                if req_state.detokenizer is None:
                    raise RuntimeError("Detokenizer is required for token output")
                if req_state.logprobs_processor is None:
                    raise RuntimeError("Logprobs processor is required for token output")
                stop_string = req_state.detokenizer.update(
                    new_token_ids, finish_reason == FinishReason.STOP
                )
                if stop_string:
                    finish_reason = FinishReason.STOP
                    stop_reason = stop_string

                req_state.logprobs_processor.update_from_output(engine_core_output)

            if request_output := req_state.make_request_output(
                new_token_ids,
                pooling_output,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
            ):
                if req_state.streaming_input:
                    request_output.finished = False

                if req_state.queue is not None:
                    req_state.queue.put(request_output)
                else:
                    request_outputs.append(request_output)

            if finish_reason is not None:
                if req_state.streaming_input:
                    if req_state.input_chunk_queue:
                        update = req_state.input_chunk_queue.popleft()
                        req_state.apply_streaming_update(update)
                    else:
                        req_state.input_chunk_queue = None
                else:
                    self._finish_request(req_state)
                    if not engine_core_output.finished:
                        reqs_to_abort.append(req_id)

                    self._update_stats_from_finished(
                        req_state, finish_reason, iteration_stats
                    )
                    if self.tracing_enabled:
                        self.do_tracing(engine_core_output, req_state, iteration_stats)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )


@register_patch("ECTransferChatProtocolFieldsPatch", ChatCompletionRequest)
class ECTransferChatProtocolFieldsPatch(VLLMPatch):
    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        _register_ec_transfer_fields()


@register_patch("ECTransferChatProtocolSamplingPatch", ChatCompletionRequest)
class ECTransferChatProtocolSamplingPatch(VLLMPatch):
    _attr_names_to_apply = ["to_sampling_params"]

    def to_sampling_params(
        self,
        max_tokens: int,
        default_sampling_params: dict,
    ):
        sampling_params = _orig_to_sampling_params(
            self,
            max_tokens=max_tokens,
            default_sampling_params=default_sampling_params,
        )

        if self.ec_transfer_params:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["ec_transfer_params"] = self.ec_transfer_params

        return sampling_params
