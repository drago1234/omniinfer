# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Reuse P-side output tokens on D with consistent output and scheduler state.

vLLM 0.25.1 has no single extension hook covering this cross-layer flow.
A custom Scheduler alone cannot deliver the first token to the frontend;
serving/engine subclasses would also require replacing their construction
sites. The existing general plugin loads this patch, but plugin registration
alone does not intercept these operations, so selected methods are wrapped.

Scope: OpenAIServingChat response metadata/assembly, AsyncLLM.generate, and
Scheduler KV completion, finish notifications and resumed-request token counts.
OnlineRenderer preprocessing is adapted through InputIdsPiggybackPatch.
Model execution and KV data transport are unchanged.

Remove when upstream provides equivalent token relay and state accounting,
or extension hooks that replace these wrappers, after validating output,
stop/boundary and request-resume behavior.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, AsyncGenerator

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.inputs import EngineInput, PromptType
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.v1.request import Request, RequestStatus
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine.async_llm import AsyncLLM

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_original_chat_completion_full_generator = OpenAIServingChat.chat_completion_full_generator
_original_preprocess_chat = OnlineRenderer.preprocess_chat
_original_update_from_kv_xfer_finished = Scheduler._update_from_kv_xfer_finished
_original_make_cached_request_data = Scheduler._make_cached_request_data
_original_generate = AsyncLLM.generate


def _get_serving_vllm_config(serving: OpenAIServingChat) -> Any:
    """Resolve the config used by HTTP admission checks."""
    for attr_name in ("_omni_vllm_config", "vllm_config"):
        vllm_config = getattr(serving, attr_name, None)
        if vllm_config is not None:
            return vllm_config

    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None:
            return vllm_config
    except (ImportError, AttributeError):
        pass

    engine_client = getattr(serving, "engine_client", None)
    return getattr(engine_client, "vllm_config", None)


def _is_kv_producer(serving: OpenAIServingChat) -> bool:
    vllm_config = _get_serving_vllm_config(serving)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and getattr(
        kv_transfer_config, "is_kv_transfer_instance", False
    ):
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        return kv_role == "kv_producer" or getattr(
            kv_transfer_config, "is_kv_producer", False
        )
    return envs.OMNI_PD_ROLE == "prefill"


async def to_async_iterator(request_output: RequestOutput) -> AsyncIterator[RequestOutput]:
    yield request_output


def _slice_first_logprobs(logprobs):
    return logprobs[:1] if logprobs else logprobs


def _first_token_cumulative_logprob(logprobs, token_id, fallback):
    first_step = logprobs[0] if logprobs else None
    if first_step:
        logprob_obj = first_step.get(token_id) or first_step.get(str(token_id))
        if isinstance(logprob_obj, dict):
            value = logprob_obj.get("logprob")
        else:
            value = getattr(logprob_obj, "logprob", None)
        if value is not None:
            return value
    return fallback


@register_patch("PrefilledTokenSkipOpenAIServingChat", OpenAIServingChat)
class OpenAIServingChatPatch(VLLMPatch):
    _attr_names_to_apply = ['chat_completion_full_generator']

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list,
        tokenizer: TokenizerLike | None,
        request_metadata,
        parser: Any = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            return self.create_error_response(str(e))

        if final_res is None:
            raise RuntimeError("result_generator should yield at least one RequestOutput.")

        reuse_prefilled_tokens = envs.OMNI_REUSE_PREFILLED_TOKENS
        skip_decode_tokenize = envs.OMNI_SKIP_DECODE_TOKENIZE

        if reuse_prefilled_tokens:
            prefilled_token_ids = getattr(request, "_omni_prefilled_token_ids", None)
            if prefilled_token_ids:
                output = final_res.outputs[0]
                prefilled_text = getattr(request, "_omni_prefilled_text", "")
                prefilled_logprobs = getattr(
                    request, "_omni_prefilled_logprobs", None)
                prefilled_cumulative_logprob = getattr(
                    request, "_omni_prefilled_cumulative_logprob", None)
                if prefilled_text and not output.text.startswith(prefilled_text):
                    output.text = prefilled_text + output.text
                if not output.token_ids or output.token_ids[0] != prefilled_token_ids[0]:
                    output.token_ids = prefilled_token_ids + list(output.token_ids)
                    if output.logprobs is not None and prefilled_logprobs:
                        output.logprobs = prefilled_logprobs + output.logprobs
                    if (prefilled_cumulative_logprob is not None
                            and output.cumulative_logprob is not None):
                        output.cumulative_logprob += prefilled_cumulative_logprob
        ## In Prefill node, the response will carry prompt_token_ids with kv_transfer_params

        if skip_decode_tokenize:
            if final_res.kv_transfer_params:
                final_res.kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids
        if reuse_prefilled_tokens:
            if final_res.kv_transfer_params:
                prefilled_token_id = final_res.outputs[0].token_ids[0]
                prefilled_logprobs = _slice_first_logprobs(final_res.outputs[0].logprobs)
                final_res.kv_transfer_params["prefilled_token"] = [prefilled_token_id]
                final_res.kv_transfer_params["stop_reasons"] = [
                    output.stop_reason for output in final_res.outputs
                ]
                final_res.kv_transfer_params["prefilled_logprobs"] = prefilled_logprobs
                final_res.kv_transfer_params[
                    "prefilled_cumulative_logprob"
                ] = _first_token_cumulative_logprob(
                    prefilled_logprobs,
                    prefilled_token_id,
                    final_res.outputs[0].cumulative_logprob,
                )

        is_prefill_node = _is_kv_producer(self)

        response_request = request
        tool_choice = getattr(request, "tool_choice", None)
        forced_tool_choice = (
            isinstance(
                tool_choice,
                ChatCompletionNamedToolChoiceParam,
            )
            or tool_choice == "required"
        )
        if is_prefill_node and forced_tool_choice:
            # The prefill result is transport data, not a complete model
            # response. It may contain only the first generated token, so the
            # normal reasoning/forced-tool response parser must not interpret
            # it as a finished reasoning/end-marker/JSON sequence.
            response_request = request.model_copy(
                update={"tool_choice": "none"},
                deep=False,
            )
            # v0.25 passes the parser per call rather than on the serving object.
            parser = None

        response = await _original_chat_completion_full_generator(
            self,
            response_request,
            to_async_iterator(final_res),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )
        return response


# ``InputIdsPiggybackPatch`` is the sole owner of
# ``OnlineRenderer.preprocess_chat``.  It calls this adapter on its fallback
# path, giving us an explicit and order-independent relay:
# InputIdsPiggyback -> PrefilledTokenSkip -> vLLM OnlineRenderer.
class OnlineRendererPrefilledTokenAdapter:

    @staticmethod
    async def preprocess_chat(
        renderer: OnlineRenderer,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: Any,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        parser: Any = None,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[Any], list[EngineInput]]:
        conversation, [engine_prompt] = await _original_preprocess_chat(
            renderer,
            request,
            messages,
            default_template,
            default_template_content_format,
            default_template_kwargs,
            tool_dicts=tool_dicts,
            parser=parser,
            skip_mm_cache=skip_mm_cache,
        )

        reuse_prefilled_tokens = envs.OMNI_REUSE_PREFILLED_TOKENS
        for attr in (
                "_omni_prefilled_token_ids",
                "_omni_prefilled_text",
                "_omni_prefilled_logprobs",
                "_omni_prefilled_cumulative_logprob",
        ):
            if hasattr(request, attr):
                delattr(request, attr)
        if reuse_prefilled_tokens:
            if request.kv_transfer_params and "prefilled_token" in request.kv_transfer_params:
                tokenizer = renderer.renderer.tokenizer
                new_tokens = tokenizer.convert_ids_to_tokens(
                    request.kv_transfer_params["prefilled_token"][0])
                delta_text = tokenizer.convert_tokens_to_string([new_tokens])
                # If the decoded text ends with '�', it indicates an incomplete UTF-8 sequence — abandon reuse.
                # If the prefilled side has already triggered a stop reason, we also fall back to normal generation.
                if (
                    delta_text.endswith("�")
                    or any(
                        stop_reason is not None
                        for stop_reason in request.kv_transfer_params["stop_reasons"]
                    )
                    or request.kv_transfer_params["prefilled_token"][0] == tokenizer.eos_token_id
                ):
                    request.kv_transfer_params.pop("prefilled_token", None)
                else:
                    engine_prompt[
                        "prefilled_token_ids"] = request.kv_transfer_params[
                            "prefilled_token"]
                    engine_prompt["prefilled_texts"] = delta_text
                    prefilled_logprobs = _slice_first_logprobs(
                        request.kv_transfer_params["prefilled_logprobs"])
                    prefilled_token_id = request.kv_transfer_params[
                        "prefilled_token"][0]
                    prefilled_cumulative_logprob = _first_token_cumulative_logprob(
                        prefilled_logprobs,
                        prefilled_token_id,
                        request.kv_transfer_params["prefilled_cumulative_logprob"],
                    )
                    engine_prompt["prefilled_logprobs"] = prefilled_logprobs
                    engine_prompt[
                        "prefilled_cumulative_logprob"] = prefilled_cumulative_logprob
                    request._omni_prefilled_token_ids = request.kv_transfer_params[
                        "prefilled_token"]
                    request._omni_prefilled_text = delta_text
                    request._omni_prefilled_logprobs = prefilled_logprobs
                    request._omni_prefilled_cumulative_logprob = (
                        prefilled_cumulative_logprob)
        return conversation, [engine_prompt]


def _append_prefilled_token(request: Request) -> bool:
    extra_args = getattr(request.sampling_params, "extra_args", None) or {}
    kv_params = extra_args.get("kv_transfer_params")
    prefilled_token = kv_params.get("prefilled_token") if kv_params else None
    if not prefilled_token:
        return False

    request.prompt_token_ids.extend(prefilled_token)
    request.append_output_token_ids(prefilled_token)
    request._omni_prefilled_token_count = len(prefilled_token)
    # Consume exactly once; the scheduler may observe the same transfer result
    # in more than one scheduling tick.
    kv_params.pop("prefilled_token", None)
    return True


def _queue_finish_notification(
    scheduler: Scheduler, request: Request, finished_status: "RequestStatus"
) -> None:
    pending = getattr(scheduler, "_omni_pending_finish_outputs", None)
    if pending is None:
        pending = []
        scheduler._omni_pending_finish_outputs = pending
    pending.append(
        (
            request.client_index,
            EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=RequestStatus.get_finished_reason(finished_status),
                stop_reason=request.stop_reason,
                events=request.take_events(),
                trace_headers=request.trace_headers,
                prefill_stats=request.take_prefill_stats(),
            ),
        )
    )


def _finish_request_and_notify_client(
    scheduler: Scheduler, request: Request, finished_status: "RequestStatus"
) -> None:
    scheduler.finish_requests(request.request_id, finished_status)
    _queue_finish_notification(scheduler, request, finished_status)


def drain_pending_finish_outputs(scheduler: Scheduler, outputs) -> None:
    """Drain queued finish notifications into update_from_output outputs."""
    pending = getattr(scheduler, "_omni_pending_finish_outputs", None)
    if not pending:
        return
    for client_index, engine_core_output in pending:
        # v0.25 uses per-client EngineCoreOutputs instead of output lists.
        if client_index not in outputs:
            outputs[client_index] = EngineCoreOutputs()
        outputs[client_index].outputs.append(engine_core_output)
    pending.clear()


@register_patch("PrefilledTokenSkipKvXferFinished", Scheduler)
class SchedulerKvXferFinishedPatch(VLLMPatch):
    _attr_names_to_apply = [
        "_update_from_kv_xfer_finished", "update_from_output", "_make_cached_request_data",
    ]

    _upstream_update_from_output = Scheduler.update_from_output

    @classmethod
    def apply(cls):
        # Relay to other update_from_output wrappers instead of replacing them.
        cls.apply_bypass_conflict("update_from_output")

    def _make_cached_request_data(
        self, running_reqs, resumed_reqs, num_scheduled_tokens,
        spec_decode_tokens, req_to_new_blocks,
    ):
        data = _original_make_cached_request_data(
            self, running_reqs, resumed_reqs, num_scheduled_tokens,
            spec_decode_tokens, req_to_new_blocks,
        )
        if self.use_v2_model_runner:
            return data

        for index, req_id in enumerate(data.req_ids):
            request = self.requests[req_id]
            prefilled_count = getattr(request, "_omni_prefilled_token_count", 0)
            if not prefilled_count:
                continue
            # MRv1 already has the relayed token in its prompt. Do not restore
            # it again as output when a paused request re-enters the batch.
            # Scheduler output budgets and async placeholders stay unchanged.
            # A streaming session update may already have reset the prompt count.
            data.num_output_tokens[index] -= min(
                prefilled_count, len(request.prompt_token_ids) - request.num_prompt_tokens)
        return data

    def _update_from_kv_xfer_finished(self, kv_connector_output) -> None:
        _original_update_from_kv_xfer_finished(self, kv_connector_output)

        reuse_prefilled_tokens = envs.OMNI_REUSE_PREFILLED_TOKENS
        if not reuse_prefilled_tokens:
            return

        for req_id in kv_connector_output.finished_recving or ():
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue

            if req_id in self.failed_recving_kv_req_ids:
                continue

            if not _append_prefilled_token(request):
                continue

            if check_stop(request, self.max_model_len):
                finished_status = request.status
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                # v0.25 finish_requests consumes the receive-completed set to
                # decide whether KV can be freed. Do not discard req_id here.
                _finish_request_and_notify_client(self, request, finished_status)

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = SchedulerKvXferFinishedPatch._upstream_update_from_output(self, scheduler_output, model_runner_output)
        drain_pending_finish_outputs(self, outputs)
        return outputs


@register_patch("PrefilledTokenSkipAsyncLLM", AsyncLLM)
class AsyncLLMPatch(VLLMPatch):
    _attr_names_to_apply = ['generate']

    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType | EngineInput,
        sampling_params,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        from collections import namedtuple
        from vllm.sampling_params import RequestOutputKind

        Logprob = namedtuple('Logprob', ['logprob', 'rank', 'decoded_token'])

        def convert_to_standard_logprobs(
            prefilled_logprobs: list[dict[str, Any]],
        ) -> list[dict[int, Logprob]]:
            if prefilled_logprobs is None:
                return None
            result = []
            for item in prefilled_logprobs:
                new_item = {}
                for token_id_str, data in item.items():
                    token_id = int(token_id_str)
                    logprob_obj = Logprob(
                        logprob=data['logprob'],
                        rank=data['rank'],
                        decoded_token=data['decoded_token']
                    )
                    new_item[token_id] = logprob_obj
                result.append(new_item)
            return result

        reuse_prefilled_tokens = envs.OMNI_REUSE_PREFILLED_TOKENS
        pending_prefilled_output = None
        if reuse_prefilled_tokens:
            prefilled_token_ids: list[int] = []
            prompt_token_ids: list[int] | None = None
            prefilled_text = ""
            prefilled_logprobs = None
            prefill_cumulative_logprob = None

            # v0.25.1 serving passes EngineInput before process_inputs runs.
            is_engine_input = (
                isinstance(prompt, Mapping)
                and "type" in prompt
                and "prefilled_token_ids" not in prompt
            )
            if isinstance(prompt, EngineCoreRequest) or is_engine_input:
                prompt_token_ids = (
                    prompt.get("prompt_token_ids") if is_engine_input
                    else prompt.prompt_token_ids
                )
                extra_args = getattr(sampling_params, "extra_args", None) or {}
                kv_transfer_params = extra_args.get("kv_transfer_params", None)
                if kv_transfer_params and "prefilled_token" in kv_transfer_params:
                    prefilled_token_ids = kv_transfer_params["prefilled_token"] or []
                    if prefilled_token_ids:
                        tokenizer = self.tokenizer
                        if tokenizer is not None:
                            token = tokenizer.convert_ids_to_tokens(prefilled_token_ids[0])
                            prefilled_text = tokenizer.convert_tokens_to_string([token])

                    prefilled_logprobs = convert_to_standard_logprobs(kv_transfer_params["prefilled_logprobs"])
                    prefill_cumulative_logprob = kv_transfer_params["prefilled_cumulative_logprob"]

            elif isinstance(prompt, Mapping):
                if "prefilled_token_ids" in prompt:
                    prefilled_token_ids = prompt["prefilled_token_ids"] or []
                    prompt_token_ids = prompt.get("prompt_token_ids")
                    prefilled_text = prompt.get("prefilled_texts", "")
                    prompt["prefilled_token_ids"] = []
                    prefilled_logprobs = convert_to_standard_logprobs(
                        prompt.get("prefilled_logprobs"))
                    prefill_cumulative_logprob = prompt.get(
                        "prefilled_cumulative_logprob")

            if prefilled_token_ids:
                synthetic_outputs = [
                    CompletionOutput(
                        index=idx,
                        cumulative_logprob=prefill_cumulative_logprob,
                        logprobs=prefilled_logprobs,
                        text=prefilled_text,
                        token_ids=prefilled_token_ids,
                    )
                    for idx in range(sampling_params.n)
                ]
                synthetic_res = RequestOutput(
                    request_id=request_id,
                    prompt=None,
                    finished=False,
                    prompt_logprobs=None,
                    prompt_token_ids=prompt_token_ids,
                    outputs=synthetic_outputs,
                )
                if getattr(sampling_params, "output_kind", None) == RequestOutputKind.DELTA:
                    yield synthetic_res
                else:
                    pending_prefilled_output = synthetic_res
        async for res in _original_generate(
            self,
            prompt,
            sampling_params,
            request_id,
            prompt_text=prompt_text,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            reasoning_ended=reasoning_ended,
            reasoning_parser_kwargs=reasoning_parser_kwargs,
        ):
            if pending_prefilled_output is not None:
                by_index = {
                    output.index: output
                    for output in pending_prefilled_output.outputs
                }
                for output in res.outputs:
                    prefilled_output = by_index.get(output.index)
                    if prefilled_output is None:
                        continue
                    prefilled_ids = list(prefilled_output.token_ids)
                    if output.token_ids and output.token_ids[0] == prefilled_ids[0]:
                        continue
                    output.token_ids = prefilled_ids + list(output.token_ids)
                    if (prefilled_output.text
                            and not output.text.startswith(prefilled_output.text)):
                        output.text = prefilled_output.text + output.text
                    if output.logprobs is not None and prefilled_output.logprobs:
                        output.logprobs = prefilled_output.logprobs + output.logprobs
                    if (output.cumulative_logprob is not None
                            and prefilled_output.cumulative_logprob is not None):
                        output.cumulative_logprob += (
                            prefilled_output.cumulative_logprob)
                pending_prefilled_output = None
            yield res
