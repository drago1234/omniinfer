# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# This patch is used for reuse prefilled tokens

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, AsyncGenerator

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.inputs import EngineInput, PromptType
from vllm.tokenizers import TokenizerLike
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreRequest
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.v1.request import Request
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.async_llm import AsyncLLM

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_original_chat_completion_full_generator = OpenAIServingChat.chat_completion_full_generator
_original_preprocess_chat = OnlineRenderer.preprocess_chat
_original_update_from_kv_xfer_finished = Scheduler._update_from_kv_xfer_finished
_original_generate = AsyncLLM.generate
logger = init_logger(__name__)
logger.info(
    "PD token relay trial 20260914 loaded: reuse=%s, skip_decode_tokenize=%s",
    envs.OMNI_REUSE_PREFILLED_TOKENS, envs.OMNI_SKIP_DECODE_TOKENIZE,
)


def _validate_trial_request(request, *, max_tokens=None):
    """Keep this A/B trial on ordinary n=1 chat generation, not structured output."""
    if getattr(request, "n", 1) not in (None, 1):
        raise ValueError("PD token relay trial requires n=1")
    if max_tokens is not None and max_tokens < 2:
        raise ValueError("PD token relay trial requires max_tokens >= 2")
    for field in (
        "stop", "tools", "structured_outputs", "use_beam_search",
    ):
        if getattr(request, field, None):
            raise ValueError("PD token relay trial does not support stop strings, tools or structured output")


def _prepend_output_once(output, token_ids, text, logprobs, cumulative):
    # An equal first token may be legitimate repetition: never deduplicate by
    # token/text equality. Mark the output object after merging instead.
    if getattr(output, "_omni_prefilled_merged", False):
        return
    output.token_ids = list(token_ids) + list(output.token_ids)
    output.text = text + output.text
    if output.logprobs is not None and logprobs:
        output.logprobs = logprobs + output.logprobs
    if output.cumulative_logprob is not None and cumulative is not None:
        output.cumulative_logprob += cumulative
    output._omni_prefilled_merged = True


def _replace_prompt_token_ids(
    engine_input: EngineInput,
    prompt_token_ids: list[int],
) -> EngineInput:
    """Replace token IDs without discarding multimodal input metadata."""
    updated_input = engine_input.copy()
    updated_input["prompt_token_ids"] = list(prompt_token_ids)
    return updated_input


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


def _add_prefilled_token_to_transfer(final_res: RequestOutput, kv_transfer_params: dict) -> None:
    """Attach the first generated token to the prefill hand-off envelope."""
    if not final_res.outputs:
        return
    prefilled_output = final_res.outputs[0]
    if not prefilled_output.token_ids:
        return

    prefilled_token_id = prefilled_output.token_ids[0]
    prefilled_logprobs = _slice_first_logprobs(prefilled_output.logprobs)
    kv_transfer_params["prefilled_token"] = [prefilled_token_id]
    kv_transfer_params["stop_reasons"] = [
        output.stop_reason for output in final_res.outputs
    ]
    kv_transfer_params["prefilled_logprobs"] = prefilled_logprobs
    kv_transfer_params["prefilled_cumulative_logprob"] = (
        _first_token_cumulative_logprob(
            prefilled_logprobs,
            prefilled_token_id,
            prefilled_output.cumulative_logprob,
        )
    )


def _can_relay_prefilled_token(request, tokenizer, token_id, text, stop_reasons) -> bool:
    """Require standalone text and a token that has not ended generation."""
    if not text or text.endswith("�"):
        return False
    if any(reason is not None for reason in stop_reasons):
        return False
    if token_id == tokenizer.eos_token_id:
        return False
    stop_token_ids = getattr(request, "stop_token_ids", None) or ()
    return token_id not in stop_token_ids


def _decode_single_token(tokenizer: TokenizerLike, token_id: int) -> str:
    """Decode a relayed token ID through the tokenizer's real decoder.

    ``TokenizersBackend`` subclasses may inherit a generic
    ``convert_tokens_to_string`` implementation that exposes ByteLevel pieces
    such as ``Ġ``.  ``decode`` uses the backend decoder and is also the public
    token-ID-to-text API.  Keep the old conversion path for lightweight or
    legacy tokenizers that do not provide it.
    """
    decode = getattr(tokenizer, "decode", None)
    if decode is not None:
        try:
            return decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            # Some custom tokenizers expose the older, narrower signature.
            return decode([token_id])

    token = tokenizer.convert_ids_to_tokens(token_id)
    return tokenizer.convert_tokens_to_string([token])


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
        *args: Any,
        **kwargs: Any,
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
                _prepend_output_once(output, prefilled_token_ids, prefilled_text,
                                     prefilled_logprobs, prefilled_cumulative_logprob)
        # The prefill response is the PD hand-off envelope.  In vLLM 0.25.1
        # ``kv_transfer_params`` is present only when this result participates
        # in a KV transfer, so standalone/non-PD requests remain untouched.
        kv_transfer_params = final_res.kv_transfer_params
        if skip_decode_tokenize and kv_transfer_params is not None:
            kv_transfer_params["prompt_token_ids"] = final_res.prompt_token_ids

        if reuse_prefilled_tokens and kv_transfer_params is not None:
            _add_prefilled_token_to_transfer(final_res, kv_transfer_params)

        response = await _original_chat_completion_full_generator(
            self,
            request,
            to_async_iterator(final_res),
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            *args,
            **kwargs,
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
        conversation, engine_inputs = await _original_preprocess_chat(
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

        for attr in (
            "_omni_prefilled_token_ids",
            "_omni_prefilled_text",
            "_omni_prefilled_logprobs",
            "_omni_prefilled_cumulative_logprob",
        ):
            if hasattr(request, attr):
                delattr(request, attr)

        if len(engine_inputs) != 1:
            return conversation, engine_inputs

        kv_params = getattr(request, "kv_transfer_params", None) or {}
        engine_input = engine_inputs[0]

        prompt_token_ids = kv_params.get("prompt_token_ids")
        if envs.OMNI_SKIP_DECODE_TOKENIZE and prompt_token_ids is not None:
            # Keep vLLM's rendered input intact: fields such as
            # ``multi_modal_data`` and ``multi_modal_inputs`` are still needed
            # on D to build mRoPE positions, even when image KV comes from P.
            # Only the token IDs must come from P so KV positions match exactly.
            engine_input = _replace_prompt_token_ids(
                engine_input,
                prompt_token_ids,
            )
            engine_inputs = [engine_input]

        prefilled_token_ids = kv_params.get("prefilled_token")
        if not envs.OMNI_REUSE_PREFILLED_TOKENS or not prefilled_token_ids:
            return conversation, engine_inputs

        max_tokens = getattr(request, "max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = getattr(request, "max_tokens", None)
        _validate_trial_request(request, max_tokens=max_tokens)

        tokenizer = getattr(renderer.renderer, "tokenizer", None)
        if tokenizer is None:
            kv_params.pop("prefilled_token", None)
            return conversation, engine_inputs

        prefilled_token_id = prefilled_token_ids[0]
        prefilled_text = _decode_single_token(tokenizer, prefilled_token_id)
        stop_reasons = kv_params.get("stop_reasons") or ()
        if not _can_relay_prefilled_token(
            request, tokenizer, prefilled_token_id, prefilled_text, stop_reasons,
        ):
            # The P token cannot be represented safely as a standalone decode
            # delta, so let D generate it normally.
            kv_params.pop("prefilled_token", None)
            return conversation, engine_inputs

        prefilled_logprobs = _slice_first_logprobs(
            kv_params.get("prefilled_logprobs")
        )
        prefilled_cumulative_logprob = _first_token_cumulative_logprob(
            prefilled_logprobs,
            prefilled_token_id,
            kv_params.get("prefilled_cumulative_logprob"),
        )
        engine_input["prefilled_token_ids"] = list(prefilled_token_ids)
        engine_input["prefilled_texts"] = prefilled_text
        engine_input["prefilled_logprobs"] = prefilled_logprobs
        engine_input["prefilled_cumulative_logprob"] = (
            prefilled_cumulative_logprob
        )
        request._omni_prefilled_token_ids = list(prefilled_token_ids)
        request._omni_prefilled_text = prefilled_text
        request._omni_prefilled_logprobs = prefilled_logprobs
        request._omni_prefilled_cumulative_logprob = (
            prefilled_cumulative_logprob
        )
        return conversation, engine_inputs


def _append_prefilled_token(request: Request) -> bool:
    extra_args = getattr(request.sampling_params, "extra_args", None) or {}
    kv_params = extra_args.get("kv_transfer_params")
    prefilled_token = kv_params.get("prefilled_token") if kv_params else None
    if not prefilled_token:
        return False

    request.prompt_token_ids.extend(prefilled_token)
    request.append_output_token_ids(prefilled_token)
    # Consume exactly once; the scheduler may observe the same transfer result
    # in more than one scheduling tick.
    kv_params.pop("prefilled_token", None)
    return True


@register_patch("PrefilledTokenSkipKvXferFinished", Scheduler)
class SchedulerKvXferFinishedPatch(VLLMPatch):
    _attr_names_to_apply = ["_update_from_kv_xfer_finished"]

    def _update_from_kv_xfer_finished(self, kv_connector_output) -> None:
        _original_update_from_kv_xfer_finished(self, kv_connector_output)

        if not envs.OMNI_REUSE_PREFILLED_TOKENS:
            return

        for request_id in kv_connector_output.finished_recving or ():
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            if request_id in self.failed_recving_kv_req_ids:
                continue
            _append_prefilled_token(request)


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

            if isinstance(prompt, EngineCoreRequest):
                prompt_token_ids = prompt.prompt_token_ids
                extra_args = getattr(sampling_params, "extra_args", None) or {}
                kv_transfer_params = extra_args.get("kv_transfer_params", None)
                if kv_transfer_params and "prefilled_token" in kv_transfer_params:
                    prefilled_token_ids = kv_transfer_params["prefilled_token"] or []
                    if prefilled_token_ids:
                        tokenizer = getattr(self, "tokenizer", None)
                        if tokenizer is None:
                            tokenizer = getattr(self.input_processor, "tokenizer", None)
                        if tokenizer is not None:
                            prefilled_text = _decode_single_token(
                                tokenizer,
                                prefilled_token_ids[0],
                            )

                    prefilled_logprobs = convert_to_standard_logprobs(kv_transfer_params["prefilled_logprobs"])
                    prefill_cumulative_logprob = kv_transfer_params["prefilled_cumulative_logprob"]

            elif isinstance(prompt, Mapping):
                if "prefilled_token_ids" in prompt:
                    prefilled_token_ids = prompt["prefilled_token_ids"] or []
                    prompt_token_ids = prompt.get("prompt_token_ids")
                    prefilled_text = prompt.get("prefilled_texts", "")
                    # Consume once to avoid repeated synthetic output.
                    prompt["prefilled_token_ids"] = []
                    prefilled_logprobs = convert_to_standard_logprobs(
                        prompt.get("prefilled_logprobs"))
                    prefill_cumulative_logprob = prompt.get(
                        "prefilled_cumulative_logprob")

            if prefilled_token_ids:
                _validate_trial_request(
                    sampling_params, max_tokens=getattr(sampling_params, "max_tokens", None))
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
        upstream = _original_generate(
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
        )
        try:
            async for res in upstream:
                if pending_prefilled_output is not None:
                    by_index = {
                        output.index: output
                        for output in pending_prefilled_output.outputs
                    }
                    for output in res.outputs:
                        prefilled_output = by_index.get(output.index)
                        if prefilled_output is None:
                            continue
                        _prepend_output_once(
                            output, prefilled_output.token_ids, prefilled_output.text,
                            prefilled_output.logprobs, prefilled_output.cumulative_logprob)
                yield res
        finally:
            await upstream.aclose()
