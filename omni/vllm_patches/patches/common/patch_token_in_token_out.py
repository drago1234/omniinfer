# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Render/generate extensions for vLLM 0.25.1 (omni-npu PR2230 + PR2257).

Upstream does not preserve is_embed masks over HTTP or support content_parts
and optional prompt-token metadata. This patch also isolates sampling params,
aligns output modes with the stream flag, relays KV metadata and reuses Omni's
PD expert helpers and four-field response format.

Scope: replace scale_out.factories.init_render_state, init_scale_out_state and
register_scale_out_api_routers to install local protocol/serving extensions
and per-app render/generate routes. Rendering, feature serde, engine processing
and full-response assembly reuse upstream implementations. Derender remains
native and cannot consume Omni's expert dictionary.

Remove when upstream or supported extension hooks cover these behaviors,
including Omni-compatible PD expert output, and render/generate, multimodal
mask, JSON/SSE and PD regressions pass without this patch.

Targets vLLM 0.25.1, upstream commit
752a3a504485790a2e8491cacbb35c137339ad34. Other versions require interface review
and regression validation; the 0.28.0 URL schema reference below does not imply
runtime compatibility.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated, Any, Literal, TypeAlias

import msgspec
import torch
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field, model_validator
from typing_extensions import Required, TypedDict

import vllm.entrypoints.scale_out.factories as scale_out_factories
from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartPromptEmbedsParam,
    MultiModalItemTracker,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.scale_out.derender.api_router import router as derender_router
from vllm.entrypoints.scale_out.derender.serving import ServingDerender
from vllm.entrypoints.scale_out.render.serving import ServingRender
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest as NativeGenerateRequest,
    GenerateResponse as NativeGenerateResponse,
    GenerateResponseChoice as NativeGenerateResponseChoice,
    GenerateResponseStreamChoice as NativeGenerateResponseStreamChoice,
    GenerateStreamResponse as NativeGenerateStreamResponse,
    MultiModalFeatures as NativeMultiModalFeatures,
    PlaceholderRangeInfo as NativePlaceholderRangeInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    load_aware_call,
    should_include_usage,
    validate_json_request,
    with_cancellation,
)
from vllm.inputs import EngineInput, TokensPrompt, mm_input
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.outputs import RequestOutput
from vllm.renderers.hf import (
    HfRenderer,
    _ensure_prompt_embeds_placeholder_token,
    rebuild_mm_uuids_from_mm_data,
    replace_vision_chunk_video_placeholder,
)
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components,
    extract_prompt_len,
)
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.collection_utils import as_list

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_routed_experts import (
    _attach_context_routed_experts,
    _build_routed_experts_payload,
    _extract_kv_routed_experts,
    _stash_prefill_routed_experts,
    _wrap_request_output_stream,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# 1. HTTP protocol
# ---------------------------------------------------------------------------

class URLMediaContentPart(TypedDict, total=False):
    """Generate URL protocol shared with omni-npu and upstream vLLM 0.28.0."""

    type: Required[Literal["image_url", "audio_url", "video_url"]]
    url: Required[str]
    uuid: str | None


# Generate uses root-level url/uuid fields; Chat uses nested URL objects.
# Keep the existing input_audio and prompt_embeds shapes separately.
# Text is already represented by token_ids and is not a second prompt source.
MediaContentPart: TypeAlias = Annotated[
    URLMediaContentPart
    | ChatCompletionContentPartInputAudioParam
    | ChatCompletionContentPartPromptEmbedsParam,
    Field(discriminator="type"),
]


class OmniPlaceholderRangeInfo(NativePlaceholderRangeInfo):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)
    is_embed: list[bool] | None = None

    @model_validator(mode="after")
    def validate_mask_length(self) -> OmniPlaceholderRangeInfo:
        if self.is_embed is not None and len(self.is_embed) != self.length:
            raise ValueError("is_embed length must equal placeholder.length")
        return self


class OmniMultiModalFeatures(NativeMultiModalFeatures):
    mm_placeholders: dict[str, list[OmniPlaceholderRangeInfo]]

    @model_validator(mode="after")
    def validate_item_alignment(self) -> OmniMultiModalFeatures:
        modalities = set(self.mm_hashes)
        if set(self.mm_placeholders) != modalities:
            raise ValueError("mm_hashes and mm_placeholders modalities must match")
        if self.kwargs_data is not None and set(self.kwargs_data) != modalities:
            raise ValueError("kwargs_data modalities must match mm_hashes")
        for modality, hashes in self.mm_hashes.items():
            if len(self.mm_placeholders[modality]) != len(hashes):
                raise ValueError(f"{modality}: hash/placeholder item counts differ")
            if self.kwargs_data is not None:
                if len(self.kwargs_data[modality]) != len(hashes):
                    raise ValueError(f"{modality}: hash/kwargs item counts differ")
        return self


class OmniGenerateRequest(NativeGenerateRequest):
    features: OmniMultiModalFeatures | None = None
    content_parts: list[MediaContentPart] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Raw media for token_ids which already include the chat template "
            "but whose media placeholders have not been expanded. "
            "URL parts use root-level type/url/optional uuid fields; "
            "prompt_embeds uses data."
        ),
    )
    return_token_ids: bool | None = Field(
        default=None,
        description="Return actual prompt token IDs and optional MM position summary.",
    )

    @model_validator(mode="after")
    def validate_input_exclusivity(self) -> OmniGenerateRequest:
        if self.features is not None and self.content_parts is not None:
            raise ValueError("features and content_parts are mutually exclusive")
        return self


class OmniGenerateResponseChoice(NativeGenerateResponseChoice):
    # Same shape/dtype/encoded-data dictionary as chat/completion.
    routed_experts: dict[str, Any] | None = None


class OmniGenerateResponseStreamChoice(NativeGenerateResponseStreamChoice):
    routed_experts: dict[str, Any] | None = None


class OmniGenerateStreamResponse(NativeGenerateStreamResponse):
    choices: list[OmniGenerateResponseStreamChoice]


class OmniGenerateResponse(NativeGenerateResponse):
    choices: list[OmniGenerateResponseChoice]
    prompt_token_ids: list[int] | None = None
    # This is a summary, NOT the input feature: deliberately no is_embed/kwargs.
    mm_placeholders: dict[str, list[NativePlaceholderRangeInfo]] | None = None


# ---------------------------------------------------------------------------
# 2. Input normalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedGenerateInput:
    engine_input: EngineInput
    prompt_token_ids: list[int] | None
    mm_placeholders: Mapping[str, Sequence[PlaceholderRange]]
    input_mode: str


def _normalized_input(
    engine_input: EngineInput,
    model_config: Any,
    input_mode: str,
) -> NormalizedGenerateInput:
    token_ids = extract_prompt_components(model_config, engine_input).token_ids
    return NormalizedGenerateInput(
        engine_input=engine_input,
        prompt_token_ids=list(token_ids) if token_ids is not None else None,
        mm_placeholders=engine_input.get("mm_placeholders", {}),
        input_mode=input_mode,
    )


def _validate_placeholder_bounds(
    placeholders: Mapping[str, Sequence[Any]], prompt_length: int
) -> None:
    for modality, ranges in placeholders.items():
        for position in ranges:
            if (
                position.offset < 0
                or position.length < 0
                or position.offset + position.length > prompt_length
            ):
                raise ValueError(f"{modality}: placeholder lies outside the final prompt")


def decode_native_features(
    request: OmniGenerateRequest, model_config: Any
) -> NormalizedGenerateInput:
    """Deserialize native kwargs; leave encoding, ordering and LoRA IDs to vLLM."""
    features = request.features
    if features is None:
        raise ValueError("decode_native_features requires features")

    placeholders = {
        modality: [
            PlaceholderRange(
                offset=p.offset,
                length=p.length,
                is_embed=(
                    torch.tensor(p.is_embed, dtype=torch.bool)
                    if p.is_embed is not None
                    else None
                ),
            )
            for p in ranges
        ]
        for modality, ranges in features.mm_placeholders.items()
    }
    _validate_placeholder_bounds(placeholders, len(request.token_ids))

    kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
    for modality, hashes in features.mm_hashes.items():
        items = (
            features.kwargs_data[modality]
            if features.kwargs_data is not None
            else [None] * len(hashes)
        )
        kwargs[modality] = [
            decode_mm_kwargs_item(item) if item is not None else None for item in items
        ]

    engine_input = mm_input(
        prompt_token_ids=list(request.token_ids),
        mm_kwargs=MultiModalKwargsItems(kwargs),
        mm_hashes=features.mm_hashes,
        mm_placeholders=placeholders,
        cache_salt=request.cache_salt,
    )
    # Preserve native metadata alongside the reconstructed model kwargs.
    if request.assistant_tokens_mask is not None:
        engine_input["assistant_tokens_mask"] = list(request.assistant_tokens_mask)
    return _normalized_input(engine_input, model_config, "native_features")


async def render_content_parts(
    request: OmniGenerateRequest, online_renderer: OnlineRenderer
) -> NormalizedGenerateInput:
    """Resolve media with the sync parser, then expand placeholders once."""
    if request.content_parts is None:
        raise ValueError("render_content_parts requires content_parts")
    renderer = online_renderer.renderer

    def resolve_and_process() -> EngineInput:
        # Keep this sync path: OpenPangu video/audio processing must agree with
        # render. Offload its blocking media I/O and processor work as a unit.
        tracker = MultiModalItemTracker(online_renderer.model_config)
        parser = tracker.create_parser()
        for part in request.content_parts:
            part_type = part["type"]
            if part_type == "image_url":
                parser.parse_image(part["url"], part.get("uuid"))
            elif part_type == "audio_url":
                parser.parse_audio(part["url"], part.get("uuid"))
            elif part_type == "video_url":
                parser.parse_video(part["url"], part.get("uuid"))
            elif part_type == "input_audio":
                parser.parse_input_audio(part["input_audio"])
            elif part_type == "prompt_embeds":
                if not part["data"]:
                    raise ValueError("prompt_embeds content part requires non-empty data")
                parser.parse_prompt_embeds(part["data"])
            else:
                raise ValueError(f"Unsupported content_parts type: {part_type!r}")
        mm_data, mm_uuids = tracker.resolve_items()
        if not mm_data:
            raise ValueError("content_parts did not resolve any media")

        prompt = TokensPrompt(prompt_token_ids=list(request.token_ids))
        if request.cache_salt is not None:
            prompt["cache_salt"] = request.cache_salt

        prompt_embeds = mm_data.get("prompt_embeds")
        if prompt_embeds is not None:
            if any(modality != "prompt_embeds" for modality in mm_data):
                raise ValueError(
                    "content_parts cannot mix prompt_embeds with image/audio/video"
                )
            placeholder_id = _ensure_prompt_embeds_placeholder_token(
                renderer.get_tokenizer()
            )
            # Native pure-mode conversion keeps the full tensor and the
            # prompt_is_token_ids mask; a bare mm_input would lose both.
            HfRenderer._apply_prompt_embeds_to_prompt(
                prompt, list(prompt_embeds), placeholder_id
            )
        else:
            # Mirror native HF handling for models with unified vision chunks.
            # This is media placeholder expansion, not another chat template.
            if (
                getattr(renderer, "use_unified_vision_chunk", False)
                and mm_uuids is not None
            ):
                mm_uuids = rebuild_mm_uuids_from_mm_data(mm_uuids, mm_data)
                prompt["prompt_token_ids"] = replace_vision_chunk_video_placeholder(
                    prompt["prompt_token_ids"],
                    mm_data,
                    getattr(online_renderer.model_config.hf_config, "video_placeholder", None),
                )
            prompt["multi_modal_data"] = mm_data
            if mm_uuids is not None:
                prompt["multi_modal_uuids"] = mm_uuids

        return renderer.process_for_engine(
            prompt, arrival_time=time.time(), skip_mm_cache=True
        )

    engine_input = await asyncio.to_thread(resolve_and_process)
    return _normalized_input(engine_input, online_renderer.model_config, "content_parts")


async def normalize_generate_input(
    request: OmniGenerateRequest, online_renderer: OnlineRenderer
) -> NormalizedGenerateInput:
    """Normalize tokens, native features, or raw media into a final engine input.

    Decode native features without re-expanding placeholders, process raw
    content_parts once, or reuse native preprocessing for token-only input.
    Return the final input together with its token IDs and MM placeholders.
    """
    if request.features is not None:
        return decode_native_features(request, online_renderer.model_config)
    if request.content_parts is not None:
        return await render_content_parts(request, online_renderer)
    (engine_input,) = await online_renderer.preprocess_completion(
        request,
        prompt_input=list(request.token_ids),
        prompt_embeds=None,
        skip_mm_cache=True,
    )
    return _normalized_input(engine_input, online_renderer.model_config, "tokens")


def _validate_final_input(
    normalized: NormalizedGenerateInput, online_renderer: OnlineRenderer
) -> None:
    """Admission checks use the final input; InputProcessor still owns its checks."""
    model_config = online_renderer.model_config
    prompt_length = extract_prompt_len(model_config, normalized.engine_input)
    if prompt_length == 0 or prompt_length > model_config.max_model_len:
        raise ValueError(
            f"Final prompt length {prompt_length} must be between 1 and "
            f"max_model_len ({model_config.max_model_len})"
        )
    _validate_placeholder_bounds(normalized.mm_placeholders, prompt_length)

    # Match 0.25.1 InputProcessor's token-vocabulary rule. EmbedsInput follows
    # native embedding validation, and skip_tokenizer_init remains supported.
    tokenizer = online_renderer.renderer.tokenizer
    if normalized.engine_input["type"] != "embeds" and tokenizer is not None:
        max_id = max(normalized.prompt_token_ids or (), default=0)
        vocab_max_id = max(tokenizer.max_token_id, model_config.get_vocab_size() - 1)
        if max_id > vocab_max_id:
            raise ValueError(f"Token id {max_id} is out of vocabulary")


def _collect_mm_placeholders(
    placeholders: Mapping[str, Sequence[Any]],
) -> dict[str, list[NativePlaceholderRangeInfo]] | None:
    summary = {
        modality: [
            NativePlaceholderRangeInfo(offset=p.offset, length=p.length)
            for p in sorted(ranges, key=lambda p: p.offset)
        ]
        for modality, ranges in placeholders.items()
        if ranges
    }
    return summary or None


def _response_placeholders(
    request: OmniGenerateRequest, normalized: NormalizedGenerateInput | None
) -> dict[str, list[NativePlaceholderRangeInfo]] | None:
    if normalized is not None:
        return _collect_mm_placeholders(normalized.mm_placeholders)
    # Supports direct generator calls with already-expanded native features.
    if request.features is not None:
        return _collect_mm_placeholders(request.features.mm_placeholders)
    return None


def _clone_sampling_params(request: OmniGenerateRequest) -> SamplingParams:
    sampling_params = request.sampling_params.clone()
    if request.kv_transfer_params is not None:
        # skip_clone=True shares extra_args; isolate its mapping before writing.
        sampling_params.extra_args = dict(sampling_params.extra_args or {})
        sampling_params.extra_args["kv_transfer_params"] = copy.deepcopy(
            request.kv_transfer_params
        )
    return sampling_params


def _to_omni_generate_request(request: NativeGenerateRequest) -> OmniGenerateRequest:
    # Read objects directly. Parent model_dump would drop a nested child's mask(is_embed).
    values = {name: getattr(request, name) for name in type(request).model_fields}
    result = OmniGenerateRequest(**values)
    provided = request._sampling_params_provided_keys
    result._sampling_params_provided_keys = (
        set(provided) if provided is not None else None
    )
    return result


# ---------------------------------------------------------------------------
# 3. Render service
# ---------------------------------------------------------------------------

class OmniServingRender(ServingRender):
    @staticmethod
    def _extract_mm_features(engine_input: EngineInput) -> OmniMultiModalFeatures | None:
        if engine_input.get("type") != "multimodal":
            return None
        placeholders = {
            modality: [
                OmniPlaceholderRangeInfo(
                    offset=p.offset,
                    length=p.length,
                    is_embed=p.is_embed.tolist() if p.is_embed is not None else None,
                )
                for p in ranges
            ]
            for modality, ranges in engine_input["mm_placeholders"].items()
        }
        kwargs_data = None
        if raw_kwargs := engine_input.get("mm_kwargs"):
            kwargs_data = {
                modality: [
                    encode_mm_kwargs_item(item) if item is not None else None
                    for item in items
                ]
                for modality, items in raw_kwargs.items()
            }
        return OmniMultiModalFeatures(
            mm_hashes=engine_input["mm_hashes"],
            mm_placeholders=placeholders,
            kwargs_data=kwargs_data,
        )

    async def render_chat_request(
        self, request: ChatCompletionRequest
    ) -> OmniGenerateRequest | ErrorResponse:
        result = await super().render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result
        return _to_omni_generate_request(result)

    async def render_completion_request(
        self, request: CompletionRequest
    ) -> list[OmniGenerateRequest] | ErrorResponse:
        result = await super().render_completion_request(request)
        if isinstance(result, ErrorResponse):
            return result
        return [_to_omni_generate_request(item) for item in result]


# ---------------------------------------------------------------------------
# 4. Generate service
# ---------------------------------------------------------------------------

class OmniServingTokens(ServingTokens):
    async def serve_tokens(
        self, request: OmniGenerateRequest, raw_request: Request | None = None
    ) -> OmniGenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        error = await self._check_model(request)
        if error is not None:
            logger.error("Error with model %s", error)
            return error
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
        model_name = self.models.model_name(lora_request)
        request_id = (
            f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
        )
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request is not None:
            raw_request.state.request_metadata = request_metadata

        max_num_seqs = self.engine_client.vllm_config.scheduler_config.max_num_seqs
        if request.sampling_params.n > max_num_seqs:
            return self.create_error_response(
                "sampling_params.n must be at most the server's max_num_seqs "
                f"({max_num_seqs}), got {request.sampling_params.n}."
            )
        try:
            msgspec.msgpack.encode(request.sampling_params)
            # Patch begin
            normalized = await normalize_generate_input(request, self.online_renderer)
            _validate_final_input(normalized, self.online_renderer)
            sampling_params = _clone_sampling_params(request)
            # Patch end

            if not request.is_sampling_param_provided("max_tokens"):
                sampling_params.max_tokens = get_max_tokens(
                    max_model_len=self.model_config.max_model_len,
                    max_tokens=None,
                    input_length=self._extract_prompt_len(normalized.engine_input),
                    default_sampling_params=self.default_sampling_params,
                    override_max_tokens=self.override_max_tokens,
                )
            if self.force_no_detokenize:
                sampling_params.detokenize = False
            # Match chat/completion: the HTTP stream flag selects the output mode.
            sampling_params.output_kind = (
                RequestOutputKind.DELTA
                if request.stream
                else RequestOutputKind.FINAL_ONLY
            )
            # Include forwarded KV in the serialization/admission check too.
            msgspec.msgpack.encode(sampling_params)
        except (ValueError, TypeError, OverflowError, msgspec.DecodeError) as error:
            return self.create_error_response(str(error))

        # model_copy retains the native PrivateAttr tracking explicitly supplied
        # sampling fields. Both the engine and response handlers see the clone.
        effective_request = request.model_copy(update={"sampling_params": sampling_params})
        self._log_inputs(
            request_id,
            normalized.engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )
        trace_headers = (
            await self._get_trace_headers(raw_request.headers)
            if raw_request is not None
            else None
        )
        result_generator = self.engine_client.generate(
            normalized.engine_input,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=self._get_data_parallel_rank(raw_request),
        )
        if request.stream:
            return self.serve_tokens_stream_generator(
                effective_request,
                result_generator,
                request_id,
                model_name,
                request_metadata,
                normalized_input=normalized,
            )
        return await self.serve_tokens_full_generator(
            effective_request,
            result_generator,
            request_id,
            model_name,
            request_metadata,
            normalized_input=normalized,
        )

    async def serve_tokens_full_generator(
        self,
        request: OmniGenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
        *,
        normalized_input: NormalizedGenerateInput | None = None,
    ) -> OmniGenerateResponse | ErrorResponse:
        prompt_token_ids: list[int] | None = None
        final_res: RequestOutput | None = None

        async def observe_outputs() -> AsyncGenerator[RequestOutput, None]:
            nonlocal prompt_token_ids, final_res
            async for result in result_generator:
                final_res = result
                if result.prompt_token_ids is not None:
                    prompt_token_ids = list(result.prompt_token_ids)
            if final_res is not None:
                # Populate P handoff before upstream copies KV into the response.
                _stash_prefill_routed_experts(self, final_res)
                yield final_res

        # Upstream owns tokens/logprobs/usage/KV/cancellation. Replace only the
        # expert field afterwards, using the same PD helpers as chat/completion.
        response = await super().serve_tokens_full_generator(
            request, observe_outputs(), request_id, model_name, request_metadata
        )
        if isinstance(response, ErrorResponse):
            return response
        if final_res is None:
            raise RuntimeError("No output received from engine")
        values = {name: getattr(response, name) for name in type(response).model_fields}
        kv_payload = _extract_kv_routed_experts(request.kv_transfer_params)
        outputs_by_index = {output.index: output for output in final_res.outputs}
        choices: list[OmniGenerateResponseChoice] = []
        for choice in response.choices:
            choice_values = {
                name: getattr(choice, name) for name in type(choice).model_fields
            }
            choice_values["routed_experts"] = _build_routed_experts_payload(
                self, outputs_by_index[choice.index], kv_payload
            )
            choices.append(OmniGenerateResponseChoice(**choice_values))
        values["choices"] = choices
        if request.return_token_ids:
            values["prompt_token_ids"] = prompt_token_ids
            values["mm_placeholders"] = _response_placeholders(request, normalized_input)
        return OmniGenerateResponse(**values)

    async def serve_tokens_stream_generator(
        self,
        request: OmniGenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
        *,
        normalized_input: NormalizedGenerateInput | None = None,
    ) -> AsyncGenerator[str, None]:
        # All state is local to this request, including across empty deltas.
        num_prompt_tokens = 0
        num_generated_tokens = [0] * request.sampling_params.n
        first_iteration = True
        num_cached_tokens = None
        prompt_token_ids_to_return: list[int] | None = None
        prompt_metadata_sent = False
        summary = _response_placeholders(request, normalized_input)
        include_usage, include_continuous_usage = should_include_usage(
            request.stream_options, False
        )

        try:
            async for res in _wrap_request_output_stream(self, request, result_generator):
                # Patch begin
                # Cache before skipping an empty delta. A prompt-only output is
                # useful even when it contains no completion choices at all.
                if res.prompt_token_ids is not None and prompt_token_ids_to_return is None:
                    prompt_token_ids_to_return = list(res.prompt_token_ids)
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    first_iteration = False
                # Patch end

                for output in res.outputs:
                    i = output.index
                    delta_token_ids = output.token_ids
                    num_generated_tokens[i] += len(delta_token_ids)
                    self._raise_if_error(output.finish_reason, request_id)
                    if not delta_token_ids:
                        continue

                    logprobs = None
                    if request.sampling_params.logprobs is not None:
                        if output.logprobs is None:
                            raise RuntimeError("Did not output logprobs")
                        logprobs = self._create_tokens_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=output.logprobs,
                            num_output_top_logprobs=request.sampling_params.logprobs,
                        )
                    choice = OmniGenerateResponseStreamChoice(
                        index=i,
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                        token_ids=as_list(delta_token_ids),
                    )
                    _attach_context_routed_experts(choice)
                    chunk = OmniGenerateStreamResponse(
                        request_id=request_id,
                        choices=[choice],
                    )
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=num_generated_tokens[i],
                            total_tokens=num_prompt_tokens + num_generated_tokens[i],
                        )
                    payload = chunk.model_dump(mode="json")

                    # Patch begin
                    if res.kv_transfer_params is not None:
                        payload["kv_transfer_params"] = res.kv_transfer_params
                    if (
                        request.return_token_ids
                        and not prompt_metadata_sent
                        and prompt_token_ids_to_return is not None
                    ):
                        payload["prompt_token_ids"] = prompt_token_ids_to_return
                        if summary is not None:
                            payload["mm_placeholders"] = {
                                modality: [p.model_dump() for p in ranges]
                                for modality, ranges in summary.items()
                            }
                        prompt_metadata_sent = True
                    # Patch end

                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            total_completion_tokens = sum(num_generated_tokens)
            final_usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=num_prompt_tokens + total_completion_tokens,
            )
            if self.enable_prompt_tokens_details and num_cached_tokens is not None:
                final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=num_cached_tokens
                )
            if include_usage:
                final_chunk = OmniGenerateStreamResponse(
                    request_id=request_id, choices=[], usage=final_usage
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
            request_metadata.final_usage_info = final_usage
        except GenerationError as error:
            yield (
                "data: "
                f"{self._convert_generation_error_to_streaming_response(error)}\n\n"
            )
        except Exception as error:
            logger.exception("Error in token generation stream.")
            yield f"data: {self.create_streaming_error_response(error)}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 5. Per-app HTTP routers and factory helpers
# ---------------------------------------------------------------------------

_ERROR_RESPONSES = {
    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
}


def _error_response(error: ErrorResponse) -> JSONResponse:
    return JSONResponse(content=error.model_dump(), status_code=error.error.code)


def _dump_generate_response(response: NativeGenerateResponse) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    # Patch begin
    for name in ("prompt_token_ids", "mm_placeholders"):
        if payload.get(name) is None:
            payload.pop(name, None)
    # Patch end
    return payload


def _build_render_router() -> APIRouter:
    # Patch the response_model form GenerateRequest to OmniGenerateRequest
    router = APIRouter()

    @router.post(
        "/v1/chat/completions/render",
        dependencies=[Depends(validate_json_request)],
        # Patch begin
        response_model=OmniGenerateRequest,
        # Patch end
        responses={
            **_ERROR_RESPONSES,
            HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
        },
    )
    async def render_chat_completion(request: ChatCompletionRequest, raw_request: Request):
        handler = getattr(raw_request.app.state, "serving_render", None)
        if handler is None:
            raise NotImplementedError("The model does not support Chat Completions Render API")
        result = await handler.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return _error_response(result)
        return JSONResponse(content=result.model_dump(mode="json"))

    @router.post(
        "/v1/completions/render",
        dependencies=[Depends(validate_json_request)],
        response_model=list[OmniGenerateRequest],
        responses=_ERROR_RESPONSES,
    )
    async def render_completion(request: CompletionRequest, raw_request: Request):
        handler = getattr(raw_request.app.state, "serving_render", None)
        if handler is None:
            raise NotImplementedError("The model does not support Completions Render API")
        result = await handler.render_completion_request(request)
        if isinstance(result, ErrorResponse):
            return _error_response(result)
        return JSONResponse(content=[item.model_dump(mode="json") for item in result])

    return router


def _build_generate_router(*, tokens_only: bool) -> APIRouter:
    # Patch vllm\entrypoints\scale_out\token_in_token_out\api_router.py
    # Patch the respomse_model from GenerateResponse to OmniGenerateResponse
    # Patch the request from GenerateRequest th OmniGenrateRequest
    router = APIRouter()

    @router.post(
        "/inference/v1/generate",
        dependencies=[Depends(validate_json_request)],
        # Patch begin
        response_model=OmniGenerateResponse,
        # Patch end
        responses={
            **_ERROR_RESPONSES,
            HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        },
    )
    @with_cancellation
    @load_aware_call
    async def generate(request: OmniGenerateRequest, raw_request: Request):
        handler = getattr(raw_request.app.state, "serving_tokens", None)
        if handler is None:
            raise NotImplementedError("The model does not support generate tokens API")
        result = await handler.serve_tokens(request, raw_request)
        if isinstance(result, ErrorResponse):
            return _error_response(result)
        if isinstance(result, NativeGenerateResponse):
            return JSONResponse(content=_dump_generate_response(result))
        return StreamingResponse(content=result, media_type="text/event-stream")

    if tokens_only:
        @router.post("/abort_requests")
        async def abort_requests(raw_request: Request):
            try:
                body = await raw_request.json()
            except json.JSONDecodeError as error:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail=f"JSON decode error: {error}",
                ) from error
            request_ids = body.get("request_ids")
            if request_ids is None:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST.value,
                    detail="Missing 'request_ids' in request body",
                )
            asyncio.create_task(raw_request.app.state.engine_client.abort(request_ids))
            return Response(status_code=HTTPStatus.OK.value)

    return router


def _init_render_state(state: Any, request_logger: Any) -> None:
    state.serving_render = OmniServingRender(
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
    )
    state.serving_derender = ServingDerender(
        state.openai_serving_models,
        state.online_derenderer,
        request_logger=request_logger,
    )


def _init_scale_out_state(
    state: Any, args: Any, engine_client: Any, request_logger: Any
) -> None:
    _init_render_state(state, request_logger)
    state.serving_tokens = OmniServingTokens(
        engine_client,
        state.openai_serving_models,
        state.online_renderer,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_log_outputs=args.enable_log_outputs,
        force_no_detokenize=args.tokens_only,
    )


def _register_scale_out_api_routers(
    app: FastAPI, supported_tasks: tuple[str, ...]
) -> None:
    if getattr(app.state, "_omni_token_in_token_out_routers_registered", False):
        return
    app.include_router(_build_render_router())
    app.include_router(derender_router)
    if "generate" in supported_tasks:
        app.include_router(
            _build_generate_router(
                tokens_only=getattr(getattr(app.state, "args", None), "tokens_only", False)
            )
        )
    app.state._omni_token_in_token_out_routers_registered = True


# ---------------------------------------------------------------------------
# 6. Single patch registration
# ---------------------------------------------------------------------------

@register_patch("TokenInTokenOutPatch", scale_out_factories)
class TokenInTokenOutPatch(VLLMPatch):
    _attr_names_to_apply = [
        "init_render_state",
        "init_scale_out_state",
        "register_scale_out_api_routers",
    ]

    init_render_state = _init_render_state
    init_scale_out_state = _init_scale_out_state
    register_scale_out_api_routers = _register_scale_out_api_routers
