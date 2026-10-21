# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Generate HTTP, media and expert-helper integration contracts.

Protocols, HTTP validation and generate response construction are real. Media/model
execution and shared expert helpers are doubled; expert codecs, PD aggregation and
OutputProcessor algorithms belong to their own tests.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
from fastapi import FastAPI
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def token_patch():
    # Import without applying the scale-out factory patch or starting an engine.
    path = (
        Path(__file__).resolve().parents[5]
        / "omni/vllm_patches/patches/common/patch_token_in_token_out.py"
    )
    name = "test_generate_contract_module"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    modules_before = set(sys.modules)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
        # Remove the expert module only if this fixture introduced it.
        experts_name = "omni_npu.vllm_patches.patches.common.patch_routed_experts"
        if experts_name not in modules_before:
            sys.modules.pop(experts_name, None)


@pytest.fixture
def serving_factory(token_patch, monkeypatch):
    """Isolate shared expert algorithms; test only generate's integration."""
    async def passthrough(serving, request, results):
        async for result in results:
            yield result

    monkeypatch.setattr(token_patch, "_stash_prefill_routed_experts", Mock())
    monkeypatch.setattr(token_patch, "_extract_kv_routed_experts", Mock(return_value=None))
    monkeypatch.setattr(token_patch, "_build_routed_experts_payload", Mock(return_value=None))
    monkeypatch.setattr(token_patch, "_wrap_request_output_stream", passthrough)
    monkeypatch.setattr(token_patch, "_attach_context_routed_experts", Mock())

    def create():
        serving = object.__new__(token_patch.OmniServingTokens)
        serving.engine_client = NS(
            vllm_config=NS(
                kv_transfer_config=NS(
                    is_kv_transfer_instance=True,
                    kv_role="kv_consumer",
                ),
            )
        )
        serving.enable_prompt_tokens_details = True
        serving.enable_log_outputs = False
        serving.request_logger = None
        return serving

    return create


def _expert_payload(label):
    # An opaque helper result. It is deliberately not an encoded expert array:
    # generate must forward it without invoking an expert codec in this UT.
    return {
        "routed_experts_shape": [1, 1, 1],
        "routed_experts_dtype": "uint16",
        "routed_experts_str_len": len(label),
        "routed_experts_str": label,
    }


def _output(tokens=(7,), index=0, finish=None):
    return NS(
        index=index,
        token_ids=list(tokens),
        routed_experts=None,
        finish_reason=finish,
        logprobs=None,
    )


def _result(outputs, kv=None, prompt=(1, 2, 3)):
    return NS(
        outputs=outputs,
        kv_transfer_params=kv,
        prompt_token_ids=list(prompt) if prompt is not None else None,
        encoder_prompt_token_ids=None,
        prompt_logprobs=None,
        num_cached_tokens=1,
    )


async def _items(*results):
    for result in results:
        await asyncio.sleep(0)
        yield result


def _request(module, *, kv=None, stream=False, n=1, return_ids=False):
    return module.OmniGenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params={
            "max_tokens": 3,
            "n": n,
            "output_kind": (
                module.RequestOutputKind.DELTA
                if stream
                else module.RequestOutputKind.CUMULATIVE
            ),
        },
        kv_transfer_params=kv,
        stream=stream,
        stream_options={"include_usage": True} if stream else None,
        return_token_ids=return_ids,
    )


def _full(module, serving, request, *results):
    metadata = NS(final_usage_info=None)
    response = asyncio.run(
        serving.serve_tokens_full_generator(
            request, _items(*results), "generate-pd", "model", metadata
        )
    )
    return response, metadata


async def _stream(serving, request, *results):
    metadata = NS(final_usage_info=None)
    events = [
        event
        async for event in serving.serve_tokens_stream_generator(
            request, _items(*results), "generate-pd", "model", metadata
        )
    ]
    assert events[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert all("error" not in payload for payload in payloads)
    return payloads, metadata


@pytest.fixture
def pipeline(token_patch, monkeypatch):
    parser = Mock()
    tracker = Mock()
    tracker.create_parser.return_value = parser
    tracker.resolve_items.return_value = ({"image": ["loaded-media"]}, {"image": [None]})
    monkeypatch.setattr(
        token_patch, "MultiModalItemTracker", Mock(return_value=tracker)
    )
    renderer = Mock()
    renderer.use_unified_vision_chunk = False
    renderer.process_for_engine.return_value = {
        "type": "multimodal", "prompt_token_ids": [1, 99, 99, 2],
        "mm_placeholders": {"image": [NS(offset=1, length=2)]},
    }
    monkeypatch.setattr(
        token_patch, "HfRenderer", NS(_apply_prompt_embeds_to_prompt=Mock())
    )
    monkeypatch.setattr(
        token_patch, "_ensure_prompt_embeds_placeholder_token", Mock(return_value=99)
    )
    return NS(
        parser=parser, tracker=tracker, renderer=renderer,
        online=NS(model_config=NS(hf_config=NS(), is_encoder_decoder=False), renderer=renderer),
    )


def _app(module, pipeline):
    from fastapi.exceptions import RequestValidationError
    from vllm.entrypoints.serve.utils.server_utils import validation_exception_handler

    class Handler:
        async def serve_tokens(self, request, raw_request):
            self.request = request
            self.normalized = await module.render_content_parts(request, pipeline.online)
            return module.NativeGenerateResponse(request_id="media-http", choices=[])

    app = FastAPI()
    app.state.args = NS(log_error_stack=False)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.state.serving_tokens = Handler()
    app.include_router(module._build_generate_router(tokens_only=False))
    return app


def _post(app, parts, *, content_type="application/json", **fields):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/inference/v1/generate",
                content=json.dumps({
                    "token_ids": [1, 99, 2], "sampling_params": {},
                    "content_parts": parts, **fields,
                }),
                headers={"Content-Type": content_type} if content_type is not None else {},
            )
    return asyncio.run(send())


@pytest.mark.parametrize("media_type", ["image_url", "audio_url", "video_url"])
@pytest.mark.parametrize("uuid_fields", [{}, {"uuid": None}, {"uuid": "client-media-1"}])
def test_root_url_http_dispatch_and_uuid(token_patch, pipeline, media_type, uuid_fields):
    app = _app(token_patch, pipeline)
    part = {"type": media_type, "url": "https://example.com/media", **uuid_fields}
    response = _post(app, [part], cache_salt="cache-1")

    assert response.status_code == 200, response.text
    assert app.state.serving_tokens.request.content_parts == [part]
    parser_method = getattr(pipeline.parser, f"parse_{media_type.removesuffix('_url')}")
    parser_method.assert_called_once_with(part["url"], uuid_fields.get("uuid"))
    assert len(pipeline.parser.method_calls) == 1
    pipeline.tracker.resolve_items.assert_called_once_with()
    pipeline.renderer.process_for_engine.assert_called_once()
    args, kwargs = pipeline.renderer.process_for_engine.call_args
    assert args[0]["prompt_token_ids"] == [1, 99, 2]
    assert args[0]["cache_salt"] == "cache-1"
    assert args[0]["multi_modal_data"] == pipeline.tracker.resolve_items.return_value[0]
    assert args[0]["multi_modal_uuids"] == pipeline.tracker.resolve_items.return_value[1]
    assert kwargs["skip_mm_cache"] is True
    assert kwargs["arrival_time"] > 0
    assert app.state.serving_tokens.normalized.prompt_token_ids == [1, 99, 99, 2]
    assert app.state.serving_tokens.request.token_ids == [1, 99, 2]


@pytest.mark.parametrize("parts", [
    [],
    [{"type": "image_url"}],
    [{"type": "image_url", "image_url": {"url": "https://example.com/image"}}],
    [{"type": "audio_url", "audio_url": {"url": "https://example.com/audio"}}],
    [{"type": "video_url", "video_url": {"url": "https://example.com/video"}}],
    [{"type": "image_url", "url": 123}],
    [{"type": "image_url", "url": None, "uuid": "cache-only"}],
    [{"type": "image_url", "url": "https://example.com/image", "uuid": 123}],
    [{"type": "unsupported", "url": "https://example.com/media"}],
])
def test_invalid_url_parts_fail_before_media_io(token_patch, pipeline, parts):
    response = _post(_app(token_patch, pipeline), parts)
    assert response.status_code == 400, response.text
    token_patch.MultiModalItemTracker.assert_not_called()
    pipeline.renderer.process_for_engine.assert_not_called()


def test_features_remain_exclusive_with_content_parts(token_patch, pipeline):
    response = _post(
        _app(token_patch, pipeline),
        [{"type": "image_url", "url": "https://example.com/image"}],
        features={"mm_hashes": {}, "mm_placeholders": {}},
    )
    assert response.status_code == 400
    token_patch.MultiModalItemTracker.assert_not_called()


def test_openapi_documents_root_url_and_optional_uuid(token_patch, pipeline):
    schemas = _app(token_patch, pipeline).openapi()["components"]["schemas"]
    media = schemas["URLMediaContentPart"]
    assert set(media["required"]) == {"type", "url"}
    assert set(media["properties"]) == {"type", "url", "uuid"}
    assert media["properties"]["url"]["type"] == "string"
    assert set(media["properties"]["type"]["enum"]) == {
        "image_url", "audio_url", "video_url",
    }
    content = schemas["OmniGenerateRequest"]["properties"]["content_parts"]
    array = next(option for option in content["anyOf"] if option.get("type") == "array")
    assert array["minItems"] == 1
    mapping = array["items"]["discriminator"]["mapping"]
    assert mapping["image_url"] == mapping["audio_url"] == mapping["video_url"]


def test_input_audio_keeps_existing_shape_and_dispatch(token_patch, pipeline):
    audio = {"data": "AA==", "format": "wav"}
    response = _post(_app(token_patch, pipeline), [{"type": "input_audio", "input_audio": audio}])
    assert response.status_code == 200, response.text
    pipeline.parser.parse_input_audio.assert_called_once_with(audio)
    assert len(pipeline.parser.method_calls) == 1


def test_prompt_embeds_keeps_data_dispatch(token_patch, pipeline):
    embeds = object()
    pipeline.tracker.resolve_items.return_value = ({"prompt_embeds": [embeds]}, None)
    pipeline.renderer.process_for_engine.return_value = {
        "type": "embeds", "prompt_token_ids": [1, 99, 99, 2], "prompt_embeds": embeds,
    }
    response = _post(
        _app(token_patch, pipeline), [{"type": "prompt_embeds", "data": "encoded-tensor"}],
    )
    assert response.status_code == 200, response.text
    pipeline.parser.parse_prompt_embeds.assert_called_once_with("encoded-tensor")
    assert len(pipeline.parser.method_calls) == 1
    token_patch.HfRenderer._apply_prompt_embeds_to_prompt.assert_called_once()
    assert token_patch.HfRenderer._apply_prompt_embeds_to_prompt.call_args.args[1] == [embeds]
    pipeline.renderer.process_for_engine.assert_called_once()


def test_empty_prompt_embeds_data_fails_before_parser(token_patch, pipeline):
    request = token_patch.OmniGenerateRequest(
        token_ids=[1, 99, 2], sampling_params={},
        content_parts=[{"type": "prompt_embeds", "data": ""}],
    )
    with pytest.raises(ValueError, match="non-empty data"):
        asyncio.run(token_patch.render_content_parts(request, pipeline.online))
    pipeline.parser.parse_prompt_embeds.assert_not_called()
    pipeline.tracker.resolve_items.assert_not_called()
    pipeline.renderer.process_for_engine.assert_not_called()


def test_prompt_embeds_and_url_media_still_cannot_mix(token_patch, pipeline):
    pipeline.tracker.resolve_items.return_value = {
        "prompt_embeds": [object()], "image": ["image"],
    }, {"image": [None]}
    request = token_patch.OmniGenerateRequest(
        token_ids=[1, 99, 2], sampling_params={}, content_parts=[
            {"type": "prompt_embeds", "data": "encoded-tensor"},
            {"type": "image_url", "url": "https://example.com/image"},
        ],
    )
    with pytest.raises(ValueError, match="cannot mix"):
        asyncio.run(token_patch.render_content_parts(request, pipeline.online))
    pipeline.renderer.process_for_engine.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_missing_experts_keeps_null(token_patch, serving_factory, stream):
    serving = serving_factory()
    request = _request(token_patch, stream=stream)
    result = _result([_output()])
    if stream:
        payloads, _ = asyncio.run(_stream(serving, request, result))
        choice = payloads[0]["choices"][0]
    else:
        response, _ = _full(token_patch, serving, request, result)
        choice = token_patch._dump_generate_response(response)["choices"][0]
    assert choice["routed_experts"] is None


def test_full_error_and_cancellation_preserved(token_patch, serving_factory, monkeypatch):
    serving = serving_factory()
    request = _request(token_patch)

    async def cancelled():
        raise asyncio.CancelledError
        yield  # pragma: no cover

    response = asyncio.run(
        serving.serve_tokens_full_generator(
            request, cancelled(), "request", "model", NS(final_usage_info=None)
        )
    )
    assert isinstance(response, token_patch.ErrorResponse)
    error = response

    async def failure(*args, **kwargs):
        return error

    monkeypatch.setattr(token_patch.ServingTokens, "serve_tokens_full_generator", failure)
    response, _ = _full(token_patch, serving, request)
    assert response is error
    token_patch._extract_kv_routed_experts.assert_not_called()
    token_patch._build_routed_experts_payload.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_generate_preserves_choice_fields_with_experts(
    token_patch, serving_factory, monkeypatch, stream,
):
    serving = serving_factory()
    payload = _expert_payload("choice-fields")
    token_patch._build_routed_experts_payload.return_value = payload
    token_patch._attach_context_routed_experts.side_effect = (
        lambda choice: setattr(choice, "routed_experts", payload)
    )
    request = _request(token_patch, stream=stream)
    request.sampling_params.logprobs = 1
    output = _output(finish="stop")
    output.logprobs = [{7: NS(logprob=-0.25)}]
    if stream:
        payloads, _ = asyncio.run(_stream(serving, request, _result([output])))
        choice = payloads[0]["choices"][0]
    else:
        response, _ = _full(token_patch, serving, request, _result([output]))
        choice = token_patch._dump_generate_response(response)["choices"][0]
    assert choice["index"] == 0
    assert choice["token_ids"] == [7]
    assert choice["finish_reason"] == "stop"
    assert choice["logprobs"]["content"][0]["token"] == "token_id:7"
    assert choice["logprobs"]["content"][0]["logprob"] == -0.25
    assert choice["routed_experts"] == payload


def test_stream_generation_error_preserved(token_patch, serving_factory):
    serving = serving_factory()
    payload = _expert_payload("before-error")
    token_patch._attach_context_routed_experts.side_effect = (
        lambda choice: setattr(choice, "routed_experts", payload)
    )

    async def failure():
        yield _result([_output()])
        raise token_patch.GenerationError("test engine failure")

    async def collect():
        return [
            event
            async for event in serving.serve_tokens_stream_generator(
                _request(token_patch, stream=True), failure(),
                "request", "model", NS(final_usage_info=None),
            )
        ]

    events = asyncio.run(collect())
    assert events[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert payloads[0]["choices"][0]["routed_experts"] == payload
    assert "error" in payloads[-1]


def test_response_models_and_openapi_use_expert_dict(token_patch):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(token_patch._build_generate_router(tokens_only=False))
    schema = app.openapi()
    choice_schema = schema["components"]["schemas"]["OmniGenerateResponseChoice"]
    field = choice_schema["properties"]["routed_experts"]
    assert {item["type"] for item in field["anyOf"]} == {"object", "null"}
    response_schema = schema["components"]["schemas"]["OmniGenerateResponse"]
    assert response_schema["properties"]["choices"]["items"]["$ref"].endswith(
        "/OmniGenerateResponseChoice"
    )
    stream_schema = token_patch.OmniGenerateStreamResponse.model_json_schema()
    stream_choice = stream_schema["$defs"]["OmniGenerateResponseStreamChoice"]
    assert {item["type"] for item in stream_choice["properties"]["routed_experts"]["anyOf"]} == {
        "object", "null"
    }


def _prepare_entry(serving, module, monkeypatch, generate):
    """Exercise the real HTTP serving entry with input rendering isolated."""
    serving.engine_client.errored = False
    serving.engine_client.vllm_config.scheduler_config = NS(max_num_seqs=4)
    serving.engine_client.generate = generate
    serving._check_model = AsyncMock(return_value=None)
    serving._maybe_get_adapters = Mock(return_value=None)
    serving._base_request_id = lambda raw, request_id: request_id
    serving._get_data_parallel_rank = lambda raw: None
    serving._log_inputs = Mock()
    serving.models = NS(model_name=lambda lora: "model")
    serving.model_config = NS(max_model_len=32)
    serving.online_renderer = NS()
    serving.force_no_detokenize = True
    normalized = NS(
        engine_input={"type": "token", "prompt_token_ids": [1, 2]},
        prompt_token_ids=[1, 2], mm_placeholders={},
    )
    monkeypatch.setattr(module, "normalize_generate_input", AsyncMock(return_value=normalized))
    monkeypatch.setattr(module, "_validate_final_input", Mock())
    # Temporary evidence hooks are outside this business regression's scope.
    if hasattr(module, "activate_api"):
        monkeypatch.setattr(module, "activate_api", lambda serving: None)
    return normalized


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("requested_kind", [None, 0, 1, 2])
@pytest.mark.parametrize("shared_params", [False, True])
def test_entry_output_mode_matches_chat_and_completion(
    token_patch, serving_factory, monkeypatch, stream, requested_kind,
    shared_params,
):
    params = {"max_tokens": 3, "extra_args": {"caller": {"value": [1]}}}
    if requested_kind is not None:
        params["output_kind"] = requested_kind
    if shared_params:
        params = token_patch.SamplingParams(**params, skip_clone=True)
    request = token_patch.OmniGenerateRequest(
        token_ids=[1, 2], sampling_params=params, stream=stream,
        kv_transfer_params={"remote_request_id": "p-core"},
        stream_options={"include_usage": True} if stream else None,
    )
    original = request.sampling_params
    original_kind = original.output_kind
    seen = {}
    serving = serving_factory()

    def generate(prompt, sampling_params, request_id, **kwargs):
        seen["params"] = sampling_params
        return _items(_result([_output(tokens=(7, 8, 9), finish="length")], prompt=(1, 2)))

    _prepare_entry(serving, token_patch, monkeypatch, generate)
    full_handler = serving.serve_tokens_full_generator
    stream_handler = serving.serve_tokens_stream_generator

    async def observe_full(effective_request, *args, **kwargs):
        seen["request"] = effective_request
        return await full_handler(effective_request, *args, **kwargs)

    def observe_stream(effective_request, *args, **kwargs):
        seen["request"] = effective_request
        return stream_handler(effective_request, *args, **kwargs)

    serving.serve_tokens_full_generator = observe_full
    serving.serve_tokens_stream_generator = observe_stream

    async def run():
        response = await serving.serve_tokens(request)
        return [event async for event in response] if stream else response

    response = asyncio.run(run())
    chat = token_patch.ChatCompletionRequest(
        model="model", messages=[{"role": "user", "content": "x"}], stream=stream,
    )
    completion = token_patch.CompletionRequest(model="model", prompt=[1, 2], stream=stream)
    for reference in (chat, completion):
        expected = reference.to_sampling_params(max_tokens=3, default_sampling_params={})
        assert seen["params"].output_kind == expected.output_kind
    assert seen["request"].sampling_params is seen["params"]
    assert seen["params"] is not original
    assert seen["params"].detokenize is False
    assert request.sampling_params is original
    assert original.output_kind == original_kind
    assert original.detokenize is True
    assert original.extra_args == {"caller": {"value": [1]}}
    assert seen["params"].extra_args["kv_transfer_params"] == request.kv_transfer_params
    assert seen["params"].extra_args["kv_transfer_params"] is not request.kv_transfer_params
    if stream:
        assert response[-1] == "data: [DONE]\n\n"
    else:
        assert response.choices[0].token_ids == [7, 8, 9]
        assert response.usage.total_tokens == 5


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_http_rejects_non_json_before_media_io(token_patch, pipeline, content_type):
    app = _app(token_patch, pipeline)
    response = _post(app, [{"type": "image_url", "url": "https://example.com/image"}],
                     content_type=content_type)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "Bad Request"
    assert response.json()["error"]["code"] == 400
    assert "Unsupported Media Type: Only 'application/json' is allowed" in response.text
    assert not hasattr(app.state.serving_tokens, "request")
    token_patch.MultiModalItemTracker.assert_not_called()
    pipeline.renderer.process_for_engine.assert_not_called()


def test_http_accepts_json_media_type_case_and_charset(token_patch, pipeline):
    response = _post(_app(token_patch, pipeline),
                     [{"type": "image_url", "url": "https://example.com/image"}],
                     content_type="Application/JSON; charset=utf-8")
    assert response.status_code == 200, response.text
    pipeline.parser.parse_image.assert_called_once_with("https://example.com/image", None)


@pytest.mark.parametrize("return_ids", [False, True])
def test_generate_full_experts_hook_wiring(
    token_patch, serving_factory, monkeypatch, return_ids,
):
    serving = serving_factory()
    request_kv = {"remote_request_id": "p-core"}
    request = _request(token_patch, kv=request_kv, n=2, return_ids=return_ids)
    partial = _result([_output(tokens=(99,))])
    outputs = [
        _output(tokens=(7, 8), index=0, finish="stop"),
        _output(tokens=(9,), index=1, finish="length"),
    ]
    final = _result(outputs, kv={"remote_request_id": "final-core"})
    handoff = _expert_payload("handoff")
    choice_payloads = {i: _expert_payload(f"choice-{i}") for i in range(2)}
    extracted = object()

    def stash(current_serving, result):
        assert current_serving is serving
        assert result is final
        result.kv_transfer_params.update(handoff)

    token_patch._stash_prefill_routed_experts.side_effect = stash
    token_patch._extract_kv_routed_experts.return_value = extracted

    def build(current_serving, output, kv_payload):
        assert current_serving is serving
        assert kv_payload is extracted
        return choice_payloads[output.index]

    token_patch._build_routed_experts_payload.side_effect = build
    original = token_patch.ServingTokens.serve_tokens_full_generator

    async def reverse_choices(current_serving, *args, **kwargs):
        response = await original(current_serving, *args, **kwargs)
        response.choices.reverse()
        return response

    monkeypatch.setattr(
        token_patch.ServingTokens, "serve_tokens_full_generator", reverse_choices,
    )
    response, metadata = _full(token_patch, serving, request, partial, final)
    wire = token_patch._dump_generate_response(response)
    token_patch._stash_prefill_routed_experts.assert_called_once_with(serving, final)
    token_patch._extract_kv_routed_experts.assert_called_once_with(request.kv_transfer_params)
    calls = token_patch._build_routed_experts_payload.call_args_list
    assert len(calls) == 2
    assert calls[0].args[1] is outputs[1]
    assert calls[1].args[1] is outputs[0]
    assert [choice["index"] for choice in wire["choices"]] == [1, 0]
    for choice in wire["choices"]:
        assert choice["routed_experts"] == choice_payloads[choice["index"]]
        assert choice["token_ids"] == ([7, 8] if choice["index"] == 0 else [9])
    assert wire["kv_transfer_params"] == {"remote_request_id": "final-core", **handoff}
    assert request.kv_transfer_params == {"remote_request_id": "p-core"}
    assert ("prompt_token_ids" in wire) is return_ids
    if return_ids:
        assert wire["prompt_token_ids"] == [1, 2, 3]
    assert wire["usage"]["completion_tokens"] == 3
    assert metadata.final_usage_info == response.usage
    token_patch.OmniGenerateResponse.model_validate(wire)


@pytest.mark.parametrize("return_ids", [False, True])
def test_generate_stream_experts_hook_wiring(
    token_patch, serving_factory, monkeypatch, return_ids,
):
    serving = serving_factory()
    request = _request(token_patch, stream=True, n=2, return_ids=return_ids)
    kv = {"remote_request_id": "p-core", **_expert_payload("handoff")}
    result = _result([
        _output(tokens=(7,), index=0, finish="stop"),
        _output(tokens=(8, 9), index=1, finish="length"),
    ], kv=kv)
    seen_results = []

    async def wrap(current_serving, current_request, upstream):
        assert current_serving is serving
        assert current_request is request
        async for current in upstream:
            seen_results.append(current)
            yield current

    wrapper = Mock(side_effect=wrap)
    monkeypatch.setattr(token_patch, "_wrap_request_output_stream", wrapper)
    choice_payloads = {i: _expert_payload(f"stream-{i}") for i in range(2)}

    def attach(choice):
        assert isinstance(choice, token_patch.OmniGenerateResponseStreamChoice)
        choice.routed_experts = choice_payloads[choice.index]

    token_patch._attach_context_routed_experts.side_effect = attach
    payloads, metadata = asyncio.run(_stream(serving, request, result))
    wrapper.assert_called_once()
    assert seen_results == [result]
    chunks = [payload for payload in payloads if payload["choices"]]
    assert [payload["choices"][0]["index"] for payload in chunks] == [0, 1]
    attached = token_patch._attach_context_routed_experts.call_args_list
    assert [call.args[0].index for call in attached] == [0, 1]
    for payload in chunks:
        choice = payload["choices"][0]
        assert choice["routed_experts"] == choice_payloads[choice["index"]]
        assert payload["kv_transfer_params"] == kv
        assert choice["token_ids"] == ([7] if choice["index"] == 0 else [8, 9])
    assert sum("prompt_token_ids" in payload for payload in payloads) == int(return_ids)
    if return_ids:
        assert chunks[0]["prompt_token_ids"] == [1, 2, 3]
    assert payloads[-1]["choices"] == []
    assert payloads[-1]["usage"]["completion_tokens"] == 3
    assert metadata.final_usage_info.completion_tokens == 3
    for payload in payloads:
        token_patch.OmniGenerateStreamResponse.model_validate(payload)
