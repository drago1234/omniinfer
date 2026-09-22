# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tests for PD token relay; APC usage cases live in test_patch_serving_apc."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.unit
PATCH_ROOT = Path(__file__).resolve().parents[5] / "omni/vllm_patches"
PD_MODULE = "omni_npu.vllm_patches.patches.common.patch_prefilled_token_skip_tokenize"
RELAY_LOGPROBS = [{"7": {"logprob": -0.2, "rank": 1, "decoded_token": " 你"}}]
RELAY_ATTRS = (
    "_omni_prefilled_token_ids", "_omni_prefilled_text",
    "_omni_prefilled_logprobs", "_omni_prefilled_cumulative_logprob",
)


async def _items(*values):
    for value in values:
        if isinstance(value, BaseException):
            raise value
        yield value


async def _collect(iterator):
    return [item async for item in iterator]


def _output(**kwargs):
    return NS(**dict(index=0, text="好", token_ids=[8], logprobs=[{8: NS(logprob=-0.3)}],
                     cumulative_logprob=-0.3, stop_reason=None, routed_experts=None) | kwargs)


def _result(**kwargs):
    return NS(**dict(outputs=[_output()], prompt_token_ids=[11, 12],
                     kv_transfer_params={}, num_cached_tokens=0) | kwargs)


@pytest.fixture
def pd(monkeypatch):
    """Stub external interfaces only; use the real patch registry and PD code."""
    modules = {}

    def module(name, **attrs):
        if name not in modules:
            parent, _, child = name.rpartition(".")
            owner = module(parent) if parent else None
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
            monkeypatch.setitem(sys.modules, name, modules[name])
            if owner is not None:
                setattr(owner, child, modules[name])
        vars(modules[name]).update(attrs)
        return modules[name]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        target = module(name)
        target.__file__, target.__package__ = str(path), name.rpartition(".")[0]
        spec.loader.exec_module(target)
        return target

    chat = type("OpenAIServingChat", (), {
        "chat_completion_full_generator": AsyncMock(),
        "chat_completion_stream_generator": _items,
    })
    module("vllm.entrypoints.openai.chat_completion.serving", OpenAIServingChat=chat)
    module("vllm.entrypoints.openai.chat_completion.protocol",
           ChatCompletionRequest=NS, ChatCompletionResponse=NS)
    module("vllm.entrypoints.openai.engine.protocol",
           ErrorResponse=type("ErrorResponse", (NS,), {}), PromptTokenUsageInfo=NS)
    module("vllm.entrypoints.openai.completion.serving",
           OpenAIServingCompletion=type("OpenAIServingCompletion", (), {
               "create_completion": AsyncMock(), "completion_stream_generator": _items,
           }))
    module("vllm.renderers.online_renderer",
           OnlineRenderer=type("OnlineRenderer", (), {"preprocess_chat": AsyncMock()}))
    module("vllm.v1.core.sched.scheduler", Scheduler=type("Scheduler", (), {
        "_update_from_kv_xfer_finished": Mock(),
    }))
    module("vllm.v1.engine.async_llm", AsyncLLM=type("AsyncLLM", (), {"generate": _items}))
    for name, attrs in {
        "inputs": dict(EngineInput=dict, PromptType=object),
        "tokenizers": dict(TokenizerLike=object),
        "v1.engine": dict(EngineCoreRequest=type("EngineCoreRequest", (NS,), {})),
        "lora.request": dict(LoRARequest=object),
        "outputs": dict(RequestOutput=NS, CompletionOutput=NS),
        "v1.request": dict(Request=object),
        "sampling_params": dict(RequestOutputKind=NS(
            DELTA="delta", CUMULATIVE="cumulative", FINAL_ONLY="final_only")),
        "logger": dict(init_logger=logging.getLogger),
    }.items():
        module("vllm." + name, **attrs)
    module("omni_npu.envs", OMNI_REUSE_PREFILLED_TOKENS=True,
           OMNI_SKIP_DECODE_TOKENIZE=True, OMNI_PD_ROLE="decode")
    manager = load("omni_npu.vllm_patches.patch_manager", PATCH_ROOT / "patch_manager.py")
    module("omni_npu.vllm_patches", PatchManager=manager.PatchManager)
    load("omni_npu.vllm_patches.core", PATCH_ROOT / "core.py")
    patch = load(PD_MODULE, PATCH_ROOT / "patches/common/patch_prefilled_token_skip_tokenize.py")
    patch.load_test_module = load
    return patch


def _preprocess(pd, request, inputs, tokenizer=None):
    conversation, parser = [{"role": "user", "content": "hello"}], object()
    renderer = NS(renderer=NS(tokenizer=tokenizer))
    pd._original_preprocess_chat.return_value = (conversation, inputs)
    args = (renderer, request, conversation, "template", "openai", {"a": 1})
    kwargs = dict(tool_dicts=[{"type": "function"}], parser=parser, skip_mm_cache=True)
    result = asyncio.run(pd.OnlineRendererPrefilledTokenAdapter.preprocess_chat(*args, **kwargs))
    pd._original_preprocess_chat.assert_awaited_once_with(*args, **kwargs)
    assert result[0] is conversation
    return result[1]


def _chat(pd, request, result, serving=None, **kwargs):
    serving = serving or NS(create_error_response=Mock(side_effect=pd.ErrorResponse))
    generator = result if hasattr(result, "__aiter__") else _items(result)
    return asyncio.run(pd.OpenAIServingChatPatch.chat_completion_full_generator(
        serving, request, generator, "req", "model", [], None, None, **kwargs))


@pytest.mark.parametrize("skip,reuse", [(False, False), (True, False), (False, True), (True, True)])
def test_preprocess_preserves_mm_fields_and_relays_only_enabled_fields(pd, skip, reuse):
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE, pd.envs.OMNI_REUSE_PREFILLED_TOKENS = skip, reuse
    kv = dict(prompt_token_ids=[101, 102], prefilled_token=[7], stop_reasons=[None],
              prefilled_logprobs=RELAY_LOGPROBS, prefilled_cumulative_logprob=-9)
    request = NS(kv_transfer_params=kv)
    metadata = {key: object() for key in (
        "multi_modal_data", "multi_modal_inputs", "mm_hashes", "mm_placeholders", "cache_salt")}
    original = dict(type="multimodal", prompt_token_ids=[11, 12], **metadata)
    tokenizer = NS(decode=Mock(return_value=" 你"), eos_token_id=99)
    result, = _preprocess(pd, request, [original], tokenizer)
    assert result["prompt_token_ids"] == ([101, 102] if skip else [11, 12])
    assert all(result[key] is value for key, value in metadata.items())
    assert original["prompt_token_ids"] == [11, 12]
    if skip:
        assert result is not original and result["prompt_token_ids"] is not kv["prompt_token_ids"]
    if reuse:
        assert result["prefilled_token_ids"] == request._omni_prefilled_token_ids == [7]
        assert result["prefilled_texts"] == request._omni_prefilled_text == " 你"
        assert result["prefilled_logprobs"] == request._omni_prefilled_logprobs == RELAY_LOGPROBS
        assert result["prefilled_cumulative_logprob"] == request._omni_prefilled_cumulative_logprob == -0.2
        tokenizer.decode.assert_called_once_with(
            [7], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    else:
        assert "prefilled_token_ids" not in result
        assert not hasattr(request, "_omni_prefilled_token_ids")
        tokenizer.decode.assert_not_called()


@pytest.mark.parametrize("count,kv", [(0, None), (2, {}), (1, None), (1, {})])
def test_preprocess_clears_stale_state_and_delegates_without_relay(pd, count, kv):
    request = NS(kv_transfer_params=kv, **dict.fromkeys(RELAY_ATTRS, "stale"))
    inputs = [{"prompt_token_ids": [i]} for i in range(count)]
    assert _preprocess(pd, request, inputs) is inputs
    assert all(not hasattr(request, attr) for attr in RELAY_ATTRS)


@pytest.mark.parametrize("text,stop,eos,stop_token_ids", [
    ("�", None, 99, None), ("ok�", None, 99, None), ("", None, 99, None),
    ("ok", 0, 99, None), ("ok", "stop", 99, None), ("ok", None, 7, None),
    ("ok", None, 99, [7]),
])
def test_preprocess_rejects_unsafe_or_finished_first_token(pd, text, stop, eos, stop_token_ids):
    request = NS(kv_transfer_params={"prefilled_token": [7], "stop_reasons": [stop]},
                 stop_token_ids=stop_token_ids)
    inputs = [{"prompt_token_ids": [11]}]
    assert _preprocess(pd, request, inputs, NS(decode=lambda *a, **kw: text, eos_token_id=eos)) is inputs
    assert "prefilled_token" not in request.kv_transfer_params
    assert all(not hasattr(request, attr) for attr in RELAY_ATTRS)


def test_preprocess_without_tokenizer_does_not_inject_first_token(pd):
    request = NS(kv_transfer_params={"prefilled_token": [7]})
    inputs = [{"prompt_token_ids": [11]}]
    assert _preprocess(pd, request, inputs) is inputs
    assert "prefilled_token" not in request.kv_transfer_params
    assert "prefilled_token_ids" not in inputs[0]


@pytest.mark.parametrize("tokenizer", [
    NS(decode=lambda ids: " 你"),
    NS(convert_ids_to_tokens=lambda token: "Ġ", convert_tokens_to_string=lambda tokens: " 你"),
])
def test_single_token_decode_supports_legacy_interfaces(pd, tokenizer):
    assert pd._decode_single_token(tokenizer, 7) == " 你"


@pytest.mark.parametrize("probs,expected", [
    (None, -9), ([], -9), ([{}], -9), ([{8: NS(logprob=-1)}], -9),
    ([{7: NS(logprob=-0.2)}], -0.2), (RELAY_LOGPROBS, -0.2),
    ([{"7": {"logprob": None}}], -9), ([{7: NS(logprob=0)}], 0),
])
def test_first_token_logprob_uses_first_step_or_fallback(pd, probs, expected):
    assert pd._first_token_cumulative_logprob(probs, 7, -9) == expected
    assert pd._slice_first_logprobs(probs) == (probs[:1] if probs else probs)


@pytest.mark.parametrize("skip,reuse", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("is_pd", [False, True])
def test_chat_full_writes_handoff_only_for_pd_and_enabled_flags(pd, skip, reuse, is_pd):
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE, pd.envs.OMNI_REUSE_PREFILLED_TOKENS = skip, reuse
    final = _result(kv_transfer_params={"existing": 1} if is_pd else None,
                    outputs=[_output(token_ids=[7, 8], logprobs=RELAY_LOGPROBS + [{8: {}}],
                                     cumulative_logprob=-0.5)])
    response = NS(choices=[NS()], usage=object())
    async def original(*args, **kwargs):
        assert await _collect(args[2]) == [final]
        assert args[3:] == ("req", "model", [], None, None)
        assert kwargs == {"parser": "parser", "mm_token_counts": {"image": 5}}
        return response
    pd._original_chat_completion_full_generator.side_effect = original
    returned = _chat(pd, NS(kv_transfer_params=None), _items(_result(), final),
                     parser="parser", mm_token_counts={"image": 5})
    assert returned is response
    pd._original_chat_completion_full_generator.assert_awaited_once()
    expected = {"existing": 1}
    if skip:
        expected["prompt_token_ids"] = [11, 12]
    if reuse:
        expected.update(prefilled_token=[7], stop_reasons=[None],
                        prefilled_logprobs=RELAY_LOGPROBS, prefilled_cumulative_logprob=-0.2)
    assert final.kv_transfer_params == (expected if is_pd else None)


@pytest.mark.parametrize("outputs", [[], [_output(token_ids=[], logprobs=None)]])
def test_chat_full_no_generated_token_does_not_add_relay(pd, outputs):
    final = _result(outputs=outputs)
    pd._original_chat_completion_full_generator.return_value = NS(choices=[])
    _chat(pd, NS(kv_transfer_params=None), final)
    assert final.kv_transfer_params == {"prompt_token_ids": [11, 12]}


@pytest.mark.parametrize("already_in_output", [False, True])
@pytest.mark.parametrize("with_logprobs", [False, True])
def test_chat_full_first_token_is_prepended_exactly_once(pd, already_in_output, with_logprobs):
    probs, cumulative = (RELAY_LOGPROBS, -0.2) if with_logprobs else (None, None)
    request = NS(kv_transfer_params=None, **dict(zip(
        RELAY_ATTRS, ([7], " 你", probs, cumulative))))
    output = _output(logprobs=[{8: NS(logprob=-0.3)}] if with_logprobs else None,
                     cumulative_logprob=-0.3 if with_logprobs else None)
    if already_in_output:
        output._omni_prefilled_merged = True
        output.token_ids, output.text = [7, 8], " 你好"
        if with_logprobs:
            output.logprobs = RELAY_LOGPROBS + output.logprobs
            output.cumulative_logprob = -0.5
    final = _result(outputs=[output], kv_transfer_params=None)
    pd._original_chat_completion_full_generator.return_value = NS(choices=[NS()], usage=object())
    _chat(pd, request, final)
    _chat(pd, request, final)
    assert output.token_ids == [7, 8] and output.text == " 你好"
    if with_logprobs:
        assert len(output.logprobs) == 2 and output.logprobs[0] == RELAY_LOGPROBS[0]
        assert output.cumulative_logprob == pytest.approx(-0.5)
    else:
        assert output.logprobs is None and output.cumulative_logprob is None


@pytest.mark.parametrize("error,message", [
    (asyncio.CancelledError(), "Client disconnected"), (ValueError("bad request"), "bad request"),
])
def test_chat_full_handles_generator_errors(pd, error, message):
    serving = NS(create_error_response=Mock(return_value="error response"))
    assert _chat(pd, NS(), _items(error), serving) == "error response"
    serving.create_error_response.assert_called_once_with(message)
    pd._original_chat_completion_full_generator.assert_not_awaited()


def test_chat_full_empty_generator_raises_runtime_error(pd):
    with pytest.raises(RuntimeError, match="at least one RequestOutput"):
        _chat(pd, NS(), _items())
    pd._original_chat_completion_full_generator.assert_not_awaited()


def test_chat_full_preserves_upstream_error(pd):
    error = pd.ErrorResponse(message="upstream error")
    pd._original_chat_completion_full_generator.return_value = error
    assert _chat(pd, NS(kv_transfer_params=None), _result()) is error


@pytest.mark.parametrize("choice,calls", [
    (None, []), (NS(function=NS(name="tool")), []),
    (NS(function=NS(name="tool")), [NS(id="call")]), ("auto", [NS(id="call")]),
])
def test_chat_full_preserves_tool_choice_parser_and_finish_reason(pd, choice, calls):
    request = NS(kv_transfer_params=None)
    if choice is not None:
        request.tool_choice = choice
    serving = NS(reasoning_parser=object())
    parser = serving.reasoning_parser
    response = NS(choices=[NS(finish_reason="stop", message=NS(tool_calls=calls))])
    async def original(self, actual_request, *args, **kwargs):
        assert self.reasoning_parser is parser and actual_request is request
        assert getattr(actual_request, "tool_choice", None) is choice
        return response
    pd._original_chat_completion_full_generator.side_effect = original
    assert _chat(pd, request, _result(), serving) is response
    assert response.choices[0].finish_reason == "stop"
    assert serving.reasoning_parser is parser


def test_chat_full_leaves_expert_outputs_to_upstream(pd):
    experts = object()
    final = _result(outputs=[_output(routed_experts=experts)], kv_transfer_params=None)
    response = NS(choices=[NS(routed_experts=experts)])
    pd._original_chat_completion_full_generator.return_value = response
    assert _chat(pd, NS(kv_transfer_params=None), final) is response
    assert final.outputs[0].routed_experts is experts
    assert response.choices[0].routed_experts is experts


def test_prefill_to_decode_preserves_prompt_ids_and_multimodal_metadata(pd):
    pd._original_chat_completion_full_generator.return_value = NS(choices=[NS()])
    prefill = _result(outputs=[_output(token_ids=[7], logprobs=RELAY_LOGPROBS)])
    _chat(pd, NS(kv_transfer_params=None), prefill)
    metadata = {"image": object()}
    request = NS(kv_transfer_params=prefill.kv_transfer_params)
    decoded, = _preprocess(pd, request, [{"prompt_token_ids": [99], "multi_modal_data": metadata}],
                           NS(decode=lambda *a, **kw: " 你", eos_token_id=99))
    assert decoded["prompt_token_ids"] == prefill.prompt_token_ids == [11, 12]
    assert decoded["multi_modal_data"] is metadata
    assert decoded["prefilled_token_ids"] == [7]
    assert request._omni_prefilled_text == " 你"


@pytest.mark.parametrize("state", ["ready", "failed", "finished", "missing", "disabled", "no_token", "no_transfer"])
def test_scheduler_consumes_first_token_once_after_original_update(pd, state):
    pd.envs.OMNI_REUSE_PREFILLED_TOKENS = state != "disabled"
    kv = {} if state == "no_token" else {"prefilled_token": [7]}
    request = NS(prompt_token_ids=[11, 12], sampling_params=NS(extra_args={"kv_transfer_params": kv}),
                 is_finished=lambda: state == "finished", append_output_token_ids=Mock())
    scheduler = NS(requests={} if state == "missing" else {"req": request},
                   failed_recving_kv_req_ids={"req"} if state == "failed" else set())
    transfer = NS(finished_recving=None if state == "no_transfer" else {"req"})
    seen = []
    pd._original_update_from_kv_xfer_finished.side_effect = lambda *a: seen.append(list(request.prompt_token_ids))
    for _ in range(2):
        pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(scheduler, transfer)
    assert seen[0] == [11, 12]  # The original scheduler must finish before the append.
    assert pd._original_update_from_kv_xfer_finished.call_args_list == [
        ((scheduler, transfer), {}), ((scheduler, transfer), {})]
    if state == "ready":
        assert request.prompt_token_ids == [11, 12, 7] and "prefilled_token" not in kv
        request.append_output_token_ids.assert_called_once_with([7])
    else:
        assert request.prompt_token_ids == [11, 12]
        request.append_output_token_ids.assert_not_called()
        assert kv == ({} if state == "no_token" else {"prefilled_token": [7]})


@pytest.mark.parametrize("extra_args", [None, {}, {"kv_transfer_params": None}])
def test_scheduler_no_relay_is_noop(pd, extra_args):
    request = NS(sampling_params=NS(extra_args=extra_args), prompt_token_ids=[11])
    assert pd._append_prefilled_token(request) is False
    assert request.prompt_token_ids == [11]


@pytest.mark.parametrize("kind", ["delta", "cumulative", "final_only"])
@pytest.mark.parametrize("engine_core", [False, True])
@pytest.mark.parametrize("with_logprobs", [False, True])
def test_generate_relays_first_token_and_forwards_0251_arguments(pd, kind, engine_core, with_logprobs):
    probs, cumulative = (RELAY_LOGPROBS, -0.2) if with_logprobs else (None, None)
    sampling = NS(n=1, max_tokens=100, output_kind=kind, extra_args={"kv_transfer_params": dict(
        prefilled_token=[7], prefilled_logprobs=probs, prefilled_cumulative_logprob=cumulative)})
    prompt = pd.EngineCoreRequest(prompt_token_ids=[11, 12]) if engine_core else dict(
        prompt_token_ids=[11, 12], prefilled_token_ids=[7], prefilled_texts=" 你",
        prefilled_logprobs=probs, prefilled_cumulative_logprob=cumulative)
    llm = NS(input_processor=NS(tokenizer=NS(decode=lambda *a, **kw: " 你")))
    final = _result(outputs=[_output(index=0)])
    pd._original_generate = Mock(return_value=_items(final))
    kwargs = dict(prompt_text="prompt", lora_request=object(), tokenization_kwargs={"a": 1},
                  trace_headers={"trace": "id"}, priority=1, data_parallel_rank=2,
                  reasoning_ended=True, reasoning_parser_kwargs={"reasoning": True})
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(llm, prompt, sampling, "req", **kwargs)))
    pd._original_generate.assert_called_once_with(llm, prompt, sampling, "req", **kwargs)
    assert results[-1] is final
    assert len(results) == (2 if kind == "delta" else 1)
    relayed = results[0]
    if kind == "delta":
        assert relayed.prompt_token_ids == [11, 12] and relayed.request_id == "req"
        assert relayed.finished is False
        assert [out.token_ids for out in final.outputs] == [[8]]
    for i, output in enumerate(relayed.outputs):
        assert output.index == i
        assert output.token_ids == ([7] if kind == "delta" else [7, 8])
        assert output.text == (" 你" if kind == "delta" else " 你好")
        expected_cumulative = cumulative if kind == "delta" else (-0.5 if with_logprobs else -0.3)
        assert output.cumulative_logprob == expected_cumulative
        if with_logprobs:
            assert output.logprobs[0][7].logprob == -0.2
            assert output.logprobs[0][7].rank == 1 and output.logprobs[0][7].decoded_token == " 你"
        elif kind == "delta":
            assert output.logprobs is None
    if not engine_core:
        assert prompt["prefilled_token_ids"] == []


@pytest.mark.parametrize("mode", ["disabled", "missing", "empty", "core_no_tokenizer"])
def test_generate_passthrough_or_missing_tokenizer(pd, mode):
    pd.envs.OMNI_REUSE_PREFILLED_TOKENS = mode != "disabled"
    prompt = {"prompt_token_ids": [11], "prefilled_token_ids": [] if mode == "empty" else [7]}
    if mode == "missing":
        prompt.pop("prefilled_token_ids")
    if mode == "core_no_tokenizer":
        prompt = pd.EngineCoreRequest(prompt_token_ids=[11])
    sampling = NS(n=1, output_kind="delta", extra_args={"kv_transfer_params": {
        "prefilled_token": [7], "prefilled_logprobs": None, "prefilled_cumulative_logprob": None}})
    final = _result()
    pd._original_generate = Mock(return_value=_items(final))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(
        NS(input_processor=NS(tokenizer=None)), prompt, sampling, "req")))
    assert results[-1] is final and final.outputs[0].token_ids == [8]
    if mode == "core_no_tokenizer":
        assert len(results) == 2 and results[0].outputs[0].text == ""
    else:
        assert results == [final]
    if mode == "disabled":
        assert prompt["prefilled_token_ids"] == [7]


def test_generate_cumulative_preserves_legitimate_repeated_tokens(pd):
    prompt = dict(prefilled_token_ids=[7], prefilled_texts=" 你")
    first = _result(outputs=[
        _output(token_ids=[7, 8], text=" 你好"), _output(index=4)])
    second = _result(outputs=[_output(token_ids=[7, 8, 9], text=" 你好呀")])
    pd._original_generate = Mock(return_value=_items(first, second))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(
        NS(), prompt, NS(n=1, output_kind="cumulative"), "req")))
    assert results == [first, second]
    assert [out.token_ids for out in first.outputs] == [[7, 7, 8], [8]]
    assert first.outputs[0].text == " 你 你好"
    assert second.outputs[0].token_ids == [7, 7, 8, 9]


def test_apc_registry_relays_to_the_single_active_pd_implementation(pd):
    apc = pd.load_test_module(
        "omni_npu.vllm_patches.patches.common.patch_serving_apc",
        PATCH_ROOT / "patches/common/patch_serving_apc.py")
    assert apc._orig_chat_full is pd.OpenAIServingChatPatch.__dict__["chat_completion_full_generator"]
    assert [name for name in sys.modules if name.endswith(".patch_prefilled_token_skip_tokenize")] == [PD_MODULE]
    manager = sys.modules["omni_npu.vllm_patches.patch_manager"].PatchManager
    assert manager.registered_patches["PrefilledTokenSkipOpenAIServingChat"] is apc.OpenAIServingChatFullAPCPatch


@pytest.mark.parametrize("fields,max_tokens", [
    ({"n": 2}, 100), ({}, 1), ({"stop": ["END"]}, 100),
    ({"tools": ["tool"]}, 100), ({"structured_outputs": {"json": {}}}, 100),
    ({"use_beam_search": True}, 100),
])
def test_trial_rejects_out_of_scope_requests_before_emitting(pd, fields, max_tokens):
    with pytest.raises(ValueError, match="PD token relay trial"):
        pd._validate_trial_request(NS(**fields), max_tokens=max_tokens)


@pytest.mark.parametrize("fields,max_tokens", [
    ({}, None),
    ({"n": None, "stop": None, "tools": None, "structured_outputs": None,
      "use_beam_search": None}, 2),
    ({"n": 1, "stop": [], "tools": [], "structured_outputs": {},
      "use_beam_search": False}, 100),
])
def test_trial_accepts_unset_or_empty_optional_features(pd, fields, max_tokens):
    pd._validate_trial_request(NS(**fields), max_tokens=max_tokens)


def test_first_delta_is_available_before_decode_engine_output(pd):
    prompt = dict(prompt_token_ids=[11, 12], prefilled_token_ids=[7], prefilled_texts=" 你")
    sampling = NS(n=1, max_tokens=100, output_kind="delta")
    final = _result(outputs=[_output(token_ids=[7, 8], text=" 你好")])
    pd._original_generate = Mock(return_value=_items(final))

    async def exercise():
        generator = pd.AsyncLLMPatch.generate(NS(), prompt, sampling, "req")
        first = await anext(generator)
        pd._original_generate.assert_not_called()
        assert first.outputs[0].token_ids == [7]
        remaining = await _collect(generator)
        assert remaining == [final]
        # A genuine second token equal to the first is not removed.
        assert first.outputs[0].token_ids + final.outputs[0].token_ids == [7, 7, 8]

    asyncio.run(exercise())


def test_close_relay_closes_upstream_generator(pd):
    closed = []

    async def upstream(*args, **kwargs):
        try:
            yield _result()
        finally:
            closed.append(True)

    pd._original_generate = upstream

    async def exercise():
        generator = pd.AsyncLLMPatch.generate(
            NS(), dict(prefilled_token_ids=[7], prefilled_texts="first"),
            NS(n=1, max_tokens=100, output_kind="delta"), "req")
        await anext(generator)
        await anext(generator)
        await generator.aclose()
        assert closed == [True]

    asyncio.run(exercise())


@pytest.mark.parametrize("received_times", [1, 2])
def test_scheduler_token_accounting_matches_0251_request_contract(pd, received_times):
    # vLLM Request keeps independent prompt, all-token and output-token lists.
    prompt_ids = list(range(266))
    all_ids, output_ids = prompt_ids.copy(), []

    def append(ids):
        all_ids.extend(ids)
        output_ids.extend(ids)

    request = NS(prompt_token_ids=prompt_ids, num_prompt_tokens=266,
                 num_computed_tokens=266, max_tokens=100,
                 sampling_params=NS(extra_args={"kv_transfer_params": {"prefilled_token": [7]}}),
                 is_finished=lambda: False, append_output_token_ids=append)
    scheduler = NS(requests={"req": request}, failed_recving_kv_req_ids=set())
    for _ in range(received_times):
        pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(
            scheduler, NS(finished_recving={"req"}))
    assert all_ids == prompt_ids == list(range(266)) + [7]
    assert output_ids == [7]
    assert request.num_prompt_tokens == request.num_computed_tokens == 266
    assert len(all_ids) - request.num_computed_tokens == 1
    assert request.max_tokens - len(output_ids) == 99


def test_generate_uses_0251_tokenizer_property(pd):
    prompt = pd.EngineCoreRequest(prompt_token_ids=[11])
    sampling = NS(n=1, max_tokens=100, output_kind="delta", extra_args={"kv_transfer_params": {
        "prefilled_token": [7], "prefilled_logprobs": None, "prefilled_cumulative_logprob": None}})
    tokenizer = NS(decode=Mock(return_value="first"))
    pd._original_generate = Mock(return_value=_items(_result()))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(
        NS(tokenizer=tokenizer), prompt, sampling, "req")))
    assert results[0].outputs[0].text == "first"


@pytest.mark.parametrize("piggyback", [False, True])
def test_input_ids_patch_keeps_pd_adapter_in_live_relay(pd, monkeypatch, piggyback):
    helpers = pd.load_test_module("trial_chain_helpers", Path(__file__).with_name(
        "test_input_ids_chain_grammar.py"))
    helpers._install_chat_stubs(monkeypatch)
    pd.envs.OMNI_PIGGYBACK_INPUT_IDS = piggyback
    pd.envs.OMNI_VALIDATE_PIGGYBACK_INPUT_IDS = False
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE = not piggyback
    pig = pd.load_test_module("omni_npu.vllm_patches.patches.common.patch_input_ids_piggyback",
                             PATCH_ROOT / "patches/common/patch_input_ids_piggyback.py")
    request = pig.ChatCompletionRequest()
    request.n, request.max_tokens = 1, 100
    request.input_ids = [99]
    request.kv_transfer_params = {"prompt_token_ids": [11], "prefilled_token": [7]}
    pd._original_preprocess_chat.return_value = ([], [{"prompt_token_ids": [11], "mm_hashes": "keep"}])
    _, inputs = asyncio.run(pig.InputIdsPiggybackPatch.preprocess_chat(
        NS(renderer=NS(tokenizer=NS(decode=lambda *a, **kw: " 你", eos_token_id=99))),
        request, [{"role": "user", "content": "hello"}], None, "auto", None))
    assert inputs[0]["prefilled_token_ids"] == [7]
    assert inputs[0]["mm_hashes"] == "keep"
    pd._original_preprocess_chat.assert_awaited_once()
