# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tests for PD token relay; APC usage cases live in test_patch_serving_apc."""

import asyncio
import copy
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
                     kv_transfer_params={"remote_decode": True}, num_cached_tokens=0) | kwargs)


def _tokenizer(text=" 你", eos=99):
    return NS(convert_ids_to_tokens=Mock(return_value="piece"),
              convert_tokens_to_string=Mock(return_value=text), eos_token_id=eos,
              decode=Mock(side_effect=AssertionError("Must use the legacy conversion API")))


def _handoff(**kwargs):
    return dict(prefilled_token=[7], stop_reasons=[None], prefilled_logprobs=None,
                prefilled_cumulative_logprob=None) | kwargs


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
           ChatCompletionRequest=NS, ChatCompletionResponse=NS,
           ChatCompletionNamedToolChoiceParam=type("NamedToolChoice", (NS,), {}))
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
        "update_from_output": Mock(return_value={}),
        "_make_cached_request_data": Mock(),
        "add_request": Mock(),
    }))
    module("vllm.v1.core.sched.utils", check_stop=Mock(return_value=False))
    module("vllm.v1.engine.async_llm", AsyncLLM=type("AsyncLLM", (), {"generate": _items}))
    for name, attrs in {
        "inputs": dict(EngineInput=dict, PromptType=object),
        "tokenizers": dict(TokenizerLike=object),
        "v1.engine": dict(EngineCoreRequest=type("EngineCoreRequest", (NS,), {}),
                          EngineCoreOutput=NS,
                          EngineCoreOutputs=lambda **kw: NS(**(dict(outputs=[]) | kw))),
        "lora.request": dict(LoRARequest=object),
        "outputs": dict(RequestOutput=NS, CompletionOutput=NS),
        "v1.request": dict(Request=object, RequestStatus=NS(
            WAITING_FOR_REMOTE_KVS="waiting_for_remote_kvs",
            get_finished_reason=lambda status: status.removeprefix("finished_"))),
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
def test_preprocess_preserves_rendered_ids_and_mm_fields(pd, skip, reuse):
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE, pd.envs.OMNI_REUSE_PREFILLED_TOKENS = skip, reuse
    kv = dict(prompt_token_ids=[101, 102], prefilled_token=[7], stop_reasons=[None],
              prefilled_logprobs=RELAY_LOGPROBS, prefilled_cumulative_logprob=-9)
    request = NS(kv_transfer_params=kv)
    metadata = {key: object() for key in (
        "multi_modal_data", "multi_modal_inputs", "mm_hashes", "mm_placeholders", "cache_salt")}
    original = dict(type="multimodal", prompt_token_ids=[11, 12], **metadata)
    tokenizer = _tokenizer()
    result, = _preprocess(pd, request, [original], tokenizer)
    # The old chat wrapper bypasses the separately registered base-serving patch.
    assert result["prompt_token_ids"] == [11, 12]
    assert all(result[key] is value for key, value in metadata.items())
    assert original["prompt_token_ids"] == [11, 12]
    assert result is original and result["prompt_token_ids"] is not kv["prompt_token_ids"]
    if reuse:
        assert result["prefilled_token_ids"] == request._omni_prefilled_token_ids == [7]
        assert result["prefilled_texts"] == request._omni_prefilled_text == " 你"
        assert result["prefilled_logprobs"] == request._omni_prefilled_logprobs == RELAY_LOGPROBS
        assert result["prefilled_cumulative_logprob"] == request._omni_prefilled_cumulative_logprob == -0.2
        tokenizer.convert_ids_to_tokens.assert_called_once_with(7)
        tokenizer.convert_tokens_to_string.assert_called_once_with(["piece"])
        assert result["prefilled_token_ids"] is kv["prefilled_token"]
    else:
        assert "prefilled_token_ids" not in result
        assert not hasattr(request, "_omni_prefilled_token_ids")
        tokenizer.convert_ids_to_tokens.assert_not_called()
    tokenizer.decode.assert_not_called()


@pytest.mark.parametrize("count,kv", [(1, None), (1, {})])
def test_preprocess_clears_stale_state_and_delegates_without_relay(pd, count, kv):
    request = NS(kv_transfer_params=kv, **dict.fromkeys(RELAY_ATTRS, "stale"))
    inputs = [{"prompt_token_ids": [i]} for i in range(count)]
    assert _preprocess(pd, request, inputs) == inputs
    assert all(not hasattr(request, attr) for attr in RELAY_ATTRS)


@pytest.mark.parametrize("text,stop,eos,stop_token_ids", [
    ("�", None, 99, None), ("ok�", None, 99, None),
    ("ok", 0, 99, None), ("ok", "stop", 99, None), ("ok", None, 7, None),
])
def test_preprocess_rejects_unsafe_or_finished_first_token(pd, text, stop, eos, stop_token_ids):
    request = NS(kv_transfer_params={"prefilled_token": [7], "stop_reasons": [stop]},
                 stop_token_ids=stop_token_ids)
    inputs = [{"prompt_token_ids": [11]}]
    assert _preprocess(pd, request, inputs, _tokenizer(text, eos)) == inputs
    assert "prefilled_token" not in request.kv_transfer_params
    assert all(not hasattr(request, attr) for attr in RELAY_ATTRS)


def test_preprocess_without_tokenizer_keeps_legacy_failure(pd):
    request = NS(kv_transfer_params={"prefilled_token": [7]})
    inputs = [{"prompt_token_ids": [11]}]
    with pytest.raises(AttributeError):
        _preprocess(pd, request, inputs)
    assert request.kv_transfer_params["prefilled_token"] == [7]
    assert "prefilled_token_ids" not in inputs[0]


@pytest.mark.parametrize("text", [" 你", "<special>"])
def test_preprocess_uses_legacy_token_conversion_not_decode(pd, text):
    tokenizer = _tokenizer(text)
    request = NS(kv_transfer_params=_handoff())
    prompt, = _preprocess(pd, request, [dict(prompt_token_ids=[11])], tokenizer)
    assert prompt["prefilled_texts"] == text
    tokenizer.convert_ids_to_tokens.assert_called_once_with(7)
    tokenizer.convert_tokens_to_string.assert_called_once_with(["piece"])
    tokenizer.decode.assert_not_called()


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
def test_chat_full_empty_handoff_does_not_add_relay(pd, outputs):
    final = _result(outputs=outputs, kv_transfer_params={})
    pd._original_chat_completion_full_generator.return_value = NS(choices=[])
    _chat(pd, NS(kv_transfer_params=None), final)
    assert final.kv_transfer_params == {}


@pytest.mark.parametrize("already_in_output", [False, True])
@pytest.mark.parametrize("with_logprobs", [False, True])
def test_chat_full_first_token_is_prepended_exactly_once(pd, already_in_output, with_logprobs):
    probs, cumulative = (RELAY_LOGPROBS, -0.2) if with_logprobs else (None, None)
    request = NS(kv_transfer_params=None, **dict(zip(
        RELAY_ATTRS, ([7], " 你", probs, cumulative))))
    output = _output(logprobs=[{8: NS(logprob=-0.3)}] if with_logprobs else None,
                     cumulative_logprob=-0.3 if with_logprobs else None)
    if already_in_output:
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


def test_prefill_to_decode_keeps_legacy_rendered_ids_and_multimodal_metadata(pd):
    pd._original_chat_completion_full_generator.return_value = NS(choices=[NS()])
    prefill = _result(outputs=[_output(token_ids=[7], logprobs=RELAY_LOGPROBS)])
    _chat(pd, NS(kv_transfer_params=None), prefill)
    metadata = {"image": object()}
    request = NS(kv_transfer_params=prefill.kv_transfer_params)
    decoded, = _preprocess(pd, request, [{"prompt_token_ids": [99], "multi_modal_data": metadata}],
                           _tokenizer())
    assert decoded["prompt_token_ids"] == [99]
    assert prefill.kv_transfer_params["prompt_token_ids"] == prefill.prompt_token_ids == [11, 12]
    assert decoded["multi_modal_data"] is metadata
    assert decoded["prefilled_token_ids"] == [7]
    assert request._omni_prefilled_text == " 你"


@pytest.mark.parametrize("state", ["ready", "failed", "finished", "missing", "disabled", "no_token", "no_transfer"])
def test_scheduler_consumes_first_token_once_after_original_update(pd, state):
    pd.envs.OMNI_REUSE_PREFILLED_TOKENS = state != "disabled"
    kv = {} if state == "no_token" else {"prefilled_token": [7]}
    request = NS(prompt_token_ids=[11, 12], sampling_params=NS(extra_args={"kv_transfer_params": kv}),
                 is_finished=lambda: state == "finished", append_output_token_ids=Mock(),
                 status="waiting_for_remote_kvs")
    scheduler = NS(requests={} if state == "missing" else {"req": request},
                   failed_recving_kv_req_ids={"req"} if state == "failed" else set(),
                   max_model_len=1024)
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
        assert request._omni_prefilled_token_count == 1
    else:
        assert request.prompt_token_ids == [11, 12]
        request.append_output_token_ids.assert_not_called()
        assert not hasattr(request, "_omni_prefilled_token_count")
        assert kv == ({} if state == "no_token" else {"prefilled_token": [7]})


@pytest.mark.parametrize("extra_args", [None, {}, {"kv_transfer_params": None}])
def test_scheduler_no_relay_is_noop(pd, extra_args):
    request = NS(sampling_params=NS(extra_args=extra_args), prompt_token_ids=[11])
    assert pd._append_prefilled_token(request) is False
    assert request.prompt_token_ids == [11]


class _RelayRequest:
    """Independent prompt/output lists, as in the v0.25 Request contract."""

    def __init__(self, request_id, client_index=0, max_tokens=1):
        self.request_id, self.client_index = request_id, client_index
        self.prompt_token_ids = [11, 12]
        self.output_token_ids = []
        self.all_token_ids = self.prompt_token_ids.copy()
        self.num_prompt_tokens = self.num_computed_tokens = 2
        self.num_output_placeholders = 0
        self.max_tokens = max_tokens
        self.sampling_params = NS(extra_args={"kv_transfer_params": {"prefilled_token": [7]}})
        self.status = "waiting_for_remote_kvs"
        self.stop_reason = None
        self.trace_headers = {"trace-id": request_id}
        self.take_events = Mock(return_value=["queued"])
        self.take_prefill_stats = Mock(return_value=NS(num_cached_tokens=2))

    @property
    def num_tokens(self):
        return len(self.all_token_ids)

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)

    def append_output_token_ids(self, ids):
        self.output_token_ids.extend(ids)
        self.all_token_ids.extend(ids)

    def is_finished(self):
        return self.status.startswith("finished_")

    def get_finished_reason(self):
        return self.status.removeprefix("finished_") if self.is_finished() else None


def _relay_scheduler(*requests):
    scheduler = NS(requests={r.request_id: r for r in requests}, max_model_len=1024,
                   finished_recving_kv_req_ids=set(), failed_recving_kv_req_ids=set(),
                   skipped_waiting=list(requests), freed=[], finished_by_client={})

    def finish(request_id, status):
        request = scheduler.requests[request_id]
        # v0.25 skips already-finished requests and delays freeing if the receive
        # completion set was cleared early. Both would leak this waiting request.
        assert not request.is_finished()
        assert request.status == "waiting_for_remote_kvs"
        assert request_id in scheduler.finished_recving_kv_req_ids
        scheduler.finished_recving_kv_req_ids.remove(request_id)
        scheduler.skipped_waiting.remove(request)
        request.status = status
        scheduler.freed.append(request_id)
        scheduler.finished_by_client.setdefault(request.client_index, set()).add(request_id)
        del scheduler.requests[request_id]

    scheduler.finish_requests = Mock(side_effect=finish)
    return scheduler


@pytest.mark.parametrize("termination", ["budget", "context", "stop"])
def test_relay_finish_cleans_up_and_notifies_in_same_tick(pd, termination):
    request = _RelayRequest("req", client_index=3, max_tokens=1 if termination == "budget" else 100)
    scheduler = _relay_scheduler(request)
    if termination == "context":
        scheduler.max_model_len = 3
    transfer = NS(finished_recving={"req"})
    pd._original_update_from_kv_xfer_finished.side_effect = (
        lambda s, kv: s.finished_recving_kv_req_ids.update(kv.finished_recving))

    def check_stop(req, max_model_len):
        assert req.output_token_ids == [7] and req.num_output_tokens == 1
        if termination == "stop":
            req.stop_reason, req.status = 7, "finished_stop"
        else:
            assert req.num_output_tokens >= req.max_tokens or req.num_tokens >= max_model_len
            req.status = "finished_length"
        return True

    pd.check_stop.side_effect = check_stop
    stats, native_output = object(), object()
    batch = pd.EngineCoreOutputs(outputs=[native_output], scheduler_stats=stats, timestamp=123)

    def update(s, sched_output, runner_output):
        assert sched_output == "schedule" and runner_output is transfer
        pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(s, runner_output)
        # The native scheduler batches finished IDs after processing KV results.
        batch.finished_requests = s.finished_by_client[3]
        return {3: batch}

    pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.side_effect = update
    outputs = pd.SchedulerKvXferFinishedPatch.update_from_output(scheduler, "schedule", transfer)
    assert outputs[3] is batch and batch.outputs[0] is native_output
    assert batch.scheduler_stats is stats and batch.timestamp == 123
    assert batch.finished_requests == {"req"}
    notification = batch.outputs[1]
    assert notification.request_id == "req" and notification.new_token_ids == []
    assert notification.finish_reason == ("stop" if termination == "stop" else "length")
    assert notification.stop_reason == (7 if termination == "stop" else None)
    assert notification.events == ["queued"] and notification.trace_headers is request.trace_headers
    assert notification.prefill_stats is request.take_prefill_stats.return_value
    request.take_events.assert_called_once()
    request.take_prefill_stats.assert_called_once()
    assert scheduler.freed == ["req"] and scheduler.requests == {}
    assert scheduler.skipped_waiting == [] and scheduler.finished_recving_kv_req_ids == set()
    assert scheduler._omni_pending_finish_outputs == []
    assert "prefilled_token" not in request.sampling_params.extra_args["kv_transfer_params"]
    scheduler.finish_requests.assert_called_once_with("req", "finished_" + notification.finish_reason)

    # A later tick must not replay the terminal output.
    pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.side_effect = None
    pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.return_value = {}
    assert pd.SchedulerKvXferFinishedPatch.update_from_output(scheduler, None, None) == {}


def test_relay_finishes_route_to_each_client_without_stopping_longer_request(pd):
    short0, short2 = _RelayRequest("short0"), _RelayRequest("short2", client_index=2)
    longer = _RelayRequest("long", max_tokens=100)
    scheduler = _relay_scheduler(short0, short2, longer)

    def check_stop(request, max_model_len):
        if request.num_output_tokens >= request.max_tokens:
            request.status = "finished_length"
            return True
        return False

    pd.check_stop.side_effect = check_stop
    transfer = NS(finished_recving={"short0", "short2", "long"})
    pd._original_update_from_kv_xfer_finished.side_effect = (
        lambda s, kv: s.finished_recving_kv_req_ids.update(kv.finished_recving))
    pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(scheduler, transfer)
    outputs = pd.SchedulerKvXferFinishedPatch.update_from_output(scheduler, None, None)
    assert set(outputs) == {0, 2}
    assert [out.request_id for out in outputs[0].outputs] == ["short0"]
    assert [out.request_id for out in outputs[2].outputs] == ["short2"]
    assert scheduler.requests == {"long": longer}
    assert scheduler.skipped_waiting == [longer]
    assert scheduler.finished_recving_kv_req_ids == {"long"}
    assert longer.output_token_ids == [7] and longer.num_computed_tokens == 2


@pytest.mark.parametrize("pending", [None, []])
def test_scheduler_output_wrapper_preserves_native_result_without_pending_finish(pd, pending):
    scheduler = NS()
    if pending is not None:
        scheduler._omni_pending_finish_outputs = pending
    expected = {4: pd.EngineCoreOutputs(outputs=[object()], finished_requests={"native"})}
    pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.return_value = expected
    assert pd.SchedulerKvXferFinishedPatch.update_from_output(scheduler, "s", "r") is expected
    pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.assert_called_once_with(scheduler, "s", "r")


def test_scheduler_patch_registers_receive_and_finish_output_together(pd):
    pd.SchedulerKvXferFinishedPatch.apply()
    assert pd.Scheduler._update_from_kv_xfer_finished is pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished
    assert pd.Scheduler.update_from_output is pd.SchedulerKvXferFinishedPatch.update_from_output
    assert pd.Scheduler._make_cached_request_data is pd.SchedulerKvXferFinishedPatch._make_cached_request_data


def _cached_request_data(pd, scheduler, running, resumed=()):
    """Model the native scheduler's output-count and token-ID wire contract."""
    requests = [*running, *resumed]
    native_data = NS(
        req_ids=[r.request_id for r in requests],
        resumed_req_ids={r.request_id for r in resumed},
        num_output_tokens=[r.num_output_tokens + r.num_output_placeholders for r in requests],
        all_token_ids={r.request_id: r.all_token_ids.copy() for r in requests},
        num_computed_tokens=[r.num_computed_tokens for r in requests],
        new_token_ids=[], new_block_ids=[None] * len(requests),
    )
    before = copy.deepcopy(native_data)
    pd._original_make_cached_request_data.return_value = native_data
    counts, drafts, blocks = {}, {}, {}
    result = pd.SchedulerKvXferFinishedPatch._make_cached_request_data(
        scheduler, running, resumed, counts, drafts, blocks)
    pd._original_make_cached_request_data.assert_called_with(
        scheduler, running, resumed, counts, drafts, blocks)
    assert result is native_data
    for field in ("req_ids", "resumed_req_ids", "all_token_ids", "num_computed_tokens",
                  "new_token_ids", "new_block_ids"):
        assert getattr(result, field) == getattr(before, field)
    return result, before


@pytest.mark.parametrize("max_tokens", range(2, 11))
@pytest.mark.parametrize("skip", [False, True])
def test_boundary_reentry_does_not_count_relay_in_prompt_and_output(pd, max_tokens, skip):
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE = skip
    request = _RelayRequest("boundary", max_tokens=max_tokens)
    model_len = 524288
    prompt_len = model_len - max_tokens
    request.prompt_token_ids = [11] * prompt_len
    request.all_token_ids = request.prompt_token_ids.copy()
    request.num_prompt_tokens = request.num_computed_tokens = prompt_len
    scheduler = _relay_scheduler(request)
    scheduler.max_model_len = model_len
    scheduler.use_v2_model_runner = False
    pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(
        scheduler, NS(finished_recving={request.request_id}))
    # Just before the final output, pause and re-enter the MRv1 persistent batch.
    decoded = list(range(20, 20 + max_tokens - 2))
    request.append_output_token_ids(decoded)
    request.num_computed_tokens = request.num_tokens - 1
    data, native = _cached_request_data(pd, scheduler, [request])

    # MRv1 restores the last num_output_tokens IDs when an async request re-enters.
    def restore(count):
        return data.all_token_ids[request.request_id][-count:] if count else []

    worker_prompt_len = len(request.prompt_token_ids)
    bad_output = restore(native.num_output_tokens[0])
    assert bad_output == [7, *decoded]
    assert worker_prompt_len + len(bad_output) == model_len
    good_output = restore(data.num_output_tokens[0])
    assert good_output == decoded
    worker_num_tokens = worker_prompt_len + len(good_output)
    assert worker_num_tokens == request.num_tokens == model_len - 1
    # The original over-count made seq_len < worker_num_tokens discard this sample.
    seq_len = request.num_computed_tokens + 1
    assert seq_len == worker_num_tokens
    assert seq_len < worker_prompt_len + len(bad_output)
    request.append_output_token_ids([99])
    assert request.num_output_tokens == request.max_tokens == max_tokens
    assert request.num_tokens == model_len
    assert scheduler.max_model_len == model_len
    scheduler.finish_requests.assert_not_called()


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("placeholders", [0, 1, 4])
def test_cached_counts_preserve_placeholders_and_scheduler_budget(pd, resumed, placeholders):
    relay = _RelayRequest("relay", max_tokens=100)
    plain = _RelayRequest("plain", max_tokens=100)
    pd._append_prefilled_token(relay)
    relay.append_output_token_ids([8, 9])
    plain.append_output_token_ids([20, 21])
    relay.num_output_placeholders = plain.num_output_placeholders = placeholders
    scheduler = _relay_scheduler(relay, plain)
    scheduler.use_v2_model_runner = False
    running, paused = ([plain], [relay]) if resumed else ([relay, plain], [])
    data, _ = _cached_request_data(pd, scheduler, running, paused)
    assert data.num_output_tokens == [2 + placeholders, 2 + placeholders]
    assert relay.output_token_ids == [7, 8, 9] and relay.num_output_tokens == 3
    assert plain.output_token_ids == [20, 21]
    assert relay.max_tokens == plain.max_tokens == 100
    # For a continuously running MRv1 request this also gives the rollback count.
    if placeholders == 0:
        optimistic_worker_output = [8, 9, -1]
        del optimistic_worker_output[data.num_output_tokens[data.req_ids.index("relay")]:]
        assert optimistic_worker_output == [8, 9]
    again, _ = _cached_request_data(pd, scheduler, running, paused)
    assert again.num_output_tokens == data.num_output_tokens


@pytest.mark.parametrize("reuse,has_token,v2", [
    (False, True, False), (True, False, False), (True, True, True),
])
def test_cached_counts_leave_non_relay_and_v2_paths_unchanged(pd, reuse, has_token, v2):
    pd.envs.OMNI_REUSE_PREFILLED_TOKENS = reuse
    request = _RelayRequest("req", max_tokens=10)
    if not has_token:
        request.sampling_params.extra_args["kv_transfer_params"].clear()
    scheduler = _relay_scheduler(request)
    scheduler.use_v2_model_runner = v2
    pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(
        scheduler, NS(finished_recving={"req"}))
    request.append_output_token_ids([8, 9])
    data, native = _cached_request_data(pd, scheduler, [request])
    assert vars(data) == vars(native)


def test_empty_cached_batch_is_unchanged(pd):
    scheduler = _relay_scheduler()
    scheduler.use_v2_model_runner = False
    data, native = _cached_request_data(pd, scheduler, [])
    assert vars(data) == vars(native)


def test_cached_count_does_not_reapply_offset_after_session_prompt_reset(pd):
    request = _RelayRequest("session", max_tokens=10)
    pd._append_prefilled_token(request)
    # The native streaming-session update folds prior outputs into the prompt
    # and resets num_prompt_tokens before scheduling the next input chunk.
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request.output_token_ids.clear()
    request.append_output_token_ids([8])
    scheduler = _relay_scheduler(request)
    scheduler.use_v2_model_runner = False
    data, native = _cached_request_data(pd, scheduler, [request])
    assert data.num_output_tokens == native.num_output_tokens == [1]


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
    llm = NS(tokenizer=_tokenizer())
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
        NS(tokenizer=None), prompt, sampling, "req")))
    assert results[-1] is final and final.outputs[0].token_ids == [8]
    if mode == "core_no_tokenizer":
        assert len(results) == 2 and results[0].outputs[0].text == ""
    else:
        assert results == [final]
    if mode == "disabled":
        assert prompt["prefilled_token_ids"] == [7]


def test_generate_keeps_legacy_cumulative_merge_rule(pd):
    prompt = dict(prefilled_token_ids=[7], prefilled_texts=" 你")
    first = _result(outputs=[
        _output(token_ids=[7, 8], text=" 你好"), _output(index=4)])
    second = _result(outputs=[_output(token_ids=[7, 8, 9], text=" 你好呀")])
    pd._original_generate = Mock(return_value=_items(first, second))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(
        NS(), prompt, NS(n=1, output_kind="cumulative"), "req")))
    assert results == [first, second]
    # Characterize the old rule; changing equal-token dedup is a separate fix.
    assert [out.token_ids for out in first.outputs] == [[7, 8], [8]]
    assert first.outputs[0].text == " 你好"
    assert second.outputs[0].token_ids == [7, 8, 9]


def test_apc_registry_relays_to_the_single_active_pd_implementation(pd):
    apc = pd.load_test_module(
        "omni_npu.vllm_patches.patches.common.patch_serving_apc",
        PATCH_ROOT / "patches/common/patch_serving_apc.py")
    assert apc._orig_chat_full is pd.OpenAIServingChatPatch.__dict__["chat_completion_full_generator"]
    assert [name for name in sys.modules if name.endswith(".patch_prefilled_token_skip_tokenize")] == [PD_MODULE]
    manager = sys.modules["omni_npu.vllm_patches.patch_manager"].PatchManager
    assert manager.registered_patches["PrefilledTokenSkipOpenAIServingChat"] is apc.OpenAIServingChatFullAPCPatch


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


def test_generate_preserves_native_failure(pd):
    error = ValueError("native generation failed")
    pd._original_generate = Mock(return_value=_items(error))
    with pytest.raises(ValueError) as exc:
        asyncio.run(_collect(pd.AsyncLLMPatch.generate(
            NS(), dict(prefilled_token_ids=[7], prefilled_texts="first"),
            NS(n=1, max_tokens=100, output_kind="delta"), "req")))
    assert exc.value is error


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
                 is_finished=lambda: False, append_output_token_ids=append,
                 status="waiting_for_remote_kvs")
    scheduler = NS(requests={"req": request}, failed_recving_kv_req_ids=set(), max_model_len=1024)
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
    tokenizer = _tokenizer("first")
    pd._original_generate = Mock(return_value=_items(_result()))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(
        NS(tokenizer=tokenizer), prompt, sampling, "req")))
    assert results[0].outputs[0].text == "first"


@pytest.mark.parametrize("piggyback", [False, True])
@pytest.mark.parametrize("has_multimodal", [False, True])
@pytest.mark.parametrize("has_caller_ids", [False, True])
def test_input_ids_patch_keeps_legacy_path_priority(
        pd, monkeypatch, piggyback, has_multimodal, has_caller_ids):
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
    request.input_ids = [99] if has_caller_ids else None
    request.kv_transfer_params = _handoff(prompt_token_ids=[11])
    pd._original_preprocess_chat.return_value = ([], [{"prompt_token_ids": [11], "mm_hashes": "keep"}])
    content = [{"type": "image_url"}] if has_multimodal else "hello"
    tokenizer = _tokenizer()
    _, inputs = asyncio.run(pig.InputIdsPiggybackPatch.preprocess_chat(
        NS(renderer=NS(tokenizer=tokenizer), model_config=object()),
        request, [{"role": "user", "content": content}], None, "auto", None))
    if piggyback and has_caller_ids and not has_multimodal:
        # The old text-only fast path takes priority even with relay metadata.
        assert inputs == [{"type": "token", "prompt_token_ids": [99]}]
        assert not hasattr(request, "_omni_prefilled_token_ids")
        assert request.kv_transfer_params["prefilled_token"] == [7]
        pd._original_preprocess_chat.assert_not_awaited()
        # New serving forwards this bare EngineInput, not an EngineCoreRequest.
        # The downstream relay must still consume P's first token from extra_args.
        sampling = NS(n=1, max_tokens=2, output_kind="delta",
                      extra_args={"kv_transfer_params": request.kv_transfer_params})
        pd._original_generate = Mock(return_value=_items(_result()))
        outputs = asyncio.run(_collect(pd.AsyncLLMPatch.generate(
            NS(tokenizer=tokenizer), inputs[0], sampling, "req")))
        assert [token for result in outputs for token in result.outputs[0].token_ids] == [7, 8]
        return
    assert inputs[0]["prompt_token_ids"] == [11]
    assert inputs[0]["prefilled_token_ids"] == [7]
    assert inputs[0]["mm_hashes"] == "keep"
    pd._original_preprocess_chat.assert_awaited_once()


@pytest.mark.parametrize("mode", ["safe", "utf8", "stop", "eos", "disabled"])
@pytest.mark.parametrize("ids_match", [False, True])
def test_input_ids_validation_runs_legacy_relay_side_effects(pd, monkeypatch, mode, ids_match):
    helpers = pd.load_test_module("trial_chain_helpers", Path(__file__).with_name(
        "test_input_ids_chain_grammar.py"))
    helpers._install_chat_stubs(monkeypatch)
    pd.envs.OMNI_PIGGYBACK_INPUT_IDS = True
    pd.envs.OMNI_VALIDATE_PIGGYBACK_INPUT_IDS = True
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE = False
    pd.envs.OMNI_REUSE_PREFILLED_TOKENS = mode != "disabled"
    pig = pd.load_test_module("omni_npu.vllm_patches.patches.common.patch_input_ids_piggyback",
                             PATCH_ROOT / "patches/common/patch_input_ids_piggyback.py")
    request = pig.ChatCompletionRequest()
    request.n, request.max_tokens = 1, 100
    request.input_ids = [11]
    request.kv_transfer_params = _handoff(
        prompt_token_ids=[77] if ids_match else [11],
        stop_reasons=[7] if mode == "stop" else [None])
    for attr in RELAY_ATTRS:
        setattr(request, attr, "stale")
    tokenizer = _tokenizer(text="�" if mode == "utf8" else "first",
                           eos=7 if mode == "eos" else 99)
    # decode is used only for the existing mismatch diagnostic, not token relay.
    tokenizer.decode = Mock(side_effect=lambda ids: str(ids[0]))
    native_ids = [11] if ids_match else [99]
    pd._original_preprocess_chat.return_value = ([], [{"prompt_token_ids": native_ids}])

    call = pig.InputIdsPiggybackPatch.preprocess_chat(
        NS(renderer=NS(tokenizer=tokenizer), model_config=object()),
        request, [{"role": "user", "content": "hello"}], None, "auto", None)

    # Compare native-rendered IDs, not KV IDs; the latter must not hide a mismatch.
    if ids_match:
        _, inputs = asyncio.run(call)
        assert inputs == [{"type": "token", "prompt_token_ids": [11]}]
        tokenizer.decode.assert_not_called()
    else:
        with pytest.raises(ValueError, match="Input IDs verification failed"):
            asyncio.run(call)
    pd._original_preprocess_chat.assert_awaited_once()
    if mode == "safe":
        assert request._omni_prefilled_token_ids == [7]
        assert request._omni_prefilled_text == "first"
        assert request.kv_transfer_params["prefilled_token"] == [7]
    else:
        assert all(not hasattr(request, attr) for attr in RELAY_ATTRS)
        if mode == "disabled":
            assert request.kv_transfer_params["prefilled_token"] == [7]
            tokenizer.convert_ids_to_tokens.assert_not_called()
        else:
            assert "prefilled_token" not in request.kv_transfer_params


@pytest.mark.parametrize("fields", [
    {"tools": [{"type": "function"}], "tool_choice": "auto"},
    {"tools": [{"type": "function"}], "tool_choice": "required"},
    {"stop": ["END"]}, {"structured_outputs": {"json": {}}},
    {"response_format": NS(type="json_object")},
    {"response_format": {"type": "json_schema"}},
    {"n": 2}, {"use_beam_search": True},
    {"max_tokens": 1}, {"max_completion_tokens": 1, "max_tokens": 100},
    {"response_format": NS(type="text")}, {"response_format": {"type": "text"}},
    {"n": None, "stop": None, "tools": None, "structured_outputs": None},
    {"n": 1, "stop": [], "tools": [], "structured_outputs": {}},
])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_preprocess_matches_legacy_relay_without_request_type_gate(pd, fields, stream, skip):
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE = skip
    kv = dict(prompt_token_ids=[11], prefilled_token=[7],
              prefilled_logprobs=RELAY_LOGPROBS, prefilled_cumulative_logprob=-0.2,
              stop_reasons=[None], remote_prefill=True, remote_engine_id="P")
    request = NS(kv_transfer_params=kv, stream=stream, **fields,
                 **dict.fromkeys(RELAY_ATTRS, "stale"))
    original = dict(prompt_token_ids=[11], multi_modal_data=object(), mm_hashes=object(),
                    mm_placeholders=object(), cache_salt="salt")
    tokenizer = _tokenizer("first")
    inputs = _preprocess(pd, request, [original], tokenizer)
    assert all(inputs[0][key] is original[key] for key in original if key != "prompt_token_ids")
    assert inputs[0]["prompt_token_ids"] == [11]
    assert inputs[0]["prefilled_token_ids"] == request._omni_prefilled_token_ids == [7]
    assert inputs[0]["prefilled_texts"] == request._omni_prefilled_text == "first"
    assert kv["prefilled_token"] == [7]
    assert kv["remote_engine_id"] == "P" and kv["remote_prefill"]
    assert all(getattr(request, key) is value for key, value in fields.items())
    assert request.stream is stream
    tokenizer.convert_ids_to_tokens.assert_called_once_with(7)
    tokenizer.decode.assert_not_called()


@pytest.mark.parametrize("count", [0, 2])
def test_multiple_engine_inputs_keep_legacy_single_prompt_contract(pd, count):
    kv = dict(prefilled_token=[7], remote_engine_id="P")
    request = NS(kv_transfer_params=kv)
    inputs = [dict(prompt_token_ids=[11], prefilled_token_ids=[7], prefilled_texts="first")
              for _ in range(count)]
    with pytest.raises(ValueError):
        _preprocess(pd, request, inputs)
    assert kv == {"prefilled_token": [7], "remote_engine_id": "P"}
    assert all(item["prefilled_token_ids"] == [7] for item in inputs)


@pytest.mark.parametrize("fields", [
    {"n": 2}, {"max_tokens": 1}, {"stop": ["END"]},
    {"structured_outputs": {"json": {}}},
])
@pytest.mark.parametrize("kind", ["delta", "cumulative", "final_only"])
@pytest.mark.parametrize("engine_core", [False, True])
def test_generate_keeps_legacy_relay_for_all_request_types(pd, fields, kind, engine_core):
    def transfer():
        return dict(prefilled_token=[7], prefilled_logprobs=RELAY_LOGPROBS,
                    prefilled_cumulative_logprob=-0.2, stop_reasons=[None],
                    prompt_token_ids=[11], remote_engine_id="P")

    kv, embedded_kv = transfer(), transfer()
    sampling = NS(**(dict(n=1, max_tokens=100, output_kind=kind,
                          extra_args={"kv_transfer_params": kv}) | fields))
    embedded_sampling = NS(extra_args={"kv_transfer_params": embedded_kv})
    prompt = (pd.EngineCoreRequest(prompt_token_ids=[11], sampling_params=embedded_sampling)
              if engine_core else dict(prompt_token_ids=[11], prefilled_token_ids=[7],
                                       prefilled_texts="first", prefilled_logprobs=RELAY_LOGPROBS,
                                       prefilled_cumulative_logprob=-0.2, multi_modal_data="keep"))
    final = _result(outputs=[_output(token_ids=[8], finish_reason="length")])
    llm = NS(tokenizer=_tokenizer("first"))
    pd._original_generate = Mock(return_value=_items(final))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(llm, prompt, sampling, "req")))
    if kind == "delta":
        assert len(results) == 2 and results[-1] is final
        assert len(results[0].outputs) == sampling.n
        assert all(output.token_ids == [7] for output in results[0].outputs)
        assert final.outputs[0].token_ids == [8]
        assert final.outputs[0].text == "好"
    else:
        assert results == [final]
        assert final.outputs[0].token_ids == [7, 8]
        assert final.outputs[0].text == "first好"
    assert final.outputs[0].finish_reason == "length"
    assert sampling.max_tokens == fields.get("max_tokens", 100)
    assert kv == transfer()
    assert pd._original_generate.call_args.args == (llm, prompt, sampling, "req")
    if engine_core:
        assert embedded_kv == kv
    else:
        assert prompt["prefilled_token_ids"] == []
        assert prompt["multi_modal_data"] == "keep"


def test_one_token_budget_merges_empty_engine_completion_exactly_once(pd):
    request = NS(kv_transfer_params=_handoff(prompt_token_ids=[11]))
    prompt, = _preprocess(pd, request, [{"prompt_token_ids": [11]}],
                          _tokenizer("first"))
    assert request._omni_prefilled_token_ids == [7]
    sampling = NS(n=1, max_tokens=1, output_kind="final_only",
                  extra_args={"kv_transfer_params": request.kv_transfer_params})
    final = _result(outputs=[_output(token_ids=[], text="", finish_reason="length",
                                    logprobs=[], cumulative_logprob=0)], kv_transfer_params=None)
    pd._original_generate = Mock(return_value=_items(final))
    results = asyncio.run(_collect(pd.AsyncLLMPatch.generate(NS(), prompt, sampling, "req")))
    assert results == [final]
    assert request._omni_prefilled_token_ids == [7]
    pd._original_chat_completion_full_generator.return_value = NS()
    _chat(pd, request, final)
    assert final.outputs[0].token_ids == [7]
    assert final.outputs[0].text == "first"
    assert final.outputs[0].finish_reason == "length"
    assert sampling.max_tokens == 1


@pytest.mark.parametrize("kind", ["delta", "cumulative", "final_only"])
@pytest.mark.parametrize("user_budget", [1, 2])
def test_one_token_prefill_handoff_respects_decode_budget(pd, kind, user_budget):
    """Join patch stages; native scheduling/output formatting remain stubs."""
    pd.envs.OMNI_PD_ROLE = "prefill"
    prefill = _result(outputs=[_output(
        token_ids=[7], text="first", finish_reason="length",
        logprobs=None, cumulative_logprob=None)])
    _chat(pd, NS(max_tokens=1), prefill)
    assert prefill.kv_transfer_params["prefilled_token"] == [7]
    assert prefill.outputs[0].finish_reason == "length"

    # Distinct P/API/engine objects model the process boundary. In particular,
    # scheduler consumption must not erase the frontend's relay metadata.
    pd.envs.OMNI_PD_ROLE = "decode"
    request = NS(max_tokens=user_budget,
                 kv_transfer_params=copy.deepcopy(prefill.kv_transfer_params))
    prompt, = _preprocess(pd, request, [dict(prompt_token_ids=[11, 12])],
                          _tokenizer("first"))
    sampling = NS(n=1, max_tokens=user_budget, output_kind=kind,
                  extra_args={"kv_transfer_params": request.kv_transfer_params})
    engine_request = _RelayRequest("req", client_index=2, max_tokens=user_budget)
    engine_request.sampling_params = copy.deepcopy(sampling)
    scheduler = _relay_scheduler(engine_request)
    transfer = NS(finished_recving={"req"})
    pd._original_update_from_kv_xfer_finished.side_effect = (
        lambda s, kv: s.finished_recving_kv_req_ids.update(kv.finished_recving))

    def budget_stop(req, max_model_len):
        # Isolate the native output-budget rule; other stop conditions have
        # separate tests. This is not a replacement for real check_stop UT.
        if req.num_output_tokens >= req.max_tokens:
            req.status = "finished_length"
            return True
        return False

    pd.check_stop.side_effect = budget_stop

    def native_update(s, sched_output, runner_output):
        pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(s, runner_output)
        return {}

    pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.side_effect = native_update
    decode_steps = []

    async def native_generate(*args, **kwargs):
        batches = pd.SchedulerKvXferFinishedPatch.update_from_output(
            scheduler, None, transfer)
        if user_budget == 1:
            terminal, = batches[2].outputs
            assert terminal.new_token_ids == []
            assert terminal.finish_reason == "length"
            assert scheduler.requests == {} and scheduler.freed == ["req"]
            assert scheduler.skipped_waiting == []
            token_ids, text = [], ""
        else:
            assert batches == {} and scheduler.requests["req"] is engine_request
            assert engine_request.output_token_ids == [7]
            scheduler.finish_requests.assert_not_called()
            # P's length finish must not terminate the user's longer request.
            decode_steps.append("native decode")
            engine_request.append_output_token_ids([8])
            token_ids, text = [8], "tail"
        yield _result(finished=True, kv_transfer_params=None, outputs=[_output(
            token_ids=token_ids, text=text, finish_reason="length",
            logprobs=None, cumulative_logprob=None)])

    pd._original_generate = native_generate

    async def collect_bounded():
        return await asyncio.wait_for(_collect(
            pd.AsyncLLMPatch.generate(NS(), prompt, sampling, "req")), timeout=1)

    results = asyncio.run(collect_bounded())
    if kind == "delta":
        assert len(results) == 2 and results[0].finished is False
        ids = [token for result in results for token in result.outputs[0].token_ids]
        text = "".join(result.outputs[0].text for result in results)
    else:
        assert len(results) == 1
        # Final HTTP formatting must not prepend the relay a second time.
        _chat(pd, request, results[0])
        ids, text = results[0].outputs[0].token_ids, results[0].outputs[0].text
    assert ids == ([7] if user_budget == 1 else [7, 8])
    assert text == ("first" if user_budget == 1 else "firsttail")
    assert results[-1].finished is True
    assert results[-1].outputs[0].finish_reason == "length"
    assert engine_request.num_output_tokens == user_budget
    assert sampling.max_tokens == request.max_tokens == user_budget
    assert len(decode_steps) == user_budget - 1
    assert request.kv_transfer_params["prefilled_token"] == [7]
    assert "prefilled_token" not in engine_request.sampling_params.extra_args["kv_transfer_params"]
    if user_budget == 1:
        assert scheduler._omni_pending_finish_outputs == []
        pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.side_effect = None
        pd.SchedulerKvXferFinishedPatch._upstream_update_from_output.return_value = {}
        assert pd.SchedulerKvXferFinishedPatch.update_from_output(scheduler, None, None) == {}


@pytest.mark.parametrize("role", ["prefill", "decode", "standalone"])
@pytest.mark.parametrize("choice", ["required", "named", "auto", "none"])
@pytest.mark.parametrize("positional_parser", [False, True])
def test_only_prefill_forced_tool_envelope_bypasses_response_parser(pd, role, choice, positional_parser):
    pd.envs.OMNI_PD_ROLE = role
    tool_choice = (pd.ChatCompletionNamedToolChoiceParam(function=NS(name="get_weather"))
                   if choice == "named" else choice)

    class Request(NS):
        def model_copy(self, *, update, deep=False):
            return Request(**(vars(self) | update))

    request = Request(tool_choice=tool_choice, tools=[{"type": "function"}], max_tokens=1)
    parser = NS(parse=Mock(side_effect=ValueError("incomplete tool JSON")))
    serving = NS(parser_cls=object())
    final = _result(kv_transfer_params={"remote_decode": True} if role != "standalone" else None)
    response = NS(choices=[NS(finish_reason="tool_calls")])

    async def original(actual_serving, actual_request, generator, *args, **kwargs):
        assert actual_serving is serving
        assert await _collect(generator) == [final]
        actual_parser = kwargs["parser"]
        if role == "prefill" and choice in ("required", "named"):
            assert actual_request is not request
            assert actual_request.tool_choice == "none"
            assert actual_request.tools is request.tools
            assert actual_parser is None
        else:
            assert actual_request is request
            assert actual_parser is parser
        assert kwargs["mm_token_counts"] == {"image": 5}
        return response

    pd._original_chat_completion_full_generator.side_effect = original
    args = (serving, request, _items(final), "req", "model", [], None, None)
    extra = (parser, {"image": 5}) if positional_parser else ()
    kwargs = {} if positional_parser else dict(parser=parser, mm_token_counts={"image": 5})
    result = asyncio.run(pd.OpenAIServingChatPatch.chat_completion_full_generator(*args, *extra, **kwargs))
    assert result is response
    assert request.tool_choice is tool_choice
    assert serving.parser_cls is not None
    assert result.choices[0].finish_reason == "tool_calls"
    parser.parse.assert_not_called()
    if final.kv_transfer_params is not None:
        assert final.kv_transfer_params["prompt_token_ids"] == [11, 12]
        assert final.kv_transfer_params["prefilled_token"] == [8]


@pytest.mark.parametrize("producer", [False, True])
def test_prefill_role_uses_engine_config_before_environment(pd, producer):
    pd.envs.OMNI_PD_ROLE = "decode" if producer else "prefill"
    config = NS(kv_transfer_config=NS(is_kv_transfer_instance=True, is_kv_producer=producer))
    assert pd._is_kv_producer(NS(engine_client=NS(vllm_config=config))) is producer
    assert pd._is_kv_producer(NS(vllm_config=config)) is producer


def test_prefill_proxy_one_token_budget_does_not_disable_ordinary_relay(pd):
    pd.envs.OMNI_PD_ROLE = "prefill"
    final = _result()
    pd._original_chat_completion_full_generator.return_value = NS()
    _chat(pd, NS(max_tokens=1), final)
    assert final.kv_transfer_params["prefilled_token"] == [8]


def _config(num_spec=3, max_model_len=128, producer=False):
    return NS(speculative_config=NS(num_speculative_tokens=num_spec),
              model_config=NS(max_model_len=max_model_len),
              kv_transfer_config=NS(is_kv_transfer_instance=True, is_kv_producer=producer))


@pytest.mark.parametrize("text,stop_ids", [("", None), ("first", [7])])
def test_token_fallback_matches_old_rules_not_extra_eligibility_gates(pd, text, stop_ids):
    request = NS(kv_transfer_params=_handoff(),
                 stop_token_ids=stop_ids)
    result, = _preprocess(pd, request, [dict(prompt_token_ids=[11])],
                          _tokenizer(text))
    assert result["prefilled_token_ids"] == [7]
    assert request._omni_prefilled_text == text


@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("skip,reuse,has_relay", [
    (False, False, False), (False, False, True),
    (True, False, True), (False, True, True), (True, True, True),
])
def test_preprocess_leaves_context_policy_and_budgets_to_native(pd, num_spec, skip, reuse, has_relay):
    pd.envs.OMNI_SKIP_DECODE_TOKENIZE, pd.envs.OMNI_REUSE_PREFILLED_TOKENS = skip, reuse
    config = _config(num_spec=num_spec)
    renderer = NS(model_config=config.model_config, renderer=NS(
        config=config, tokenizer=_tokenizer("first")))
    # Valid under the configured 128-token context, but not under the old
    # global 3 * num_spec + 1 reservation. Relay must not add that policy.
    prompt_ids = [11] * 118
    kv = _handoff(prompt_token_ids=prompt_ids) if has_relay else None
    request = NS(max_tokens=10, max_completion_tokens=10, thinking_token_budget=4,
                 kv_transfer_params=kv)
    original = dict(prompt_token_ids=prompt_ids.copy(), multi_modal_data=object())
    native_inputs = [original]
    pd._original_preprocess_chat.return_value = ([], native_inputs)

    _, inputs = asyncio.run(pd.OnlineRendererPrefilledTokenAdapter.preprocess_chat(
        renderer, request, [], None, "auto", None))

    pd._original_preprocess_chat.assert_awaited_once()
    assert request.max_tokens == request.max_completion_tokens == 10
    assert request.thinking_token_budget == 4
    assert config.model_config.max_model_len == 128
    assert inputs[0]["prompt_token_ids"] == prompt_ids
    assert inputs[0]["multi_modal_data"] is original["multi_modal_data"]
    assert ("prefilled_token_ids" in inputs[0]) is (has_relay and reuse)
    if not has_relay or not (skip or reuse):
        assert inputs == native_inputs


@pytest.mark.parametrize("has_relay", [False, True])
def test_preprocess_preserves_native_validation_failure(pd, has_relay):
    request = NS(kv_transfer_params={"prefilled_token": [7]} if has_relay else None)
    error = ValueError("native context validation failed")
    pd._original_preprocess_chat.side_effect = error
    with pytest.raises(ValueError) as exc:
        asyncio.run(pd.OnlineRendererPrefilledTokenAdapter.preprocess_chat(
            NS(), request, [], None, "auto", None))
    assert exc.value is error
    assert not hasattr(request, "_omni_prefilled_token_ids")


@pytest.mark.parametrize("num_spec", [0, 3])
@pytest.mark.parametrize("reuse,has_token", [(False, True), (True, False), (True, True)])
def test_scheduler_retains_native_context_limit_and_admission(pd, num_spec, reuse, has_token):
    pd.envs.OMNI_REUSE_PREFILLED_TOKENS = reuse
    original_add_request = pd.Scheduler.add_request
    pd.SchedulerKvXferFinishedPatch.apply()
    assert pd.Scheduler.add_request is original_add_request

    request = _RelayRequest("req", max_tokens=10)
    request.prompt_token_ids = [11] * 118
    request.all_token_ids = request.prompt_token_ids.copy()
    request.num_prompt_tokens = request.num_computed_tokens = 118
    if not has_token:
        request.sampling_params.extra_args["kv_transfer_params"].clear()
    scheduler = _relay_scheduler(request)
    scheduler.max_model_len = 128
    scheduler.vllm_config = _config(num_spec=num_spec)
    pd.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(
        scheduler, NS(finished_recving={"req"}))

    if reuse and has_token:
        pd.check_stop.assert_called_once_with(request, 128)
        assert request.num_tokens == 119 and request.num_output_tokens == 1
    else:
        pd.check_stop.assert_not_called()
        assert request.num_tokens == 118 and request.num_output_tokens == 0
    assert request.max_tokens == 10
    scheduler.finish_requests.assert_not_called()
