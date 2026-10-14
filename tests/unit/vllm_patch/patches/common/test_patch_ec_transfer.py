# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline unit tests for the EC transfer vLLM backport."""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

PATCH_PATH = next(
    p for p in Path(__file__).resolve().parents
    if (p / "omni" / "vllm_patches").is_dir()
) / "omni/vllm_patches/patches/common/patch_ec_transfer.py"


class HashableNamespace(types.SimpleNamespace):
    __hash__ = object.__hash__


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)
    return module


def _install_stubs():
    for name in (
        "vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched",
        "vllm.v1.engine", "vllm.entrypoints", "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion", "omni_npu",
        "omni_npu.vllm_patches", "omni_npu.vllm_patches.patches",
        "omni_npu.vllm_patches.patches.common",
    ):
        _module(name)

    class RequestOutput:
        def __init__(self, **kwargs):
            self.original_init = kwargs
            self.added = None

        def add(self, next_output, aggregate):
            self.added = (next_output, aggregate)

    class Request:
        SamplingParams = object
        PoolingParams = object
        torch = types.SimpleNamespace(Tensor=object)
        MultiModalFeatureSpec = object
        LoRARequest = object
        BlockHash = object

        def __init__(self, **kwargs):
            self.original_init = kwargs

    class SchedulerInterface:
        pass

    class Scheduler:
        def update_from_output(self, *_args):
            self._free_request(self.request_to_free)
            return {0: types.SimpleNamespace(outputs=[
                types.SimpleNamespace(request_id=self.request_to_free.request_id)
            ])}

    class EngineCoreOutput:
        __struct_fields__ = ()

    class EngineCoreOutputs:
        pass

    class EngineCore:
        def __init__(self):
            self.added = None

        def add_request(self, request, request_wave=0):
            self.added = (request, request_wave)
            return "added"

    class RequestState:
        def make_request_output(self, *args):
            return self.original_output

    class OutputProcessor:
        pass

    class OutputProcessorOutput:
        def __init__(self, request_outputs, reqs_to_abort):
            self.request_outputs = request_outputs
            self.reqs_to_abort = reqs_to_abort

    class FinishReason:
        STOP = "stop"

    class ChatCompletionRequest:
        model_fields = {}
        field_names = "cached"
        rebuilds = 0

        @classmethod
        def model_rebuild(cls, force=False):
            cls.rebuilds += int(force)

        def to_sampling_params(self, *, max_tokens, default_sampling_params):
            return types.SimpleNamespace(extra_args=None)

    class ChatCompletionResponse:
        model_fields = {}
        field_names = "cached"
        rebuilds = 0

        @classmethod
        def model_rebuild(cls, force=False):
            cls.rebuilds += int(force)

    envs = _module("vllm.envs", os=os, environment_variables={})
    _module("vllm.outputs", RequestOutput=RequestOutput)
    _module("vllm.v1.request", Request=Request)
    _module("vllm.v1.core.sched.interface", SchedulerInterface=SchedulerInterface)
    scheduler_mod = _module("vllm.v1.core.sched.scheduler", Scheduler=Scheduler)
    engine_mod = sys.modules["vllm.v1.engine"]
    engine_mod.EngineCoreOutput = EngineCoreOutput
    engine_mod.EngineCoreOutputs = EngineCoreOutputs
    core_mod = _module("vllm.v1.engine.core", EngineCore=EngineCore)
    coordinator_mod = _module("vllm.v1.engine.coordinator")
    core_client_mod = _module("vllm.v1.engine.core_client")
    _module(
        "vllm.v1.engine.output_processor", FinishReason=FinishReason,
        OutputProcessor=OutputProcessor, OutputProcessorOutput=OutputProcessorOutput,
        RequestState=RequestState,
    )
    _module(
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionRequest=ChatCompletionRequest,
        ChatCompletionResponse=ChatCompletionResponse,
    )
    _module("vllm.logger", init_logger=lambda _name: types.SimpleNamespace(warning=lambda *_a: None))

    registered = []

    class VLLMPatch:
        _attr_names_to_apply = []

    def register_patch(name, target):
        def decorate(cls):
            registered.append((name, target))
            return cls
        return decorate

    _module("omni_npu.vllm_patches.core", VLLMPatch=VLLMPatch,
            register_patch=register_patch)
    return types.SimpleNamespace(
        envs=envs, scheduler_mod=scheduler_mod, engine_mod=engine_mod,
        core_mod=core_mod, coordinator_mod=coordinator_mod,
        core_client_mod=core_client_mod, RequestOutput=RequestOutput,
        Request=Request, ChatCompletionRequest=ChatCompletionRequest,
        ChatCompletionResponse=ChatCompletionResponse, FinishReason=FinishReason,
        registered=registered,
    )


@pytest.fixture(scope="module")
def ec_patch():
    saved = dict(sys.modules)
    stubs = _install_stubs()
    try:
        spec = importlib.util.spec_from_file_location("_patch_ec_transfer", PATCH_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module, stubs
    finally:
        for name in list(sys.modules):
            if name not in saved:
                sys.modules.pop(name)
        for name, original in saved.items():
            sys.modules[name] = original


def test_registers_all_ec_transfer_patches(ec_patch):
    _patch, stubs = ec_patch
    names = {name for name, _target in stubs.registered}
    assert names == {
        "ECTransferEnvVarsPatch", "ECTransferRequestOutputPatch",
        "ECTransferRequestPatch", "ECTransferSchedulerPatch",
        "ECTransferEngineCorePatch", "ECTransferEngineInitPatch",
        "ECTransferRequestStatePatch", "ECTransferOutputProcessorPatch",
        "ECTransferChatProtocolFieldsPatch", "ECTransferChatProtocolSamplingPatch",
    }


def test_env_patch_sets_defaults_and_rejects_invalid_mapping(ec_patch):
    patch, stubs = ec_patch
    patch.ECTransferEnvVarsPatch.apply()
    assert stubs.envs.environment_variables["VLLM_EC_SIDE_CHANNEL_HOST"]() == "localhost"
    assert stubs.envs.environment_variables["VLLM_EC_SIDE_CHANNEL_PORT"]() == 5601
    stubs.envs.environment_variables = []
    with pytest.raises(TypeError, match="must be a dict"):
        patch.ECTransferEnvVarsPatch.apply()


def test_protocol_fields_are_added_once_and_sampling_forwards_params(ec_patch):
    patch, stubs = ec_patch
    patch._register_ec_transfer_fields()
    assert set(stubs.ChatCompletionRequest.model_fields) == {"ec_transfer_params"}
    assert set(stubs.ChatCompletionResponse.model_fields) == {"ec_transfer_params"}
    assert stubs.ChatCompletionRequest.field_names is None
    assert stubs.ChatCompletionResponse.field_names is None
    rebuilds = (stubs.ChatCompletionRequest.rebuilds, stubs.ChatCompletionResponse.rebuilds)
    patch._register_ec_transfer_fields()
    assert rebuilds == (stubs.ChatCompletionRequest.rebuilds, stubs.ChatCompletionResponse.rebuilds)

    request = stubs.ChatCompletionRequest()
    request.ec_transfer_params = {"mm_hash_endpoints": {"h": "tcp://p"}}
    sampling = patch.ECTransferChatProtocolSamplingPatch.to_sampling_params(request, 3, {})
    assert sampling.extra_args["ec_transfer_params"] == request.ec_transfer_params
    request.ec_transfer_params = None
    assert patch.ECTransferChatProtocolSamplingPatch.to_sampling_params(request, 3, {}).extra_args is None


def test_request_and_request_output_preserve_ec_transfer_params(ec_patch):
    patch, stubs = ec_patch
    output = stubs.RequestOutput.__new__(stubs.RequestOutput)
    patch.ECTransferRequestOutputPatch.__init__(
        output, "id", None, None, None, [], False,
        ec_transfer_params={"out": 1}, kv_transfer_params={"kv": 1}, extra="kept",
    )
    assert output.original_init["kv_transfer_params"] == {"kv": 1}
    assert output.ec_transfer_params == {"out": 1}
    next_output = types.SimpleNamespace(ec_transfer_params={"next": 2})
    patch.ECTransferRequestOutputPatch.add(output, next_output, True)
    assert output.added == (next_output, True)
    assert output.ec_transfer_params == {"next": 2}

    request = stubs.Request.__new__(stubs.Request)
    sampling = types.SimpleNamespace(extra_args={"ec_transfer_params": {"in": 1}})
    patch.ECTransferRequestPatch.__init__(request, "id", [], sampling, None)
    assert request.ec_transfer_params == {"in": 1}
    request = stubs.Request.__new__(stubs.Request)
    patch.ECTransferRequestPatch.__init__(request, "id", [], None, None)
    assert request.ec_transfer_params is None


def _scheduler(request, connector=None):
    freed = []
    scheduler = types.SimpleNamespace(
        request_to_free=request, ec_connector=connector, _inflight_prefills=set(),
        _connector_finished=lambda _request: (False, {"kv": 1}),
        encoder_cache_manager=types.SimpleNamespace(free=lambda item: freed.append(("cache", item))),
        finished_req_ids=set(), finished_req_ids_dict=[set()],
        _free_blocks=lambda item: freed.append(("blocks", item)),
    )
    return scheduler, freed


def test_scheduler_free_and_update_attach_ec_params(ec_patch):
    patch, _stubs = ec_patch
    request = HashableNamespace(request_id="r1", client_index=0, is_finished=lambda: True)
    connector = types.SimpleNamespace(request_finished=lambda _request: (False, {"ec": 1}))
    scheduler, freed = _scheduler(request, connector)
    kv, ec = patch.ECTransferSchedulerPatch._free_request(scheduler, request)
    assert (kv, ec) == ({"kv": 1}, {"ec": 1})
    assert scheduler.finished_req_ids == {"r1"}
    assert scheduler.finished_req_ids_dict[0] == {"r1"}
    assert freed == [("cache", request), ("blocks", request)]

    scheduler, _freed = _scheduler(request, connector)

    def _noop_free_request(*_args):
        return None

    scheduler._free_request = _noop_free_request
    result = patch.ECTransferSchedulerPatch.update_from_output(scheduler, object(), object())
    assert result[0].outputs[0].ec_transfer_params == {"ec": 1}
    assert scheduler._free_request is not None
    assert patch.ECTransferSchedulerPatch.get_ec_connector(scheduler) is connector


def test_engine_core_warns_without_connector_and_engine_init_updates_aliases(ec_patch):
    patch, stubs = ec_patch
    warnings = []

    def _record_warning(message):
        warnings.append(message)

    patch.logger.warning = _record_warning
    request = types.SimpleNamespace(ec_transfer_params={"x": 1})
    engine = types.SimpleNamespace(scheduler=types.SimpleNamespace(get_ec_connector=lambda: None))
    assert patch.ECTransferEngineCorePatch.add_request(engine, request, 2) == "added"
    assert engine.added == (request, 2)
    assert warnings

    patch.ECTransferEngineInitPatch.apply()
    assert stubs.engine_mod.EngineCoreOutput.__module__ == patch.__name__
    for module in (stubs.core_mod, stubs.core_client_mod, stubs.coordinator_mod, stubs.scheduler_mod):
        assert module.EngineCoreOutputs is stubs.engine_mod.EngineCoreOutputs


def test_request_state_and_output_processor_forward_ec_params(ec_patch):
    patch, stubs = ec_patch
    original = types.SimpleNamespace()
    state = types.SimpleNamespace(original_output=original)
    assert patch.ECTransferRequestStatePatch.make_request_output(
        state, [1], None, None, None, {"kv": 1}, {"ec": 1}
    ) is original
    assert original.ec_transfer_params == {"ec": 1}

    received = []
    result_output = types.SimpleNamespace(finished=True)
    req_state = types.SimpleNamespace(
        is_prefilling=True, num_cached_tokens=None, routed_experts_chunks=[],
        detokenizer=None, logprobs_processor=None, streaming_input=False, queue=None,
        make_request_output=lambda *args: received.append(args) or result_output,
    )
    engine_output = types.SimpleNamespace(
        request_id="r", new_token_ids=[1], pooling_output=object(), finish_reason=None,
        stop_reason=None, kv_transfer_params={"kv": 1}, ec_transfer_params={"ec": 1},
        routed_experts=None, prefill_stats=types.SimpleNamespace(num_cached_tokens=4),
        finished=True,
    )
    processor = types.SimpleNamespace(
        request_states={"r": req_state}, tracing_enabled=False,
        _update_stats_from_output=lambda *_args: None,
        _finish_request=lambda *_args: None,
        _update_stats_from_finished=lambda *_args: None,
    )
    result = patch.ECTransferOutputProcessorPatch.process_outputs(processor, [engine_output])
    assert result.request_outputs == [result_output]
    assert received[0][-1] == {"ec": 1}
    assert req_state.num_cached_tokens == 4


def test_output_processor_handles_stop_and_abort(ec_patch):
    patch, stubs = ec_patch
    queued = []
    updates = []
    state = types.SimpleNamespace(
        is_prefilling=False, routed_experts_chunks=[], streaming_input=False,
        queue=types.SimpleNamespace(put=queued.append),
        detokenizer=types.SimpleNamespace(update=lambda *_args: "stop text"),
        logprobs_processor=types.SimpleNamespace(update_from_output=lambda *_args: None),
        make_request_output=lambda *_args: types.SimpleNamespace(finished=True),
    )
    output = types.SimpleNamespace(
        request_id="r", new_token_ids=[1], pooling_output=None, finish_reason=None,
        stop_reason=None, kv_transfer_params=None, ec_transfer_params=None,
        routed_experts=[3], prefill_stats=None, finished=False,
    )
    processor = types.SimpleNamespace(
        request_states={"r": state}, tracing_enabled=True,
        _update_stats_from_output=lambda *_args: None,
        _finish_request=lambda value: updates.append(value),
        _update_stats_from_finished=lambda *_args: None,
        do_tracing=lambda *_args: None,
    )
    result = patch.ECTransferOutputProcessorPatch.process_outputs(processor, [output])
    assert result.request_outputs == []
    assert result.reqs_to_abort == ["r"]
    assert output.routed_experts == [3]
    assert state.routed_experts_chunks == [[3]]
    assert len(queued) == 1
    assert updates == [state]
