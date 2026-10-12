# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU regressions for expert export on vLLM 0.25.1.

Migrated from omni-npu/tests/unit/vllm_patch/common/test_routed_experts_patch.py.
Legacy shared-memory capturer and full scheduler-backport cases are excluded:
these responsibilities belong to native vLLM in the upgraded implementation.
"""

import asyncio
import argparse
import json
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from unit.vllm_patch.patches.common.test_prefilled_token_skip_tokenize import (
    PATCH_ROOT,
    PD_MODULE,
    RELAY_LOGPROBS,
    _collect,
    _items,
    _output,
    _result,
    pd as pd,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("init_order", ["expert_first", "other_first"])
def test_scheduler_init_patch_preserves_other_init_patch(
    expert_patch, monkeypatch, init_order
):
    scheduler_cls = expert_patch.Scheduler
    config = NS(
        model_config=NS(enable_return_routed_experts=True),
        parallel_config=NS(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        kv_transfer_config=NS(
            is_kv_transfer_instance=True, is_kv_producer=False
        ),
    )
    calls = []

    def original_init(self, *, vllm_config):
        calls.append("original")
        self.vllm_config = vllm_config
        self.kv_cache_config = NS()
        self.routed_experts_mgr = NS(attn_gid=2)

    monkeypatch.setattr(scheduler_cls, "__init__", original_init)
    monkeypatch.setattr(expert_patch, "_get_attention_kv_cache_gid", lambda _: 0)

    class OtherSchedulerInitPatch(expert_patch.SchedulerInitPatch.__bases__[0]):
        _target = scheduler_cls
        _attr_names_to_apply = ["__init__"]

        def __init__(self, *args, **kwargs):
            OtherSchedulerInitPatch._upstream__init__(self, *args, **kwargs)
            calls.append("other")
            self.other_patch_initialized = True

        @classmethod
        def apply(cls):
            cls.apply_bypass_conflict("__init__")

    patches = (expert_patch.SchedulerInitPatch, OtherSchedulerInitPatch)
    if init_order == "other_first":
        patches = reversed(patches)
    for patch in patches:
        patch.apply()

    scheduler = scheduler_cls(vllm_config=config)
    assert calls == ["original", "other"]
    assert scheduler.other_patch_initialized is True
    assert scheduler.routed_experts_attn_gid == 2
    assert scheduler.routed_experts_mgr.is_pd_disaggregation is True
    assert scheduler.routed_experts_mgr.is_pd_prefill is False


def test_scheduler_patch_keeps_pd_finish_notification_chain(pd, expert_patch):
    pd.SchedulerKvXferFinishedPatch.apply()
    expert_patch.SchedulerRoutedExpertsPatch.apply()
    scheduler_cls = expert_patch.Scheduler
    scheduler = scheduler_cls()
    request = NS(
        request_id="req",
        num_tokens=7,
        num_prompt_tokens=4,
        num_cached_tokens=0,
        sampling_params=NS(output_kind="cumulative"),
    )
    output = NS(
        request_id="req", new_token_ids=[8], finish_reason=None, routed_experts=None
    )
    pending = NS(request_id="finished", finish_reason="stop", routed_experts=None)
    expected = {0: NS(outputs=[output])}
    scheduler.requests = {"req": request}
    scheduler.routed_experts_mgr = NS(
        is_pd_disaggregation=True,
        is_pd_prefill=False,
        routed_experts_by_slot=np.arange(256).reshape(256, 1, 1),
    )
    scheduler.vllm_config = NS(model_config=NS(enable_return_routed_experts=True))
    scheduler.kv_cache_manager = NS(
        get_blocks=lambda req_id: NS(get_block_ids=lambda: [[10, 11]])
    )
    scheduler.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=4))])
    scheduler._omni_pending_finish_outputs = [(0, pending)]
    pd._original_update_from_output.return_value = expected
    scheduled, runner = NS(num_scheduled_tokens={"req": 1}), NS()
    result = scheduler.update_from_output(scheduled, runner)
    assert result is expected
    pd._original_update_from_output.assert_called_once_with(
        scheduler, scheduled, runner
    )
    assert result[0].outputs == [output, pending]
    assert scheduler._omni_pending_finish_outputs == []
    np.testing.assert_array_equal(output.routed_experts.ravel(), [44, 45])


def test_cli_expert_patch_keeps_lopt_and_repetition_wrappers(
    pd, expert_patch, monkeypatch
):
    class EngineArgs:
        @staticmethod
        def add_cli_args(parser):
            return parser

        @classmethod
        def from_cli_args(cls, args):
            return cls()

        def create_engine_config(self, usage_context=None, headless=False):
            return expert_patch.VllmConfig()

        def create_model_config(self):
            return NS()

    def module(name, **attrs):
        stub = ModuleType(name)
        stub.__path__ = []
        vars(stub).update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
        return stub

    monkeypatch.setattr(sys.modules["vllm"], "EngineArgs", EngineArgs, raising=False)
    monkeypatch.setattr(
        sys.modules["vllm.config"],
        "ModelConfig",
        type("ModelConfig", (), {}),
        raising=False,
    )
    monkeypatch.setattr(
        pd.envs, "OMNI_REPETITION_DETECTION_CONFIG", None, raising=False
    )
    module("vllm.utils.argparse_utils", FlexibleArgumentParser=argparse.ArgumentParser)
    module("vllm.renderers.base", BaseRenderer=type("BaseRenderer", (), {}))
    module(
        "vllm.v1.engine.input_processor",
        InputProcessor=type(
            "InputProcessor", (), {"process_inputs": lambda *args, **kwargs: None}
        ),
    )
    sampling = sys.modules["vllm.sampling_params"]
    monkeypatch.setattr(
        sampling,
        "RepetitionDetectionParams",
        type("RepetitionDetectionParams", (NS,), {}),
        raising=False,
    )
    monkeypatch.setattr(
        sampling, "SamplingParams", type("SamplingParams", (NS,), {}), raising=False
    )
    prefix = "omni_npu.vllm_patches.patches.common."
    lopt = pd.load_test_module(
        prefix + "patch_lopt", PATCH_ROOT / "patches/common/patch_lopt.py"
    )
    repetition = pd.load_test_module(
        prefix + "patch_repetition_detection_config",
        PATCH_ROOT / "patches/common/patch_repetition_detection_config.py",
    )
    routed = pd.load_test_module(
        prefix + "patch_routed_experts_args",
        PATCH_ROOT / "patches/common/patch_routed_experts_args.py",
    )
    lopt.EngineArgsLoptPatch.apply()
    repetition.EngineArgsRepetitionDetectionPatch.apply()
    before = EngineArgs.add_cli_args(argparse.ArgumentParser())
    assert "--enable-lopt" in before._option_string_actions
    assert "--repetition-detection" in before._option_string_actions
    routed.RoutedExpertsArgsEngineArgsPatch.apply()
    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    assert "--enable-lopt" in parser._option_string_actions
    assert "--repetition-detection" in parser._option_string_actions
    args = parser.parse_args(
        [
            "--enable-lopt",
            "--lopt-pool-size",
            "3",
            "--repetition-detection",
            '{"max_pattern_size":10,"min_pattern_size":2,"min_count":3}',
            "--routed-experts-serialization-mode",
            "base64",
        ]
    )
    engine = EngineArgs.from_cli_args(args)
    assert engine.enable_lopt is True and engine.lopt_pool_size == 3
    assert engine.repetition_detection.max_pattern_size == 10
    cfg = engine.create_engine_config()
    assert cfg.routed_experts_serialization_mode == "base64"
    assert repetition._PROCESS_DEFAULT is engine.repetition_detection


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning": "think", "content": "answer"},
        {
            "reasoning_content": "think",
            "tool_calls": [{"function": {"name": "tool", "arguments": "{}"}}],
        },
    ],
)
def test_split_mixed_sse_carries_experts_only_once(
    pd, expert_patch, serving_factory, monkeypatch, delta
):
    logger_module = sys.modules["vllm.logger"]
    monkeypatch.setattr(
        logger_module, "logger", logger_module.init_logger("split-test"), raising=False
    )
    prefix = "omni_npu.vllm_patches.patches.common."
    pd.load_test_module(
        prefix + "patch_serving_apc", PATCH_ROOT / "patches/common/patch_serving_apc.py"
    )
    split = pd.load_test_module(
        prefix + "patch_split_reasoning_content",
        PATCH_ROOT / "patches/common/patch_split_reasoning_content.py",
    )
    serving = serving_factory()
    routing = np.arange(4, dtype=np.uint8).reshape(1, 2, 2)
    payload = _payload(serving, routing)
    raw = (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": delta, "routed_experts": payload}]}
        )
        + "\n\n"
    )
    events = split._maybe_split_sse_line(raw, None)
    assert len(events) == 2
    choices = [
        json.loads(event.removeprefix("data: "))["choices"][0] for event in events
    ]
    assert sum(choice.get("routed_experts") is not None for choice in choices) == 1
    assert choices[0].get("routed_experts") is None
    np.testing.assert_array_equal(
        _restore(serving, choices[1]["routed_experts"]), routing
    )


@pytest.fixture
def expert_patch(pd, monkeypatch):
    """Load actual expert code, mocking only external vLLM interfaces."""

    def module(name, **attrs):
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent_name and parent is None:
            parent = module(parent_name)
        stub = ModuleType(name)
        previous = sys.modules.get(name)
        if previous is not None:
            vars(stub).update(vars(previous))
        stub.__path__ = []
        vars(stub).update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
        if parent is not None:
            monkeypatch.setattr(parent, child, stub, raising=False)
        return stub

    for name, choices in (
        (
            "chat_completion",
            ("ChatCompletionResponseChoice", "ChatCompletionResponseStreamChoice"),
        ),
        ("completion", ("CompletionResponseChoice", "CompletionResponseStreamChoice")),
    ):
        module(
            f"vllm.entrypoints.openai.{name}.protocol",
            **{
                choice: type(
                    choice,
                    (NS,),
                    {"model_dump": lambda self, **kwargs: vars(self).copy()},
                )
                for choice in choices
            },
        )
    completion = sys.modules[
        "vllm.entrypoints.openai.completion.serving"
    ].OpenAIServingCompletion
    completion.request_output_to_completion_response = Mock()
    module(
        "vllm.config",
        VllmConfig=type("VllmConfig", (), {"__post_init__": lambda self: None}),
    )
    module("vllm.outputs", RequestOutput=type("RequestOutput", (NS,), {"add": Mock()}))
    module(
        "vllm.entrypoints.generate.base.serving",
        GenerateBaseServing=type("GenerateBaseServing", (), {}),
    )
    module("vllm.v1.core.sched.output", SchedulerOutput=NS)
    module("vllm.v1.outputs", ModelRunnerOutput=NS)
    module(
        "omni_npu.plugin_decorators", update_from_output_decorator=lambda method: method
    )
    loaded = pd.load_test_module(
        "omni_npu.vllm_patches.patches.common.patch_routed_experts",
        PATCH_ROOT / "patches/common/patch_routed_experts.py",
    )
    loaded.envs.OMNI_PD_ROLE = "standalone"
    return loaded


@pytest.fixture
def serving_factory(expert_patch):
    def make(role=None, mode="zip_base64"):
        config = NS(
            is_kv_transfer_instance=role is not None,
            kv_role=role,
            is_kv_producer=role == "kv_producer",
            is_kv_consumer=role == "kv_consumer",
        )
        serving = NS(
            engine_client=NS(
                vllm_config=NS(
                    kv_transfer_config=config, routed_experts_serialization_mode=mode
                )
            )
        )
        patch = expert_patch.RoutedExpertsSerializationPatch
        for name in patch._attr_names_to_apply:
            setattr(serving, name, patch.__dict__[name].__get__(serving))
        return serving

    return make


def _payload(serving, data):
    payload = {}
    serving.add_ndarray_info_to_dict(data, payload)
    return payload


def _restore(serving, payload):
    return serving.restore_str_to_ndarray(
        payload["routed_experts_str"],
        tuple(payload["routed_experts_shape"]),
        payload["routed_experts_dtype"],
    )


def _proxy_payloads(patch, proxy):
    return [
        patch.CompletionResponseChoice(index=output.index, text="text")
        .model_dump()
        .get("routed_experts")
        for output in proxy.outputs
    ]


@pytest.mark.parametrize(
    "role,expected", [("prefill", True), ("decode", False), ("standalone", False)]
)
def test_pd_role_helpers_fall_back_to_environment(
    expert_patch, serving_factory, role, expected
):
    expert_patch.envs.OMNI_PD_ROLE = role
    manager = NS()
    expert_patch._set_pd_flags(manager)
    assert manager.is_pd_disaggregation is (role != "standalone")
    assert manager.is_pd_prefill is expected
    assert expert_patch._is_prefill_node(serving_factory()) is expected


@pytest.mark.parametrize(
    "role,expected", [("kv_producer", True), ("kv_consumer", False)]
)
def test_pd_role_helpers_prefer_kv_transfer_config(
    expert_patch, serving_factory, role, expected
):
    expert_patch.envs.OMNI_PD_ROLE = "decode" if expected else "prefill"
    serving = serving_factory(role)
    manager = NS()
    expert_patch._set_pd_flags(manager, serving.engine_client.vllm_config)
    assert manager.is_pd_disaggregation is True
    assert manager.is_pd_prefill is expected
    assert expert_patch._is_prefill_node(serving) is expected


@pytest.mark.parametrize("mode", ["zip_base64", "base64"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32])
@pytest.mark.parametrize("strided", [False, True])
def test_serialization_helpers_round_trip(serving_factory, mode, dtype, strided):
    serving = serving_factory(mode=mode)
    data = np.arange(24, dtype=dtype).reshape(4, 2, 3)
    if dtype == np.uint16:
        data += 256
    if strided:
        data = data[::2]
    payload = _payload(serving, data)
    assert set(payload) == {
        "routed_experts_shape",
        "routed_experts_dtype",
        "routed_experts_str_len",
        "routed_experts_str",
    }
    assert payload["routed_experts_shape"] == data.shape
    assert payload["routed_experts_dtype"] == str(data.dtype)
    assert payload["routed_experts_str_len"] == len(payload["routed_experts_str"])
    restored = _restore(serving, payload)
    assert restored.dtype == data.dtype
    np.testing.assert_array_equal(restored, data)


def test_restore_rejects_serialization_mode_mismatch(serving_factory):
    zipped = serving_factory(mode="zip_base64")
    plain = serving_factory(mode="base64")
    data = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    with pytest.raises(ValueError):
        _restore(plain, _payload(zipped, data))


@pytest.mark.parametrize("operation", ["convert", "restore"])
def test_serialization_rejects_unknown_mode(serving_factory, operation):
    serving = serving_factory(mode="unsupported")
    with pytest.raises(
        ValueError, match="Unsupported routed experts serialization mode"
    ):
        if operation == "convert":
            serving.convert_ndarray_to_str(np.zeros((1, 2, 2), dtype=np.uint8))
        else:
            serving.restore_str_to_ndarray("AA==", (1,), "uint8")


@pytest.mark.parametrize("mismatch", ["rank", "shape", "dtype"])
def test_concatenate_rejects_incompatible_routing(serving_factory, mismatch):
    serving = serving_factory()
    prefill = np.zeros((1, 2, 2), dtype=np.uint8)
    decode = {
        "rank": np.zeros((2, 2), dtype=np.uint8),
        "shape": np.zeros((1, 3, 2), dtype=np.uint8),
        "dtype": np.zeros((1, 2, 2), dtype=np.uint16),
    }[mismatch]
    with pytest.raises(ValueError):
        serving.concatenate_dict_and_ndarray(_payload(serving, prefill), decode)


@pytest.mark.parametrize("mode", ["zip_base64", "base64"])
def test_build_payload_merges_prefill_and_decode(expert_patch, serving_factory, mode):
    serving = serving_factory(mode=mode)
    prefill = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    decode = np.arange(8, 16, dtype=np.int32).reshape(2, 2, 2)
    payload = expert_patch._build_routed_experts_payload(
        serving, _output(routed_experts=decode), _payload(serving, prefill)
    )
    np.testing.assert_array_equal(
        _restore(serving, payload), np.concatenate([prefill, decode])
    )


@pytest.mark.parametrize("data", [None, np.empty((0, 2, 2), dtype=np.int32)])
def test_build_payload_without_decode_keeps_prefill(
    expert_patch, serving_factory, data
):
    serving = serving_factory()
    kv = _payload(serving, np.zeros((1, 2, 2), dtype=np.int32))
    assert (
        expert_patch._build_routed_experts_payload(
            serving, _output(routed_experts=data), kv
        )
        is kv
    )
    assert (
        expert_patch._build_routed_experts_payload(
            serving, _output(routed_experts=data)
        )
        is None
    )


@pytest.mark.parametrize("mode", ["zip_base64", "base64"])
def test_stream_attaches_prefill_once_per_choice(expert_patch, serving_factory, mode):
    expert_patch._ensure_choice_patches_applied()
    serving = serving_factory("kv_consumer", mode)
    prefill = np.zeros((1, 2, 2), dtype=np.int32)
    request = NS(kv_transfer_params=_payload(serving, prefill))
    rows = [np.full((1, 2, 2), i, dtype=np.int32) for i in range(1, 5)]
    results = [
        NS(
            outputs=[
                _output(index=i, routed_experts=rows[2 * step + i]) for i in range(2)
            ],
            kv_transfer_params=None,
        )
        for step in range(2)
    ]
    proxies = asyncio.run(
        _collect(
            expert_patch._wrap_request_output_stream(serving, request, _items(*results))
        )
    )
    for step, proxy in enumerate(proxies):
        for choice, payload in enumerate(_proxy_payloads(expert_patch, proxy)):
            row = rows[2 * step + choice]
            expected = np.concatenate([prefill, row]) if step == 0 else row
            np.testing.assert_array_equal(_restore(serving, payload), expected)


@pytest.mark.parametrize("first_has_tokens", [False, True])
def test_stream_prefilled_or_empty_chunk_does_not_duplicate_prompt(
    expert_patch, serving_factory, first_has_tokens
):
    expert_patch._ensure_choice_patches_applied()
    serving = serving_factory("kv_consumer")
    prefill = np.zeros((1, 2, 2), dtype=np.int32)
    decode = np.ones((1, 2, 2), dtype=np.int32)
    request = NS(kv_transfer_params=_payload(serving, prefill))
    results = [
        NS(
            outputs=[
                _output(routed_experts=None, token_ids=[7] if first_has_tokens else [])
            ],
            kv_transfer_params=None,
        ),
        NS(outputs=[_output(routed_experts=decode)], kv_transfer_params=None),
    ]
    proxies = asyncio.run(
        _collect(
            expert_patch._wrap_request_output_stream(serving, request, _items(*results))
        )
    )
    first = _proxy_payloads(expert_patch, proxies[0])[0]
    second = _proxy_payloads(expert_patch, proxies[1])[0]
    if first_has_tokens:
        np.testing.assert_array_equal(_restore(serving, first), prefill)
        np.testing.assert_array_equal(_restore(serving, second), decode)
    else:
        assert first is None
        np.testing.assert_array_equal(
            _restore(serving, second), np.concatenate([prefill, decode])
        )


def test_indexed_stream_attaches_prefill_once_per_prompt(expert_patch, serving_factory):
    expert_patch._ensure_choice_patches_applied()
    serving = serving_factory("kv_consumer")
    prefill = np.zeros((1, 2, 2), dtype=np.int32)
    rows = [np.full((1, 2, 2), i, dtype=np.int32) for i in range(1, 4)]
    results = [
        (index, NS(outputs=[_output(routed_experts=row)], kv_transfer_params=None))
        for index, row in zip([0, 1, 0], rows)
    ]
    wrapped = asyncio.run(
        _collect(
            expert_patch._wrap_indexed_request_output_stream(
                serving,
                NS(kv_transfer_params=_payload(serving, prefill)),
                _items(*results),
            )
        )
    )
    for step, (_, proxy) in enumerate(wrapped):
        expected = np.concatenate([prefill, rows[step]]) if step < 2 else rows[step]
        np.testing.assert_array_equal(
            _restore(serving, _proxy_payloads(expert_patch, proxy)[0]), expected
        )


def test_proxy_restores_routing_context_after_iteration(expert_patch):
    expert_patch._ensure_choice_patches_applied()
    payload = {"routed_experts_str": "payload"}
    proxy = expert_patch._RequestOutputProxy(
        NS(outputs=[_output()]), lambda output: payload
    )
    assert _proxy_payloads(expert_patch, proxy) == [payload]
    assert expert_patch._CURRENT_ROUTED_EXPERTS_PAYLOAD.get() is None
    assert (
        "routed_experts"
        not in expert_patch.CompletionResponseChoice(index=1).model_dump()
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", None])
def test_stash_routing_only_creates_handoff_on_prefill(
    expert_patch, serving_factory, role
):
    serving = serving_factory(role)
    data = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    result = NS(outputs=[_output(routed_experts=data)], kv_transfer_params=None)
    expert_patch._stash_prefill_routed_experts(serving, result)
    if role == "kv_producer":
        np.testing.assert_array_equal(
            _restore(serving, result.kv_transfer_params), data
        )
    else:
        assert result.kv_transfer_params is None


@pytest.mark.parametrize(
    "kind,stopped,pd_decode,sent,expected",
    [
        ("delta", False, False, True, [45, 46]),
        ("cumulative", False, False, False, [40, 41, 42, 43, 44, 45, 46]),
        ("final_only", True, True, False, [45, 46]),
    ],
)
def test_slot_mapping_selects_cached_and_decode_rows(
    expert_patch, kind, stopped, pd_decode, sent, expected
):
    manager = NS(
        is_pd_disaggregation=pd_decode,
        is_pd_prefill=False,
        routed_experts_by_slot=np.arange(256).reshape(256, 1, 1),
    )
    request = NS(
        num_tokens=8,
        num_prompt_tokens=5,
        num_cached_tokens=3,
        sampling_params=NS(output_kind=kind),
        _omni_stream_prompt_routed_experts_sent=sent,
    )
    result = expert_patch._get_routed_experts_from_slot_mapping(
        manager, request, np.arange(40, 48), stopped=stopped, num_new_tokens=2
    )
    np.testing.assert_array_equal(result.ravel(), expected)


def test_delta_slot_mapping_sends_prompt_only_once(expert_patch):
    manager = NS(
        is_pd_disaggregation=False,
        routed_experts_by_slot=np.arange(256).reshape(256, 1, 1),
    )
    request = NS(
        num_tokens=8,
        num_prompt_tokens=5,
        num_cached_tokens=0,
        sampling_params=NS(output_kind=expert_patch.RequestOutputKind.DELTA),
    )
    slots = np.arange(40, 48)
    first = expert_patch._get_routed_experts_from_slot_mapping(
        manager, request, slots, stopped=False, num_new_tokens=2
    )
    np.testing.assert_array_equal(first.ravel(), np.arange(40, 47))
    request.num_tokens = 9
    second = expert_patch._get_routed_experts_from_slot_mapping(
        manager, request, slots, stopped=False, num_new_tokens=1
    )
    np.testing.assert_array_equal(second.ravel(), [47])


@pytest.mark.parametrize("cached,error", [(-1, RuntimeError), (8, ValueError)])
def test_slot_mapping_rejects_invalid_cache_count(expert_patch, cached, error):
    request = NS(
        num_tokens=8,
        num_prompt_tokens=5,
        num_cached_tokens=cached,
        sampling_params=NS(output_kind=expert_patch.RequestOutputKind.CUMULATIVE),
    )
    with pytest.raises(error):
        expert_patch._get_routed_experts_from_slot_mapping(
            NS(), request, np.arange(8), stopped=False, num_new_tokens=1
        )


def test_request_slot_mapping_uses_attention_group(expert_patch):
    blocks = NS(get_block_ids=lambda: [[1], [10, 11]])
    scheduler = NS(
        routed_experts_attn_gid=1,
        kv_cache_manager=NS(get_blocks=lambda req_id: blocks),
        kv_cache_config=NS(
            kv_cache_groups=[
                NS(kv_cache_spec=NS(block_size=128)),
                NS(kv_cache_spec=NS(block_size=4)),
            ]
        ),
    )
    np.testing.assert_array_equal(
        expert_patch._get_request_slot_mapping(scheduler, NS(request_id="req")),
        np.arange(40, 48),
    )


@pytest.mark.parametrize(
    "enabled,manager",
    [(False, None), (True, None), (True, NS(is_pd_disaggregation=False))],
)
def test_scheduler_preserves_native_output_when_no_pd_decode(
    expert_patch, monkeypatch, enabled, manager
):
    expected = {0: NS(outputs=[])}
    native = Mock(return_value=expected)
    monkeypatch.setattr(
        expert_patch.SchedulerRoutedExpertsPatch, "_upstream_update_from_output", native
    )
    scheduler = NS(
        routed_experts_mgr=manager,
        vllm_config=NS(model_config=NS(enable_return_routed_experts=enabled)),
    )
    scheduled, runner = NS(num_scheduled_tokens={}), NS()
    result = expert_patch.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler, scheduled, runner
    )
    assert result is expected
    native.assert_called_once_with(scheduler, scheduled, runner)


@pytest.mark.parametrize("stopped", [False, True])
def test_pd_scheduler_replaces_only_decode_routing_after_native_cleanup(
    expert_patch, monkeypatch, stopped
):
    request = NS(
        request_id="req",
        num_tokens=7,
        num_prompt_tokens=4,
        num_cached_tokens=0,
        sampling_params=NS(output_kind=expert_patch.RequestOutputKind.CUMULATIVE),
    )
    output = NS(
        request_id="req",
        finish_reason="stop" if stopped else None,
        new_token_ids=[8],
        routed_experts="native-last-step",
    )
    expected = {0: NS(outputs=[output])}
    scheduler = NS(
        requests={"req": request},
        routed_experts_mgr=NS(
            is_pd_disaggregation=True,
            is_pd_prefill=False,
            routed_experts_by_slot=np.arange(256).reshape(256, 1, 1),
        ),
        vllm_config=NS(model_config=NS(enable_return_routed_experts=True)),
        kv_cache_manager=NS(
            get_blocks=lambda req_id: NS(get_block_ids=lambda: [[10, 11]])
        ),
        kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=4))]),
    )

    def native(current, scheduled, runner):
        if stopped:
            current.requests.pop("req")
        return expected

    monkeypatch.setattr(
        expert_patch.SchedulerRoutedExpertsPatch, "_upstream_update_from_output", native
    )
    result = expert_patch.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler, NS(num_scheduled_tokens={"req": 1}), NS()
    )
    assert result is expected
    np.testing.assert_array_equal(output.routed_experts.ravel(), [44, 45])


@pytest.mark.parametrize("scenario", ["aggregate", "replace", "append"])
def test_request_output_add_preserves_native_behavior_and_routing(
    expert_patch, monkeypatch, scenario
):
    first_data = np.arange(4, dtype=np.int32).reshape(1, 2, 2)
    next_data = np.arange(8, 16, dtype=np.int32).reshape(2, 2, 2)
    old = _output(text="a", token_ids=[7], routed_experts=first_data)
    incoming = _output(
        index=1 if scenario == "append" else 0,
        text="bc",
        token_ids=[8, 9],
        routed_experts=next_data,
    )
    current, following = (
        NS(outputs=[old], finished=False),
        NS(outputs=[incoming], finished=True),
    )

    def native(output, next_output, aggregate):
        output.finished = next_output.finished
        if scenario == "append":
            output.outputs.append(incoming)
        elif aggregate:
            old.text += incoming.text
            old.token_ids += incoming.token_ids
            old.routed_experts = incoming.routed_experts
        else:
            output.outputs[0] = incoming

    original = Mock(side_effect=native)
    monkeypatch.setattr(expert_patch, "_original_add", original)
    aggregate = scenario != "replace"
    expert_patch.RequestOutputRoutedExpertsAggregationPatch.add(
        current, following, aggregate
    )
    original.assert_called_once_with(current, following, aggregate)
    assert current.finished is True
    if scenario == "aggregate":
        assert old.text == "abc" and old.token_ids == [7, 8, 9]
        np.testing.assert_array_equal(
            old.routed_experts, np.concatenate([first_data, next_data])
        )
    else:
        assert current.outputs[-1] is incoming
        np.testing.assert_array_equal(incoming.routed_experts, next_data)


def test_expert_apc_registry_relays_to_the_single_active_pd_implementation(
    pd, expert_patch
):
    experts = expert_patch
    apc = pd.load_test_module(
        "omni_npu.vllm_patches.patches.common.patch_serving_apc",
        PATCH_ROOT / "patches/common/patch_serving_apc.py",
    )
    pd_method = pd.OpenAIServingChatPatch.__dict__["chat_completion_full_generator"]
    assert (
        apc._orig_chat_full
        is experts.ExpertIdServingChatFinal.__dict__["chat_completion_full_generator"]
    )
    assert experts._ORIGINAL_CHAT_FULL is pd_method
    assert [
        name
        for name in sys.modules
        if name.endswith(".patch_prefilled_token_skip_tokenize")
    ] == [PD_MODULE]
    manager = sys.modules["omni_npu.vllm_patches.patch_manager"].PatchManager
    assert (
        manager.registered_patches["PrefilledTokenSkipOpenAIServingChat"]
        is apc.OpenAIServingChatFullAPCPatch
    )
    manager().apply_patch("PrefilledTokenSkipOpenAIServingChat")
    chat = sys.modules[
        "vllm.entrypoints.openai.chat_completion.serving"
    ].OpenAIServingChat
    assert chat._omni_npu_applied_patches == {
        "chat_completion_full_generator": "OpenAIServingChatFullAPCPatch"
    }


@pytest.mark.parametrize("role", ["prefill", "decode", "standalone"])
@pytest.mark.parametrize("mode", ["zip_base64", "base64"])
def test_apc_expert_pd_chain_preserves_routing_and_handoff(
    pd, expert_patch, role, mode
):
    import numpy as np

    experts = expert_patch
    apc = pd.load_test_module(
        "omni_npu.vllm_patches.patches.common.patch_serving_apc",
        PATCH_ROOT / "patches/common/patch_serving_apc.py",
    )
    pd.envs.OMNI_PD_ROLE = role
    serving = NS(
        enable_prompt_tokens_details=True,
        engine_client=NS(vllm_config=NS(routed_experts_serialization_mode=mode)),
    )
    serialization = experts.RoutedExpertsSerializationPatch
    for name in serialization._attr_names_to_apply:
        setattr(serving, name, serialization.__dict__[name].__get__(serving))
    prompt_routing = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    decode_routing = np.arange(8, 12, dtype=np.uint8).reshape(1, 2, 2)
    kv_payload = {}
    serving.add_ndarray_info_to_dict(prompt_routing, kv_payload)
    request = NS(kv_transfer_params=kv_payload if role == "decode" else None)
    if role == "decode":
        request._omni_prefilled_token_ids = [7]
        request._omni_prefilled_text = "first"
        request._omni_prefilled_logprobs = RELAY_LOGPROBS
        request._omni_prefilled_cumulative_logprob = -0.2
    output = _output(
        token_ids=[8] if role == "decode" else [7],
        routed_experts=decode_routing if role == "decode" else prompt_routing,
    )
    final = _result(
        outputs=[output],
        num_cached_tokens=1,
        kv_transfer_params={"remote_decode": True} if role == "prefill" else None,
    )

    async def native(actual_serving, actual_request, generator, *args, **kwargs):
        assert actual_serving is serving
        assert await _collect(generator) == [final]
        # Copy the handoff as a serializer would: routing must be stashed first.
        handoff = (
            dict(final.kv_transfer_params)
            if final.kv_transfer_params is not None
            else None
        )
        return NS(
            choices=[NS(index=0)],
            kv_transfer_params=handoff,
            usage=NS(prompt_tokens=2, prompt_tokens_details=None),
        )

    pd._original_chat_completion_full_generator.side_effect = native
    intermediate = _result(num_cached_tokens=0)
    response = asyncio.run(
        apc.OpenAIServingChatFullAPCPatch.chat_completion_full_generator(
            serving,
            request,
            _items(intermediate, final),
            "req",
            "model",
            [],
            None,
            None,
        )
    )
    pd._original_chat_completion_full_generator.assert_awaited_once()
    payload = response.choices[0].routed_experts
    actual = serving.restore_str_to_ndarray(
        payload["routed_experts_str"],
        tuple(payload["routed_experts_shape"]),
        payload["routed_experts_dtype"],
    )
    expected = (
        np.concatenate([prompt_routing, decode_routing])
        if role == "decode"
        else prompt_routing
    )
    np.testing.assert_array_equal(actual, expected)
    assert response.usage.prompt_tokens_details.cached_tokens == 1
    if role == "prefill":
        assert response.kv_transfer_params["prompt_token_ids"] == [11, 12]
        assert response.kv_transfer_params["prefilled_token"] == [7]
        assert (
            response.kv_transfer_params["routed_experts_str"]
            == payload["routed_experts_str"]
        )
    elif role == "decode":
        assert output.token_ids == [7, 8]
        assert len(output.logprobs) == 2
    else:
        assert response.kv_transfer_params is None


@pytest.mark.parametrize(
    "failure", [ValueError("generation failed"), asyncio.CancelledError()]
)
def test_expert_chat_delegates_generation_errors_to_pd(pd, expert_patch, failure):
    experts = expert_patch
    serving = NS(
        create_error_response=Mock(
            side_effect=lambda message: pd.ErrorResponse(message=message)
        )
    )
    response = asyncio.run(
        experts.ExpertIdServingChatFinal.chat_completion_full_generator(
            serving,
            NS(kv_transfer_params=None),
            _items(failure),
            "req",
            "model",
            [],
            None,
            None,
        )
    )
    assert isinstance(response, pd.ErrorResponse)
    serving.create_error_response.assert_called_once()
    pd._original_chat_completion_full_generator.assert_not_awaited()


def test_expert_chat_preserves_pd_empty_generator_error(pd, expert_patch):
    experts = expert_patch
    with pytest.raises(RuntimeError, match="at least one RequestOutput"):
        asyncio.run(
            experts.ExpertIdServingChatFinal.chat_completion_full_generator(
                NS(),
                NS(kv_transfer_params=None),
                _items(),
                "req",
                "model",
                [],
                None,
                None,
            )
        )
    pd._original_chat_completion_full_generator.assert_not_awaited()


def test_expert_chat_preserves_native_error_response(pd, expert_patch):
    experts = expert_patch
    error = pd.ErrorResponse(message="native response failed")
    pd._original_chat_completion_full_generator.return_value = error
    response = asyncio.run(
        experts.ExpertIdServingChatFinal.chat_completion_full_generator(
            NS(),
            NS(kv_transfer_params=None),
            _items(_result()),
            "req",
            "model",
            [],
            None,
            None,
        )
    )
    assert response is error
    pd._original_chat_completion_full_generator.assert_awaited_once()


@pytest.mark.parametrize("role", ["prefill", "decode", "standalone"])
@pytest.mark.parametrize("choice", ["required", "named", "auto", "none"])
@pytest.mark.parametrize("positional_parser", [False, True])
def test_expert_wrapper_preserves_prefill_forced_tool_parser_guard(
    pd, expert_patch, monkeypatch, role, choice, positional_parser
):
    monkeypatch.setattr(
        pd.OpenAIServingChatPatch,
        "chat_completion_full_generator",
        expert_patch.ExpertIdServingChatFinal.chat_completion_full_generator,
    )
    pd.envs.OMNI_PD_ROLE = role
    tool_choice = (
        pd.ChatCompletionNamedToolChoiceParam(function=NS(name="get_weather"))
        if choice == "named"
        else choice
    )

    class Request(NS):
        def model_copy(self, *, update, deep=False):
            return Request(**(vars(self) | update))

    request = Request(
        tool_choice=tool_choice, tools=[{"type": "function"}], max_tokens=1
    )
    parser = NS(parse=Mock(side_effect=ValueError("incomplete tool JSON")))
    serving = NS(parser_cls=object())
    final = _result(
        kv_transfer_params={"remote_decode": True} if role != "standalone" else None
    )
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
    kwargs = (
        {} if positional_parser else dict(parser=parser, mm_token_counts={"image": 5})
    )
    result = asyncio.run(
        pd.OpenAIServingChatPatch.chat_completion_full_generator(
            *args, *extra, **kwargs
        )
    )
    assert result is response
    assert request.tool_choice is tool_choice
    assert serving.parser_cls is not None
    assert result.choices[0].finish_reason == "tool_calls"
    parser.parse.assert_not_called()
    if final.kv_transfer_params is not None:
        assert final.kv_transfer_params["prompt_token_ids"] == [11, 12]
        assert final.kv_transfer_params["prefilled_token"] == [8]
