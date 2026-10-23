# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tests.unit.moe_layer_stub import as_moe_runner, moe_layer


pytestmark = pytest.mark.unit


def _make_module(monkeypatch, name, is_package=False):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def hifloat8_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "src"))

    torch_npu = _make_module(monkeypatch, "torch_npu")
    torch_npu.hifloat8 = object()

    _make_module(monkeypatch, "vllm", is_package=True)
    config_module = _make_module(monkeypatch, "vllm.config")
    config_module.get_current_vllm_config = lambda: SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace())
    )

    distributed_module = _make_module(monkeypatch, "vllm.distributed")
    distributed_module.get_tensor_model_parallel_rank = lambda: 0
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    distributed_module.tensor_model_parallel_all_gather = lambda x, dim=0: x
    distributed_module.tensor_model_parallel_all_reduce = lambda x: x

    forward_context_module = _make_module(monkeypatch, "vllm.forward_context")
    forward_context_module._fwctx = SimpleNamespace(additional_kwargs={})
    forward_context_module.get_forward_context = lambda: forward_context_module._fwctx
    forward_context_module.is_forward_context_available = lambda: True

    _make_module(monkeypatch, "vllm.utils", is_package=True)
    torch_utils_module = _make_module(monkeypatch, "vllm.utils.torch_utils")
    torch_utils_module.direct_register_custom_op = lambda **_kw: None

    logger_module = _make_module(monkeypatch, "vllm.logger")
    logger_module.init_logger = lambda _name: MagicMock()

    _make_module(monkeypatch, "vllm.model_executor", is_package=True)
    _make_module(monkeypatch, "vllm.model_executor.layers", is_package=True)

    fused_moe_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe", is_package=True
    )

    class FusedMoE:
        pass

    class FusedMoEMethodBase:
        def __init__(self, *_args, **_kwargs):
            pass

    class RoutedExperts:
        pass

    fused_moe_module.FusedMoE = FusedMoE
    fused_moe_module.FusedMoEMethodBase = FusedMoEMethodBase
    fused_moe_module.RoutedExperts = RoutedExperts
    fused_moe_module.FusedMoeWeightScaleSupported = SimpleNamespace(
        CHANNEL=SimpleNamespace(value="channel")
    )

    fused_moe_layer_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.fused_moe.layer"
    )
    fused_moe_layer_module.FusedMoeWeightScaleSupported = SimpleNamespace(
        CHANNEL=SimpleNamespace(value="channel")
    )

    linear_module = _make_module(monkeypatch, "vllm.model_executor.layers.linear")

    class LinearBase:
        pass

    class LinearMethodBase:
        pass

    class UnquantizedLinearMethod:
        pass

    linear_module.LinearBase = LinearBase
    linear_module.LinearMethodBase = LinearMethodBase
    linear_module.UnquantizedLinearMethod = UnquantizedLinearMethod

    quantization_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization", is_package=True
    )
    quantization_module.QuantizationMethods = str
    quantization_module.register_quantization_config = lambda _name: lambda cls: cls

    quantization_base_module = _make_module(
        monkeypatch, "vllm.model_executor.layers.quantization.base_config"
    )

    class QuantizationConfig:
        def __init__(self, *_args, **_kwargs):
            pass

    class QuantizeMethodBase:
        pass

    quantization_base_module.QuantizationConfig = QuantizationConfig
    quantization_base_module.QuantizeMethodBase = QuantizeMethodBase

    parameter_module = _make_module(monkeypatch, "vllm.model_executor.parameter")
    parameter_module.ChannelQuantScaleParameter = object
    parameter_module.ModelWeightParameter = object

    utils_module = _make_module(monkeypatch, "vllm.model_executor.utils")
    utils_module.set_weight_attrs = (
        lambda param, attrs: [setattr(param, key, value) for key, value in attrs.items()]
    )

    platforms_module = _make_module(monkeypatch, "vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    omni_pkg = _make_module(monkeypatch, "omni_npu", is_package=True)
    omni_pkg.__path__ = [str(repo_root / "omni")]
    layers_pkg = _make_module(monkeypatch, "omni_npu.layers", is_package=True)
    layers_pkg.__path__ = [str(repo_root / "omni" / "layers")]
    quant_pkg = _make_module(monkeypatch, "omni_npu.layers.quantization", is_package=True)
    quant_pkg.__path__ = [str(repo_root / "omni" / "layers" / "quantization")]

    fused_moe_pkg = _make_module(monkeypatch, "omni_npu.layers.fused_moe", is_package=True)
    # Point the package at disk so the (lightweight, torch-only) shared_expert
    # submodule imports for real; the other submodules stay stubbed via sys.modules.
    fused_moe_pkg.__path__ = [str(repo_root / "omni" / "layers" / "fused_moe")]
    fused_moe_config_module = _make_module(monkeypatch, "omni_npu.layers.fused_moe.config")
    fused_moe_config_module.hifloat8_moe_quant_config = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )

    npu_fused_moe_base_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe_method_base"
    )

    class NPUFusedMoEMethodBase:
        def __init__(self, *_args, **_kwargs):
            pass

    npu_fused_moe_base_module.NPUFusedMoEMethodBase = NPUFusedMoEMethodBase

    npu_fused_moe_layer_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.layer"
    )

    class NPUFusedMoE:
        pass

    npu_fused_moe_layer_module.NPUFusedMoE = NPUFusedMoE

    from dataclasses import dataclass, field
    from typing import Callable, Optional

    layer_utils_module = _make_module(monkeypatch, "omni_npu.layers.utils")
    layer_utils_module.named_stream = lambda _name: _DummyStream()
    layer_utils_module.CUBE_SIDE_TASKS_KEY = "cube_side_tasks"
    layer_utils_module.SIDE_STREAM_NAME = "side_stream"
    layer_utils_module.CUBE_SIDE_STREAM_NAME = layer_utils_module.SIDE_STREAM_NAME

    @dataclass
    class _CubeSideTask:
        fn: Callable[[], None]
        done_event: Optional[object] = None

    layer_utils_module.CubeSideTask = _CubeSideTask

    model_config_pkg = _make_module(
        monkeypatch, "omni_npu.model_config", is_package=True
    )
    config_loader_pkg = _make_module(
        monkeypatch, "omni_npu.model_config.config_loader", is_package=True
    )
    config_loader_module = _make_module(
        monkeypatch, "omni_npu.model_config.config_loader.loader"
    )
    config_loader_module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            shared_expert_multi_stream=True,
            shared_expert_parallel_schedule="with_finalize",
            enable_moe_init_routing_v2=False,
            enable_moe_init_routing_v3=False,
        ),
    )

    _make_module(monkeypatch, "omni_npu.v1", is_package=True)
    _make_module(monkeypatch, "omni_npu.v1.layers", is_package=True)
    v1_linear_module = _make_module(monkeypatch, "omni_npu.v1.layers.linear")

    class FlashCommLinearMethodBase:
        pass

    class FlashCommLinearBase:
        pass

    class UnquantizedFlashCommLinearMethod:
        pass

    v1_linear_module.FlashCommLinearMethodBase = FlashCommLinearMethodBase
    v1_linear_module.FlashCommLinearBase = FlashCommLinearBase
    v1_linear_module.UnquantizedFlashCommLinearMethod = (
        UnquantizedFlashCommLinearMethod
    )

    fused_mlp_module = _make_module(
        monkeypatch, "omni_npu.v1.layers.fused_mlp.layer"
    )

    class FusedMLP:
        pass

    class FusedMLPMethodBase:
        pass

    fused_mlp_module.FusedMLP = FusedMLP
    fused_mlp_module.FusedMLPMethodBase = FusedMLPMethodBase

    v1_utils_module = _make_module(monkeypatch, "omni_npu.v1.utils")
    v1_utils_module.on_ascend950 = lambda: False

    fused_moe_impl_module = _make_module(
        monkeypatch, "omni_npu.layers.fused_moe.fused_moe"
    )
    fused_moe_impl_module.fused_experts_tp = MagicMock()

    prefetch_module = _make_module(monkeypatch, "omni_npu.layers.prefetch")
    prefetch_module.PrefetchManager = MagicMock

    v1_layers_utils_module = _make_module(
        monkeypatch, "omni_npu.v1.layers.utils"
    )
    v1_layers_utils_module.get_npu_execution_type = lambda _label: _NoopStreamCtx()

    # omni_npu.layers.mhc — load the real cube_side_task_ops module from disk
    # so torch.ops.vllm.cube_side_run / cube_side_wait can be bound to its
    # Python implementations for the runner tests.
    mhc_pkg = _make_module(monkeypatch, "omni_npu.layers.mhc", is_package=True)
    mhc_pkg.__path__ = [str(repo_root / "omni" / "layers" / "mhc")]
    cube_side_ops_name = "omni_npu.layers.mhc.cube_side_task_ops"
    monkeypatch.delitem(sys.modules, cube_side_ops_name, raising=False)
    cube_side_ops_module = importlib.import_module(cube_side_ops_name)
    cube_side_ops_module = importlib.reload(cube_side_ops_module)

    # Bind torch.ops.vllm.cube_side_run / cube_side_wait to the Python impls
    # so Hifloat8 quant methods can call them through the torch.ops dispatch.
    torch_ops_vllm = SimpleNamespace(
        cube_side_run=cube_side_ops_module.cube_side_run,
        cube_side_wait=cube_side_ops_module.cube_side_wait,
    )
    monkeypatch.setattr(torch.ops, "vllm", torch_ops_vllm, raising=False)

    module_name = "omni_npu.layers.quantization.hifloat8"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    # Re-expose constants previously imported directly into hifloat8 — kept
    # accessible via the hifloat8 module to minimise test churn.
    module.CUBE_SIDE_TASKS_KEY = cube_side_ops_module.CUBE_SIDE_TASKS_KEY
    module.CUBE_SIDE_PENDING_KEY = cube_side_ops_module.CUBE_SIDE_PENDING_KEY
    module.CubeSideTask = cube_side_ops_module.CubeSideTask
    return module


@pytest.fixture
def mock_torch_npu(monkeypatch, hifloat8_module):
    ns = SimpleNamespace(
        hifloat8=object(),
        npu_trans_quant_param=MagicMock(side_effect=lambda x: x + 1),
        npu_grouped_matmul=MagicMock(),
        npu_swiglu=MagicMock(side_effect=lambda x: x[:, : x.shape[-1] // 2]),
        npu_dtype_cast=MagicMock(side_effect=lambda x, dtype: x.to(torch.int8)),
        npu_quant_matmul=MagicMock(
            side_effect=lambda x1, x2, scale, pertoken_scale, bias, x1_dtype,
            x2_dtype, output_dtype: torch.zeros(
                x1.shape[0], x2.shape[1], dtype=output_dtype
            )
        ),
    )
    monkeypatch.setattr(hifloat8_module, "torch_npu", ns)
    return ns


class _NoopStreamCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStream:
    def __init__(self):
        self.wait_stream_calls = []
        self.wait_event_calls = []

    def wait_stream(self, other):
        self.wait_stream_calls.append(other)

    def wait_event(self, event):
        self.wait_event_calls.append(event)


def _patch_npu_streams(monkeypatch):
    """Stub torch.npu.{Event,current_stream,stream} for tests touching side streams."""
    main_stream = _DummyStream()

    class _DummyEvent:
        def __init__(self):
            self.recorded = False
            self.waits = []

        def record(self):
            self.recorded = True

        def wait(self, stream):
            self.waits.append(stream)

    npu_ns = SimpleNamespace(
        Event=_DummyEvent,
        current_stream=lambda: main_stream,
        stream=lambda _stream: _NoopStreamCtx(),
    )
    monkeypatch.setattr(torch, "npu", npu_ns, raising=False)
    return main_stream


def _linear_layer(weight_shape=(2, 3), scale_shape=(2, 1)):
    layer = torch.nn.Module()
    weight = torch.arange(
        weight_shape[0] * weight_shape[1], dtype=torch.int64, device="cpu"
    ).reshape(weight_shape).to(torch.uint8)
    layer.weight = torch.nn.Parameter(
        weight,
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(scale_shape, dtype=torch.float32, device="cpu")
    )
    layer.orig_dtype = torch.bfloat16
    return layer


@pytest.mark.parametrize(
    ("patterns", "prefix", "expected"),
    [
        (["model.layer"], "model.layer", True),
        (["layer"], "model.layer", True),
        (["re:.*expert.*"], "model.expert.0", True),
        (["re:^expert"], "model.expert.0", False),
    ],
)
def test_is_layer_ignored(hifloat8_module, patterns, prefix, expected):
    assert hifloat8_module._is_layer_ignored(prefix, patterns) is expected


def test_hifloat8_config_metadata(hifloat8_module):
    config = hifloat8_module.Hifloat8Config.from_config({"ignore": ["lm_head"]})

    assert config.ignored_layers == ["lm_head"]
    assert hifloat8_module.Hifloat8Config.from_config({}).ignored_layers is None
    assert config.get_name() == "hifloat8"
    assert config.get_supported_act_dtypes() == [torch.bfloat16]
    assert config.get_config_filenames() == []
    with pytest.raises(NotImplementedError):
        config.get_min_capability()


def test_hifloat8_config_requires_custom_model_plugin(
    hifloat8_module, monkeypatch
):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    with pytest.raises(NotImplementedError):
        hifloat8_module.Hifloat8Config().get_quant_method(torch.nn.Module(), "x")


def test_hifloat8_config_routes_to_custom_method(hifloat8_module, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    config = hifloat8_module.Hifloat8Config()
    config.get_quant_method_custom = MagicMock(return_value="method")

    assert config.get_quant_method(torch.nn.Module(), "x") == "method"


def test_hifloat8_config_dispatches_supported_layers(hifloat8_module):
    from omni_npu.v1.layers.fused_mlp.layer import FusedMLP
    from omni_npu.v1.layers.linear import FlashCommLinearBase
    from vllm.model_executor.layers.linear import LinearBase

    config = hifloat8_module.Hifloat8Config()
    assert isinstance(
        config.get_quant_method_custom(FlashCommLinearBase(), "fc"),
        hifloat8_module.Hifloat8FCLinearMethod,
    )
    assert isinstance(
        config.get_quant_method_custom(LinearBase(), "linear"),
        hifloat8_module.Hifloat8LinearMethod,
    )
    assert isinstance(
        config.get_quant_method_custom(FusedMLP(), "mlp"),
        hifloat8_module.Hifloat8MlpMethod,
    )
    assert config.get_quant_method_custom(torch.nn.Module(), "unknown") is None


def test_hifloat8_config_dispatches_ignored_layers(hifloat8_module):
    from omni_npu.v1.layers.linear import (
        FlashCommLinearBase,
        UnquantizedFlashCommLinearMethod,
    )
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    config = hifloat8_module.Hifloat8Config(ignored_layers=["skip"])
    assert isinstance(
        config.get_quant_method_custom(FlashCommLinearBase(), "skip"),
        UnquantizedFlashCommLinearMethod,
    )
    assert isinstance(
        config.get_quant_method_custom(LinearBase(), "skip"),
        UnquantizedLinearMethod,
    )


@pytest.mark.parametrize(
    "method_class",
    ["Hifloat8LinearMethod", "Hifloat8FCLinearMethod"],
)
def test_linear_create_weights(hifloat8_module, monkeypatch, method_class):
    monkeypatch.setattr(
        hifloat8_module,
        "ModelWeightParameter",
        lambda data, **_kwargs: torch.nn.Parameter(data, requires_grad=False),
    )
    monkeypatch.setattr(
        hifloat8_module,
        "ChannelQuantScaleParameter",
        lambda data, **_kwargs: torch.nn.Parameter(data, requires_grad=False),
    )
    layer = torch.nn.Module()
    method = getattr(hifloat8_module, method_class)(object())

    method.create_weights(
        layer,
        input_size_per_partition=4,
        output_partition_sizes=[2, 3],
        input_size=4,
        output_size=5,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_args: None,
    )

    assert layer.weight.shape == (5, 4)
    assert layer.weight_scale.shape == (5, 1)
    assert layer.orig_dtype == torch.bfloat16


def test_hifloat8_mlp_pipeline(hifloat8_module, mock_torch_npu):
    method = hifloat8_module.Hifloat8MlpMethod(object())
    layer = moe_layer(
        gate_up_proj=MagicMock(return_value=(torch.ones(2, 4), None)),
        act_fn=MagicMock(return_value=torch.ones(2, 2)),
        down_proj=MagicMock(return_value=(torch.ones(2, 3), None)),
    )
    layer.gate_up_proj.weight_scale = torch.nn.Parameter(torch.ones(2))

    method.process_weights_after_loading(layer)
    output = method.apply(layer, torch.ones(2, 3), stream_label="mlp")

    assert output.shape == (2, 3)
    mock_torch_npu.npu_dtype_cast.assert_called()
    layer.gate_up_proj.assert_called_once()
    # The activation runs npu_swiglu + HiF8 cast directly instead of act_fn.
    layer.act_fn.assert_not_called()
    mock_torch_npu.npu_swiglu.assert_called_once()
    down_input = layer.down_proj.call_args.args[0]
    assert set(down_input) == {"x_hif8"}
    layer.down_proj.assert_called_once()


def test_linear_process_weights_transforms_scale(hifloat8_module, mock_torch_npu):
    layer = _linear_layer()
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    method.process_weights_after_loading(layer)

    mock_torch_npu.npu_trans_quant_param.assert_called_once()
    assert layer.weight.shape == (3, 2)
    assert torch.equal(layer.weight_scale, torch.full_like(layer.weight_scale, 2.0))
    assert layer.weight.requires_grad is False
    assert layer.weight_scale.requires_grad is False


def test_flashcomm_linear_process_weights_transforms_scale(
    hifloat8_module, mock_torch_npu
):
    layer = _linear_layer(weight_shape=(4, 2), scale_shape=(4, 1))
    method = hifloat8_module.Hifloat8FCLinearMethod(quant_config=object())

    method.process_weights_after_loading(layer)

    mock_torch_npu.npu_trans_quant_param.assert_called_once()
    assert layer.weight.shape == (2, 4)
    assert torch.equal(layer.weight_scale, torch.full_like(layer.weight_scale, 2.0))
    assert layer.weight.requires_grad is False
    assert layer.weight_scale.requires_grad is False


def test_moe_apply_experts_flattens_scale_and_uses_swiglu_cast(
    hifloat8_module, mock_torch_npu
):
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    layer = as_moe_runner(
        SimpleNamespace(
            w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
            w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
            w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
            w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        ),
        use_ep=False,
    )
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    result = method.apply_experts(layer, prepare_result)

    assert result.shape == (4, 4)
    assert mock_torch_npu.npu_grouped_matmul.call_count == 2
    first_kwargs = mock_torch_npu.npu_grouped_matmul.call_args_list[0].kwargs
    second_kwargs = mock_torch_npu.npu_grouped_matmul.call_args_list[1].kwargs
    assert first_kwargs["group_list_type"] == 0
    assert second_kwargs["group_list_type"] == 0
    assert first_kwargs["per_token_scale"][0].shape == (4,)
    assert second_kwargs["per_token_scale"] is None
    mock_torch_npu.npu_swiglu.assert_called_once()
    mock_torch_npu.npu_dtype_cast.assert_called_once()
    cast_args = mock_torch_npu.npu_dtype_cast.call_args.args
    assert cast_args[0].shape == (4, 4)
    assert cast_args[1] is mock_torch_npu.hifloat8


def test_moe_apply_experts_reads_weights_from_routed_experts(
    hifloat8_module, mock_torch_npu
):
    """vLLM 0.25 hands apply_experts the MoERunner; weights live on routed_experts."""
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    experts = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
    )
    runner = moe_layer(routed_experts=experts)
    assert not hasattr(runner, "w13_weight")
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(4, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=None,
    )
    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    method.apply_experts(runner, prepare_result)

    first, second = mock_torch_npu.npu_grouped_matmul.call_args_list
    assert first.args[1][0] is experts.w13_weight
    assert first.kwargs["scale"][0] is experts.w13_weight_scale
    assert second.args[1][0] is experts.w2_weight
    assert second.kwargs["scale"][0] is experts.w2_weight_scale


def test_linear_apply_with_dict_input_skips_dtype_cast(
    hifloat8_module, mock_torch_npu
):
    layer = moe_layer(
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())
    x_hif8 = torch.zeros(4, 3, dtype=torch.int8)

    out = method.apply(layer, x={"x_hif8": x_hif8}, bias=None)

    mock_torch_npu.npu_dtype_cast.assert_not_called()
    mock_torch_npu.npu_quant_matmul.assert_called_once()
    assert mock_torch_npu.npu_quant_matmul.call_args.kwargs["x1"] is x_hif8
    assert out.dtype == layer.orig_dtype


def test_linear_apply_with_tensor_input_casts_to_hifloat8(
    hifloat8_module, mock_torch_npu
):
    layer = moe_layer(
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())
    x = torch.ones(4, 3, dtype=torch.bfloat16)
    bias = torch.zeros(2, dtype=torch.bfloat16)

    method.apply(layer, x=x, bias=bias)

    mock_torch_npu.npu_dtype_cast.assert_called_once()
    cast_args = mock_torch_npu.npu_dtype_cast.call_args.args
    assert torch.equal(cast_args[0], x)
    assert cast_args[1] is mock_torch_npu.hifloat8
    qm_kwargs = mock_torch_npu.npu_quant_matmul.call_args.kwargs
    assert qm_kwargs["bias"].dtype == torch.float32


def test_moe_process_weights_keeps_float_scales_for_grouped_matmul(
    hifloat8_module, mock_torch_npu
):
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    layer = moe_layer(
        w13_weight=torch.nn.Parameter(
            torch.zeros(2, 8, 4, dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(2, 4, 8, dtype=torch.uint8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.ones(2, 8, 1, dtype=torch.float32), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.ones(2, 4, 1, dtype=torch.float32), requires_grad=False
        ),
        _ensure_moe_quant_config_init=MagicMock(),
    )

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 4, 8)
    assert layer.w2_weight.shape == (2, 8, 4)
    mock_torch_npu.npu_trans_quant_param.assert_not_called()
    assert layer.w13_weight_scale.shape == (2, 8)
    assert layer.w2_weight_scale.shape == (2, 4)
    assert layer.w13_weight_scale.dtype == torch.float32
    assert layer.w2_weight_scale.dtype == torch.float32
    layer._ensure_moe_quant_config_init.assert_called_once()


def test_mlp_process_weights_is_noop(hifloat8_module):
    method = hifloat8_module.Hifloat8MlpMethod(quant_config=object())
    original_scale = torch.nn.Parameter(torch.ones(4, 1, dtype=torch.bfloat16))
    layer = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight_scale=original_scale),
    )

    method.process_weights_after_loading(layer)

    assert layer.gate_up_proj.weight_scale is original_scale


def test_mlp_activation_uses_swiglu_then_hifloat8_cast(
    hifloat8_module, mock_torch_npu
):
    method = hifloat8_module.Hifloat8MlpMethod(quant_config=object())
    gate_up = torch.ones(2, 8, dtype=torch.bfloat16)

    result = method.apply_part2_activation_on_stream(
        SimpleNamespace(), gate_up
    )

    mock_torch_npu.npu_swiglu.assert_called_once()
    assert mock_torch_npu.npu_swiglu.call_args.args[0] is gate_up
    mock_torch_npu.npu_dtype_cast.assert_called_once()
    cast_input, cast_dtype = mock_torch_npu.npu_dtype_cast.call_args.args
    assert torch.equal(cast_input, gate_up[:, :4])
    assert cast_dtype is mock_torch_npu.hifloat8
    assert set(result) == {"x_hif8"}
    assert result["x_hif8"].dtype == torch.int8


def test_mlp_apply_forwards_stream_label_by_keyword(hifloat8_module):
    method = hifloat8_module.Hifloat8MlpMethod(quant_config=object())
    stream = object()
    x = torch.ones(2, 4)
    x_hif8 = torch.zeros(2, 4, dtype=torch.int8)
    x_scale = torch.ones(2)
    method.apply_quant = MagicMock(return_value=(x_hif8, x_scale))
    method.apply_part1_gate_up_on_stream = MagicMock(return_value=torch.ones(2, 8))
    method.apply_part2_activation_on_stream = MagicMock(
        return_value={"x_hif8": torch.zeros(2, 4, dtype=torch.int8)}
    )
    method.apply_part3_down_on_stream = MagicMock(return_value=torch.ones(2, 4))

    method.apply(SimpleNamespace(), x, stream_label=stream)

    method.apply_quant.assert_called_once_with(x, stream_label=stream)


def test_mlp_apply_quant_accepts_prequantized_hif8_mapping(hifloat8_module):
    method = hifloat8_module.Hifloat8MlpMethod(quant_config=object())
    x_hif8 = torch.zeros(2, 4, dtype=torch.int8)
    x_scale = torch.ones(2)

    result, scale = method.apply_quant(
        {"x_hif8": x_hif8, "pertoken_scale": x_scale}
    )

    assert result is x_hif8
    assert scale is x_scale


def test_mlp_apply_quant_rejects_invalid_prequantized_mapping(hifloat8_module):
    method = hifloat8_module.Hifloat8MlpMethod(quant_config=object())

    with pytest.raises(ValueError, match="x_hif8"):
        method.apply_quant({"x_int8": torch.zeros(2, 4, dtype=torch.int8)})


@pytest.mark.parametrize("tuple_output", [False, True])
def test_moe_apply_experts_with_routed_experts_cv_returns_tuple(
    hifloat8_module, mock_torch_npu, monkeypatch, tuple_output
):
    """Cover apply_experts side-stream branch when shared_expert_parallel_schedule == 'with_routed_experts_cv'."""
    main_stream = _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_routed_experts_cv",
    )

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()

    shared_experts = SimpleNamespace(
        act_fn=MagicMock(side_effect=lambda x: x),
        down_proj=MagicMock(return_value=torch.full((2, 4), 9.0, dtype=torch.bfloat16)),
    )
    if tuple_output:
        shared_experts.down_proj.return_value = (shared_experts.down_proj.return_value, None)
    layer = moe_layer(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        shared_experts=shared_experts,
        _shared_experts=SimpleNamespace(_layer=shared_experts),
    )

    shared_event = torch.npu.Event()
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
        shared_expert_gate_up=torch.ones(2, 4, dtype=torch.bfloat16),
        shared_expert_finished_event=shared_event,
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    result = method.apply_experts(layer, prepare_result)

    assert isinstance(result, tuple)
    routed_out, shared_out = result
    assert routed_out.shape == (4, 4)
    assert torch.equal(shared_out, torch.full((2, 4), 9.0, dtype=torch.bfloat16))
    # event waited on the main stream and the shared down_proj ran
    assert shared_event.waits == [main_stream]
    shared_experts.act_fn.assert_called_once()
    shared_experts.down_proj.assert_called_once()
    # main stream waits on the side stream twice (around down_proj and again at end)
    assert main_stream.wait_stream_calls.count(method.shared_experts_stream) >= 2


def test_moe_apply_experts_with_routed_experts_cv_grouped_finalize_routing(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """When use_grouped_matmul_finalize_routing=True, apply_experts returns early but
    shared expert side stream still produces results that we can't return; cover line 508."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_routed_experts_cv",
    )

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()

    shared_experts = SimpleNamespace(
        act_fn=MagicMock(side_effect=lambda x: x),
        down_proj=MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16)),
    )
    layer = moe_layer(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        shared_experts=shared_experts,
        _shared_experts=SimpleNamespace(_layer=shared_experts),
    )

    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(4, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=None,
        shared_expert_gate_up=torch.ones(2, 4, dtype=torch.bfloat16),
        shared_expert_finished_event=torch.npu.Event(),
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
    ]

    intermediate, pertoken_scale = method.apply_experts(
        layer, prepare_result, use_grouped_matmul_finalize_routing=True
    )

    assert intermediate.shape == (4, 4)
    assert pertoken_scale is None
    assert mock_torch_npu.npu_grouped_matmul.call_count == 1


def test_moe_init_records_shared_experts_stream_via_named_stream(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover Hifloat8MoEMethod.__init__ — specifically that shared_experts_stream
    is acquired from named_stream('sub_stream')."""
    captured = []

    def fake_named_stream(name):
        captured.append(name)
        return SimpleNamespace(label=name)

    monkeypatch.setattr(hifloat8_module, "named_stream", fake_named_stream)

    layer = moe_layer(
        moe_config=SimpleNamespace(num_experts=8),
        layer_name="layer_0",
    )
    method = hifloat8_module.Hifloat8MoEMethod(quant_config=object(), layer=layer)

    assert captured == ["sub_stream"]
    assert method.shared_experts_stream.label == "sub_stream"
    assert method.n_routed_experts == 8
    assert method.prefix == "layer_0"
    assert method.on_ascend950 is False
    assert method.model_prefetch is not None


def _build_apply_method(hifloat8_module, monkeypatch):
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.tp_size = 1
    method.tp_rank = 0
    method.shared_experts_stream = _DummyStream()
    method.select_communication_strategy = MagicMock(
        return_value=("agrs", SimpleNamespace())
    )
    method.apply_prepare_permute = MagicMock(
        return_value=SimpleNamespace(row_idx_type=0)
    )
    method.apply_unpermute_finalize = MagicMock(
        return_value=torch.zeros(2, 4, dtype=torch.bfloat16)
    )
    # Patch NPUFusedMoE.select_experts to avoid requiring full vllm gate.
    fake_npu_moe = SimpleNamespace(
        select_experts=lambda **_kwargs: (
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(hifloat8_module, "NPUFusedMoE", fake_npu_moe)
    return method


@pytest.mark.parametrize("with_shared", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("strategy", ["agrs", "all2all", "dispatch_combine"])
@pytest.mark.parametrize("custom_model", [False, True])
def test_moe_apply_non_ep_a5_uses_hifloat_tp_path(
    hifloat8_module, monkeypatch, with_shared, tp_size, strategy, custom_model
):
    # use_ep lives on layer.moe_config.moe_parallel_config under vLLM 0.25, and
    # the shared MLP on layer._shared_experts._layer (not the runner wrapper).
    _patch_npu_streams(monkeypatch)
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models" if custom_model else "omni_npu_patches")
    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.tp_size = tp_size
    method.tp_rank = tp_size - 1
    method.select_communication_strategy.return_value = (strategy, SimpleNamespace())
    method.on_ascend950 = True
    method.model_prefetch = MagicMock()
    # An odd token count also detects accidental padding before TP execution.
    hidden_states = torch.ones(3, 4)
    router_logits = torch.zeros(3, 3)
    topk_weights = torch.ones(3, 1)
    topk_ids = torch.zeros(3, 1, dtype=torch.int32)
    select_experts = MagicMock(return_value=(topk_weights, topk_ids))
    monkeypatch.setattr(hifloat8_module.NPUFusedMoE, "select_experts", select_experts)
    routed = torch.full((3, 4), 3.0)
    fused_tp = MagicMock(return_value=routed)
    monkeypatch.setattr(hifloat8_module, "fused_experts_tp", fused_tp)
    shared_mlp = MagicMock(return_value=torch.full((3, 4), 2.0))
    wrapper = MagicMock(spec=["__call__"])
    layer = as_moe_runner(
        SimpleNamespace(
            gate=None,
            shared_experts=wrapper if with_shared else None,
            _shared_experts=SimpleNamespace(_layer=shared_mlp) if with_shared else None,
        ),
        use_ep=False,
    )

    result = method.apply(
        layer=layer,
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=1,
        renormalize=False,
    )

    fused_tp.assert_called_once()
    assert fused_tp.call_args.kwargs["x"] is hidden_states
    assert fused_tp.call_args.kwargs["topk_ids"] is topk_ids
    assert fused_tp.call_args.kwargs["topk_weights"] is topk_weights
    assert select_experts.call_args.kwargs["router_logits"] is router_logits
    method.apply_prepare_permute.assert_not_called()
    if with_shared:
        shared_mlp.assert_called_once_with(hidden_states)
        wrapper.assert_not_called()
        shared_out, model_output = result
        assert torch.equal(shared_out, shared_mlp.return_value)
        # Built-in Pangu adds shared_out in its caller; custom models consume
        # the combined second result directly. Both must add it exactly once.
        total = model_output if custom_model else model_output + shared_out
        assert torch.equal(total, routed + shared_out)
    else:
        assert result is routed


@pytest.mark.parametrize("schedule", ["with_finalize", "with_routed_experts_cv"])
def test_hifloat_tp_executes_real_expert_pipeline_without_agrs_events(
    hifloat8_module, mock_torch_npu, monkeypatch, schedule
):
    _patch_npu_streams(monkeypatch)
    config = hifloat8_module.model_extra_config
    monkeypatch.setattr(config, "dtype", torch.bfloat16, raising=False)
    monkeypatch.setattr(config.operator_opt_config, "shared_expert_parallel_schedule", schedule)
    monkeypatch.setattr(config.operator_opt_config, "enable_moe_init_routing_v2", True, raising=False)
    prepare = _make_module(monkeypatch, "omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize")
    # This is the TP result contract: no AGRS shared-event or gate-up fields.
    prepare.PreparePermuteResult = lambda **kw: SimpleNamespace(avg_tokens_per_expert=None, **kw)
    prepare._uses_hifloat_activation = lambda cfg: cfg.use_hifloat8_w8a8
    prepare._uses_mxfp_activation = lambda cfg: False
    prepare._sorted_topk_weights = MagicMock()
    module_path = Path(__file__).resolve().parents[3] / "omni/layers/fused_moe/fused_moe.py"
    spec = importlib.util.spec_from_file_location("hifloat_tp_pipeline_under_test", module_path)
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    monkeypatch.setattr(fused, "torch_npu", mock_torch_npu)
    routed = torch.full((3, 4), 3.0, dtype=torch.bfloat16)
    mock_torch_npu.npu_moe_init_routing_v2 = MagicMock(return_value=(
        torch.ones(3, 4, dtype=torch.bfloat16), torch.arange(3), torch.tensor([3]), None,
    ))
    mock_torch_npu.npu_moe_finalize_routing = MagicMock(side_effect=lambda x, **kw: x.squeeze(0))
    mock_torch_npu.npu_grouped_matmul.side_effect = [[torch.ones(3, 8)], [routed]]
    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()
    method.moe_quant_config = SimpleNamespace(use_hifloat8_w8a8=True)
    experts = SimpleNamespace(
        global_num_experts=1, quant_method=method, quant_config=object(),
        activation=SimpleNamespace(value="silu"), w13_weight=torch.ones(1), w2_weight=torch.ones(1),
        w13_weight_scale=torch.ones(1), w2_weight_scale=torch.ones(1),
    )
    layer = as_moe_runner(experts, use_ep=False)
    activate_shared = MagicMock(side_effect=AssertionError("TP must not consume AGRS shared metadata"))
    monkeypatch.setattr(hifloat8_module, "activate_shared_expert_on_side_stream", activate_shared)

    output = fused.fused_experts_tp(
        layer, torch.ones(3, 4), torch.zeros(3, 1, dtype=torch.int32), torch.ones(3, 1),
    )

    assert torch.equal(output, routed)
    assert mock_torch_npu.npu_grouped_matmul.call_count == 2
    mock_torch_npu.npu_moe_finalize_routing.assert_called_once()
    activate_shared.assert_not_called()


def test_moe_apply_with_routed_experts_cv_unpacks_tuple(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply lines 430-431: shared_expert_parallel_schedule == 'with_routed_experts_cv'
    consumes a (routed_output, shared_output) tuple from apply_experts."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_routed_experts_cv",
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    routed = torch.zeros(2, 4, dtype=torch.bfloat16)
    shared = torch.full((2, 4), 7.0, dtype=torch.bfloat16)
    method.apply_experts = MagicMock(return_value=(routed, shared))

    layer = moe_layer(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(tp_size=1),
        ),
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    out = method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    # Tuple (shared_output, routed_output) returned because shared_output is not None.
    assert isinstance(out, tuple)
    shared_out, routed_out = out
    assert torch.equal(shared_out, shared)
    assert torch.equal(routed_out, method.apply_unpermute_finalize.return_value)


def test_moe_apply_default_with_finalize_runs_shared_experts_locally(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply lines 433-438, 440 default branch (with_finalize) and 446-450 all-reduce."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_finalize",
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.apply_experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))

    shared_call = MagicMock(return_value=torch.full((2, 4), 5.0, dtype=torch.bfloat16))
    shared_experts = MagicMock(side_effect=shared_call)
    shared_experts.gate_up_proj = SimpleNamespace(tp_size=2)

    all_reduce_calls = []

    def fake_all_reduce(t):
        all_reduce_calls.append(t)
        return t * 2

    monkeypatch.setattr(
        hifloat8_module, "tensor_model_parallel_all_reduce", fake_all_reduce
    )

    layer = moe_layer(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=shared_experts,
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    out = method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    # tp_size > 1 path: shared_experts(hidden_states) called and result all-reduced.
    shared_experts.assert_called_once()
    assert torch.equal(shared_experts.call_args.args[0], hidden)
    assert len(all_reduce_calls) == 1
    assert torch.equal(all_reduce_calls[0], shared_call.return_value)
    assert isinstance(out, tuple)


def test_moe_apply_default_with_finalize_shared_tp1_uses_x_slice(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """Cover apply line 440: tp_size == 1 branch invokes shared_experts(x_slice)."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_finalize",
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.apply_experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))

    shared_experts = MagicMock(return_value=torch.full((2, 4), 3.0, dtype=torch.bfloat16))
    shared_experts.gate_up_proj = SimpleNamespace(tp_size=1)

    layer = moe_layer(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=shared_experts,
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    shared_experts.assert_called_once()
    # x_slice == hidden_states when tp_size == 1 (no slicing needed).
    assert torch.equal(shared_experts.call_args.args[0], hidden)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_moe_apply_without_multi_stream_runs_shared_mlp_on_main_stream(
    hifloat8_module, mock_torch_npu, monkeypatch, tp_size
):
    """Multi-stream off: the shared MLP from _shared_experts._layer runs inline."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_multi_stream",
        False,
    )
    monkeypatch.setattr(
        hifloat8_module, "tensor_model_parallel_all_reduce", lambda t: t
    )

    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.apply_experts = MagicMock(return_value=torch.zeros(2, 4, dtype=torch.bfloat16))

    shared_mlp = MagicMock(return_value=torch.full((2, 4), 3.0, dtype=torch.bfloat16))
    shared_mlp.gate_up_proj = SimpleNamespace(tp_size=tp_size)
    wrapper = MagicMock(spec=["__call__"])
    layer = moe_layer(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None),
        shared_experts=wrapper,
        _shared_experts=SimpleNamespace(_layer=shared_mlp),
    )
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    out = method.apply(
        layer=layer,
        hidden_states=hidden,
        router_logits=None,
        top_k=1,
        renormalize=False,
    )

    shared_mlp.assert_called_once()
    wrapper.assert_not_called()
    # hidden_states and x_slice coincide here (no slicing), so both branches see it.
    assert torch.equal(shared_mlp.call_args.args[0], hidden)
    assert isinstance(out, tuple)
    assert torch.equal(out[0], shared_mlp.return_value)


def test_moe_apply_experts_default_with_finalize_no_shared_path(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    """When schedule is 'with_finalize' (default), apply_experts must not touch shared_experts."""
    _patch_npu_streams(monkeypatch)
    monkeypatch.setattr(
        hifloat8_module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule",
        "with_finalize",
    )

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.shared_experts_stream = _DummyStream()

    layer = moe_layer(
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
        shared_experts=SimpleNamespace(act_fn=MagicMock(), down_proj=MagicMock()),
    )
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )

    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    result = method.apply_experts(layer, prepare_result)

    assert not isinstance(result, tuple)
    assert result.shape == (4, 4)
    layer.shared_experts.act_fn.assert_not_called()
    layer.shared_experts.down_proj.assert_not_called()


@pytest.mark.parametrize("shared_tp", [1, 2])
@pytest.mark.parametrize("prefetch", [False, True])
def test_agrs_overlap_launches_shared_between_finalize_and_reduce_scatter(
    hifloat8_module, mock_torch_npu, monkeypatch, shared_tp, prefetch
):
    main_stream = _patch_npu_streams(monkeypatch)
    config = hifloat8_module.model_extra_config.operator_opt_config
    monkeypatch.setattr(config, "shared_expert_parallel_schedule", "with_finalize")
    monkeypatch.setattr(config, "shared_expert_multi_stream", True)
    monkeypatch.setattr(config, "enable_prefetch", prefetch, raising=False)
    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.enable_agrs_finalize_metadata_overlap = True
    method.model_prefetch = MagicMock()
    events = []
    metadata = object()

    def prepare(*args):
        events.append("metadata")
        return metadata

    def experts(**kwargs):
        events.append("experts")
        return torch.zeros(2, 4, dtype=torch.bfloat16)

    def shared(x):
        events.append("shared")
        return x * 3

    def finalize(*args, finalize_metadata=None, post_finalize_callback=None, **kwargs):
        assert finalize_metadata is metadata
        assert main_stream.wait_stream_calls[-1] is method.shared_experts_stream
        events.append("finalize")
        post_finalize_callback()
        events.append("reduce_scatter")
        return torch.zeros(2, 4, dtype=torch.bfloat16)

    method.prepare_finalize_metadata = MagicMock(side_effect=prepare)
    method.apply_experts = MagicMock(side_effect=experts)
    method.apply_unpermute_finalize = MagicMock(side_effect=finalize)
    shared_experts = MagicMock(side_effect=shared)
    shared_experts.gate_up_proj = SimpleNamespace(tp_size=shared_tp)
    layer = moe_layer(
        gate=lambda x: (torch.zeros(x.shape[0], 4), None), shared_experts=shared_experts,
    )
    output = method.apply(
        layer=layer, hidden_states=torch.ones(2, 4, dtype=torch.bfloat16),
        router_logits=None, top_k=1, renormalize=False,
    )
    assert events == ["metadata", "experts", "finalize", "shared", "reduce_scatter"]
    shared_experts.assert_called_once()
    assert isinstance(output, tuple)
    assert method.model_prefetch.prefetch.call_count == (2 if prefetch else 0)


def _set_cube_side_tasks(hifloat8_module, tasks):
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    fwctx.additional_kwargs = {hifloat8_module.CUBE_SIDE_TASKS_KEY: tasks}


def test_linear_apply_runs_registered_cube_side_task(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = moe_layer(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    _set_cube_side_tasks(hifloat8_module, {layer.prefix: task})
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    out = method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16))

    assert flag["called"] is True
    assert task.done_event is not None and task.done_event.recorded
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY] == {}
    mock_torch_npu.npu_quant_matmul.assert_called_once()
    assert out.dtype == layer.orig_dtype


def test_linear_apply_skips_cube_side_task_when_prefix_not_registered(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    _set_cube_side_tasks(hifloat8_module, {"some.other.layer": task})
    layer = moe_layer(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16))

    assert flag["called"] is False
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert "some.other.layer" in fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY]


def test_linear_apply_no_op_when_no_cube_side_tasks_dict(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    fwctx.additional_kwargs = {}
    layer = moe_layer(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    method = hifloat8_module.Hifloat8LinearMethod(quant_config=object())

    method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16))

    mock_torch_npu.npu_quant_matmul.assert_called_once()


def test_apply_experts_runs_registered_cube_side_task(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = moe_layer(
        layer_name="layers.0.mlp.experts",
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
    )
    _set_cube_side_tasks(hifloat8_module, {layer.layer_name: task})
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )
    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
        [torch.zeros(4, 4, dtype=torch.bfloat16)],
    ]

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.apply_experts(layer, prepare_result)

    assert flag["called"] is True
    assert task.done_event is not None and task.done_event.recorded
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY] == {}


def test_apply_experts_skips_cube_side_task_in_grouped_matmul_finalize_routing(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)
    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = moe_layer(
        layer_name="layers.0.mlp.experts",
        w13_weight=torch.zeros(2, 4, 8, dtype=torch.uint8),
        w13_weight_scale=torch.ones(2, 8, dtype=torch.float32),
        w2_weight=torch.zeros(2, 4, 4, dtype=torch.uint8),
        w2_weight_scale=torch.ones(2, 4, dtype=torch.float32),
    )
    _set_cube_side_tasks(hifloat8_module, {layer.layer_name: task})
    prepare_result = SimpleNamespace(
        hidden_states_sorted_by_experts=torch.zeros(2, 2, 4, dtype=torch.int8),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        avg_tokens_per_expert=[2],
        dynamic_scale=torch.ones(4, 1, dtype=torch.float32),
    )
    mock_torch_npu.npu_grouped_matmul.side_effect = [
        [torch.zeros(4, 8, dtype=torch.bfloat16)],
    ]

    method = hifloat8_module.Hifloat8MoEMethod.__new__(hifloat8_module.Hifloat8MoEMethod)
    method.apply_experts(layer, prepare_result, use_grouped_matmul_finalize_routing=True)

    assert flag["called"] is False
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert layer.layer_name in fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY]


def test_flashcomm_apply_runs_registered_cube_side_task(
    hifloat8_module, mock_torch_npu, monkeypatch
):
    _patch_npu_streams(monkeypatch)

    # Hifloat8FCLinearMethod.apply imports these lazily; stub the module so the
    # default (x_transform=None) path imports without the real comm extension.
    v1_distributed = types.ModuleType("omni_npu.v1.distributed")
    v1_distributed.__path__ = []
    comm_ext = types.ModuleType("omni_npu.v1.distributed.communication_op_ext")
    comm_ext.layer_parallel_all_gather = lambda x, *a, **k: x
    comm_ext.layer_parallel_all2all_single = lambda x, *a, **k: x
    monkeypatch.setitem(sys.modules, "omni_npu.v1.distributed", v1_distributed)
    monkeypatch.setitem(
        sys.modules, "omni_npu.v1.distributed.communication_op_ext", comm_ext
    )

    flag = {"called": False}

    def side_fn():
        flag["called"] = True

    task = hifloat8_module.CubeSideTask(fn=side_fn)
    layer = moe_layer(
        prefix="layers.0.self_attn.o_proj",
        weight=torch.zeros(3, 2, dtype=torch.uint8),
        weight_scale=torch.ones(2, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
    )
    _set_cube_side_tasks(hifloat8_module, {layer.prefix: task})
    method = hifloat8_module.Hifloat8FCLinearMethod(quant_config=object())

    out = method.apply(layer, x=torch.ones(4, 3, dtype=torch.bfloat16), bias=None)

    # layer_key resolves from layer.prefix; the registered cube-side task fires
    # via torch.ops.vllm.cube_side_run and is awaited via cube_side_wait.
    assert flag["called"] is True
    assert task.done_event is not None and task.done_event.recorded
    fwctx = sys.modules["vllm.forward_context"]._fwctx
    assert fwctx.additional_kwargs[hifloat8_module.CUBE_SIDE_TASKS_KEY] == {}
    mock_torch_npu.npu_quant_matmul.assert_called_once()
    assert out.dtype == layer.orig_dtype


@pytest.mark.parametrize("force_balance", [None, False, True])
@pytest.mark.parametrize("use_ep", [False, True])
def test_moe_apply_forwards_force_load_balance(
    hifloat8_module, monkeypatch, force_balance, use_ep
):
    _patch_npu_streams(monkeypatch)
    method = _build_apply_method(hifloat8_module, monkeypatch)
    method.on_ascend950 = True
    opts = hifloat8_module.model_extra_config.operator_opt_config
    monkeypatch.setattr(opts, "shared_expert_multi_stream", False)
    monkeypatch.setattr(opts, "enable_prefetch", False, raising=False)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    select = MagicMock(return_value=(weights, ids))
    monkeypatch.setattr(hifloat8_module.NPUFusedMoE, "select_experts", select)
    layer = as_moe_runner(SimpleNamespace(gate=None), use_ep=use_ep)
    if force_balance is not None:
        layer.use_moe_force_load_balance = force_balance
    method.apply_experts = MagicMock(return_value=torch.zeros(2, 4))
    fused_tp = hifloat8_module.fused_experts_tp
    fused_tp.return_value = torch.zeros(2, 4)

    method.apply(
        layer=layer, hidden_states=torch.ones(2, 4),
        router_logits=torch.zeros(2, 4), top_k=2, renormalize=False,
    )

    assert select.call_args.kwargs["force_load_balance"] is (force_balance is True)
    if use_ep:
        assert method.apply_prepare_permute.call_args.args[3] is ids
        assert method.apply_unpermute_finalize.call_args.args[3] is ids
        assert method.apply_unpermute_finalize.call_args.args[4] is weights
    else:
        assert fused_tp.call_args.kwargs["topk_ids"] is ids
        assert fused_tp.call_args.kwargs["topk_weights"] is weights
