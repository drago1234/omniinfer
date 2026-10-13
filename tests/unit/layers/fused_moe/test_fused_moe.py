# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _moe_layer(**kwargs):
    layer = SimpleNamespace(**kwargs)
    if getattr(layer, "routed_experts", None) is None:
        layer.routed_experts = layer
    if getattr(layer, "activation", None) is None:
        layer.activation = SimpleNamespace(value="silu")
    return layer


@pytest.fixture
def fused_module(monkeypatch):
    torch_npu = MagicMock()
    torch_npu.npu_moe_gating_top_k_softmax = MagicMock()
    torch_npu.npu_moe_finalize_routing = MagicMock(side_effect=lambda *args, **kwargs: args[0])
    torch_npu.npu_moe_init_routing = MagicMock()
    torch_npu.npu_moe_init_routing_v2 = MagicMock()
    torch_npu.npu_moe_compute_expert_tokens = MagicMock()
    torch_npu.npu_dynamic_quant = MagicMock()
    torch_npu.npu_dynamic_mx_quant = MagicMock()
    torch_npu.npu_dtype_cast = MagicMock()
    torch_npu.hifloat8 = object()

    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")

    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.get_ep_group = lambda: SimpleNamespace(world_size=1)
    distributed_module.get_dp_group = lambda: SimpleNamespace(
        world_size=1,
        all_gather=lambda tensor, dim=0: tensor,
        reduce_scatter=lambda tensor, dim=0: tensor,
    )
    distributed_module.get_tp_group = lambda: SimpleNamespace(all_reduce=lambda x: x)
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1

    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)

    orig_arange = torch.arange
    orig_tensor = torch.tensor
    orig_ones = torch.ones
    orig_zeros = torch.zeros
    orig_full = torch.full

    def _coerce_device(kwargs):
        dev = kwargs.get("device")
        if dev is None:
            try:
                if orig_tensor([]).device.type == "npu":
                    kwargs["device"] = "cpu"
            except Exception:
                pass
            return
        if isinstance(dev, torch.device):
            if dev.type == "npu":
                kwargs["device"] = "cpu"
        elif dev == "npu":
            kwargs["device"] = "cpu"

    def _safe_arange(*args, **kwargs):
        _coerce_device(kwargs)
        return orig_arange(*args, **kwargs)

    def _safe_tensor(*args, **kwargs):
        _coerce_device(kwargs)
        return orig_tensor(*args, **kwargs)

    def _safe_ones(*args, **kwargs):
        _coerce_device(kwargs)
        return orig_ones(*args, **kwargs)

    def _safe_zeros(*args, **kwargs):
        _coerce_device(kwargs)
        return orig_zeros(*args, **kwargs)

    def _safe_full(*args, **kwargs):
        _coerce_device(kwargs)
        return orig_full(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", _safe_arange)
    monkeypatch.setattr(torch, "tensor", _safe_tensor)
    monkeypatch.setattr(torch, "ones", _safe_ones)
    monkeypatch.setattr(torch, "zeros", _safe_zeros)
    monkeypatch.setattr(torch, "full", _safe_full)
    sys.modules.pop("omni_npu.layers.fused_moe.fused_moe", None)
    module = importlib.import_module("omni_npu.layers.fused_moe.fused_moe")
    importlib.reload(module)

    stubs = SimpleNamespace(torch_npu=torch_npu)
    return module, stubs


@pytest.mark.unit
def test_fused_topk_renormalizes(fused_module):
    module, stubs = fused_module
    weights = torch.ones(2, 2)
    stubs.torch_npu.npu_moe_gating_top_k_softmax.return_value = (
        weights.clone(),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.zeros(2, 2, dtype=torch.int32),
    )

    topk_weights, topk_ids, row_idx = module.fused_topk(torch.zeros(2, 4), topk=2, renormalize=True)

    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(2))
    assert topk_ids.shape == (2, 2)
    assert row_idx.shape == (2, 2)


@pytest.mark.unit
def test_grouped_topk_sigmoid_and_bias(fused_module):
    module, _ = fused_module
    gating_output = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
    bias = torch.tensor([0.1, 0.2, 0.3, 0.4])

    weights, ids, row_idx = module.grouped_topk(
        gating_output=gating_output,
        topk=1,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )

    assert weights.shape == (1, 1)
    assert ids.dtype == torch.int32
    assert row_idx.shape == (1, 1)


@pytest.mark.unit
def test_grouped_topk_invalid_scoring_func_raises(fused_module):
    module, _ = fused_module
    with pytest.raises(ValueError, match="Unsupported scoring function"):
        module.grouped_topk(
            gating_output=torch.zeros(1, 2),
            topk=1,
            renormalize=False,
            num_expert_group=1,
            topk_group=1,
            scoring_func="invalid",
        )


@pytest.mark.unit
def test_fused_experts_tp_no_quant(fused_module):
    module, stubs = fused_module
    layer = _moe_layer(
        global_num_experts=4,
        quant_config=None,
        quant_method=MagicMock(),
    )
    sorted_tokens = torch.ones(3, 2)
    expanded_src_to_dst_row = torch.zeros(3, 1, dtype=torch.int32)
    expanded_expert_idx = torch.zeros(3, 1, dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing.return_value = (
        sorted_tokens,
        expanded_src_to_dst_row,
        expanded_expert_idx,
    )
    stubs.torch_npu.npu_moe_compute_expert_tokens.return_value = torch.tensor([1, 2])
    layer.quant_method.apply_experts.return_value = torch.full((3, 2), 5.0)
    stubs.torch_npu.npu_moe_finalize_routing.side_effect = None
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.full((3, 2), 6.0)

    output = module.fused_experts_tp(
        layer=layer,
        x=torch.ones(3, 2),
        topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        topk_weights=torch.ones(3, 1),
    )

    stubs.torch_npu.npu_moe_init_routing.assert_called_once()
    stubs.torch_npu.npu_moe_init_routing_v2.assert_not_called()
    stubs.torch_npu.npu_moe_compute_expert_tokens.assert_called_once()
    layer.quant_method.apply_experts.assert_called_once()
    finalize_call = stubs.torch_npu.npu_moe_finalize_routing.call_args
    assert "drop_pad_mode" not in finalize_call.kwargs
    assert torch.equal(output, torch.full((3, 2), 6.0))


@pytest.mark.unit
def test_fused_experts_tp_uses_routing_v2(fused_module, monkeypatch):
    module, stubs = fused_module
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "enable_moe_init_routing_v2",
        True,
    )
    layer = _moe_layer(
        global_num_experts=4,
        quant_config=object(),
        quant_method=MagicMock(),
    )
    layer.quant_method.moe_quant_config = SimpleNamespace(
        use_hifloat8_w8a8=False,
        use_mxfp8_w8a8=False,
        use_mxfp4_w4a8=False,
    )
    x = torch.ones(3, 2)
    topk_ids = torch.zeros(3, 1, dtype=torch.int64)
    topk_weights = torch.ones(3, 1)
    sorted_tokens = torch.ones(3, 2)
    expanded_src_to_dst_row = torch.tensor(
        [[0], [1], [2]],
        dtype=torch.int32,
    )
    expert_tokens = torch.tensor([1, 2, 3, 3], dtype=torch.int64)
    routing_map = torch.empty(0, dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        sorted_tokens,
        expanded_src_to_dst_row,
        expert_tokens,
        routing_map,
    )
    stubs.torch_npu.npu_dynamic_quant.return_value = (
        sorted_tokens,
        torch.ones(3),
    )
    expert_output = torch.full((3, 2), 5.0)
    layer.quant_method.apply_experts.return_value = expert_output
    finalized_output = torch.full((3, 2), 6.0)
    stubs.torch_npu.npu_moe_finalize_routing.side_effect = None
    stubs.torch_npu.npu_moe_finalize_routing.return_value = finalized_output

    output = module.fused_experts_tp(
        layer=layer,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )

    stubs.torch_npu.npu_moe_init_routing_v2.assert_called_once()
    stubs.torch_npu.npu_moe_init_routing.assert_not_called()
    stubs.torch_npu.npu_moe_compute_expert_tokens.assert_not_called()
    stubs.torch_npu.npu_dynamic_quant.assert_called_once_with(sorted_tokens)
    stubs.torch_npu.npu_dynamic_mx_quant.assert_not_called()
    stubs.torch_npu.npu_dtype_cast.assert_not_called()

    init_call = stubs.torch_npu.npu_moe_init_routing_v2.call_args
    assert init_call.args == (x,)
    assert init_call.kwargs["expert_num"] == 4
    assert init_call.kwargs["active_expert_range"] == [0, 4]
    assert init_call.kwargs["expert_tokens_num_type"] == 0
    assert init_call.kwargs["expert_tokens_num_flag"] is True
    assert init_call.kwargs["active_num"] == topk_ids.numel()
    assert init_call.kwargs["drop_pad_mode"] == 0
    assert init_call.kwargs["row_idx_type"] == 0
    assert init_call.kwargs["quant_mode"] == -1
    assert init_call.kwargs["expert_idx"].dtype == torch.int32

    apply_call = layer.quant_method.apply_experts.call_args
    prepare_result = apply_call.kwargs["prepare_permute_result"]
    assert torch.equal(
        prepare_result.hidden_states_sorted_by_experts,
        sorted_tokens,
    )
    assert torch.equal(prepare_result.expert_tokens, expert_tokens)

    finalize_call = stubs.torch_npu.npu_moe_finalize_routing.call_args
    assert torch.equal(finalize_call.args[0], expert_output.unsqueeze(0))
    assert finalize_call.kwargs["skip1"] is None
    assert finalize_call.kwargs["skip2"] is None
    assert finalize_call.kwargs["bias"] is None
    assert finalize_call.kwargs["drop_pad_mode"] == 3
    assert torch.equal(
        finalize_call.kwargs["scales"],
        topk_weights.to(expert_output.dtype),
    )
    assert torch.equal(
        finalize_call.kwargs["expanded_src_to_dst_row"],
        expanded_src_to_dst_row,
    )
    assert finalize_call.kwargs["export_for_source_row"].dtype == torch.int32
    assert torch.equal(
        output,
        finalized_output.to(module.model_extra_config.dtype),
    )


@pytest.mark.unit
def test_fused_experts_tp_with_quant(fused_module):
    module, stubs = fused_module
    layer = _moe_layer(
        global_num_experts=4,
        quant_config=object(),
        quant_method=MagicMock(),
    )
    # Real MoE quant configs carry bool flags; a bare MagicMock attribute is truthy.
    layer.quant_method.moe_quant_config.use_mxfp8_w8a8 = False
    layer.quant_method.moe_quant_config.use_mxfp4_w4a8 = False
    layer.quant_method.moe_quant_config.use_hifloat8_w8a8 = False
    sorted_tokens = torch.ones(3, 2)
    expanded_src_to_dst_row = torch.zeros(3, 1, dtype=torch.int32)
    expanded_expert_idx = torch.zeros(3, 1, dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing.return_value = (
        sorted_tokens,
        expanded_src_to_dst_row,
        expanded_expert_idx,
    )
    stubs.torch_npu.npu_moe_compute_expert_tokens.return_value = torch.tensor([1, 2])
    stubs.torch_npu.npu_dynamic_quant.return_value = (sorted_tokens, torch.ones(3))
    layer.quant_method.apply_experts.return_value = torch.full((3, 2), 5.0)
    stubs.torch_npu.npu_moe_finalize_routing.side_effect = None
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.full((3, 2), 6.0)

    output = module.fused_experts_tp(
        layer=layer,
        x=torch.ones(3, 2),
        topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        topk_weights=torch.ones(3, 1),
    )

    layer.quant_method.apply_experts.assert_called_once()
    stubs.torch_npu.npu_dynamic_quant.assert_called_once()
    assert torch.equal(output, torch.full((3, 2), 6.0))


@pytest.mark.unit
def test_fused_experts_tp_with_mxfp_quant(fused_module):
    """MXFP8 activations use the training-aligned scale algorithm."""
    module, stubs = fused_module
    experts = _moe_layer(
        global_num_experts=4,
        quant_config=object(),
        quant_method=MagicMock(),
    )
    layer = SimpleNamespace(routed_experts=experts)
    experts.quant_method.moe_quant_config.use_mxfp8_w8a8 = True
    experts.quant_method.moe_quant_config.use_hifloat8_w8a8 = False
    sorted_tokens = torch.ones(3, 2)
    expanded_src_to_dst_row = torch.zeros(3, 1, dtype=torch.int32)
    expanded_expert_idx = torch.zeros(3, 1, dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing.return_value = (
        sorted_tokens,
        expanded_src_to_dst_row,
        expanded_expert_idx,
    )
    stubs.torch_npu.npu_moe_compute_expert_tokens.return_value = torch.tensor([1, 2])
    stubs.torch_npu.npu_dynamic_mx_quant.return_value = (
        sorted_tokens,
        torch.ones(3),
    )
    experts.quant_method.apply_experts.return_value = torch.full((3, 2), 5.0)
    stubs.torch_npu.npu_moe_finalize_routing.side_effect = None
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.full((3, 2), 6.0)

    output = module.fused_experts_tp(
        layer=layer,
        x=torch.ones(3, 2),
        topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        topk_weights=torch.ones(3, 1),
    )

    experts.quant_method.apply_experts.assert_called_once()
    stubs.torch_npu.npu_dynamic_mx_quant.assert_called_once()
    _, kwargs = stubs.torch_npu.npu_dynamic_mx_quant.call_args
    assert kwargs["scale_alg"] == 1
    stubs.torch_npu.npu_dynamic_quant.assert_not_called()
    assert torch.equal(output, torch.full((3, 2), 6.0))


@pytest.mark.unit
def test_fused_experts_tp_with_hifloat8_casts_without_scale(fused_module):
    """HiF8 TP casts routed activations and does not create a per-token scale."""
    module, stubs = fused_module
    experts = _moe_layer(
        global_num_experts=4,
        quant_config=object(),
        quant_method=MagicMock(),
    )
    layer = SimpleNamespace(routed_experts=experts)
    experts.quant_method.moe_quant_config.use_hifloat8_w8a8 = True
    experts.quant_method.moe_quant_config.use_mxfp8_w8a8 = False
    experts.quant_method.moe_quant_config.use_mxfp4_w4a8 = False
    sorted_tokens = torch.ones(3, 2)
    cast_tokens = torch.zeros(3, 2, dtype=torch.int8)
    stubs.torch_npu.npu_moe_init_routing.return_value = (
        sorted_tokens,
        torch.zeros(3, 1, dtype=torch.int32),
        torch.zeros(3, 1, dtype=torch.int32),
    )
    stubs.torch_npu.npu_moe_compute_expert_tokens.return_value = torch.tensor([1, 2])
    stubs.torch_npu.npu_dtype_cast.return_value = cast_tokens
    experts.quant_method.apply_experts.return_value = torch.full((3, 2), 5.0)
    stubs.torch_npu.npu_moe_finalize_routing.side_effect = None
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.full((3, 2), 6.0)

    module.fused_experts_tp(
        layer=layer,
        x=torch.ones(3, 2),
        topk_ids=torch.zeros(3, 1, dtype=torch.int32),
        topk_weights=torch.ones(3, 1),
    )

    stubs.torch_npu.npu_dtype_cast.assert_called_once_with(
        sorted_tokens, stubs.torch_npu.hifloat8
    )
    stubs.torch_npu.npu_dynamic_quant.assert_not_called()
    stubs.torch_npu.npu_dynamic_mx_quant.assert_not_called()
    prepare_result = experts.quant_method.apply_experts.call_args.kwargs[
        "prepare_permute_result"
    ]
    assert torch.equal(prepare_result.hidden_states_sorted_by_experts, cast_tokens)
    assert prepare_result.dynamic_scale is None
