# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Latent expert dimensions, clamp numerics and BF16 communication paths."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from omni_npu.v1.models.pangu import pangu_v2_moe as model

pytestmark = pytest.mark.unit


@pytest.fixture
def build_moe(monkeypatch, default_vllm_config):
    group = SimpleNamespace(
        world_size=1,
        rank_in_group=0,
        rank=lambda: 0,
        size=lambda: 1,
        all_gather=lambda x, dim=0: x,
        all_reduce=lambda x: x,
        reduce_scatter=lambda x, dim=0: x,
        _get_backend=lambda device: SimpleNamespace(get_hccl_comm_name=lambda rank: "test"),
    )
    group.device_group = group
    monkeypatch.setattr(model, "get_tp_group", lambda: group)
    monkeypatch.setattr(model, "get_ep_group", lambda: group)
    monkeypatch.setattr(model, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(model, "on_ascend950", lambda: False)
    monkeypatch.setattr(model, "on_ascend910b", lambda: False)
    monkeypatch.setattr(model.OpenPanguV2MOE, "_is_graph_mode", property(lambda self: False))
    monkeypatch.setitem(sys.modules, "omni_training_custom_ops", SimpleNamespace())

    def swiglu_group(x, *, weight, group_index, clamp_limit):
        assert group_index is None
        gate, up = x.chunk(2, dim=-1)
        output = (
            torch.nn.functional.silu(gate.float().clamp(max=clamp_limit))
            * up.float().clamp(-clamp_limit, clamp_limit)
        )
        if weight is not None:
            output = output * weight
        return output.to(x.dtype)

    fused_activation = Mock(side_effect=swiglu_group)
    monkeypatch.setattr(torch.ops.custom, "npu_ai_infra_swiglu_group", fused_activation, raising=False)

    class Linear(nn.Linear):
        def __init__(self, size, output, return_bias=True, **kwargs):
            super().__init__(size, sum(output) if isinstance(output, list) else output, bias=False)
            self.return_bias = return_bias

        def forward(self, x):
            value = super().forward(x)
            return (value, None) if self.return_bias else value

    for name in ("ReplicatedLinear", "MergedColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(model, name, Linear)
    monkeypatch.setattr(model, "RMSNorm", lambda size, eps: nn.RMSNorm(size, eps=eps))

    def fused(**kwargs):
        width, inter = kwargs["hidden_size"], kwargs["intermediate_size"]
        experts = SimpleNamespace(
            w13_weight=torch.randn(1, width, 2 * inter),
            w2_weight=torch.randn(1, inter, width),
            top_k=1,
            topk_group=1,
            num_expert_group=1,
        )
        return SimpleNamespace(routed_experts=experts)

    monkeypatch.setattr(model, "FusedMoE", fused)
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )

    def build(latent=2, norm=True, limit=10, quant=None):
        config = SimpleNamespace(
            hidden_size=4,
            n_routed_experts=1,
            n_shared_experts=1,
            hidden_act="silu",
            routed_scaling_factor=1.0,
            moe_intermediate_size=3,
            num_experts_per_tok=1,
            norm_topk_prob=True,
            routed_expert_hidden_size=latent,
            latent_moe_use_norm=norm,
            swiglu_limit=limit,
            rms_norm_eps=1e-5,
        )
        module = model.OpenPanguV2MOE(config, parallel, quant_config=quant, prefix="model.layers.0.mlp")
        module.fused_activation_calls = fused_activation
        return module

    return build


@pytest.mark.parametrize(
    "latent,norm,limit",
    [(None, False, None), (None, False, 0), (None, True, 10), (2, False, None), (2, True, -1), (2, True, 10)],
)
def test_configuration_and_projection(build_moe, monkeypatch, latent, norm, limit):
    def swiglu(x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    monkeypatch.setattr(model.torch_npu, "npu_swiglu", swiglu)
    moe = build_moe(latent, norm, limit)
    assert moe.gate.in_features == moe.shared_experts.gate_up_proj.in_features == 4
    assert moe.experts.w13_weight.shape == (1, latent or 4, 6)
    assert isinstance(moe.shared_experts.act_fn, model.SiluAndMulWithClamp) == (limit is not None and limit > 0)
    if limit is None or limit <= 0:
        packed = torch.tensor([[-20., 20., 3., -4.]])
        torch.testing.assert_close(moe.act_fn(packed), swiglu(packed))
    x = torch.randn(3, 4)
    y = moe._project_routed_input(x)
    if latent is None:
        assert y is x and moe._project_routed_output(y) is x
    else:
        expected = x @ moe.routed_expert_down_proj.weight.T
        torch.testing.assert_close(y, expected)
        if norm:
            expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-5)
        expected = expected @ moe.routed_expert_up_proj.weight.T
        torch.testing.assert_close(moe._project_routed_output(y), expected)


@pytest.mark.parametrize(
    "latent,limit,quant,error",
    [
        (0, 0, None, ValueError),
        (2, 0, object(), NotImplementedError),
        (None, 10, object(), NotImplementedError),
    ],
)
def test_unsupported_configuration(build_moe, latent, limit, quant, error):
    with pytest.raises(error):
        build_moe(latent=latent, limit=limit, quant=quant)


@pytest.mark.parametrize("mtp", [False, True])
def test_latent_weight_loading(build_moe, monkeypatch, mtp):
    from omni_npu.v1.models.pangu import pangu_v2_moe_mtp as mtp_model

    module = mtp_model if mtp else model
    cls = mtp_model.OpenPanguV2MTP if mtp else model.OpenPanguV2ForCausalLM
    wrapper = cls.__new__(cls)
    nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(
        n_routed_experts=1,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    wrapper.num_redundant_experts = 0
    wrapper.model = nn.Module()
    wrapper.model.layers = nn.ModuleList([nn.Module()])
    wrapper.model.layers[0].mlp = build_moe()
    monkeypatch.setattr(module, "fused_moe_make_expert_params_mapping", lambda *args, **kw: [])
    monkeypatch.setattr(module, "is_pp_missing_parameter", lambda *args: False, raising=False)
    monkeypatch.setattr(module, "maybe_remap_kv_scale_name", lambda name, params: name, raising=False)
    if mtp:
        wrapper.model.mtp_start_layer_idx = 0
        monkeypatch.setattr(wrapper, "get_spec_layer", lambda name: 0)
        monkeypatch.setattr(wrapper, "_rewrite_spec_layer_name", lambda index, name: name)
    weights = {name: torch.randn_like(param) for name, param in wrapper.named_parameters() if "routed_expert_" in name}
    assert len(weights) == 3
    assert wrapper.load_weights(weights.items()) == set(weights)
    for name, param in wrapper.named_parameters():
        if name in weights:
            torch.testing.assert_close(param, weights[name])
    packed = wrapper.model.layers[0].mlp.shared_experts.gate_up_proj.weight
    packed.weight_loader = lambda param, value, shard: param.data.chunk(2)[shard].copy_(value)
    prefix = "model.layers.0.mlp.shared_experts"
    halves = {f"{prefix}.{name}.weight": torch.randn(3, 4) for name in ("gate_proj", "up_proj")}
    assert wrapper.load_weights(halves.items()) == {f"{prefix}.gate_up_proj.weight"}
    torch.testing.assert_close(packed, torch.cat(list(halves.values())))


@pytest.mark.parametrize(
    "path,latent", [("allreduce", None), ("allreduce", 2), ("allgather", 2), ("dispatch", 2), ("all2allv", 2)]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_bf16_paths(build_moe, monkeypatch, path, latent, enabled):
    monkeypatch.setattr(model.model_extra_config.operator_opt_config, "use_swiglu_group_fusion_op", enabled)
    torch.manual_seed(5)
    moe = build_moe(latent=latent)
    moe.bfloat16()
    moe.gate.float()
    moe.e_score_correction_bias.data = moe.e_score_correction_bias.data.float()
    moe.experts.w13_weight = moe.experts.w13_weight.bfloat16()
    moe.experts.w2_weight = moe.experts.w2_weight.bfloat16()
    x = torch.randn(3, 4, dtype=torch.bfloat16) * 20
    shared = moe.shared_experts(x)
    routed_input = moe._project_routed_input(x)
    packed = routed_input @ moe.experts.w13_weight[0]
    gate, up = packed.chunk(2, dim=-1)
    if enabled:
        gate, up = gate.float(), up.float()
    gate, up = gate.clamp(max=10), up.clamp(-10, 10)
    expected = gate * torch.sigmoid(gate) * up
    probabilities = torch.tensor([[0.2], [0.4], [0.8]])
    if enabled:
        expected = expected * probabilities
    expected = expected.to(x.dtype) @ moe.experts.w2_weight[0]
    if not enabled:
        expected = (expected.float() * probabilities).to(x.dtype)
    expected = moe._project_routed_output(expected) + shared
    seen = []
    moe.gate.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    moe.shared_experts.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    counts = torch.tensor([len(x)], dtype=torch.int64)
    indices = torch.arange(len(x), dtype=torch.int32)
    ops = model.torch_npu
    monkeypatch.setattr(
        ops,
        "npu_moe_gating_top_k",
        lambda *args, **kw: (
            probabilities,
            torch.zeros(len(x), 1, dtype=torch.int32),
            None,
        ),
    )
    def init_routing(value, *args, **kwargs):
        order = torch.arange(len(value) - 1, -1, -1, dtype=torch.int32)
        return value.flip(0), order, torch.tensor([len(value)]), None

    monkeypatch.setattr(ops, "npu_moe_init_routing_v2", init_routing)
    monkeypatch.setattr(ops, "npu_grouped_matmul", lambda inputs, weights, **kw: [inputs[0] @ weights[0][0]])

    def finalize(value, *args, **kwargs):
        scales = kwargs["scales"] if "scales" in kwargs else args[3]
        torch.testing.assert_close(scales, torch.ones_like(probabilities) if enabled else probabilities)
        return (value.reshape(len(x), -1).flip(0).float() * scales).to(value.dtype)

    monkeypatch.setattr(ops, "npu_moe_finalize_routing", finalize)
    def dispatch(**kwargs):
        num_tokens = kwargs["x"].shape[0]
        batch_counts = torch.tensor([num_tokens], dtype=torch.int64)
        scales = kwargs.get("expert_scales")
        return (
            kwargs["x"].flip(0), None, torch.arange(num_tokens, dtype=torch.int32),
            batch_counts, batch_counts, batch_counts,
            scales.flatten().flip(0) if scales is not None else torch.empty(0),
        )

    monkeypatch.setattr(ops, "npu_moe_distribute_dispatch_v2", dispatch)

    def combine(**kwargs):
        scales = kwargs["expert_scales"]
        assert bool(torch.all(scales == 1)) == enabled
        value = kwargs["expand_x"]
        return (value.flip(0).float() * scales).to(value.dtype)

    monkeypatch.setattr(ops, "npu_moe_distribute_combine_v2", combine)
    monkeypatch.setattr(
        ops, "npu_moe_re_routing", lambda value, *args, **kw: (value.flip(0), None, indices.flip(0), counts)
    )
    monkeypatch.setattr(model.dist, "all_to_all_single", lambda output, value, *args, **kw: output.copy_(value))
    monkeypatch.setattr(moe, "_get_mc2_mask", lambda value: None)
    moe.moe_comm_strategy = {
        "allreduce": "allreduce",
        "allgather": "allgather_reducescatter",
        "dispatch": "dispatch_combine",
        "all2allv": "all2allv",
    }[path]
    if path == "dispatch":
        # Exercise the chunked dispatch branch as well as projection placement.
        moe.moe_dispatch_combine_max_batch_size = 2
    # Exercise strategy selection too: TP=1 must retain latent projections and clamp.
    moe.fused_activation_calls.reset_mock()
    actual = moe._forward_single(x)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)
    assert moe.fused_activation_calls.call_count == ((3 if path == "dispatch" else 2) if enabled else 0)
    if enabled:
        routed_calls = [call for call in moe.fused_activation_calls.call_args_list if call.kwargs["weight"] is not None]
        assert len(routed_calls) == (2 if path == "dispatch" else 1)
        if path == "dispatch":
            torch.testing.assert_close(routed_calls[0].kwargs["weight"].flatten(), probabilities[:2].flatten().flip(0))
            torch.testing.assert_close(routed_calls[1].kwargs["weight"].flatten(), probabilities[2:].flatten())
        else:
            torch.testing.assert_close(
                routed_calls[0].kwargs["weight"].flatten(),
                probabilities.flatten() if path == "all2allv" else probabilities.flatten().flip(0),
            )
    assert len(seen) == 2
    torch.testing.assert_close(seen[0], x.float())
    torch.testing.assert_close(seen[1], x)


@pytest.mark.parametrize(
    "enabled,limit,alpha,beta",
    [(True, 10.0, 1.0, 0.0), (False, 10.0, 1.0, 0.0), (True, None, 1.0, 0.0), (True, 10.0, 1.702, 1.0)],
)
def test_swiglu_group_switch_and_clamp_semantics(monkeypatch, default_vllm_config, enabled, limit, alpha, beta):
    from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig, model_extra_config

    assert not ModelOperatorOptConfig().use_swiglu_group_fusion_op
    monkeypatch.setattr(
        model_extra_config, "operator_opt_config", ModelOperatorOptConfig(use_swiglu_group_fusion_op=enabled)
    )
    monkeypatch.setitem(sys.modules, "omni_training_custom_ops", SimpleNamespace())
    x = torch.tensor([[-20.0, 2.0, 20.0, -30.0, 30.0, -3.0]])
    gate, up = x.chunk(2, dim=-1)
    native = Mock(side_effect=lambda value: torch.nn.functional.silu(value[..., :3]) * value[..., 3:])
    monkeypatch.setattr(model.torch_npu, "npu_swiglu", native)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    expected = gate * torch.sigmoid(alpha * gate) * (up + beta)
    fused = Mock(return_value=expected)
    monkeypatch.setattr(torch.ops.custom, "npu_ai_infra_swiglu_group", fused, raising=False)
    activation = model.SiluAndMulWithClamp(limit, alpha, beta) if limit is not None else model.torch_npu.npu_swiglu
    actual = activation(x)
    torch.testing.assert_close(actual, expected)
    if enabled and limit is not None and alpha == 1.0 and beta == 0.0:
        fused.assert_called_once_with(x, weight=None, group_index=None, clamp_limit=limit)
        fused.reset_mock()
        empty = activation(x[:0])
        assert empty.shape == (0, 3)
        fused.assert_not_called()
    else:
        fused.assert_not_called()
    if limit is None:
        native.assert_called_once_with(x)
    else:
        native.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_swiglu_expanded_topk_weights_with_remote_experts(build_moe, monkeypatch, enabled):
    monkeypatch.setattr(model.model_extra_config.operator_opt_config, "use_swiglu_group_fusion_op", enabled)
    torch.manual_seed(19)
    moe = build_moe()
    moe.physical_expert_start, moe.physical_expert_end, moe.n_physical_experts = 1, 3, 3
    moe.experts.w13_weight = torch.randn(2, 2, 6)
    moe.experts.w2_weight = torch.randn(2, 3, 2)
    moe.experts.top_k = 2
    x = torch.randn(3, 4)
    ids = torch.tensor([[0, 1], [2, 1], [0, 2]], dtype=torch.int32)
    probabilities = torch.tensor([[.11, .22], [.33, .44], [.55, .66]])
    # Original token/expert pairs -> local expert rows; remote expert entries are -1.
    indices = torch.tensor([-1, 0, 2, 1, -1, 3], dtype=torch.int32)
    counts = torch.tensor([2, 2])
    monkeypatch.setattr(model.torch_npu, "npu_moe_gating_top_k", lambda *args, **kw: (probabilities, ids, None))

    def init_routing(value, *args, **kwargs):
        sorted_tokens = torch.zeros(6, value.shape[-1])
        sorted_tokens[:4] = value[torch.tensor([0, 1, 1, 2])]
        return sorted_tokens, indices, counts, None

    def grouped_matmul(inputs, weights, **kwargs):
        result = inputs[0].new_zeros((6, weights[0].shape[-1]))
        for expert in range(2):
            rows = slice(2 * expert, 2 * expert + 2)
            result[rows] = inputs[0][rows] @ weights[0][expert]
        return [result]

    def finalize(value, *args, **kwargs):
        scales = args[3]
        torch.testing.assert_close(scales, torch.ones_like(probabilities) if enabled else probabilities)
        value = value.flatten(1)
        result = torch.zeros(3, 2)
        for pair, row in enumerate(indices):
            if row >= 0:
                result[pair // 2] += value[row] * scales.flatten()[pair]
        return result

    monkeypatch.setattr(model.torch_npu, "npu_moe_init_routing_v2", init_routing)
    monkeypatch.setattr(model.torch_npu, "npu_grouped_matmul", grouped_matmul)
    monkeypatch.setattr(model.torch_npu, "npu_moe_finalize_routing", finalize)
    routed = moe._project_routed_input(x)
    expected = torch.zeros(3, 2)
    for token in range(3):
        for k in range(2):
            expert = int(ids[token, k]) - 1
            if expert < 0:
                continue
            gate, up = (routed[token] @ moe.experts.w13_weight[expert]).chunk(2)
            activated = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            if enabled:
                activated = activated * probabilities[token, k]
            output = activated @ moe.experts.w2_weight[expert]
            expected[token] += output if enabled else output * probabilities[token, k]
    expected = moe._project_routed_output(expected) + moe.shared_experts(x)
    torch.testing.assert_close(moe._forward_allgather(x, use_allreduce=True), expected)
