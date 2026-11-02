# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional

import torch
import torch_npu
from vllm.platforms import current_platform
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import (
    _uses_hifloat_activation,
    _uses_mxfp_activation,
    _sorted_topk_weights,
)


def _apply_experts(
    layer: torch.nn.Module, h: torch.Tensor, expert_tokens: torch.Tensor,
    pertoken_scale: torch.Tensor | None = None, expanded_topk_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import PreparePermuteResult

    experts = layer.routed_experts
    return experts.quant_method.apply_experts(
        layer=layer,
        prepare_permute_result=PreparePermuteResult(
            hidden_states_sorted_by_experts=h,
            expert_tokens=expert_tokens,
            dynamic_scale=pertoken_scale,
            expanded_topk_weights=expanded_topk_weights,
        ),
        activation=experts.activation.value,
    )


def fused_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(gating_output, k=topk)

    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, row_idx


def grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
):
    gating_output = gating_output.float()
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = topk_ids.int()
    # adapt add row_idx
    row_idx = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=topk_ids.dtype)
    row_idx = row_idx.reshape(topk_ids.shape[1], topk_ids.shape[0]).transpose(1, 0).contiguous()
    # adapt end

    return topk_weights, topk_ids, row_idx


def fused_experts_tp(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
):
    experts = layer.routed_experts
    n_routed_experts = experts.global_num_experts
    use_routing_v2 = (model_extra_config.operator_opt_config.enable_moe_init_routing_v2)
    use_routing_v3 = (model_extra_config.operator_opt_config.enable_moe_init_routing_v3)

    if use_routing_v2 or use_routing_v3:
        topk_ids = topk_ids.int()
        routing_input = x
        if use_routing_v3 and input_scale is None:
            routing_input, input_scale = torch_npu.npu_dynamic_quant(x)
        (
            sorted_tokens,
            expanded_src_to_dst_row,
            expert_tokens,
            pertoken_scale,
        ) = torch_npu.npu_moe_init_routing_v2(
            routing_input,
            expert_idx=topk_ids,
            expert_num=n_routed_experts,
            active_expert_range=[0, n_routed_experts],
            expert_tokens_num_type=0,
            expert_tokens_num_flag=True,
            active_num=topk_ids.numel(),
            drop_pad_mode=0,
            row_idx_type=0,
            quant_mode=-1,
            scale=input_scale,
        )
        expert_tokens = expert_tokens.to(torch.int64)
    else:
        row_idx = (
            torch.arange(
                topk_ids.numel(),
                device=current_platform.device_type,
                dtype=torch.int32,
            )
            .view(-1, x.shape[0])
            .transpose(0, 1)
        )
        num_tokens = x.shape[0]
        (
            sorted_tokens,
            expanded_src_to_dst_row,
            expanded_expert_idx,
        ) = torch_npu.npu_moe_init_routing(
            x,
            row_idx,
            topk_ids,
            num_tokens,
        )
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx,
            n_routed_experts,
        ).to(torch.int64)

    if experts.quant_config is None:
        pertoken_scale = None
    elif not use_routing_v3:
        moe_quant_config = getattr(experts.quant_method, "moe_quant_config", None)
        if _uses_hifloat_activation(moe_quant_config):
            sorted_tokens = torch_npu.npu_dtype_cast(sorted_tokens, torch_npu.hifloat8)
            sorted_tokens = sorted_tokens.view(dtype=torch.int8)
            pertoken_scale = None
        elif _uses_mxfp_activation(moe_quant_config):
            sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                sorted_tokens,
                dst_type=torch.float8_e4m3fn,
                scale_alg=1
            )
        else:
            sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)

    expanded_topk_weights = None
    if layer.use_swiglu_group_fusion:
        # V1 numbers source rows in top-k-major order; V2 uses token-major order.
        source_topk_weights = topk_weights if use_routing_v2 else topk_weights.T.contiguous()
        expanded_topk_weights = _sorted_topk_weights(source_topk_weights, expanded_src_to_dst_row)
        topk_weights = torch.ones_like(topk_weights)
    out = _apply_experts(layer, sorted_tokens, expert_tokens, pertoken_scale, expanded_topk_weights)

    if use_routing_v2 or use_routing_v3:
        return torch_npu.npu_moe_finalize_routing(
            out.unsqueeze(0),
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights.to(out.dtype),
            expanded_src_to_dst_row=expanded_src_to_dst_row,
            export_for_source_row=topk_ids,
            drop_pad_mode=3,
        ).to(model_extra_config.dtype)

    return torch_npu.npu_moe_finalize_routing(
        out, None, None, None, topk_weights, expanded_src_to_dst_row, topk_ids
    ).to(model_extra_config.dtype)


def fused_experts_tp_with_shared(
    layer: torch.nn.Module,
    x_slice: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    shared_experts,
):
    routed_output = fused_experts_tp(
        layer=layer,
        x=x_slice,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    if shared_experts is not None:
        shared_output = shared_experts(x_slice)
        return shared_output, routed_output + shared_output
    return routed_output
