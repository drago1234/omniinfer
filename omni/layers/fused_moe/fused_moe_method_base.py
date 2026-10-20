# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC, abstractmethod

import torch

from omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize import (
    CommunicationStrategySelector,
    FusedMoEPreparePermuteAndUnpermuteFinalize,
    PreparePermuteOptions,
    PreparePermuteResult,
)


def npu_grouped_moe_experts_bf16(
    hidden_states: torch.Tensor,
    experts: torch.nn.Module,
    expert_tokens: torch.Tensor,
    group_list_type: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply unquantized BF16 routed experts via grouped matmul.

    w13 grouped matmul -> swiglu -> w2 grouped matmul. ``group_list_type``
    picks the interpretation of ``expert_tokens``: 1 is per-expert counts
    (EP path), 0 is the cumulative sum (TP-only path). ``output_dtype`` is
    only forwarded when explicitly set, matching call sites that omit it.
    """
    # Resolve torch_npu at call time so tests that stub sys.modules
    # ["torch_npu"] after this module was imported still take effect.
    import torch_npu

    matmul_kwargs: dict = {
        "bias": None,
        "group_list": expert_tokens,
        "split_item": 3,
        "group_type": 0,
        "group_list_type": group_list_type,
    }
    if output_dtype is not None:
        matmul_kwargs["output_dtype"] = output_dtype

    gate_up_proj = torch_npu.npu_grouped_matmul(
        [hidden_states],
        [experts.w13_weight],
        **matmul_kwargs,
    )[0]
    intermediate_hidden_states = torch_npu.npu_swiglu(gate_up_proj)
    return torch_npu.npu_grouped_matmul(
        [intermediate_hidden_states],
        [experts.w2_weight],
        **matmul_kwargs,
    )[0]


class NPUFusedMoEMethodBase(ABC):
    def __init__(self):
        super().__init__()
        self.communication_strategy_selector = None

    def make_communication_strategy_selector(
        self,
        moe: torch.nn.Module,
    ):
        self.communication_strategy_selector = CommunicationStrategySelector(moe)

    def select_communication_strategy(self, num_tokens: int):
        return self.communication_strategy_selector.select_communication_strategy(num_tokens)

    def apply_prepare_permute(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        options: PreparePermuteOptions | None = None,
    ):
        return prepare_permute_and_unpermute_finalize.prepare_permute(layer, x, topk_ids, options)

    def prepare_finalize_metadata(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        topk_weights: torch.Tensor,
        prepare_permute_result: PreparePermuteResult,
    ):
        return prepare_permute_and_unpermute_finalize.prepare_finalize_metadata(
            layer, topk_weights, prepare_permute_result
        )

    def apply_unpermute_finalize(
        self,
        prepare_permute_and_unpermute_finalize: FusedMoEPreparePermuteAndUnpermuteFinalize,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        prepare_permute_result: PreparePermuteResult,
        finalize_params=None,
        finalize_metadata=None,
        post_finalize_callback=None,
    ):
        kwargs = {
            "finalize_params": finalize_params,
            "finalize_metadata": finalize_metadata,
        }
        # Only AGRS exposes the scheduling point. Other communication
        # strategies keep their existing unpermute_finalize signatures.
        if post_finalize_callback is not None:
            kwargs["post_finalize_callback"] = post_finalize_callback
        return prepare_permute_and_unpermute_finalize.unpermute_finalize(
            layer,
            hidden_states,
            topk_ids,
            topk_weights,
            prepare_permute_result,
            **kwargs,
        )

    @abstractmethod
    def apply_experts(
        self,
        layer: torch.nn.Module,
        prepare_permute_result: PreparePermuteResult,
        activation: str = "silu",
    ):
        raise NotImplementedError
