# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.utils import on_ascend950


def shared_expert_core_limit(enabled: bool = True):
    """Give shared experts all Cube cores while dispatch/combine uses Vector."""
    if (
        enabled
        and model_extra_config.task_config.graph_mode == "acl_graph"
        and on_ascend950()
    ):
        return torch.npu.npugraph_ex.scope.limit_core_num(32, 8)
    return nullcontext()


def activate_shared_expert_on_side_stream(
    side_stream,
    layer: torch.nn.Module,
    prepare_permute_result,
    quant_fn: Callable[[torch.Tensor], object],
):
    """Activate + quantize the shared-expert gate_up projection on a side stream.

    Used by the ``with_routed_experts_cv`` schedule, where the shared expert
    runs concurrently with the routed-expert cube ops. Waits on the
    prepare/permute finished event, then applies ``act_fn`` followed by the
    caller-supplied ``quant_fn`` on ``side_stream``.

    ``quant_fn`` keeps this helper agnostic of the quantization scheme:
    HiFloat8 casts the activation, MXFP8 dynamically quantizes it (returning a
    scale alongside the activation). Whatever ``quant_fn`` returns is returned
    verbatim.
    """
    prepare_permute_result.shared_expert_finished_event.wait(torch.npu.current_stream())
    side_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(side_stream):
        shared_expert_gate_up = prepare_permute_result.shared_expert_gate_up
        if isinstance(shared_expert_gate_up, tuple):
            # Keep compatibility with producers that still return the
            # FlashComm (output, bias) pair.
            shared_expert_gate_up = shared_expert_gate_up[0]
        shared_expert_act = layer._shared_experts._layer.act_fn(shared_expert_gate_up)
        return quant_fn(shared_expert_act)
