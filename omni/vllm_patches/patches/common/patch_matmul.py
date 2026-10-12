# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Patch PyTorch matrix operations with precision-consistency NPU operators."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

logger = logging.getLogger(__name__)


class _NpuAiInfraMatmul(torch.autograd.Function):
    """Add first-order autograd support to npu_ai_infra_matmul."""

    @staticmethod
    def forward(ctx, input_tensor, other):
        ctx.save_for_backward(input_tensor, other)
        return _npu_ai_infra_matmul(input_tensor, other)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, other = ctx.saved_tensors
        input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = _npu_ai_infra_matmul(
                grad_output_2d, other.transpose(0, 1)
            ).reshape_as(input_tensor)

        grad_other = None
        if ctx.needs_input_grad[1]:
            grad_other = _npu_ai_infra_matmul(
                input_2d.transpose(0, 1), grad_output_2d
            )

        return grad_input, grad_other


def _npu_ai_infra_matmul(input_tensor, other):
    """Call the forward-only NPU matmul op for supported input shapes."""
    if input_tensor.numel() == 0:
        return input_tensor.new_zeros((*input_tensor.shape[:-1], other.shape[1]))

    if input_tensor.dim() == 2:
        output = torch.ops.custom.npu_ai_infra_matmul(input_tensor, other)
    else:
        batch_size, seq_length, hidden_size = input_tensor.shape
        flattened_input = input_tensor.reshape(batch_size * seq_length, hidden_size)
        output = torch.ops.custom.npu_ai_infra_matmul(flattened_input, other)
        output = output.reshape(batch_size, seq_length, other.shape[1])
    return output


def matmul_npu(input_tensor, other, *, out=None):
    if out is not None:
        raise ValueError("torch.ops.custom.npu_ai_infra_matmul does not support out")
    if other.dim() != 2:
        raise ValueError(
            "torch.ops.custom.npu_ai_infra_matmul requires a 2-D right-hand tensor, "
            f"but got other.dim()={other.dim()}"
        )
    if input_tensor.dim() not in (2, 3):
        raise ValueError(
            "torch.ops.custom.npu_ai_infra_matmul supports only 2-D or 3-D left-hand "
            f"tensors, but got input_tensor.dim()={input_tensor.dim()}"
        )
    return _NpuAiInfraMatmul.apply(input_tensor, other)


def linear_npu(input_tensor, weight, bias=None):
    output = matmul_npu(input_tensor, weight.t())
    if bias is not None:
        output = output + bias
    return output


def _custom_op_available() -> bool:
    try:
        import omni_custom_ops  # noqa: F401
        return getattr(torch.ops.custom, "npu_ai_infra_matmul", None) is not None
    except (ModuleNotFoundError, AttributeError, RuntimeError):
        return False


def patch_matmul() -> None:
    """Replace PyTorch matmul and linear with the NPU implementations.

    This function is intentionally not registered as a vLLM patch. Call it
    after model extra config is loaded when precision strong consistency is
    enabled.
    """
    if not _custom_op_available():
        logger.error("torch.ops.custom.npu_ai_infra_matmul is unavailable; skip matmul patch")
        return

    if torch.matmul is torch._C._VariableFunctions.matmul:
        torch.matmul = matmul_npu
        logger.info("Installed patch: torch.matmul -> matmul_npu")
    else:
        logger.warning("Skip matmul_npu patch because torch.matmul is already patched.")

    if F.linear is torch._C._nn.linear:
        F.linear = linear_npu
        logger.info("Installed patch: F.linear -> linear_npu")
    else:
        logger.warning("Skip linear_npu patch because F.linear is already patched.")
