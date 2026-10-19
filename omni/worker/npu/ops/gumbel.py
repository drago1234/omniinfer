# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Shared Triton-Ascend Gumbel helpers for MRv2 NPU sampling."""

from vllm.triton_utils import tl, triton


@triton.jit
def _npu_gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    uniform_ptr,
    uniform_stride,
    uniform_row,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    # Callers promote logits to FP32; MRv2 temperature and uniforms are FP32.
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(
        temp_ptr + req_state_idx, mask=is_valid_req, other=0.0
    )
    if APPLY_TEMPERATURE:
        if temp != 0.0:
            # Match the behavior of the upstream _temperature_kernel.
            logits = logits / temp

    if processed_logits_ptr is not None:
        if processed_logits_col_ptr is not None:
            col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    if temp != 0.0:
        # Uniform rows follow the caller's sampling rows, not state slots:
        # token_idx for draft sampling, req_idx for rejected/bonus sampling.
        r = tl.load(
            uniform_ptr + uniform_row * uniform_stride + block,
            mask=mask & is_valid_req, other=0.5,
        )
        # Keep the original transform, including its endpoint handling.
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx
