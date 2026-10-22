# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""FP32 Murmur3 Gumbel helpers from vLLM 52a65d7b19e9.

Source: vllm/v1/worker/gpu/sample/gumbel.py, commit
52a65d7b19e9e3ad86bf1e3b6954e4e11eeec7da.
The RNG/noise helpers add unsigned-shift masks for Triton-Ascend and omit FP64.
The wrappers retain the installed vLLM processed-logits interface and semantics.
Murmur3 uses low-32-bit positions as upstream does; request seeds retain all
64 bits.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def spec_step_seed(seed, step):
    """Mix a speculative step into the high bits of a request seed."""
    return seed ^ (tl.cast(step, tl.int64) << 32)


@triton.jit
def _murmur3_rotl32(value, shift: tl.constexpr):
    return (value << shift) | ((value >> (32 - shift)) & ((1 << shift) - 1))


@triton.jit
def _murmur3_mix(h, key):
    key *= 0xCC9E2D51
    key = _murmur3_rotl32(key, 15)
    key *= 0x1B873593
    h ^= key
    h = _murmur3_rotl32(h, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _murmur3_fmix32(h):
    h ^= ((h >> 16) & 0xFFFF)
    h *= 0x85EBCA6B
    h ^= ((h >> 13) & 0x7FFFF)
    h *= 0xC2B2AE35
    return h ^ ((h >> 16) & 0xFFFF)


@triton.jit
def murmur3_hash32(seed, pos, offset, domain: tl.constexpr = 0):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    offset = offset.to(tl.uint32)
    # Keep the request-wide prefix scalar until the token offset is mixed in.
    h = (seed ^ seed).to(tl.uint32)
    h ^= domain
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, offset)
    return _murmur3_fmix32(h ^ 16)


@triton.jit
def murmur3_uniform32(seed, pos, offset):
    random32 = murmur3_hash32(seed, pos, offset)
    # Split the uint32 before converting to fp32 so backends without a native
    # uint32-to-float conversion can still use all 32 source bits. Both 16-bit
    # halves convert exactly; their sum is the correctly rounded fp32 value of
    # (random32 + 0.5) * 2**-32. In particular, the u -> 0 winning tail keeps
    # the full 32-bit source resolution instead of being truncated to 24 bits.
    hi16 = ((random32 >> 16) & 0xFFFF).to(tl.int32)
    lo16 = (random32 & 0xFFFF).to(tl.int32)
    return (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


@triton.jit
def _log1p_neg_stable(value):
    # Preserve precision for the positive Gumbel tail without relying on a
    # backend-specific libdevice log1p. The degree-8 series has absolute error
    # below 6e-7 on [0, 0.25]; subtraction is well-conditioned elsewhere for
    # the part of the distribution that can win the argmax.
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial

    direct = tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08))
    return tl.where(value < 0.25, series, direct)


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    `keys` indexes the noise, so the same token draws the same noise wherever it
    appears; `pos` and `seed` place the draw in the request's stream, which is
    what lets a draft and its verification agree.
    """
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    if temp != 0.0:
        if IS_DRAFTING:
            pos = pos + (1 << 30)
        u = murmur3_uniform32(seed, pos, keys)
        gumbel_noise = -tl.log(-_log1p_neg_stable(u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)



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
    seeds_ptr,
    pos_ptr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    # Draft is identified by the installed vLLM caller's cache-column argument.
    # Callers promote logits to FP32; MRv2 temperature is FP32.
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

    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    if processed_logits_ptr is not None and processed_logits_col_ptr is not None:
        # Upstream spec_step_seed: separate speculative steps in seed high bits.
        seed = spec_step_seed(seed, col)
    return gumbel_noised_argmax(
        logits, block, mask, seed, pos, temp,
        IS_DRAFTING=IS_DRAFTING, USE_FP64=False, APPLY_TEMPERATURE=False,
    )



@triton.jit(
    do_not_specialize=[
        "local_argmax_stride",
        "local_max_stride",
        "processed_logits_stride",
        "logits_stride",
        "vocab_size",
    ]
)
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = _npu_gumbel_block_argmax(
        logits, block, mask, token_idx,
        expanded_idx_mapping_ptr, temp_ptr,
        processed_logits_ptr, processed_logits_stride, processed_logits_col_ptr,
        vocab_size, seeds_ptr, pos_ptr,
        IS_DRAFTING=processed_logits_col_ptr is not None,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE, USE_FP64=False,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Gumbel-max sample per token row.

    ``temp == 0`` rows are greedy (argmax, no noise); ``temp != 0`` rows get
    Gumbel noise after optional logit temperature scaling. Mixed batches are handled
    correctly — each row is decided independently by its own ``temp``.

    Community Murmur3 uniforms and Gumbel noise are generated inside the
    sampling kernel from request seed, position and vocabulary index.
    """
    if use_fp64:
        raise NotImplementedError(
            "[omni-npu/mrv2] FP64 Gumbel sampling is not supported on NPU."
        )
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    local_max = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE: Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
