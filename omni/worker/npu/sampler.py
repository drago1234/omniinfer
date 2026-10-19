# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU sampling ops for the V2 runner.

Two symbols are injected by
``omni/vllm_patches/patches/common/patch_mrv2_sampler.py``:

1. ``gumbel_sample`` — ported from vllm-ascend's
   ``vllm_ascend/worker/v2/sample/gumbel.py`` (the Ascend-native adaptation of
   upstream ``vllm/v1/worker/gpu/sample/gumbel.py``). Uses batched stateless
   Philox uniforms indexed by request seed and position, and the ``+1e-20`` form
   ``-log(-log(r + 1e-20) + 1e-20)`` for the Gumbel transform — this avoids
   ``tldevice.log1p`` which is a returning-None stub on triton-ascend 3.2.2.

2. ``apply_top_k_top_p`` uses the CANN fused filter and preserves the input
   logits dtype. It returns filtered logits; token selection stays in the
   existing Gumbel/rejection sampler.
"""

from __future__ import annotations

import torch
import torch_npu

from vllm.triton_utils import tl, triton

from omni_npu.worker.npu.ops.gumbel import _npu_gumbel_block_argmax
from omni_npu.worker.npu.ops.position_sampling_rng import (
    make_uniform,
    position_rng_enabled,
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
def _gumbel_sample_kernel(  # pragma: no cover
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
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    uniform_ptr,
    uniform_stride,
    vocab_size,
    EXTERNAL_UNIFORM: tl.constexpr,
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
        expanded_idx_mapping_ptr, temp_ptr, seeds_ptr, pos_ptr,
        processed_logits_ptr, processed_logits_stride, processed_logits_col_ptr,
        vocab_size, uniform_ptr, uniform_stride, token_idx,
        EXTERNAL_UNIFORM=EXTERNAL_UNIFORM,
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
    Gumbel noise scaled by the row's temperature. Mixed batches are handled
    correctly — each row is decided independently by its own ``temp``.

    Set OMNI_MRV2_POSITION_RNG=0 before graph capture to use the original
    inline RNG for comparison. The default is batched, position-indexed RNG.
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
    external_uniform = position_rng_enabled()
    uniform = (
        make_uniform(num_tokens, vocab_size, logits.device,
                     expanded_idx_mapping, seed, pos)
        if external_uniform else logits
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
        seed,
        pos,
        temperature,
        uniform,
        uniform.stride(0),
        vocab_size,
        EXTERNAL_UNIFORM=external_uniform,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE: Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Filter logits with CANN; leave token selection to the existing sampler.

    Requires a CANN runtime supporting the input dtype, including FP32. Do not
    downcast logits: rounding can change the top-k/top-p mask and distribution.

    Uses CANN semantics: top-k keeps k entries even at tied logits; p=1 disables
    top-p filtering. These boundary cases differ from the PyTorch sort fallback.
    """
    if k is None and p is None:
        return logits

    rows, vocab_size = logits.shape
    if k is None:
        k = torch.full((rows,), vocab_size, dtype=torch.int32, device=logits.device)
    else:
        k = k.to(dtype=torch.int32).contiguous()
    if p is None:
        p = torch.ones(rows, dtype=logits.dtype, device=logits.device)
    else:
        p = p.to(dtype=logits.dtype).contiguous()

    _, filtered_logits = torch_npu.npu_top_k_top_p_sample(
        logits.contiguous(), k, p, q=None, is_need_logits=True
    )
    return filtered_logits.to(dtype=logits.dtype)
