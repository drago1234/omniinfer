# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU sampling ops for the V2 runner.

Two symbols are injected by
``omni/vllm_patches/patches/common/patch_mrv2_sampler.py``:

1. ``gumbel_sample`` uses NPU ``uniform_`` for noise and Triton only for the
   Gumbel transform and reduction. The global NPU RNG advances on graph replay;
   request seed/position arguments remain for API compatibility, not per-request
   reproducibility. Greedy rows still ignore noise.

2. ``apply_top_k_top_p`` uses CANN's fused filtering operation, avoiding the
   full-vocabulary sort/scatter path. It returns masked logits; token selection
   remains in the Gumbel/rejection sampler.
"""

from __future__ import annotations

import torch
import torch_npu

from vllm.triton_utils import tl, triton

from omni_npu.worker.npu.ops.gumbel import _npu_gumbel_block_argmax


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
    uniform_noise_ptr,
    uniform_noise_stride,
    temp_ptr,
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
        uniform_noise_ptr, uniform_noise_stride, token_idx,
        processed_logits_ptr, processed_logits_stride, processed_logits_col_ptr,
        vocab_size, APPLY_TEMPERATURE=APPLY_TEMPERATURE, USE_FP64=False,
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
    Gumbel noise after optional temperature scaling. Noise comes from the
    global NPU generator; ``seed`` and ``pos`` no longer control its sequence.
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
    uniform_noise = torch.empty(
        (num_tokens, vocab_size), dtype=torch.float32, device=logits.device
    ).uniform_()
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
        uniform_noise,
        uniform_noise.stride(0),
        temperature,
        vocab_size,
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
    """CANN top-k/top-p filter; discard the operator's sampled token output.

    Keep FP32 logits on CANN versions supporting FP32 (including this runtime),
    avoiding the unnecessary BF16 cast in the older MRv1 compatibility path.
    """
    if p is None and k is None:
        return logits
    rows, vocab = logits.shape
    k = (torch.full((rows,), vocab, dtype=torch.int32, device=logits.device)
         if k is None else k.to(dtype=torch.int32).contiguous())
    p = (torch.ones(rows, dtype=logits.dtype, device=logits.device)
         if p is None else p.to(dtype=logits.dtype).contiguous())
    _, filtered = torch_npu.npu_top_k_top_p_sample(
        logits.contiguous(), k, p, q=None, is_need_logits=True
    )
    return filtered.to(dtype=logits.dtype)
