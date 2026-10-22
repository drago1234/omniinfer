# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU sampling entry points for the V2 runner.

Gumbel sampling is implemented in ``ops.gumbel`` and re-exported here for
existing callers. Top-k/top-p uses the CANN fused filter and preserves the
input logits dtype; token selection stays in the Gumbel/rejection sampler.
"""

from __future__ import annotations

import torch
import torch_npu

from omni_npu.worker.npu.ops.gumbel import gumbel_sample as gumbel_sample


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
    if p is None and k is None:
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
