# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Thinking-token budget for the V2 runner on NPU.

Port of upstream vLLM 0.30.0 ``vllm/v1/worker/gpu/sample/thinking_budget.py``
(vllm#46727, plus the spec-decode TPOT follow-up vllm#46662) onto the 0.25.1
ModelRunnerV2 stack. Semantics: token counting starts at the last reasoning
start marker; once the reasoning token count reaches the request's
``thinking_token_budget``, the next token of the forced end sequence
(``reasoning_end_token_ids``, possibly a multi-token transition phrase) gets
its logit set to 1e9 until the sequence is fully emitted. A model that ends
reasoning naturally (``natural_reasoning_end_token_ids``) is never forced.

Differences from the upstream CUDA implementation:

The upstream ``UvaBackedTensor`` uses the existing MRv2 NPU buffer patches
for staged H2D copies or the optional UVA view.

1. The marker-cache kernel is launched over logits rows and gated on
   ``local_pos == 0`` (exactly one row per request per step) instead of a
   separate ``req_ids`` tensor, because the 0.25.1
   ``Sampler.apply_sampling_params`` signature does not carry ``idx_mapping``.
2. The cold scan (new/resumed request) is a flag-guarded for-loop instead of
   upstream's ``while`` loop: ``scf.while`` with dynamic indexing crashes the
   triton-ascend 3.2.2 pipeline (cf. ``omni/worker/npu/sampler.py``).
   Its bound uses the committed length; blocks after a marker hit skip loads.
3. Upstream's ``_load_effective_token`` @triton.jit device function (early
   return under ``if``) cannot be legalized by triton-ascend's
   ConvertTritonIRToLinalgIR pass ("failed to legalize unresolved
   materialization from memref<?xi32> to !tt.ptr<i32>"; the call site then
   reports "tt.call does not reference a valid function"). The committed /
   draft-prefix load is inlined branch-free at each use instead (clamped
   dual load + tl.where select).
4. Jitted code uses no ``and``/``or`` at all: conditions are combined with
   ``&``/``|`` on int1 scalars. Triton's AST-to-TTIR pass rejects chained
   boolean operators with ``UnsupportedLanguageConstruct`` (a fatal, engine-killing
   error at first request), so the kernels stay on the bitwise form.
   ``tests/unit/worker/npu/test_thinking_budget.py`` pins this with an AST scan,
   since the failure only shows up at kernel compile time on device.

The state is attached to the V2 ``Sampler`` by ``NPUModelRunnerV2`` and
driven through the patched ``Sampler.add_request`` /
``Sampler.apply_staged_writes`` / ``Sampler.apply_sampling_params`` (see
``omni/vllm_patches/patches/common/patch_mrv2_thinking_budget.py``).
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

_INT32_MAX = np.iinfo(np.int32).max
_COLD_SCAN_BLOCK = 1024


class NPUThinkingBudgetState:
    """Model Runner V2 state for per-request thinking token budgets."""

    def __init__(
        self,
        req_states: RequestState,
        reasoning_config,
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device

        start_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_start_token_ids or []
        )
        end_ids = (
            []
            if reasoning_config is None
            else reasoning_config.reasoning_end_token_ids or []
        )
        # getattr: a native (non-Omni-superset) ReasoningConfig predates the
        # natural-end field and must disable the feature, not raise.
        natural_end_ids = (
            []
            if reasoning_config is None
            else getattr(reasoning_config, "natural_reasoning_end_token_ids", None)
            or []
        )
        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        if not self.enabled:
            logger.info(
                "[omni-npu/mrv2] thinking budget disabled: reasoning start/end/"
                "natural-end token ids are not all available"
            )
            return

        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(-1)
        self.thinking_token_budget.copy_to_uva()
        self.use_thinking_budget = np.zeros(self.max_num_reqs, dtype=bool)

        self.cached_last_start = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_last_end = torch.full(
            (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
        )
        self.cached_scan_pos = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._reset_reqs: list[int] = []
        self._budget_dirty = False

        self.reasoning_start_token_ids = torch.tensor(
            start_ids, dtype=torch.int32, device=self.device
        )
        self.natural_reasoning_end_token_ids = torch.tensor(
            natural_end_ids, dtype=torch.int32, device=self.device
        )
        self.reasoning_end_token_ids = torch.tensor(
            end_ids, dtype=torch.int32, device=self.device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        if not self.enabled:
            return
        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = budget is not None
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self._reset_reqs.append(req_idx)
        if self.thinking_token_budget.np[req_idx] != budget:
            self.thinking_token_budget.np[req_idx] = budget
            self._budget_dirty = True

    def apply_staged_writes(self) -> None:
        if not self.enabled:
            return
        if self._reset_reqs:
            idx = async_tensor_h2d(
                self._reset_reqs, dtype=torch.int64, device=self.device
            )
            self.cached_last_start.index_fill_(0, idx, -1)
            self.cached_last_end.index_fill_(0, idx, -1)
            self.cached_scan_pos.index_fill_(0, idx, 0)
            self._reset_reqs.clear()
        if self._budget_dirty:
            self.thinking_token_budget.copy_to_uva()
            self._budget_dirty = False

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        """Force the reasoning-end sequence onto over-budget FP32 logits."""
        if not self.enabled or not np.any(
            self.use_thinking_budget[idx_mapping_np]
        ):
            return

        apply_thinking_budget(
            logits,
            expanded_idx_mapping,
            expanded_local_pos,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            self.cached_last_start,
            self.cached_last_end,
            self.cached_scan_pos,
            self.reasoning_start_token_ids,
            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
        )


@triton.jit
def _update_committed_marker_cache_kernel(  # pragma: no cover - triton
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per logits row; only the first row of each request
    # (local_pos == 0) refreshes that request's committed-marker cache.
    token_idx = tl.program_id(0)
    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    if local_pos != 0:
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        # The request was preempted and resumed with a shorter committed
        # history; rescan from scratch.
        scan_pos = 0
        last_start = -1
        last_end = -1

    if (scan_pos == 0) & (last_start < 0) & (last_end < 0):
        # Ascend-compatible cold scan; skip loads after the first marker hit.
        for b in tl.range(0, tl.cdiv(total_len, BLOCK)):
            block_hi = total_len - b * BLOCK
            if (block_hi > 0) & (last_start < 0) & (last_end < 0):
                block_lo = block_hi - BLOCK
                if block_lo < 0:
                    block_lo = 0
                offs = block_lo + tl.arange(0, BLOCK)

                start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + offs
                        + j,
                        mask=offs + j < total_len,
                        other=-1,
                    )
                    start_match = start_match & (actual == expected)

                end_match = (offs < block_hi) & (
                    offs + NATURAL_END_LEN <= total_len
                )
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + offs
                        + j,
                        mask=offs + j < total_len,
                        other=-1,
                    )
                    end_match = end_match & (actual == expected)

                last_start = tl.max(tl.where(start_match, offs, -1), axis=0)
                last_end = tl.max(tl.where(end_match, offs, -1), axis=0)
    else:
        # Hot path: only the tokens committed since the previous step.
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + i
                        + j
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + i
                        + j
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)


@triton.jit
def _thinking_budget_kernel(  # pragma: no cover - triton
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    start_lo = total_len - START_LEN + 1
    if start_lo < 0:
        start_lo = 0
    for i in tl.range(start_lo, effective_len - START_LEN + 1):
        start_match = True
        for j in tl.static_range(0, START_LEN):
            expected = tl.load(reasoning_start_token_ids_ptr + j)
            pos = i + j
            is_committed = pos < total_len
            committed = tl.load(
                all_token_ids_ptr
                + req_state_idx * all_token_ids_stride
                + tl.where(is_committed, pos, 0)
            )
            pending = tl.load(
                input_ids_ptr
                + tl.where(
                    is_committed,
                    0,
                    (cur_req_first_pos + pos - total_len + 1).to(tl.int32),
                )
            )
            actual = tl.where(is_committed, committed, pending)
            start_match = start_match & (actual == expected)
        if start_match:
            last_start = i

    end_lo = total_len - NATURAL_END_LEN + 1
    if end_lo < 0:
        end_lo = 0
    for i in tl.range(end_lo, effective_len - NATURAL_END_LEN + 1):
        end_match = True
        for j in tl.static_range(0, NATURAL_END_LEN):
            expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
            pos = i + j
            is_committed = pos < total_len
            committed = tl.load(
                all_token_ids_ptr
                + req_state_idx * all_token_ids_stride
                + tl.where(is_committed, pos, 0)
            )
            pending = tl.load(
                input_ids_ptr
                + tl.where(
                    is_committed,
                    0,
                    (cur_req_first_pos + pos - total_len + 1).to(tl.int32),
                )
            )
            actual = tl.where(is_committed, committed, pending)
            end_match = end_match & (actual == expected)
        if end_match:
            last_end = i

    if (last_start < 0) | (last_start <= last_end):
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    # If the tail already ends with a prefix of the forced end sequence
    # (even from a resumed prompt), continue from the next marker token.
    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len < max_prefix_len:
        max_prefix_len = effective_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    pos = suffix_start + j
                    is_committed = pos < total_len
                    committed = tl.load(
                        all_token_ids_ptr
                        + req_state_idx * all_token_ids_stride
                        + tl.where(is_committed, pos, 0)
                    )
                    pending = tl.load(
                        input_ids_ptr
                        + tl.where(
                            is_committed,
                            0,
                            (cur_req_first_pos + pos - total_len + 1).to(tl.int32),
                        )
                    )
                    actual = tl.where(is_committed, committed, pending)
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def apply_thinking_budget(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]

    _update_committed_marker_cache_kernel[(num_tokens,)](
        expanded_idx_mapping,
        expanded_local_pos,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        cached_last_start,
        cached_last_end,
        cached_scan_pos,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )

    _thinking_budget_kernel[(num_tokens,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        thinking_token_budget,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        input_ids,
        expanded_local_pos,
        cached_last_start,
        cached_last_end,
        reasoning_start_token_ids,
        natural_reasoning_end_token_ids,
        reasoning_end_token_ids,
        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
    )
