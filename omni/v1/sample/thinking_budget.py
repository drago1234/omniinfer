# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRv1 backport of vLLM #46662 (32e632dfeb) for vLLM 0.25.1.

The thinking-ban patch composes these processors with Omni's ban mask. Budget
state consumes committed outputs separately from drafts, avoiding history copies
and the unused repeat-index H2D on the budget-only rejection path.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words, apply_bad_words_with_drafts
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def sampler_apply_logits_processors(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    predict_bonus_token: bool,
) -> torch.Tensor:
    bad_words_token_ids = sampling_metadata.bad_words_token_ids
    any_penalties_or_bad_words = (
        bool(bad_words_token_ids) or not sampling_metadata.no_penalties
    )
    output_token_ids = sampling_metadata.output_token_ids
    if predict_bonus_token and any_penalties_or_bad_words:
        output_token_ids = self._combine_outputs_with_spec_tokens(
            output_token_ids, sampling_metadata.spec_token_ids
        )

    if sampling_metadata.allowed_token_ids_mask is not None:
        logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))
    if bad_words_token_ids:
        apply_bad_words(logits, bad_words_token_ids, output_token_ids)
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)
    logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)

    holder = sampling_metadata.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        holder.update_state(
            sampling_metadata.output_token_ids,
            sampling_metadata.spec_token_ids,
            repeat_indices=None,
        )
        logits = holder.apply_to_logits(
            logits, predict_bonus_token, sampling_metadata.spec_token_ids
        )
    return logits


def rejection_apply_logits_processors(
    self,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    metadata: SpecDecodeMetadata,
) -> torch.Tensor:
    has_penalties = not sampling_metadata.no_penalties
    output_token_ids = sampling_metadata.output_token_ids
    if sampling_metadata.bad_words_token_ids or has_penalties:
        output_token_ids = self._combine_outputs_with_spec_tokens(
            output_token_ids, sampling_metadata.spec_token_ids
        )

    need_repeat_indices = (
        sampling_metadata.allowed_token_ids_mask is not None or has_penalties
    )
    if need_repeat_indices:
        num_requests = len(metadata.num_draft_tokens)
        num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
        original_indices = torch.arange(num_requests, device="cpu")
        repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
        repeat_indices = repeat_indices_cpu.to(device=logits.device, non_blocking=True)
        logits = self.apply_penalties(
            logits, sampling_metadata, metadata, repeat_indices, output_token_ids
        )
        if sampling_metadata.allowed_token_ids_mask is not None:
            token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
            logits.masked_fill_(token_mask, float("-inf"))

    if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
        apply_bad_words_with_drafts(
            logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
        )
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        if isinstance(processor, MinTokensLogitsProcessor):
            logits = processor.apply_with_spec_decode(logits, metadata.num_draft_tokens)

    holder = sampling_metadata.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        logits = holder.apply_to_logits(
            logits,
            predict_bonus_token=False,
            spec_token_ids=sampling_metadata.spec_token_ids,
        )
    return logits


def update_thinking_budget_state(
    self,
    output_token_ids: list[list[int]],
    spec_token_ids: list[list[int]] | None,
    repeat_indices: torch.Tensor | None = None,
) -> None:
    """Update the native holder with committed output rows, never strip drafts."""
    if not self.is_enabled or not self._state:
        return

    spec_lists = spec_token_ids or []
    last_row_for_req: dict[int, int] | None = None
    if repeat_indices is not None:
        last_row_for_req = {}
        for batch_row, req_i in enumerate(repeat_indices.cpu().tolist()):
            last_row_for_req[req_i] = batch_row

    for seq_idx, state in list(self._state.items()):
        if last_row_for_req is not None:
            output_row = last_row_for_req.get(seq_idx)
            if output_row is None or output_row >= len(output_token_ids):
                continue
            state["output_tok_ids"] = output_token_ids[output_row]
        elif seq_idx >= len(output_token_ids):
            continue
        else:
            state["output_tok_ids"] = output_token_ids[seq_idx]
        state["spec_token_ids"] = (
            list(spec_lists[seq_idx]) if seq_idx < len(spec_lists) else []
        )
        state["in_spec_mode"] = self.in_spec_mode
        state["force_index"] = []
        self._update_think_state(state)
