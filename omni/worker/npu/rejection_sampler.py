# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backport vLLM #58784 without changing padded model inputs or cache layout."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from omni_npu.worker.npu.ops.rejection_sampler_utils import rejection_sample


@triton.jit
def _gather_draft_sampled_kernel(  # pragma: no cover - triton
    draft_sampled_ptr, pos_ptr, input_ids_ptr, positions_ptr, logits_indices_ptr,
    expanded_idx_mapping_ptr, expanded_local_pos_ptr, prefill_len_ptr, num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask)
    pos = tl.load(positions_ptr + token_idx, mask=mask)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask)
    # Drafts are only proposed after sampling past the prefill, so any draft
    # slots in a step starting within the prefill are placeholders (e.g. the
    # padded first step after a P/D remote KV load).
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(  # pragma: no cover - triton
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the input token and position of each logits row.

    Draft rows of requests that have not yet sampled past their prefill are
    set to -1 so that the rejection kernels reject them.
    """
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    BLOCK_SIZE = 1024
    _gather_draft_sampled_kernel[(triton.cdiv(num_logits, BLOCK_SIZE),)](
        draft_sampled, pos, input_ids, positions, logits_indices, expanded_idx_mapping,
        expanded_local_pos, prefill_len, num_logits, BLOCK_SIZE=BLOCK_SIZE,
    )
    return draft_sampled, pos


def rejection_sampler_call(
    self,
    logits: torch.Tensor,
    input_batch: InputBatch,
    draft_logits: torch.Tensor | None = None,
) -> SamplerOutput:
    # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
    # that num_nans is computed before applying penalties and temperature.
    num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

    draft_sampled, pos = gather_draft_sampled(
        input_batch.input_ids,
        input_batch.positions,
        input_batch.logits_indices,
        input_batch.expanded_idx_mapping,
        input_batch.expanded_local_pos,
        self.sampler.req_states.prefill_len.gpu,
    )
    processed_logits = self.sampler.apply_sampling_params(
        logits,
        input_batch.expanded_idx_mapping,
        input_batch.idx_mapping_np,
        pos,
        draft_sampled,
        input_batch.expanded_local_pos,
    )
    sampled, num_sampled = rejection_sample(
        processed_logits,
        draft_logits,
        draft_sampled,
        input_batch.cu_num_logits,
        pos,
        input_batch.idx_mapping,
        input_batch.expanded_idx_mapping,
        input_batch.expanded_local_pos,
        self.sampler.sampling_states.temperature.gpu,
        self.sampler.sampling_states.seeds.gpu,
        self.num_speculative_steps,
        self.synthetic_conditional_rates,
        use_fp64=self.sampler.use_fp64_gumbel,
        use_block_verification=self.use_block_verification,
    )
    logprobs_tensors = self._get_logprobs_tensors(
        input_batch,
        sampled,
        num_sampled,
        processed_logits
        if self.sampler.logprobs_mode == "processed_logprobs"
        else logits,
    )

    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled,
        input_batch.seq_lens,
        input_batch.cu_num_logits,
        input_batch.idx_mapping,
        self.sampler.req_states.prefill_len.gpu,
    )

    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
