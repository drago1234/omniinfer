# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Optional
import torch
import torch_npu

from vllm.logger import init_logger

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.rejection_sampler import (
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    GREEDY_TEMPERATURE,
    RejectionSampler
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.triton_utils import HAS_TRITON

from omni_npu import envs
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.sample.ops.topk_topp_sampler import apply_top_k_top_p_npu
from omni_npu.sample.ops.block_acceptance import compute_block_acceptance_triton
from omni_npu.sample.ops.block_selection import select_block_tokens_triton
from omni_npu.v1.utils import on_ascend950

logger = init_logger(__name__)

NEW_GREEDY_TEMPERATURE = 1e-6 # convert greedy cases into random cases


class NPURejectionSampler(RejectionSampler):
    def __init__(self, sampler: Sampler, speculative_config):
        super().__init__(sampler)
        # Share the main sampler's side stream. This step's sampling noise is
        # produced exactly ONCE in forward() below and consumed on the default
        # stream as contiguous copies, so the bonus and verify consumers no
        # longer interleave producers on this stream (IJTE2J).

        rejection_sample_method = speculative_config.rejection_sample_method
        if rejection_sample_method not in ("standard", "block"):
            raise ValueError(f"Unsupported rejection sample method: {rejection_sample_method}")

        self.rejection_sample_method = rejection_sample_method
        self.dsa_stream = sampler.dsa_stream
        self.use_npu_sample = not (
            model_extra_config.operator_opt_config.disable_npu_top_k_top_p_sample
            or on_ascend950()
        )
        self.not_support_float = (
            envs.OMNI_NPU_TOP_K_TOP_P_SAMPLE_NOT_SUPPORT_FLOAT
        )
        self.rollback_dummy = None


    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: Optional[torch.Tensor],  # shape: (num_tokens, vocab_size)
        logits: torch.Tensor,  # shape: (num_tokens + batch_size, vocab_size)
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        cur_stream = torch.npu.current_stream()
        # Generate ALL spec-decode sampling noise once on the dedicated side
        # stream (one producer per step), then hand DEFAULT-stream contiguous
        # copies to the two consumers (bonus sampling + target verify
        # sampling). This removes the per-consumer side-stream allocations
        # whose blocks raced pending default-stream work (IJTE2J).
        # NOTE: generate_random_sequence allocates float32 explicitly; do NOT
        # derive the noise dtype from `logits` (bf16 exponential_ is a
        # 128-value table on NPU and silently degrades sampling to greedy,
        # see IJTLFW).
        target_q = None
        saved_generator_states = {}
        if draft_probs is None and not sampling_metadata.all_greedy:
            # Save per-request RNG states so we can roll them back after the
            # first rejected position is known (rejected tokens must not
            # consume RNG, matching non-MTP sampling order).
            if model_extra_config.operator_opt_config.enable_mtp_invariant:
                for i, generator in sampling_metadata.generators.items():
                    saved_generator_states[i] = generator.get_state()
            if model_extra_config.operator_opt_config.sampler_multi_stream:
                with torch.npu.stream(self.dsa_stream):
                    q = generate_random_sequence(
                        logits, sampling_metadata, metadata,
                    )
                cur_stream.wait_stream(self.dsa_stream)
                # q is consumed on the default stream below; keep its block
                # from being reused early by the next side-stream allocation.
                q.record_stream(cur_stream)
            else:
                q = generate_random_sequence(
                    logits, sampling_metadata, metadata,
                )
            self.sampler.bonus_q = q[bonus_logits_indices].contiguous()
            target_q = q[target_logits_indices].contiguous()

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()

        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )

        if draft_probs is None:
            target_token_ids, target_logits = compute_probs_and_sample(
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
                target_q,
                self.use_npu_sample,
                self.not_support_float,
            )

            output_token_ids, accepted_num = simple_verify(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                target_token_ids,
                bonus_token_ids,
                sampling_metadata,
            )

            if (model_extra_config.operator_opt_config.enable_mtp_invariant
                    and not sampling_metadata.all_greedy):
                # Roll back per-request RNGs: only accepted positions plus the
                # first rejected position (or the bonus when all accepted) should
                # have consumed RNG. `accepted_num` already counts accepted draft
                # tokens; consumed draws = accepted_num + 1 for every request.
                vocab_size = logits.shape[-1]
                if self.rollback_dummy is None:
                    self.rollback_dummy = torch.empty(
                                                (1, vocab_size),
                                                dtype=torch.float32,
                                                device=logits.device,
                                            )
                for i, generator in sampling_metadata.generators.items():
                    n = metadata.num_draft_tokens[i]
                    if accepted_num[i].item() >= n:
                        # All accepted (or n == 0): generator already advanced
                        # exactly n + 1 draws, no rollback needed.
                        continue
                    consumed = int(accepted_num[i].item()) + 1
                    generator.set_state(saved_generator_states[i])

                    for _ in range(consumed):
                        self.rollback_dummy.exponential_(generator=generator)
        else:
            # NOTE(woosuk): `target_logits` can be updated in place inside the
            # `compute_probs` function.
            target_logits = compute_probs(  # shape: (num_tokens, vocab_size)
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
                self.use_npu_sample,
            )
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

            output_token_ids = rejection_sample(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target_probs,
                bonus_token_ids,
                sampling_metadata,
                self.dsa_stream,
                self.rejection_sample_method,
            )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
        )


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=dtype,
        device=device,
    )

    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        if n == 0:
            continue

        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)

        start_idx = end_idx

    return uniform_probs


def rejection_sample(
    draft_token_ids: torch.Tensor,  # shape: (num_tokens)
    num_draft_tokens: list[int],  # shape: (batch_size)
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,  # shape: (batch_size)
    draft_probs: Optional[torch.Tensor],  # shape: (num_tokens, vocab_size)
    target_probs: torch.Tensor,  # shape: (num_tokens, vocab_size)
    bonus_token_ids: torch.Tensor,  # shape: (batch_size, 1)
    sampling_metadata: SamplingMetadata,
    stream: torch.npu.Stream,
    rejection_sample_method: str = "standard",
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)
    cur_stream = torch.npu.current_stream()
    multi_stream = model_extra_config.operator_opt_config.sampler_multi_stream
    uniform_dtype = (
        torch.float32
        if (
            rejection_sample_method == "block"
            and HAS_TRITON
            and target_probs.device.type == "npu"
        )
        else torch.float64
    )

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1).to(torch.int32)
        rejection_greedy_sample_native(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling. shape: (num_tokens)
    if multi_stream:
        with torch.npu.stream(stream):
            uniform_probs = generate_uniform_probs(
                num_tokens,
                num_draft_tokens,
                sampling_metadata.generators,
                device,
                dtype=uniform_dtype,
            )
        cur_stream.wait_stream(stream)
        # Consumed on the default stream below; keep its block from being
        # reused early by the next side-stream allocation.
        uniform_probs.record_stream(cur_stream)
    else:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
            dtype=uniform_dtype,
        )

    # Rejection sampling for random sampling requests.
    rejection_random_sample_native(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        num_draft_tokens,
        sampling_metadata,
        stream,
        device,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        rejection_sample_method,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return output_token_ids


# exactly same with compute_probs from vllm/vllm/v1/sample/rejection_sampler.py
# to use apply_top_k_top_p_npu
def compute_probs(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    use_npu_sample: bool = True,
) -> torch.Tensor:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    if not use_npu_sample:
        logits = apply_top_k_top_p(logits, top_k, top_p)
    else:
        logits = apply_top_k_top_p_npu(logits, top_k, top_p)
    return logits


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    replaced_x = torch.where(x == replace_from, replace_to, x)
    num_tokens_tensor = cu_num_tokens.clone()
    num_tokens_tensor[1:] -= cu_num_tokens[:-1]
    expanded_x = torch.repeat_interleave(
        replaced_x,
        num_tokens_tensor,
        output_size=num_tokens,
    )
    return expanded_x


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,  # shape: (batch_size)
    draft_token_ids: torch.Tensor,  # shape: (num_tokens)
    draft_probs: Optional[torch.Tensor],  # shape: (num_tokens, vocab_size)
    target_probs: torch.Tensor,  # shape: (num_tokens, vocab_size)
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    stream: torch.npu.Stream,
    target_scale=None,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    cur_stream = torch.npu.current_stream()
    multi_stream = model_extra_config.operator_opt_config.sampler_multi_stream
    if multi_stream:
        with torch.npu.stream(stream):
            q = torch.empty(
                (batch_size, vocab_size),
                dtype=torch.float32,
                device=device,
            )
            q.exponential_()
            for i, generator in sampling_metadata.generators.items():
                # Do not generate random numbers for requests with no draft tokens.
                # This can be important for reproducibility.
                if num_draft_tokens[i] > 0:
                    q[i].exponential_(generator=generator)
        cur_stream.wait_stream(stream)
        # Consumed on the default stream below; keep its block from being
        # reused early by the next side-stream allocation.
        q.record_stream(cur_stream)
    else:
        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        q.exponential_()
        for i, generator in sampling_metadata.generators.items():
            # Do not generate random numbers for requests with no draft tokens.
            # This can be important for reproducibility.
            if num_draft_tokens[i] > 0:
                q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_native(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        vocab_size, # next_power_of_2(vocab_size) is useless in native
        NO_DRAFT_PROBS=draft_probs is None,
        target_scale=target_scale,
    )
    return recovered_token_ids


def sample_recovered_tokens_native(
    recovered_token_ids, # [num_tokens]
    cu_num_draft_tokens, # [batch_size]
    draft_token_ids, # [num_tokens]
    draft_probs, # [num_tokens, vocab_size] or None
    target_probs, # [num_tokens, vocab_size]
    q, # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE, # next_power_of_2(vocab_size) is useless in native
    NO_DRAFT_PROBS,
    target_scale=None,
):
    num_tokens = target_probs.shape[0]

    if draft_probs is None:
        sample_probs = target_probs.clone()
        offset = torch.arange(
            num_tokens, dtype=draft_token_ids.dtype, device=draft_token_ids.device,
        ) * sample_probs.shape[1]
        sample_probs.view(-1)[draft_token_ids.view(-1) + offset] -= 1
    else:
        if target_scale is None:
            sample_probs = target_probs - draft_probs
        else:
            sample_probs = (
                target_scale[:, None] * target_probs - draft_probs
            ).clamp_min(0)
            # A padded boundary has no proposal: sample directly from P.
            sample_probs = torch.where(
                draft_token_ids[:, None] < 0, target_probs, sample_probs
            )

    num_draft_tokens = cu_num_draft_tokens.clone()
    num_draft_tokens[1:] -= cu_num_draft_tokens[:-1]
    q = torch.repeat_interleave(
        q, num_draft_tokens, dim=0, output_size=num_tokens,
    )
    recovered_token_ids[:] = (sample_probs / q).argmax(dim=-1).to(torch.int32)


def rejection_greedy_sample_native(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    is_greedy,  # [batch_size] or None
    max_spec_len,
):
    accepted = draft_token_ids == target_argmax

    return select_tokens_by_accepted(
        output_token_ids,
        accepted,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )


def rejection_random_sample_native(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    num_draft_tokens,
    sampling_metadata,
    stream,
    device,
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    rejection_sample_method,
    NO_DRAFT_PROBS,
):
    num_tokens = target_probs.shape[0]
    target_scale = None

    if rejection_sample_method == "standard":
        if draft_probs is None:
            draft_prob = torch.ones(
                num_tokens,
                dtype=target_probs.dtype,
                device=device,
            )
        else:
            draft_prob = draft_probs.gather(1, draft_token_ids.view(-1, 1)).view(-1)

        target_prob = target_probs.gather(1, draft_token_ids.view(-1, 1)).view(-1)
        accepted = target_prob / draft_prob >= uniform_probs

    elif rejection_sample_method == "block":
        if HAS_TRITON and target_probs.device.type == "npu":
            tau, accepted, target_scale = compute_block_acceptance_triton(
                draft_token_ids=draft_token_ids,
                cu_num_draft_tokens=cu_num_draft_tokens,
                draft_probs=draft_probs,
                target_probs=target_probs,
                uniform_probs=uniform_probs,
                max_num_draft_tokens=max_spec_len,
                is_greedy=is_greedy,
            )
        else:
            tau, accepted, target_scale = compute_block_acceptance_torch(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens,
                draft_probs=draft_probs,
                target_probs=target_probs,
                uniform_probs=uniform_probs,
                device=device,
                is_greedy=is_greedy,
            )

    # Sample recovered tokens for each position.
    recovered_token_ids = sample_recovered_tokens(  # shape: (num_tokens)
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
        stream,
        target_scale,
    )

    if (
        rejection_sample_method == "block"
        and HAS_TRITON
        and target_probs.device.type == "npu"
    ):
        return select_block_tokens_triton(
            output_token_ids=output_token_ids,
            tau=tau,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_token_ids=draft_token_ids,
            recovered_token_ids=recovered_token_ids,
            bonus_token_ids=bonus_token_ids,
            fill_this_time=~is_greedy,
            max_spec_len=max_spec_len,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
        )

    return select_tokens_by_accepted(
        output_token_ids,
        accepted,
        cu_num_draft_tokens,
        draft_token_ids,
        recovered_token_ids,
        bonus_token_ids,
        ~is_greedy,
        max_spec_len,
    )


def compute_block_acceptance_torch(
    draft_token_ids,    # [D]
    num_draft_tokens,   # list[int]，每个请求的 draft 数
    draft_probs,       # [D, V]，Q
    target_probs,      # [D, V]，P
    uniform_probs,     # [D]，由调用方生成
    device,
    is_greedy=None,    # [B]，True 表示交给原 greedy 分支处理
):
    batch_size = len(num_draft_tokens)

    tau = torch.zeros(batch_size, dtype=torch.int32, device=device)
    accepted = torch.zeros_like(draft_token_ids, dtype=torch.bool)
    target_scale = torch.ones(
        draft_token_ids.numel(), dtype=torch.float32, device=device
    )

    start = 0
    for request, k in enumerate(num_draft_tokens):
        if k == 0:
            continue

        end = start + k
        P = target_probs[start:end].float()
        Q = draft_probs[start:end].float()
        ids = draft_token_ids[start:end].long()[:, None]

        # 1. 每个位置只取实际 draft token 对应的 P/Q。
        # Only the contiguous real prefix participates in verification.
        valid_prefix = (ids.squeeze(1) >= 0).to(torch.int32).cumprod(0).bool()
        safe_ids = ids.clamp_min(0)
        p = P.gather(1, safe_ids).squeeze(1)
        q = Q.gather(1, safe_ids).squeeze(1)
        # Neutral factors carry c through padding without log(0) - log(0).
        p = torch.where(valid_prefix, p, 1.0)
        q = torch.where(valid_prefix, q, 1.0)

        # Greedy 行不参与概率验证，避免其所选 token 的 Q=0。
        if is_greedy is not None:
            p = torch.where(is_greedy[request], 1.0, p)
            q = torch.where(is_greedy[request], 1.0, q)

        # 2. log c_i = min(0, log c_(i-1) + log P_i - log Q_i)。
        log_ratio = p.log() - q.log()
        log_c = torch.empty_like(p)
        previous = P.new_zeros(())  # log(c_0) = log(1) = 0

        for i in range(k):
            previous = (previous + log_ratio[i]).clamp_max(0)
            log_c[i] = previous

        c = log_c.exp()

        # 3. 前缀 i 的 Z，使用下一位置的整行 P/Q。
        Z = (
            c[:-1, None] * P[1:] - Q[1:]
        ).clamp_min(0).sum(dim=-1)

        denominator = Z + (1 - c[:-1])
        safe_denominator = torch.where(
            denominator == 0, 1.0, denominator
        )

        h = torch.empty_like(c)
        h[:-1] = torch.where(
            denominator == 0, 1.0, Z / safe_denominator
        )
        # The last real candidate uses c, not a residual from a padded Q.
        h[:-1] = torch.where(valid_prefix[1:], h[:-1], c[:-1])
        h[-1] = c[-1]

        # 4. 选择最长通过前缀。
        lengths = torch.arange(
            1, k + 1, dtype=torch.int32, device=device
        )
        passed = valid_prefix & (uniform_probs[start:end] <= h) & (h > 0)
        request_tau = torch.where(passed, lengths, 0).max()

        # 此 helper 不决定 greedy 请求的接受结果。
        if is_greedy is not None:
            request_tau = torch.where(
                is_greedy[request], 0, request_tau
            )

        tau[request] = request_tau
        accepted[start:end] = lengths <= request_tau

        # 5. 恢复位置 j 使用 c_(j-1)：[1, c_1, ..., c_(k-1)]。
        target_scale[start + 1:end] = c[:-1]

        start = end

    return tau, accepted, target_scale


def select_tokens_by_accepted(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    accepted, # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    recovered_token_ids,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    fill_this_time, # [batch_size] or None
    max_spec_len, # int
):
    batch_size = cu_num_draft_tokens.shape[0]
    num_tokens = accepted.shape[0]
    device = accepted.device

    num_draft_tokens = cu_num_draft_tokens.clone()
    num_draft_tokens[1:] -= cu_num_draft_tokens[:-1]

    # get arange
    pre_cu_num_draft_tokens = torch.zeros_like(cu_num_draft_tokens)
    pre_cu_num_draft_tokens[1:] = cu_num_draft_tokens[:-1]
    index_offset = (
        torch.arange(
            batch_size,
            dtype=cu_num_draft_tokens.dtype,
            device=cu_num_draft_tokens.device,
        )
        * (max_spec_len + 1)
        - pre_cu_num_draft_tokens
    )
    offsets = torch.repeat_interleave(
        index_offset,
        num_draft_tokens,
        dim=0,
        output_size=num_tokens,
    )
    arange = torch.arange(num_tokens, dtype=cu_num_draft_tokens.dtype, device=cu_num_draft_tokens.device) + offsets

    accepted_mat = torch.zeros(batch_size, (max_spec_len + 1), dtype=accepted.dtype, device=device)
    accepted_mat.view(-1)[arange] = accepted
    accepted_mat = accepted_mat.to(torch.int32).cumprod(dim=-1)
    accepted_num = accepted_mat.sum(dim=-1)
    accepted_mat = accepted_mat.to(torch.bool)
    accepted = accepted_mat.view(-1)[arange]

    recovered_mat = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    recovered_mat.view(-1)[arange] = recovered_token_ids
    recovered_mat.scatter_(1, num_draft_tokens.view(-1, 1), bonus_token_ids.view(-1, 1))

    draft_token_ids = torch.where(accepted, draft_token_ids, PLACEHOLDER_TOKEN_ID)
    # Create output buffer.
    output = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output.view(-1)[arange] = draft_token_ids
    output.scatter_(1, accepted_num.view(-1, 1), recovered_mat.gather(1, accepted_num.view(-1, 1)))

    if fill_this_time is not None:
        output_token_ids[:] = torch.where(fill_this_time[:, None], output, output_token_ids)
    else:
        output_token_ids[:] = output

    return accepted_num


def compute_probs_and_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
    q: Optional[torch.Tensor],  # [num_tokens, vocab_size] f32, default-stream copy
    use_npu_sample,
    not_support_float,
) -> torch.Tensor:

    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1

    if sampling_metadata.all_greedy:
        return logits.argmax(dim=-1).to(torch.int32), logits

    num_tokens = logits.shape[0]
    if sampling_metadata.temperature is None:
        temperature = torch.ones((num_tokens,), device=logits.device, dtype=torch.float32)
    else:
        temperature = expand_batch_to_tokens(
            sampling_metadata.temperature,
            cu_num_draft_tokens,
            num_tokens,
            replace_from=GREEDY_TEMPERATURE,
            replace_to=NEW_GREEDY_TEMPERATURE,
        )
    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        ).type(torch.int32)
    elif use_npu_sample:
        top_k = torch.ones((logits.shape[0],), dtype=torch.int32, device=logits.device) * logits.shape[1]
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )
    elif use_npu_sample:
        top_p = torch.ones(logits.shape[0], dtype=logits.dtype, device=logits.device)

    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # `q` was generated once in NPURejectionSampler.forward (on the side
    # stream, already synced) and sliced here as a default-stream copy.
    if not use_npu_sample:
        logits = apply_top_k_top_p(logits, top_k, top_p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        token_ids = probs.div_(q).argmax(dim=-1).view(-1)
        return token_ids.to(torch.int32), logits
    if not_support_float:
        # This is adapt for cann 8.5.1 as not support logits.dtype is float in npu_top_k_top_p_sample.
        # Normalize logits before casting to fp16/bf16. Mixed MTP batches can
        # replace greedy temperature with a tiny value, which amplifies logits and
        # may overflow fp16 during top-k/top-p sampling. Subtracting the row max
        # preserves softmax/argmax semantics while keeping values non-positive.
        logits = logits - logits.max(dim=-1, keepdim=True)[0]
        logits = logits.type(model_extra_config.dtype)
        top_p = top_p.type(model_extra_config.dtype)
    res = torch_npu.npu_top_k_top_p_sample(
        logits,
        top_k,
        top_p,
        q,
        is_need_logits=True,
    )
    return res[0].to(torch.int32), res[1]


def simple_verify(
    draft_token_ids: torch.Tensor,  # shape: (num_tokens)
    num_draft_tokens: list[int],  # shape: (batch_size)
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,  # shape: (batch_size)
    target_token_ids: torch.Tensor,  # shape: (num_tokens, 1)
    bonus_token_ids: torch.Tensor,  # shape: (batch_size, 1)
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert draft_token_ids.ndim == 1
    assert cu_num_draft_tokens.ndim == 1
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    device = target_token_ids.device
    assert draft_token_ids.is_contiguous()
    assert bonus_token_ids.is_contiguous()

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    accepted_num = rejection_greedy_sample_native(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        None,
        max_spec_len,
    )
    return output_token_ids, accepted_num


def generate_random_sequence(
    probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    spec_metadata: Optional[SpecDecodeMetadata] = None,
):
    generators = sampling_metadata.generators
    batchsize = probs.shape[0] if spec_metadata is None else len(spec_metadata.num_draft_tokens)
    req_arange = list(range(batchsize + 1))
    if spec_metadata is not None:
        # Combined layout: per request, num_draft_tokens target rows + 1 bonus row.
        for i in range(batchsize):
            req_arange[i + 1] = req_arange[i] + spec_metadata.num_draft_tokens[i] + 1
    q = torch.empty_like(probs, dtype=torch.float32)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != batchsize:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            if model_extra_config.operator_opt_config.enable_mtp_invariant:
                for row in range(req_arange[i], req_arange[i + 1]):
                    q[row:row + 1].exponential_(generator=generator)
            else:
                if req_arange[i + 1] > req_arange[i]:
                    q[req_arange[i]:req_arange[i + 1]].exponential_(generator=generator)
    return q
