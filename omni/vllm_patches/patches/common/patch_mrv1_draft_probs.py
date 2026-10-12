# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

SpecDecodeBaseProposer_original_init = SpecDecodeBaseProposer.__init__


@register_patch("MRv1ProbabilisticDraftProposerPatch", SpecDecodeBaseProposer)
class MRv1ProbabilisticDraftProposerPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        SpecDecodeBaseProposer_original_init(
            self,
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=pass_hidden_states_to_model,
            runner=runner,
        )
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method in ("standard", "block")
            and self.speculative_config.draft_sample_method == "probabilistic"
        )


@register_patch("MRv1ProbabilisticDraftDummySamplerPatch", GPUModelRunner)
class MRv1ProbabilisticDraftDummySamplerPatch(VLLMPatch):
    _attr_names_to_apply = ["_dummy_sampler_run"]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.

        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # MM Encoder only model no need to run sampler.
            return torch.tensor([])

        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(
                logits=logits, sampling_metadata=dummy_metadata
            )
            # Also warm forward_native (taken when generators dict is non-empty),
            # but skip the extra call in 'processed_logits' / 'processed_logprobs'
            # modes — there TopKTopPSampler binds forward = forward_native at
            # init time, so the warmup call is redundant and only inflates peak
            # memory during profile_run.
            # No .clone() of logits: warmup output is discarded, so any in-place
            # mutation by forward_native does not affect correctness.
            if self.sampler.logprobs_mode not in (
                "processed_logits",
                "processed_logprobs",
            ):
                self.sampler(
                    logits=logits,
                    sampling_metadata=replace(
                        dummy_metadata,
                        generators={
                            0: torch.Generator(device=self.device).manual_seed(0)
                        },
                    ),
                )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            draft_probs = None
            if (
                ### patch start for block dummy draft probabilities
                self.speculative_config.rejection_sample_method in ("standard", "block")
                ### patch end for block dummy draft probabilities
                and self.speculative_config.draft_sample_method == "probabilistic"
            ):
                draft_probs = torch.rand(
                    num_tokens,
                    logits.shape[-1],
                    device=self.device,
                    dtype=torch.float32,
                )
                draft_probs = torch.softmax(draft_probs, dim=-1)
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output
