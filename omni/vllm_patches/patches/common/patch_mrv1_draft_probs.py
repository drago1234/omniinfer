# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Target: vLLM 0.25.1 MRv1, including vendor builds such as 0.25.1+empty.
# Integration: project review requires both adaptations in the patch registry.
# NPUModelRunner inheritance is possible but intentionally not used here; this
# is an explicit project exception to the inheritance-first preference. There
# is no upstream plugin entry point for these two MRv1 draft-Q decisions.
# Original methods are captured only during registered apply(), not at import.
# Exit: remove after this NPU path migrates to upstream MRv2 block support
# (https://github.com/vllm-project/vllm/pull/46781), with draft-Q collection and
# sampler warmup verified. A version bump alone is not proof of support.
# MRv1 https://github.com/vllm-project/vllm/pull/40819 was closed, not merged.

import torch
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


class _MRv1OriginalMethodPatch(VLLMPatch):
    @classmethod
    def apply(cls):
        upstream = getattr(cls._target, cls._attr_names_to_apply[0])
        # Save only after apply succeeds to preserve the original on conflicts.
        super().apply()
        cls._upstream = staticmethod(upstream)


@register_patch("MRv1ProbabilisticDraftProposerPatch", SpecDecodeBaseProposer)
class MRv1ProbabilisticDraftProposerPatch(_MRv1OriginalMethodPatch):
    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args, **kwargs):
        __class__._upstream(self, *args, **kwargs)
        if (
            self.speculative_config.rejection_sample_method == "block"
            and self.speculative_config.draft_sample_method == "probabilistic"
        ):
            self._enable_probabilistic_draft_probs = True


@register_patch("MRv1ProbabilisticDraftDummySamplerPatch", GPUModelRunner)
class MRv1ProbabilisticDraftDummySamplerPatch(_MRv1OriginalMethodPatch):
    _attr_names_to_apply = ["_dummy_sampler_run"]

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor):
        config = self.speculative_config
        if not (
            config is not None
            and config.rejection_sample_method == "block"
            and config.draft_sample_method == "probabilistic"
        ):
            return __class__._upstream(self, hidden_states)

        def add_dummy_draft_probs(_module, args, kwargs):
            metadata, draft_probs, logits, sampling_metadata = args
            draft_probs = torch.rand(
                metadata.draft_token_ids.numel(),
                logits.shape[-1],
                dtype=torch.float32,
                device=logits.device,
            ).softmax(dim=-1)
            return (metadata, draft_probs, logits, sampling_metadata), kwargs

        handle = self.rejection_sampler.register_forward_pre_hook(
            add_dummy_draft_probs, with_kwargs=True
        )
        try:
            return __class__._upstream(self, hidden_states)
        finally:
            handle.remove()
