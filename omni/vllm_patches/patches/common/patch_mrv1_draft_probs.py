# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Target: vLLM 0.25.1 MRv1, including vendor builds such as 0.25.1+empty.
# Integration: project review requires both adaptations in the patch registry.
# NPUModelRunner inheritance is possible but intentionally not used here; this
# is an explicit project exception to the inheritance-first preference. There
# is no upstream plugin entry point for these two MRv1 draft-Q decisions.
# Original methods are kept on each patch class before registration applies
# replacements; installation and duplicate checks remain with VLLMPatch.
# Exit: remove after this NPU path migrates to upstream MRv2 block support
# (https://github.com/vllm-project/vllm/pull/46781), with draft-Q collection and
# sampler warmup verified. A version bump alone is not proof of support.
# MRv1 https://github.com/vllm-project/vllm/pull/40819 was closed, not merged.

import torch
from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)

try:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
except ImportError:
    logger.warning(
        "[omni-npu/mrv1] SpecDecodeBaseProposer unavailable; "
        "MRv1ProbabilisticDraftProposerPatch not registered",
        exc_info=True,
    )
else:
    _original_init = getattr(SpecDecodeBaseProposer, "__init__", None)
    # A removed constructor can resolve to object.__init__ through inheritance.
    if not callable(_original_init) or _original_init is object.__init__:
        logger.warning(
            "[omni-npu/mrv1] SpecDecodeBaseProposer.__init__ unavailable; "
            "MRv1ProbabilisticDraftProposerPatch not registered"
        )
    else:
        @register_patch("MRv1ProbabilisticDraftProposerPatch", SpecDecodeBaseProposer)
        class MRv1ProbabilisticDraftProposerPatch(VLLMPatch):
            _attr_names_to_apply = ["__init__"]
            _original_init = staticmethod(_original_init)

            def __init__(self, *args, **kwargs):
                __class__._original_init(self, *args, **kwargs)
                if (
                    self.speculative_config.rejection_sample_method == "block"
                    and self.speculative_config.draft_sample_method == "probabilistic"
                ):
                    self._enable_probabilistic_draft_probs = True

try:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
except ImportError:
    logger.warning(
        "[omni-npu/mrv1] GPUModelRunner unavailable; "
        "MRv1ProbabilisticDraftDummySamplerPatch not registered",
        exc_info=True,
    )
else:
    _original_dummy_sampler_run = getattr(GPUModelRunner, "_dummy_sampler_run", None)
    if not callable(_original_dummy_sampler_run):
        logger.warning(
            "[omni-npu/mrv1] GPUModelRunner._dummy_sampler_run unavailable; "
            "MRv1ProbabilisticDraftDummySamplerPatch not registered"
        )
    else:
        @register_patch("MRv1ProbabilisticDraftDummySamplerPatch", GPUModelRunner)
        class MRv1ProbabilisticDraftDummySamplerPatch(VLLMPatch):
            _attr_names_to_apply = ["_dummy_sampler_run"]
            _original_dummy_sampler_run = staticmethod(_original_dummy_sampler_run)

            @torch.inference_mode()
            def _dummy_sampler_run(self, hidden_states: torch.Tensor):
                config = self.speculative_config
                if not (
                    config is not None
                    and config.rejection_sample_method == "block"
                    and config.draft_sample_method == "probabilistic"
                ):
                    return __class__._original_dummy_sampler_run(self, hidden_states)

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
                    return __class__._original_dummy_sampler_run(self, hidden_states)
                finally:
                    handle.remove()
