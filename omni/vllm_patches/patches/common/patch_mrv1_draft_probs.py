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

import logging

from packaging.version import InvalidVersion, Version
import torch
import vllm

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = logging.getLogger(__name__)


def _needs_block_draft_probs(owner):
    config = getattr(owner, "speculative_config", None)
    return (
        getattr(config, "rejection_sample_method", None) == "block"
        and getattr(config, "draft_sample_method", None) == "probabilistic"
    )


class _GuardedMRv1Patch(VLLMPatch):
    @classmethod
    def apply(cls):
        version = getattr(vllm, "__version__", None)
        try:
            supported = Version(str(version)).base_version == "0.25.1"
        except InvalidVersion:
            supported = False
        if not supported:
            logger.warning("%s skipped: unvalidated vLLM %s", cls.__name__, version)
            return
        name = cls._attr_names_to_apply[0]
        upstream = getattr(cls._target, name, None)
        if not callable(upstream):
            logger.warning("%s skipped: upstream %s is unavailable", cls.__name__, name)
            return
        # Let the registry handle conflicts without saving our own wrapper as
        # the upstream method on a repeated apply (which would recurse).
        if name not in getattr(cls._target, "_omni_npu_applied_patches", {}):
            cls._upstream = staticmethod(upstream)
        super().apply()


try:
    from vllm.v1.spec_decode import llm_base_proposer
except ImportError:
    llm_base_proposer = None

_proposer = getattr(llm_base_proposer, "SpecDecodeBaseProposer", None)
if _proposer is None:
    logger.warning("MRv1 block proposer patch skipped: SpecDecodeBaseProposer is unavailable")
else:
    @register_patch("MRv1ProbabilisticDraftProposerPatch", _proposer)
    class MRv1ProbabilisticDraftProposerPatch(_GuardedMRv1Patch):
        _attr_names_to_apply = ["__init__"]

        def __init__(self, *args, **kwargs):
            # VLLMPatch replaces a method; it does not change the target MRO.
            # Call the original initializer, not super() on the patch class.
            __class__._upstream(self, *args, **kwargs)
            if _needs_block_draft_probs(self):
                if hasattr(self, "_enable_probabilistic_draft_probs"):
                    self._enable_probabilistic_draft_probs = True
                else:
                    logger.warning("MRv1 block draft-Q adaptation skipped: unsupported proposer")


try:
    from vllm.v1.worker import gpu_model_runner
except ImportError:
    gpu_model_runner = None

_runner = getattr(gpu_model_runner, "GPUModelRunner", None)
if _runner is None:
    logger.warning("MRv1 block dummy patch skipped: GPUModelRunner is unavailable")
else:
    @register_patch("MRv1ProbabilisticDraftDummySamplerPatch", _runner)
    class MRv1ProbabilisticDraftDummySamplerPatch(_GuardedMRv1Patch):
        _attr_names_to_apply = ["_dummy_sampler_run"]

        @torch.inference_mode()
        def _dummy_sampler_run(self, hidden_states):
            if not _needs_block_draft_probs(self):
                return __class__._upstream(self, hidden_states)
            sampler = getattr(self, "rejection_sampler", None)
            register_hook = getattr(sampler, "register_forward_pre_hook", None)
            if not callable(register_hook):
                logger.warning("MRv1 block dummy-Q skipped: sampler pre-hook API is unavailable")
                return __class__._upstream(self, hidden_states)

            def add_dummy_draft_probs(module, args, kwargs):
                if kwargs or len(args) != 4:
                    logger.warning("MRv1 block dummy-Q skipped: sampler call signature changed")
                    return None
                metadata, draft_probs, logits, sampling_metadata = args
                if draft_probs is not None:
                    return None
                draft_ids = getattr(metadata, "draft_token_ids", None)
                if draft_ids is None:
                    logger.warning("MRv1 block dummy-Q skipped: draft_token_ids is unavailable")
                    return None
                draft_probs = torch.rand(
                    draft_ids.numel(), logits.shape[-1],
                    dtype=torch.float32, device=logits.device,
                ).softmax(dim=-1)
                return (metadata, draft_probs, logits, sampling_metadata), kwargs

            handle = register_hook(add_dummy_draft_probs, with_kwargs=True)
            try:
                return __class__._upstream(self, hidden_states)
            finally:
                handle.remove()
