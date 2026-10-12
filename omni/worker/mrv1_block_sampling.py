# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Target: vLLM 0.25.1 MRv1, including vendor builds such as 0.25.1+empty.
# Integration: use NPUModelRunner inheritance instead of a monkey-patch. Its
# constructor owns the drafter, and its dummy sampler override can call super.
# The upstream MRv1 proposer and dummy sampler only supply Q for standard;
# no upstream class or module is modified by this compatibility helper.
# Exit: remove after migrating this NPU path to upstream MRv2 block support
# (https://github.com/vllm-project/vllm/pull/46781), including working draft-Q
# collection and warmup on NPU. A version bump alone is not proof of support.
# MRv1 proposal https://github.com/vllm-project/vllm/pull/40819 was closed,
# not merged; do not treat it as a released MRv1 fix.

import logging

from packaging.version import InvalidVersion, Version
import torch
import vllm

logger = logging.getLogger(__name__)


def _needs_block_draft_probs(runner):
    config = getattr(runner, "speculative_config", None)
    return (
        getattr(config, "rejection_sample_method", None) == "block"
        and getattr(config, "draft_sample_method", None) == "probabilistic"
    )


def _supported_version():
    version = getattr(vllm, "__version__", None)
    try:
        if Version(str(version)).base_version == "0.25.1":
            return True
    except InvalidVersion:
        pass
    logger.warning(
        "MRv1 block draft-Q adaptation skipped for unvalidated vLLM %s; "
        "check upstream draft-Q collection and sampler warmup support.", version,
    )
    return False


def enable_block_draft_probs(runner):
    if not _needs_block_draft_probs(runner) or not _supported_version():
        return
    drafter = getattr(runner, "drafter", None)
    if drafter is None:  # Pipeline stages without a drafter need no adaptation.
        return
    if not hasattr(drafter, "_enable_probabilistic_draft_probs"):
        logger.warning(
            "MRv1 drafter no longer exposes _enable_probabilistic_draft_probs; "
            "block draft-Q adaptation skipped. Check upstream support."
        )
        return
    drafter._enable_probabilistic_draft_probs = True


@torch.inference_mode()
def run_block_dummy_sampler(runner, hidden_states, upstream_run):
    if not callable(upstream_run):
        logger.warning(
            "GPUModelRunner._dummy_sampler_run is unavailable; MRv1 block "
            "sampler warmup skipped. Check the upstream warmup API."
        )
        return torch.tensor([])
    if not _needs_block_draft_probs(runner) or not _supported_version():
        return upstream_run(hidden_states)
    sampler = getattr(runner, "rejection_sampler", None)
    register_hook = getattr(sampler, "register_forward_pre_hook", None)
    if not callable(register_hook):
        logger.warning(
            "MRv1 rejection sampler has no forward pre-hook API; block "
            "dummy-Q adaptation skipped. Check upstream sampler support."
        )
        return upstream_run(hidden_states)

    def add_dummy_draft_probs(module, args, kwargs):
        if kwargs or len(args) != 4:
            logger.warning(
                "MRv1 dummy rejection sampler arguments changed; "
                "dummy-Q adaptation skipped. Check the upstream call signature."
            )
            return None
        metadata, draft_probs, logits, sampling_metadata = args
        if draft_probs is not None:
            return None
        draft_token_ids = getattr(metadata, "draft_token_ids", None)
        if draft_token_ids is None:
            logger.warning("MRv1 dummy draft_token_ids missing; dummy-Q skipped.")
            return None
        draft_probs = torch.rand(
            draft_token_ids.numel(), logits.shape[-1],
            dtype=torch.float32, device=logits.device,
        ).softmax(dim=-1)
        return (metadata, draft_probs, logits, sampling_metadata), kwargs

    # Startup warmup is synchronous. Remove the instance-local hook even on OOM;
    # real requests must never receive synthetic proposal probabilities.
    handle = register_hook(add_dummy_draft_probs, with_kwargs=True)
    try:
        return upstream_run(hidden_states)
    finally:
        handle.remove()
