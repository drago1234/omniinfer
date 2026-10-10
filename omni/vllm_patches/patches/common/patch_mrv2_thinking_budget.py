# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict thinking budgets for Omni's NPU V2 sampler and MTP verification."""

from copy import copy

from vllm import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.worker.gpu.sample.sampler import Sampler

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_validate_params = InputProcessor._validate_params
_apply_sampling_params = Sampler.apply_sampling_params


@register_patch("MRv2ThinkingBudgetValidationPatch", InputProcessor)
class MRv2ThinkingBudgetValidationPatch(VLLMPatch):
    _attr_names_to_apply = ["_validate_params"]

    def _validate_params(self, params, supported_tasks):
        if (self.use_v2_model_runner and isinstance(params, SamplingParams)
                and params.thinking_token_budget is not None):
            if (self.speculative_config is not None and
                    self.speculative_config.rejection_sample_method != "standard"):
                raise ValueError(
                    "MRv2 thinking_token_budget requires standard rejection sampling."
                )
            cfg = self.vllm_config.reasoning_config
            if cfg is not None and cfg.enabled and (
                not cfg.reasoning_start_token_ids or not cfg.reasoning_end_token_ids
            ):
                raise ValueError(
                    "MRv2 thinking_token_budget requires reasoning start/end token IDs."
                )
            # Keep every native check, except the unsupported-runner guard.
            # A shallow copy avoids changing the shared input processor.
            validator = copy(self)
            validator.use_v2_model_runner = False
            return _validate_params(validator, params, supported_tasks)
        return _validate_params(self, params, supported_tasks)


@register_patch("MRv2ThinkingBudgetSamplerPatch", Sampler)
class MRv2ThinkingBudgetSamplerPatch(VLLMPatch):
    _attr_names_to_apply = ["apply_sampling_params"]

    def apply_sampling_params(self, logits, expanded_idx_mapping, idx_mapping_np,
                              pos, input_ids, expanded_local_pos,
                              skip_top_k_top_p=False):
        logits = _apply_sampling_params(
            self, logits, expanded_idx_mapping, idx_mapping_np, pos, input_ids,
            expanded_local_pos, skip_top_k_top_p=skip_top_k_top_p
        )
        budget = getattr(self, "thinking_budget", None)
        if budget is not None:
            logits = budget.apply(
                logits, self.req_states, idx_mapping_np,
                expanded_idx_mapping, input_ids,
            )
        return logits
