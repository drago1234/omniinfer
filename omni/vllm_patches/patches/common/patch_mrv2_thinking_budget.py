# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport vllm#46727's sampler lifecycle to the 0.25.1 V2 runner.

NPUModelRunnerV2 attaches the state after load_model creates the sampler.
The sampling hook writes returned FP32 logits after bias, penalties, bad
words, temperature and min_p. Ordinary sampling applies top-k/top-p next;
speculative verification has already applied those filters, so this hook
restores the forced token afterwards. Upstream instead forces before
temperature; this wrapper preserves the 0.25.1 sampling interface.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

try:
    import vllm.v1.worker.gpu.sample.sampler as up_sampler
except ImportError:
    # Log and skip: this directory is imported wholesale by the plugin and
    # exec_module has no try/except, so raising here takes down MRv1 too.
    logger.error(
        "[omni-npu/mrv2] thinking budget patch target unavailable; not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2ThinkingBudgetSamplerPatch", up_sampler.Sampler)
    class MRv2ThinkingBudgetSamplerPatch(VLLMPatch):
        """Forward lifecycle events when the NPU runner attached a state."""

        _attr_names_to_apply = [
            "add_request",
            "apply_staged_writes",
            "apply_sampling_params",
        ]

        @classmethod
        def apply(cls):
            # Bypass the already-patched conflict check so later patches can
            # still chain onto these methods via ``_upstream_<name>``.
            cls.apply_bypass_conflict(
                "add_request",
                "apply_staged_writes",
                "apply_sampling_params",
            )

        def add_request(self, req_idx, prompt_len, sampling_params):
            MRv2ThinkingBudgetSamplerPatch._upstream_add_request(
                self, req_idx, prompt_len, sampling_params
            )
            state = getattr(self, "thinking_budget_state", None)
            if state is not None:
                state.add_request(req_idx, sampling_params)

        def apply_staged_writes(self):
            MRv2ThinkingBudgetSamplerPatch._upstream_apply_staged_writes(self)
            state = getattr(self, "thinking_budget_state", None)
            if state is not None:
                state.apply_staged_writes()

        def apply_sampling_params(
            self,
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p: bool = False,
        ):
            logits = MRv2ThinkingBudgetSamplerPatch._upstream_apply_sampling_params(
                self,
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                skip_top_k_top_p,
            )
            state = getattr(self, "thinking_budget_state", None)
            if state is not None:
                state.apply(
                    logits,
                    expanded_idx_mapping,
                    idx_mapping_np,
                    input_ids,
                    expanded_local_pos,
                )
            return logits
