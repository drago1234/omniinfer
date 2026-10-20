# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""V2 thinking-token budget hooks. Implementation: omni/worker/npu/thinking_budget.py.

Upstream 0.25.1's ModelRunnerV2 Sampler predates thinking-budget support
(vllm#46727, released in 0.30.0), so the NPU state is attached by
``NPUModelRunnerV2._init_thinking_budget_state`` and driven through the three
lifecycle methods wrapped here:

* ``add_request`` — stage the per-request budget;
* ``apply_staged_writes`` — sync budgets / reset marker caches (upstream
  ``model_runner`` already calls this every step after ``req_states`` staged
  writes, so ``all_token_ids`` / ``total_len`` are current);
* ``apply_sampling_params`` — force the reasoning-end token onto logits of
  over-budget rows. The write happens on the returned FP32 logits, i.e. after
  logit bias / penalties / bad_words / temperature / min_p. A 1e9 write is
  the row's global maximum, so the subsequent top-k/top-p filter can never
  drop the forced token, and being after logit bias is what lets it override
  min_tokens masking (cf. upstream's ``restores_masked_end_token`` test).
  This wrapper also covers the spec-decode path: the V2 RejectionSampler
  calls ``sampler.apply_sampling_params`` for the target logits, and the
  kernels consume the expanded per-draft-step rows through
  ``expanded_local_pos``.

``apply_bypass_conflict`` keeps the chain extendable: another patch may wrap
these methods later and reach this one through ``_upstream_<name>``.
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
        """Sampler lifecycle hooks for NPUThinkingBudgetState.

        The state attribute may be absent (e.g. pooling models, non-last PP
        ranks, or a plain upstream runner in tests); every hook no-ops then.
        """

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
