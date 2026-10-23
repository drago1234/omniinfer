# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Select NPU speculative decoding implementations."""

from vllm.v1.worker.gpu.spec_decode import init_speculator as vllm_init_speculator


def init_speculator(vllm_config, device):
    speculative_config = vllm_config.speculative_config
    if speculative_config.method == "mtp" and not speculative_config.use_gemma4_mtp():
        from .mtp_speculator import NPUMTPSpeculator

        return NPUMTPSpeculator(vllm_config, device)
    if speculative_config.method in ("dflash", "dspark"):
        from .dflash_speculator import (
            NPUDFlash2Speculator,
            NPUDSparkSpeculator,
            NPUDFlashSpeculator,
        )

        if speculative_config.method == "dspark":
            return NPUDSparkSpeculator(vllm_config, device)
        hf_config = speculative_config.draft_model_config.hf_config
        knobs = getattr(hf_config, "dflash_config", None) or {}
        proposal = knobs.get("proposal", {}) or {}
        if proposal.get("strategy") == "selector":
            return NPUDFlash2Speculator(vllm_config, device)
        return NPUDFlashSpeculator(vllm_config, device)
    return vllm_init_speculator(vllm_config, device)
