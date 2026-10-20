# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Select NPU speculative decoding implementations."""

from vllm.v1.worker.gpu.spec_decode import init_speculator as vllm_init_speculator


def init_speculator(vllm_config, device):
    speculative_config = vllm_config.speculative_config
    if speculative_config.method == "mtp" and not speculative_config.use_gemma4_mtp():
        from .mtp_speculator import NPUMTPSpeculator

        return NPUMTPSpeculator(vllm_config, device)
    return vllm_init_speculator(vllm_config, device)
