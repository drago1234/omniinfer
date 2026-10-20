# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.config.speculative import SpeculativeConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any


_origin_compose_draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides


def _ultra_draft_hf_override(hf_config: PretrainedConfig) -> PretrainedConfig:
    # vLLM excludes target dict overrides from the draft config. Preserve the
    # explicitly selected Pangu architecture without copying target-only fields.
    # Keep this callable at module scope so spawned workers can unpickle it.
    if hf_config.model_type == "openpangu_v2":
        hf_config.architectures = ["PanguUltraMoEForCausalLM"]
    return SpeculativeConfig.hf_config_override(hf_config)


@register_patch("PanguUltraDraftHFOverridesPatch", SpeculativeConfig)
class PanguUltraDraftHFOverridesPatch(VLLMPatch):
    """Preserve the Ultra draft architecture only in high-throughput mode."""

    _attr_names_to_apply = ['compose_draft_hf_overrides']

    @staticmethod
    def compose_draft_hf_overrides(target_hf_overrides):
        if (
            isinstance(target_hf_overrides, dict)
            and target_hf_overrides.get("architectures")
            == ["PanguUltraMoEForCausalLM"]
        ):
            return _ultra_draft_hf_override
        return _origin_compose_draft_hf_overrides(target_hf_overrides)
