# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Literal

from pydantic.dataclasses import rebuild_dataclass
from vllm.config.speculative import SpeculativeConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("DraftSampleMethodPatch", SpeculativeConfig)
class DraftSampleMethodPatch(VLLMPatch):
    """Allow temperature-only draft sampling through normal config validation."""

    @classmethod
    def apply(cls):
        method = Literal["greedy", "probabilistic", "temperature_only"]
        SpeculativeConfig.__annotations__["draft_sample_method"] = method
        SpeculativeConfig.__dataclass_fields__["draft_sample_method"].type = method
        SpeculativeConfig.__pydantic_fields__["draft_sample_method"].annotation = method
        rebuild_dataclass(SpeculativeConfig, force=True)
