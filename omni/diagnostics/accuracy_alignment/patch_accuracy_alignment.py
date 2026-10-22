# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in NPUModelRunner patch for the accuracy-alignment tool."""

from typing import ClassVar

from omni_npu.diagnostics.accuracy_alignment.config import alignment_requested

if alignment_requested():
    from omni_npu.diagnostics.accuracy_alignment.installer import (
        install_accuracy_alignment,
    )
    from omni_npu.diagnostics.accuracy_alignment.train_ref import dummy_run_guard
    from omni_npu.vllm_patches.core import VLLMPatch, register_patch
    from omni_npu.worker.npu_model_runner import NPUModelRunner

    # Keep the first-stage registration name so existing explicit
    # OMNI_VLLM_PATCHES lists continue to work for all three modes.
    @register_patch("AccuracyAlignmentTrainRefPatch", NPUModelRunner)
    class AccuracyAlignmentPatch(VLLMPatch):
        _attr_names_to_apply: ClassVar[list[str]] = [
            "load_model",
            "_dummy_run",
            "profile_run",
        ]

        @classmethod
        def apply(cls):
            cls.apply_bypass_conflict("load_model", "_dummy_run", "profile_run")

        def load_model(self, *args, **kwargs):
            result = AccuracyAlignmentPatch._upstream_load_model(self, *args, **kwargs)
            install_accuracy_alignment(self)
            return result

        def _dummy_run(self, *args, **kwargs):
            with dummy_run_guard():
                return AccuracyAlignmentPatch._upstream_dummy_run(self, *args, **kwargs)

        def profile_run(self, *args, **kwargs):
            with dummy_run_guard():
                return AccuracyAlignmentPatch._upstream_profile_run(
                    self, *args, **kwargs
                )
