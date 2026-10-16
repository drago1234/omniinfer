# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 DFlash compatibility patches for vLLM 0.25.1+empty.

Why patch upstream internals: DFlashSpeculator.load_draft_model calls the
module-local load_dflash_model, which unwraps multimodal targets for embeddings
but reads lm_head from the outer model. The upstream speculator factory creates
DFlashSpeculator directly, and neither that module call nor
_build_draft_attn_metadata exposes a plugin hook for Omni's per-batch NPU graph
context. Subclassing a model cannot change these speculator call sites.

Removal criteria: drop the loader patch when upgrading to vLLM >= 0.26.0,
whose DFlash loader uses get_target_lm_head for multimodal targets; see
https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu/spec_decode/dflash/utils.py.
Drop the metadata patch when DFlash exposes a per-batch graph-context hook that
Omni NPU can use instead of replacing _build_draft_attn_metadata; track the
upstream implementation at
https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py.
"""

from types import SimpleNamespace

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def _ensure_acl_update_stream(manager):
    if hasattr(manager, "_omni_acl_update_stream"):
        return

    import torch

    manager._omni_acl_update_stream = torch.npu.Stream()


def _refresh_draft_graph_context(speculator, metadata):
    manager = speculator.query_cudagraph_manager
    if manager is None:
        return

    _ensure_acl_update_stream(manager)
    speculator._omni_runtime_draft_attn_metadata = metadata

    def context_provider(desc):
        return SimpleNamespace(
            attn_metadata=speculator._omni_runtime_draft_attn_metadata,
            batch_descriptor=SimpleNamespace(
                num_tokens=desc.num_tokens, num_reqs=desc.num_reqs
            ),
        )

    manager._omni_acl_context_provider = context_provider


try:
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as up_dflash_speculator
except ImportError:
    logger.warning(
        "[omni-npu/mrv2] dflash speculator unavailable; DFlash patches not registered"
    )
else:
    _upstream_load_dflash_model = getattr(
        up_dflash_speculator, "load_dflash_model", None
    )
    if not callable(_upstream_load_dflash_model):
        logger.warning(
            "[omni-npu/mrv2] DFlash load_dflash_model unavailable; "
            "loader patch not registered"
        )
    else:
        def load_dflash_model_with_target_head(target_model, vllm_config):
            # Upstream still decides whether the draft owns its weights.
            if hasattr(target_model, "get_language_model"):
                target_model = target_model.get_language_model()
            return _upstream_load_dflash_model(target_model, vllm_config)

        @register_patch("MRv2DflashSpeculatorLoaderPatch", up_dflash_speculator)
        class MRv2DflashSpeculatorLoaderPatch(VLLMPatch):
            _attr_names_to_apply = ["load_dflash_model"]

            load_dflash_model = load_dflash_model_with_target_head

    _dflash_speculator_cls = getattr(up_dflash_speculator, "DFlashSpeculator", None)
    _upstream_build_draft_attn_metadata = getattr(
        _dflash_speculator_cls, "_build_draft_attn_metadata", None
    )
    if (
        not isinstance(_dflash_speculator_cls, type)
        or "_build_draft_attn_metadata" not in vars(_dflash_speculator_cls)
        or not callable(_upstream_build_draft_attn_metadata)
    ):
        logger.warning(
            "[omni-npu/mrv2] DFlashSpeculator._build_draft_attn_metadata "
            "unavailable; metadata refresh patch not registered"
        )
    else:
        @register_patch("MRv2DflashMetadataRefreshPatch", _dflash_speculator_cls)
        class MRv2DflashMetadataRefreshPatch(VLLMPatch):
            _attr_names_to_apply = ["_build_draft_attn_metadata"]
            _upstream_build_draft_attn_metadata = _upstream_build_draft_attn_metadata

            def _build_draft_attn_metadata(self, *args, **kwargs):
                metadata = (
                    MRv2DflashMetadataRefreshPatch._upstream_build_draft_attn_metadata(
                        self, *args, **kwargs
                    )
                )
                _refresh_draft_graph_context(self, metadata)
                return metadata
