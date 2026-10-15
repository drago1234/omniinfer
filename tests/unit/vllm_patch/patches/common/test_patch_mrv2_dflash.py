# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 DFlash model loading and draft metadata patch tests."""

import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omni_npu.vllm_patches.patches.common import patch_mrv2_dflash as patch_mod
from unit.vllm_patch.patches.patch_test_utils import applied_patches


@pytest.mark.parametrize("multimodal", [False, True])
def test_dflash_loading_uses_language_model_without_overriding_sharing_policy(
    monkeypatch, multimodal
):
    language = SimpleNamespace(lm_head=object())
    target = (
        SimpleNamespace(get_language_model=lambda: language) if multimodal else language
    )
    upstream = Mock(return_value=object())
    monkeypatch.setattr(patch_mod, "_upstream_load_dflash_model", upstream)
    config = object()
    assert (
        patch_mod.load_dflash_model_with_target_head(target, config)
        is upstream.return_value
    )
    upstream.assert_called_once_with(language, config)


def test_draft_graph_provider_tracks_current_metadata(monkeypatch):
    import torch

    cls = patch_mod.MRv2DflashMetadataRefreshPatch
    upstream = Mock(side_effect=[{"layer": object()}, {"layer": object()}])
    monkeypatch.setattr(cls, "_upstream_build_draft_attn_metadata", upstream)
    monkeypatch.setattr(torch.npu, "Stream", Mock(return_value=object()))
    manager = SimpleNamespace()
    speculator = SimpleNamespace(query_cudagraph_manager=manager)
    desc = SimpleNamespace(num_tokens=8, num_reqs=1)
    first = cls._build_draft_attn_metadata(speculator)
    provider = manager._omni_acl_context_provider
    assert provider.__name__ == "context_provider"
    assert provider(desc).attn_metadata is first
    second = cls._build_draft_attn_metadata(speculator)
    assert provider(desc).attn_metadata is second
    assert provider(desc).batch_descriptor.num_tokens == 8
    assert provider(desc).batch_descriptor.num_reqs == 1
    assert torch.npu.Stream.call_count == 1


def test_dflash_patches_bind_to_upstream_speculator():
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream

    classes = [
        patch_mod.MRv2DflashSpeculatorLoaderPatch,
        patch_mod.MRv2DflashMetadataRefreshPatch,
    ]
    with applied_patches(classes):
        assert upstream.load_dflash_model is patch_mod.load_dflash_model_with_target_head
        assert (
            upstream.DFlashSpeculator._build_draft_attn_metadata
            is classes[1].__dict__["_build_draft_attn_metadata"]
        )


def test_draft_metadata_build_without_graph_manager_preserves_upstream_call(
    monkeypatch,
):
    cls = patch_mod.MRv2DflashMetadataRefreshPatch
    metadata = {"draft": object()}
    upstream = Mock(return_value=metadata)
    monkeypatch.setattr(cls, "_upstream_build_draft_attn_metadata", upstream)
    speculator = SimpleNamespace(query_cudagraph_manager=None)
    token_positions = object()

    result = cls._build_draft_attn_metadata(
        speculator, token_positions, step=2,
    )

    assert result is metadata
    upstream.assert_called_once_with(speculator, token_positions, step=2)
    assert not hasattr(speculator, "_omni_runtime_draft_attn_metadata")


@pytest.mark.parametrize(
    ("missing", "skipped"),
    [
        ("load_dflash_model", "MRv2DflashSpeculatorLoaderPatch"),
        ("DFlashSpeculator", "MRv2DflashMetadataRefreshPatch"),
        ("_build_draft_attn_metadata", "MRv2DflashMetadataRefreshPatch"),
    ],
)
def test_missing_upstream_symbol_skips_only_affected_patch(
    monkeypatch, missing, skipped,
):
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as upstream
    from omni_npu.vllm_patches import core

    if missing == "_build_draft_attn_metadata":
        monkeypatch.delattr(upstream.DFlashSpeculator, missing)
    else:
        monkeypatch.delattr(upstream, missing)

    registered = []

    def record_patch(name, target):
        registered.append(name)
        return lambda cls: cls

    monkeypatch.setattr(core, "register_patch", record_patch)
    namespace = runpy.run_path(patch_mod.__file__)

    assert skipped not in registered
    assert skipped not in namespace
    assert len(registered) == 1
