# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""High-throughput draft overrides must not leak into the shared base patch."""

import pickle
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from omni_npu.vllm_patches.patches.models.high_throughout import (
    patch_speculative as patch_mod,
)
from omni_npu.vllm_patches.patches.models.pangu_v2_base.patch_speculative import (
    PanguV2MoeSpeculativeConfigPatch,
)


def test_ultra_target_dict_selects_matching_mtp_and_is_picklable(monkeypatch):
    monkeypatch.setattr(
        patch_mod.SpeculativeConfig,
        "hf_config_override",
        PanguV2MoeSpeculativeConfigPatch.hf_config_override,
    )
    override = patch_mod.PanguUltraDraftHFOverridesPatch.compose_draft_hf_overrides(
        {"architectures": ["PanguUltraMoEForCausalLM"], "num_hidden_layers": 7}
    )
    override = pickle.loads(pickle.dumps(override))
    config = PretrainedConfig(
        architectures=["OpenPanguV2ForCausalLM"],
        num_nextn_predict_layers=3,
        num_hidden_layers=50,
    )
    config.model_type = "openpangu_v2"
    result = override(config)
    assert result.model_type == "openpangu_mtp"
    assert result.architectures == ["OpenPanguMTPModel"]
    assert result.n_predict == 3
    assert result.num_hidden_layers == 50


@pytest.mark.parametrize(
    "override",
    [None, {}, {"architectures": ["OpenPanguV2ForCausalLM"]}, lambda config: config],
)
def test_other_target_overrides_keep_upstream_composition(override):
    sentinel = object()
    with patch.object(
        patch_mod, "_origin_compose_draft_hf_overrides", return_value=sentinel
    ) as original:
        assert (
            patch_mod.PanguUltraDraftHFOverridesPatch.compose_draft_hf_overrides(override)
            is sentinel
        )
    original.assert_called_once_with(override)


def test_ultra_callback_uses_current_mapping_and_keeps_other_architectures():
    config = PretrainedConfig(architectures=["OtherArchitecture"])
    config.model_type = "other_model"
    sentinel = object()
    with patch.object(
        patch_mod.SpeculativeConfig, "hf_config_override", return_value=sentinel
    ) as mapping:
        assert patch_mod._ultra_draft_hf_override(config) is sentinel
    mapping.assert_called_once_with(config)
    assert config.architectures == ["OtherArchitecture"]
