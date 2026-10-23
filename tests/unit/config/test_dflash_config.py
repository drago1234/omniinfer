# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from types import SimpleNamespace

import pytest

from omni_npu.model_config.config_loader import loader


CONFIG_PATH = (
    "high_throughout/openpangu_v2/"
    "openpangu_v2_omni_30b_bf16_a2_hybrid_dflash.json"
)


@pytest.mark.parametrize("config_path,expected_tiling", [
    (CONFIG_PATH, False),
    (CONFIG_PATH.replace("_hybrid_dflash.json", "_hybrid.json"), True),
])
def test_custom_config_bypasses_automatic_selection(monkeypatch, config_path, expected_tiling):
    monkeypatch.setenv("OMNI_CUSTOM_MODEL_CONFIG_PATH", config_path)
    monkeypatch.setattr(loader, "model_extra_config", SimpleNamespace())

    def unexpected_auto_selection(task):
        raise AssertionError("DFlash custom config should bypass automatic selection")

    monkeypatch.setattr(loader, "_get_best_practice_config", unexpected_auto_selection)
    task = SimpleNamespace()
    loader._init_model_extra_config(task)

    actual = loader.model_extra_config
    assert actual.task_config is task
    assert actual.operator_opt_config.use_aicpu_fa_tiling is expected_tiling
    assert actual.operator_opt_config.kv_nz is True
    assert actual.operator_opt_config.enable_moe_init_routing_v2 is True
    assert Path(loader.default_config_path, config_path).is_file()


def test_non_speculative_automatic_config_keeps_aicpu_tiling_enabled(monkeypatch):
    monkeypatch.delenv("OMNI_CUSTOM_MODEL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("CUSTOM_MODEL_CONFIG_PATH", raising=False)
    monkeypatch.setattr(loader, "model_extra_config", SimpleNamespace())
    task = SimpleNamespace(
        model_name="openpangu_v2_omni_30B",
        hardware_platform="A2", quant_type="bf16",
        enable_low_latency=False, is_pd_disaggregation=False,
        enable_pd_elastic_scaling=False,
    )
    loader._init_model_extra_config(task)
    assert loader.model_extra_config.operator_opt_config.use_aicpu_fa_tiling is True
