# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "omni" / "model_config" / "configs"
MODEL_NAME = "openpangu_v2_omni_30B"


def _load(relative_path):
    with (CONFIG_ROOT / relative_path).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def test_openpangu_v2_30b_model_fingerprint_is_registered():
    match_configs = _load("match_hf_configs.json")

    assert match_configs[MODEL_NAME] == {
        "model_type": "openpangu_v2_omni",
        "architectures": ["OpenPanguUltraOmniForConditionalGeneration"],
        "hidden_size": 2560,
        "num_attention_heads": 24,
        "vocab_size": 151552,
        "intermediate_size": 6144,
        "n_routed_experts": 384,
        "n_shared_experts": 2,
        "moe_intermediate_size": 256,
    }


def test_openpangu_v2_30b_low_latency_config_is_registered():
    best_practice_configs = _load(
        "low_latency/best_practice_configs.json"
    )
    matches = [
        entry
        for entry in best_practice_configs
        if entry["model"] == MODEL_NAME
        and entry["hardware"] == "A2"
        and entry["precision"] == "w8a8c16"
    ]

    assert len(matches) == 1
    relative_path = matches[0]["configs"]["hybrid"]["config_file"]
    config = _load("low_latency/" + relative_path)

    assert config["model_parallel_config"] == {}
    assert config["operator_optimization_config"] == {
        "kv_nz": True,
        "enable_moe_init_routing_v2": True,
    }
