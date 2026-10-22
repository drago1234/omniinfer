# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Config-matching chain tests for the pangu-2B kv_nz opt-in.

The kv_nz flag only reaches the model when the full chain resolves:

    hf_config --(_match_model_name)--> model_name
    model_name + hardware + precision --(_get_best_practice_config)
        --> openpangu/openpangu_dense.json --> operator_opt_config.kv_nz

``pangu-2B`` is registered at both hops. These tests exercise the real
loader functions against the real json assets shipped in the repo (pure
CPU, no NPU): the 2B hf_config resolves exactly to ``pangu-2B`` without
colliding with ``pangu-7B``, the 7B entry keeps resolving unchanged, and
an unregistered shape/model falls back safely (default kv_nz=False).
"""

from types import SimpleNamespace

import pytest

import omni_npu.model_config.config_loader.loader as loader_mod
from omni_npu.model_config.config_loader.loader import (
    TaskConfig,
    _get_best_practice_config,
    _match_model_name,
)


pytestmark = pytest.mark.unit


def _hf_config(**overrides):
    """PanguEmbedded 2B hf_config values (from the 2B hf_config)."""
    base = dict(
        model_type="PanguEmbedded",
        hidden_size=2048,
        num_attention_heads=16,
        vocab_size=151552,
        intermediate_size=6144,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _task_config(model_name, quant_type="bf16"):
    return TaskConfig(
        model_name=model_name,
        hardware_platform="A2",
        quant_type=quant_type,
        is_pd_disaggregation=False,
        is_prefill_node=None,
    )


@pytest.mark.unit
def test_pangu_2b_hf_config_resolves_to_pangu_2b():
    assert _match_model_name(_hf_config(), vars(_hf_config())) == "pangu-2B"


@pytest.mark.unit
def test_pangu_2b_does_not_shadow_pangu_7b():
    """The 2B entry must not steal the 7B match and vice versa."""
    hf_2b = _hf_config()
    hf_7b = _hf_config(
        hidden_size=4096,
        num_attention_heads=32,
        vocab_size=153376,
        intermediate_size=16384,
    )
    names = {
        _match_model_name(hf_2b, vars(hf_2b)),
        _match_model_name(hf_7b, vars(hf_7b)),
    }
    assert names == {"pangu-2B", "pangu-7B"}


@pytest.mark.unit
def test_unregistered_shape_falls_back_to_model_type():
    """An unknown shape falls back to model_type (no false match)."""
    hf = _hf_config(hidden_size=5120)  # not in match_hf_configs.json
    assert _match_model_name(hf, vars(hf)) == "PanguEmbedded"


@pytest.mark.unit
def test_pangu_2b_best_practice_config_enables_kv_nz():
    """(pangu-2B, A2, bf16, hybrid) must load openpangu_dense.json with kv_nz."""
    config = _get_best_practice_config(_task_config("pangu-2B"))
    assert config is not None, (
        "pangu-2B must be registered in best_practice_configs.json; "
        "otherwise kv_nz never reaches the model"
    )
    assert config["operator_optimization_config"]["kv_nz"] is True


@pytest.mark.unit
def test_pangu_7b_best_practice_lookup_unchanged():
    """The 7B entry keeps resolving to its own config file (no regression)."""
    config = _get_best_practice_config(_task_config("pangu-7B"))
    assert config is not None


@pytest.mark.unit
def test_unregistered_model_best_practice_returns_none():
    """Fallback models get the default config (kv_nz=False), not a crash."""
    assert _get_best_practice_config(_task_config("PanguEmbedded")) is None


@pytest.mark.unit
def test_default_operator_opt_config_has_kv_nz_off():
    """The dataclass default is kv_nz=False: the flag is strictly opt-in."""
    assert loader_mod.ModelOperatorOptConfig().kv_nz is False
