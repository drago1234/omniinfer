# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for OmniAdditionalConfig typed accessor over vllm_config.additional_config."""
from types import SimpleNamespace

import pytest

from omni_npu.configs.additional_config import OmniAdditionalConfig


def _vllm(additional_config):
    return SimpleNamespace(additional_config=additional_config)


def test_known_keys_mapped():
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({
        "enable_full_async_rl": True,
        "combine_block": 2,
        "npugraph_ex_config": {"enable": True},
        "enable_omni_cache": True,
    }))
    assert cfg.enable_full_async_rl is True
    assert cfg.combine_block == 2
    assert cfg.npugraph_ex_config == {"enable": True}
    assert cfg.enable_omni_cache is True


def test_npugraph_ex_config_is_a_free_dict_passthrough():
    sub = {
        "enable": True,
        "static_kernel_compile": True,
        "static_kernel_cache_enable": True,
        "static_kernel_cache_dir": "/cache/root",
    }
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({"npugraph_ex_config": sub}))
    assert cfg.npugraph_ex_config == sub


def test_unknown_shared_key_is_ignored_by_omni_accessor():
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({"totally_unknown_key": 123}))
    assert cfg.combine_block == 1  # Default value.


def test_defaults_when_empty():
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({}))
    assert cfg.enable_full_async_rl is False
    assert cfg.npugraph_ex_config == {}


def test_none_additional_config():
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm(None))
    assert cfg.enable_low_latency is False


def test_invalid_field_name_raises_at_construct():
    with pytest.raises(TypeError):
        OmniAdditionalConfig(typo_field=True)


def test_all_supported_keys_present():
    fields = {f.name for f in __import__("dataclasses").fields(OmniAdditionalConfig)}
    expected = {
        "enable_full_async_rl",
        "combine_block", "npugraph_ex_config",
        "enable_pd_elastic_scaling", "enable_low_latency", "enable_omni_cache",
        "msprobe_dump_config_path", "msprobe_dump_config",
    }
    assert expected == fields


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("enable_full_async_rl", "false"),
        ("enable_pd_elastic_scaling", "true"),
        ("enable_low_latency", 0),
        ("enable_omni_cache", []),
        ("combine_block", True),
        ("npugraph_ex_config", []),
        ("msprobe_dump_config", "not-a-dict"),
        ("msprobe_dump_config_path", 0),
    ],
)
def test_known_keys_reject_wrong_types(name, value):
    with pytest.raises(TypeError, match=name):
        OmniAdditionalConfig.from_vllm_config(_vllm({name: value}))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("combine_block", 0),
        ("combine_block", -1),
    ],
)
def test_positive_integer_keys_reject_nonpositive_values(name, value):
    with pytest.raises(ValueError, match=name):
        OmniAdditionalConfig.from_vllm_config(_vllm({name: value}))


@pytest.mark.parametrize("raw", [[], "", 0, False])
def test_additional_config_rejects_non_dict_values(raw):
    with pytest.raises(TypeError, match="additional_config must be dict"):
        OmniAdditionalConfig.from_vllm_config(_vllm(raw))


def test_vllm_config_is_parsed_on_each_access():
    vllm_config = _vllm({"combine_block": 2})

    first = OmniAdditionalConfig.from_vllm_config(vllm_config)
    vllm_config.additional_config["combine_block"] = 8
    second = OmniAdditionalConfig.from_vllm_config(vllm_config)

    assert second is not first
    assert second.combine_block == 8


# ---------------------------------------------------------------------------
# msprobe dump config fields (msprobe_dump_config_path / msprobe_dump_config)
# ---------------------------------------------------------------------------

def test_dump_config_defaults_none():
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({}))
    assert cfg.msprobe_dump_config_path is None
    assert cfg.msprobe_dump_config is None


def test_dump_config_and_path_mutually_exclusive():
    with pytest.raises(ValueError, match="Only one of"):
        OmniAdditionalConfig.from_vllm_config(_vllm({
            "msprobe_dump_config_path": "/data/msprobe_config.json",
            "msprobe_dump_config": {"task": "statistics"},
        }))


@pytest.mark.parametrize("value", ["/data/cfg.json", "/another/path.json"])
def test_dump_config_path_passthrough(value):
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({
        "msprobe_dump_config_path": value,
    }))
    assert cfg.msprobe_dump_config_path == value
    assert cfg.msprobe_dump_config is None


def test_dump_config_passthrough_dict():
    inline = {"task": "statistics", "level": "L1"}
    cfg = OmniAdditionalConfig.from_vllm_config(_vllm({
        "msprobe_dump_config": inline,
    }))
    assert cfg.msprobe_dump_config == inline
    assert cfg.msprobe_dump_config_path is None


@pytest.mark.parametrize("value", ["not-a-dict", 0, False, []])
def test_dump_config_must_be_dict(value):
    with pytest.raises(TypeError, match="msprobe_dump_config must be dict"):
        OmniAdditionalConfig.from_vllm_config(
            _vllm({"msprobe_dump_config": value})
        )


@pytest.mark.parametrize("value", [0, False, [], {}])
def test_dump_config_path_must_be_str(value):
    # Note: {} / [] / 0 / False are not str and not None -> TypeError.
    with pytest.raises(TypeError, match="msprobe_dump_config_path must be str"):
        OmniAdditionalConfig.from_vllm_config(_vllm({
            "msprobe_dump_config_path": value,
        }))
