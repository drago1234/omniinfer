# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# pylint: skip-file
"""Tests for routed-experts serialization configuration."""
import argparse
import importlib.util
from pathlib import Path

import pytest

_PATCH_PATH = (
    Path(__file__).resolve().parents[5]
    / "omni"
    / "vllm_patches"
    / "patches"
    / "common"
    / "patch_routed_experts_args.py"
)


@pytest.fixture(scope="module")
def patch_mod():
    spec = importlib.util.spec_from_file_location(
        "_t_patch_routed_experts_args", _PATCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeVllmConfig:
    """Minimal replace-compatible configuration."""

    def __init__(self, model_config=None, cache_config=None, parallel_config=None):
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config


def _make_engine_args(patch_mod, cli_value=None):
    """Build a minimal EngineArgs-compatible object."""
    engine_args_cls = type("FakeEngineArgs", (), {})
    engine_args_cls.routed_experts_serialization_mode = (
        patch_mod.DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )
    instance = engine_args_cls()
    if cli_value is not None:
        instance.routed_experts_serialization_mode = cli_value
    return instance


def _run_create_engine_config(patch_mod, engine_args, vllm_config):
    """Invoke the patched configuration factory against fakes."""
    patch_cls = patch_mod.RoutedExpertsArgsEngineArgsPatch
    original = patch_cls._upstream_create_engine_config

    def fake_orig(self, usage_context=None, headless=False):
        return vllm_config

    patch_cls._upstream_create_engine_config = fake_orig
    try:
        method = patch_mod.RoutedExpertsArgsEngineArgsPatch.__dict__[
            "create_engine_config"
        ]
        return method(engine_args)
    finally:
        patch_cls._upstream_create_engine_config = original


def _strict_replace(cfg_cls, instance, **kwargs):
    """Rebuild a config using only declared fields."""
    declared = {"model_config", "cache_config", "parallel_config"}
    data = {
        k: v
        for k, v in instance.__dict__.items()
        if k in declared
    }
    data.update(kwargs)
    return cfg_cls(**data)


def test_create_engine_config_does_not_leak_instance_keys(patch_mod):
    cfg = _FakeVllmConfig()
    cfg.model_config = "m"
    cfg.cache_config = "c"
    cfg.parallel_config = "p"
    engine_args = _make_engine_args(patch_mod, cli_value="base64")

    out = _run_create_engine_config(patch_mod, engine_args, cfg)

    assert out.routed_experts_serialization_mode == "base64"
    rebuilt = _strict_replace(_FakeVllmConfig, out)
    assert rebuilt.routed_experts_serialization_mode == "base64"
    assert set(out.__dict__) <= {
        "model_config",
        "cache_config",
        "parallel_config",
    }


def test_default_mode_used_when_flag_absent(patch_mod):
    cfg = _FakeVllmConfig()
    engine_args = _make_engine_args(patch_mod, cli_value=None)

    out = _run_create_engine_config(patch_mod, engine_args, cfg)

    assert (
        out.routed_experts_serialization_mode
        == patch_mod.DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )
    assert "routed_experts_serialization_mode" not in out.__dict__


def test_cli_flag_definitions_are_wellformed(patch_mod, monkeypatch):
    """The serialization flag accepts supported modes and its default."""
    called = []

    def upstream(parser):
        called.append(parser)
        return parser

    monkeypatch.setattr(
        patch_mod.RoutedExpertsArgsEngineArgsPatch, "_upstream_add_cli_args", upstream
    )
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve")
    serve.add_argument("model")

    patch_mod.RoutedExpertsArgsEngineArgsPatch.add_cli_args(parser)
    assert called == [parser]
    args = parser.parse_args(
        [
            "--routed-experts-serialization-mode",
            "base64",
        ]
    )
    assert args.routed_experts_serialization_mode == "base64"

    args = parser.parse_args([])
    assert (
        args.routed_experts_serialization_mode
        == patch_mod.DEFAULT_ROUTED_EXPERTS_SERIALIZATION_MODE
    )
