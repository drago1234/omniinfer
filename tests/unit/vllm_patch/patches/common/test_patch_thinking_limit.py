# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_thinking_limit (vLLM 0.25.1).

Pure-stdlib: every vLLM and omni_npu dependency is stubbed, so this needs no
vllm, torch, pydantic or NPU — same strategy as
test_patch_repetition_detection_config.py. Run with pytest or directly.

The stubs reproduce the 0.25.1 facts the patch leans on:
``thinking_token_budget`` is a real SamplingParams field, native
create_engine_config already feeds ``self.reasoning_config`` into VllmConfig,
and process_inputs takes params positionally and clones it internally.
"""

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

PATCH_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / "omni" / "vllm_patches").is_dir())
    / "omni/vllm_patches/patches/common/patch_thinking_limit.py"
)

_TESTS_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "unit" / "vllm_patch").is_dir()
)
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from unit.vllm_patch.patches.patch_test_utils import run_standalone_tests  # noqa: E402

REGISTERED = []


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


@dataclass
class _ReasoningConfig:
    """Stands in for omni/v1/config/reasoning.py::ReasoningConfig.

    Carries the fields the patch reads (thinking_token_budget) plus the ones
    _coerce_reasoning_config may receive from JSON, so dict construction
    round-trips like the real dataclass.
    """

    reasoning_parser: str = ""
    reasoning_start_str: str = ""
    reasoning_end_str: str = ""
    think_start_str: str = ""
    think_end_str: str = ""
    thinking_token_budget: "int | None" = None
    ban_tool_start_in_thinking: bool = False
    ban_tool_end_in_thinking: bool = False

    def initialize_token_ids(self, model_config) -> None:
        self.initialized_with = model_config


class _SamplingParams:
    """Mirrors vllm/sampling_params.py: a real ``thinking_token_budget`` attr."""

    def __init__(self, thinking_token_budget=None):
        self.thinking_token_budget = thinking_token_budget


class _PoolingParams:
    pass


@dataclass
class _ModelConfigStub:
    model = "dummy-model"


@dataclass
class _VllmConfig:
    """Stands in for the pydantic dataclass: reasoning_config + model_config."""

    reasoning_config: object = None
    model_config: object = None


class _EngineArgs:
    """Only the bits the patch touches.

    ``reasoning_config`` is a plain instance attribute exactly as in 0.25.1
    (a dataclass field there, but attribute semantics suffice for the patch).
    """

    def __init__(self, reasoning_config=None):
        self.reasoning_config = reasoning_config

    def create_engine_config(self, usage_context=None, headless=False):
        # Native 0.25.1 feeds self.reasoning_config into VllmConfig, whose
        # construction runs initialize_token_ids against the new model_config.
        cfg = self.reasoning_config
        model_config = _ModelConfigStub()
        if cfg is not None:
            cfg.initialize_token_ids(model_config)
        return _VllmConfig(reasoning_config=cfg, model_config=model_config)


class _InputProcessor:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def process_inputs(self, request_id, prompt, params, **kwargs):
        # 0.25.1 clones the params (input_processor.py:315); the clone is what
        # reaches the engine core, so the test asserts on it.
        if isinstance(params, _SamplingParams):
            clone = _SamplingParams(params.thinking_token_budget)
        else:
            clone = params
        return {"request_id": request_id, "params": clone, "kwargs": kwargs}


class _VLLMPatch:
    """Faithful mini-version of vllm_patches/core.py::VLLMPatch."""

    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        target = cls._target
        if not hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches = {}
        for name in cls._attr_names_to_apply:
            if name not in cls.__dict__:
                raise ValueError(f"cannot find {name} in PatchClass {cls.__name__}")
            if name in target._omni_npu_applied_patches:
                raise ValueError(f"{target}.{name} already patched")
            target._omni_npu_applied_patches[name] = cls.__name__
            setattr(target, name, cls.__dict__[name])

    @classmethod
    def apply_bypass_conflict(cls, *upstream):
        target = cls._target
        for name in upstream:
            attr = getattr(target, name)
            if name == "from_cli_args":
                attr = attr.__func__
            setattr(cls, f"_upstream_{name}", attr)
        for name in cls._attr_names_to_apply:
            if name in cls.__dict__:
                setattr(target, name, cls.__dict__[name])


def _register_patch(name, target):
    def decorator(cls):
        cls._target = target
        # Dedupe so a double module load (module-level + fixture) doesn't
        # double-fill REGISTERED.
        if not any(existing_name == name for existing_name, _, _ in REGISTERED):
            REGISTERED.append((name, target, cls))
        return cls

    return decorator


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _Envs:
    OMNI_REASONING_CONFIG = None


ENVS = _Envs()


def _install_stubs():
    for pkg in (
        "vllm",
        "vllm.v1",
        "vllm.v1.engine",
        "omni_npu",
        "omni_npu.v1",
        "omni_npu.vllm_patches",
    ):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))

    sys.modules["vllm"].EngineArgs = _EngineArgs
    sys.modules["vllm"].SamplingParams = _SamplingParams
    _module("vllm.config", VllmConfig=_VllmConfig)
    _module("vllm.logger", init_logger=lambda name: logging.getLogger(name))
    _module("vllm.v1.engine.input_processor", InputProcessor=_InputProcessor)
    sys.modules["omni_npu"].envs = ENVS
    _module("omni_npu.v1.config", ReasoningConfig=_ReasoningConfig)
    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=_VLLMPatch,
        register_patch=_register_patch,
    )


def _load_patch():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("patch_thinking_limit_test", PATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["patch_thinking_limit_test"] = mod
    spec.loader.exec_module(mod)
    # Most tests exercise the patch methods directly. Seed the same upstream
    # that apply() would capture in production.
    patch_cls = mod.ThinkingLimitEngineArgsPatch
    patch_cls._upstream_create_engine_config = _EngineArgs.create_engine_config
    return mod


try:
    import pytest
except ImportError:
    pytest = None


if pytest is not None:

    @pytest.fixture(scope="module", autouse=True)
    def _restore_sys_modules():
        # snapshot sys.modules *before* _load_patch() pollutes them
        keys = (
            "vllm", "vllm.v1", "vllm.v1.engine", "vllm.config", "vllm.logger",
            "vllm.v1.engine.input_processor",
            "omni_npu", "omni_npu.v1", "omni_npu.v1.config",
            "omni_npu.vllm_patches", "omni_npu.vllm_patches.core",
            "patch_thinking_limit_test",
        )
        saved_modules = {k: sys.modules.get(k) for k in keys}
        vllm_now = sys.modules.get("vllm")
        omni_now = sys.modules.get("omni_npu")
        saved_vllm_engine_args = (
            (hasattr(vllm_now, "EngineArgs"), getattr(vllm_now, "EngineArgs", None))
            if vllm_now
            else (False, None)
        )
        saved_vllm_sampling_params = (
            (hasattr(vllm_now, "SamplingParams"), getattr(vllm_now, "SamplingParams", None))
            if vllm_now
            else (False, None)
        )
        saved_omni_envs = (
            (hasattr(omni_now, "envs"), getattr(omni_now, "envs", None))
            if omni_now
            else (False, None)
        )
        global PATCH
        PATCH = _load_patch()
        yield
        # teardown: restore from snapshot taken before _load_patch()
        for k, v in saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        vllm_now = sys.modules.get("vllm")
        if vllm_now is not None:
            had, val = saved_vllm_engine_args
            if had:
                vllm_now.EngineArgs = val
            else:
                try:
                    delattr(vllm_now, "EngineArgs")
                except AttributeError:
                    pass
            had, val = saved_vllm_sampling_params
            if had:
                vllm_now.SamplingParams = val
            else:
                try:
                    delattr(vllm_now, "SamplingParams")
                except AttributeError:
                    pass
        omni_now = sys.modules.get("omni_npu")
        if omni_now is not None:
            had, val = saved_omni_envs
            if had:
                omni_now.envs = val
            else:
                try:
                    delattr(omni_now, "envs")
                except AttributeError:
                    pass


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

GOOD_JSON = (
    '{"reasoning_start_str":"<|thinking|>","reasoning_end_str":"<|/thinking|>",'
    '"thinking_token_budget":128}'
)


def _reset():
    ENVS.OMNI_REASONING_CONFIG = None


def _engine_args_with(reasoning_config):
    return _EngineArgs(reasoning_config)


def _create_config(ea):
    return PATCH.ThinkingLimitEngineArgsPatch.create_engine_config(ea)


def _process(processor, params):
    proc = PATCH.ThinkingLimitInputProcessorPatch.process_inputs
    return proc(processor, "req-0", "hello", params, supported_tasks=("generate",))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_every_declared_attr_exists():
    """VLLMPatch.apply() raises if a name in _attr_names_to_apply is missing."""
    for _name, target, cls in REGISTERED:
        for attr in cls._attr_names_to_apply:
            assert attr in cls.__dict__, f"{cls.__name__} declares missing {attr!r}"
        assert cls._target is target


def test_patch_targets():
    names = {name: target for name, target, _ in REGISTERED}
    assert names["ThinkingLimitEngineArgsPatch"] is _EngineArgs
    assert names["ThinkingLimitInputProcessorPatch"] is _InputProcessor


# --- _normalize_thinking_token_budget --------------------------------------


def test_normalize_accepts_non_negative_ints():
    assert PATCH._normalize_thinking_token_budget(0) == 0
    assert PATCH._normalize_thinking_token_budget(128) == 128


def test_normalize_rejects_none_bool_and_negative():
    assert PATCH._normalize_thinking_token_budget(None) is None
    assert PATCH._normalize_thinking_token_budget(True) is None
    assert PATCH._normalize_thinking_token_budget(False) is None
    assert PATCH._normalize_thinking_token_budget(-1) is None


def test_normalize_rejects_non_int_types():
    assert PATCH._normalize_thinking_token_budget("128") is None
    assert PATCH._normalize_thinking_token_budget(12.5) is None


# --- _coerce_reasoning_config ----------------------------------------------


def test_coerce_passes_through_none_and_instances():
    assert PATCH._coerce_reasoning_config(None) is None
    cfg = _ReasoningConfig()
    assert PATCH._coerce_reasoning_config(cfg) is cfg


def test_coerce_parses_json_string():
    cfg = PATCH._coerce_reasoning_config(GOOD_JSON)
    assert isinstance(cfg, _ReasoningConfig)
    assert cfg.reasoning_start_str == "<|thinking|>"
    assert cfg.reasoning_end_str == "<|/thinking|>"
    assert cfg.thinking_token_budget == 128


def test_coerce_parses_dict():
    cfg = PATCH._coerce_reasoning_config(
        {"reasoning_start_str": "<|thinking|>", "thinking_token_budget": 32}
    )
    assert isinstance(cfg, _ReasoningConfig)
    assert cfg.thinking_token_budget == 32


# --- _reasoning_config_from_env --------------------------------------------


def test_env_none_when_unset():
    _reset()
    assert PATCH._reasoning_config_from_env() is None


def test_env_parses_valid_json():
    _reset()
    ENVS.OMNI_REASONING_CONFIG = GOOD_JSON
    cfg = PATCH._reasoning_config_from_env()
    assert isinstance(cfg, _ReasoningConfig)
    assert cfg.thinking_token_budget == 128


def test_env_bad_json_returns_none():
    _reset()
    ENVS.OMNI_REASONING_CONFIG = "{not json"
    assert PATCH._reasoning_config_from_env() is None


# --- ThinkingLimitEngineArgsPatch.create_engine_config ----------------------


def test_cli_instance_config_is_kept_and_initialized():
    _reset()
    cfg_in = _ReasoningConfig(thinking_token_budget=64)
    vllm_config = _create_config(_engine_args_with(cfg_in))
    assert vllm_config.reasoning_config is cfg_in
    # The native VllmConfig construction ran initialize_token_ids.
    assert cfg_in.initialized_with is vllm_config.model_config


def test_dict_reasoning_config_is_coerced():
    _reset()
    ea = _engine_args_with({"reasoning_start_str": "<|thinking|>"})
    vllm_config = _create_config(ea)
    assert isinstance(vllm_config.reasoning_config, _ReasoningConfig)
    assert vllm_config.reasoning_config.reasoning_start_str == "<|thinking|>"
    # The EngineArgs attribute itself is normalized in place.
    assert isinstance(ea.reasoning_config, _ReasoningConfig)


def test_json_string_reasoning_config_is_coerced():
    _reset()
    ea = _engine_args_with(GOOD_JSON)
    vllm_config = _create_config(ea)
    assert vllm_config.reasoning_config.thinking_token_budget == 128


def test_env_beats_cli():
    _reset()
    cli_cfg = _ReasoningConfig(thinking_token_budget=64)
    ENVS.OMNI_REASONING_CONFIG = GOOD_JSON  # budget 128
    vllm_config = _create_config(_engine_args_with(cli_cfg))
    assert vllm_config.reasoning_config is not cli_cfg
    assert vllm_config.reasoning_config.thinking_token_budget == 128


def test_bad_env_leaves_cli_value_in_place():
    _reset()
    cli_cfg = _ReasoningConfig(thinking_token_budget=64)
    ENVS.OMNI_REASONING_CONFIG = "{not json"
    vllm_config = _create_config(_engine_args_with(cli_cfg))
    assert vllm_config.reasoning_config is cli_cfg
    assert vllm_config.reasoning_config.thinking_token_budget == 64


def test_bad_cli_dict_falls_back_to_none():
    """A garbage value (neither dict/str/ReasoningConfig) is logged, not raised,
    and never reaches the native VllmConfig construction."""
    _reset()
    ea = _engine_args_with(12345)  # uncoercible type
    vllm_config = _create_config(ea)
    assert vllm_config.reasoning_config is None


def test_none_config_leaves_vllm_config_untouched():
    _reset()
    vllm_config = _create_config(_engine_args_with(None))
    assert vllm_config.reasoning_config is None


def test_create_engine_config_forwards_usage_context_and_headless():
    _reset()
    calls = []
    original = _EngineArgs.create_engine_config

    def _recording(self, usage_context=None, headless=False):
        calls.append((usage_context, headless))
        return original(self, usage_context, headless)

    PATCH.ThinkingLimitEngineArgsPatch._upstream_create_engine_config = _recording
    try:
        _create_config(_engine_args_with(None))
    finally:
        PATCH.ThinkingLimitEngineArgsPatch._upstream_create_engine_config = original
    assert calls == [(None, False)]


# --- ThinkingLimitInputProcessorPatch.process_inputs ------------------------


def test_injection_fills_in_the_server_default():
    _reset()
    vllm_config = _VllmConfig(
        reasoning_config=_ReasoningConfig(thinking_token_budget=128),
        model_config=_ModelConfigStub(),
    )
    out = _process(_InputProcessor(vllm_config), _SamplingParams())
    assert out["params"].thinking_token_budget == 128


def test_request_value_wins_over_server_default():
    _reset()
    vllm_config = _VllmConfig(
        reasoning_config=_ReasoningConfig(thinking_token_budget=128),
        model_config=_ModelConfigStub(),
    )
    out = _process(_InputProcessor(vllm_config), _SamplingParams(32))
    assert out["params"].thinking_token_budget == 32


def test_zero_request_budget_wins_over_server_default():
    """A request explicitly asking for budget 0 is a value, not 'unset'."""
    _reset()
    vllm_config = _VllmConfig(
        reasoning_config=_ReasoningConfig(thinking_token_budget=128),
        model_config=_ModelConfigStub(),
    )
    out = _process(_InputProcessor(vllm_config), _SamplingParams(0))
    assert out["params"].thinking_token_budget == 0


def test_no_default_configured_is_a_no_op():
    _reset()
    vllm_config = _VllmConfig(
        reasoning_config=_ReasoningConfig(), model_config=_ModelConfigStub()
    )
    out = _process(_InputProcessor(vllm_config), _SamplingParams())
    assert out["params"].thinking_token_budget is None


def test_no_reasoning_config_is_a_no_op():
    _reset()
    vllm_config = _VllmConfig(reasoning_config=None, model_config=_ModelConfigStub())
    out = _process(_InputProcessor(vllm_config), _SamplingParams())
    assert out["params"].thinking_token_budget is None


def test_invalid_server_budget_is_not_injected():
    """Negative / bool budgets in the config are ignored, not injected."""
    _reset()
    for bad in (-1, True):
        vllm_config = _VllmConfig(
            reasoning_config=_ReasoningConfig(thinking_token_budget=bad),
            model_config=_ModelConfigStub(),
        )
        out = _process(_InputProcessor(vllm_config), _SamplingParams())
        assert out["params"].thinking_token_budget is None, f"bad={bad!r}"


def test_pooling_params_are_left_alone():
    _reset()
    vllm_config = _VllmConfig(
        reasoning_config=_ReasoningConfig(thinking_token_budget=128),
        model_config=_ModelConfigStub(),
    )
    pooling = _PoolingParams()
    out = _process(_InputProcessor(vllm_config), pooling)
    assert out["params"] is pooling
    assert not hasattr(pooling, "thinking_token_budget")


def test_original_process_inputs_still_receives_everything():
    _reset()
    vllm_config = _VllmConfig(reasoning_config=None, model_config=_ModelConfigStub())
    out = _process(_InputProcessor(vllm_config), _SamplingParams())
    assert out["request_id"] == "req-0"
    assert out["kwargs"]["supported_tasks"] == ("generate",)


def test_params_passed_as_kwarg_is_also_found():
    """_find_sampling_params must also see params= in kwargs (0.25.1 callers)."""
    _reset()
    vllm_config = _VllmConfig(
        reasoning_config=_ReasoningConfig(thinking_token_budget=128),
        model_config=_ModelConfigStub(),
    )
    out = PATCH.ThinkingLimitInputProcessorPatch.process_inputs(
        _InputProcessor(vllm_config),
        request_id="req-kw",
        prompt="hello",
        params=_SamplingParams(),
        supported_tasks=("generate",),
    )
    assert out["params"].thinking_token_budget == 128


# --- apply() / relay-chain integration --------------------------------------


def test_apply_actually_installs_the_attributes():
    """Applying after another EngineArgs patch preserves the whole chain."""
    _reset()

    # Model a previously-applied patch (LoPT / repetition-detection). Production
    # captures the active create_engine_config at apply time; verify the chain.
    upstream_create_engine_config = _EngineArgs.create_engine_config
    prior_calls = []

    def prior_create_engine_config(self, usage_context=None, headless=False):
        cfg = upstream_create_engine_config(self, usage_context, headless)
        prior_calls.append((usage_context, headless))
        return cfg

    _EngineArgs.create_engine_config = prior_create_engine_config

    PATCH.ThinkingLimitEngineArgsPatch.apply()
    PATCH.ThinkingLimitInputProcessorPatch.apply()
    try:
        assert _EngineArgs.create_engine_config is not prior_create_engine_config
        assert (
            PATCH.ThinkingLimitEngineArgsPatch._upstream_create_engine_config
            is prior_create_engine_config
        )
        ea = _EngineArgs(_ReasoningConfig(thinking_token_budget=64))
        vllm_config = ea.create_engine_config(usage_context="test", headless=True)
        assert prior_calls == [("test", True)]
        assert vllm_config.reasoning_config.thinking_token_budget == 64

        # InputProcessor.apply() installed process_inputs on the stub.
        out = _InputProcessor(vllm_config).process_inputs(
            "req-1", "hi", _SamplingParams(), supported_tasks=("generate",)
        )
        assert out["params"].thinking_token_budget == 64
    finally:
        _EngineArgs.create_engine_config = upstream_create_engine_config
        if hasattr(_EngineArgs, "_omni_npu_applied_patches"):
            _EngineArgs._omni_npu_applied_patches.clear()
        if hasattr(_InputProcessor, "_omni_npu_applied_patches"):
            _InputProcessor._omni_npu_applied_patches.clear()
        # Restore the class-level seed for later direct-call tests.
        PATCH.ThinkingLimitEngineArgsPatch._upstream_create_engine_config = (
            _EngineArgs.create_engine_config
        )


if __name__ == "__main__":
    # Standalone (no pytest): load the patch first, then run every test_*.
    PATCH = _load_patch()
    run_standalone_tests(globals())
