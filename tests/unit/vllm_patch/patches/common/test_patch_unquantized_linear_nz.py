# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for patch_unquantized_linear_nz (vLLM 0.25.1).

Pure-stdlib: every vLLM / torch / torch_npu / omni_npu dependency is stubbed,
so this needs no vllm, torch or NPU. Run with pytest or directly.

The patch registers against upstream UnquantizedLinearMethod, apply() swaps
the method, and the FRACTAL_NZ cast + is_weight_nz mark follow the
unquant_bmm_nz switch: only 16-bit float weights are cast (fp32 stays ND:
MHC phi feeds aclnnAiInfra* ops that require ND format), and an already
marked weight is never cast twice.
"""

import contextlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

PATCH_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / "omni" / "vllm_patches").is_dir())
    / "omni/vllm_patches/patches/common/patch_unquantized_linear_nz.py"
)

_TESTS_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "unit" / "vllm_patch").is_dir()
)
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from unit.vllm_patch.patches.patch_test_utils import run_standalone_tests  # noqa: E402


_STUB_NAMES = (
    "torch",
    "torch_npu",
    "vllm",
    "vllm.logger",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.linear",
    "vllm.model_executor.utils",
    "omni_npu",
    "omni_npu.model_config",
    "omni_npu.model_config.config_loader",
    "omni_npu.model_config.config_loader.loader",
    "omni_npu.vllm_patches",
    "omni_npu.vllm_patches.core",
)
_MISSING = object()


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _install_stubs():
    """Fresh stub state for one patch load; returns a SimpleNamespace."""
    registered = []
    orig_calls = []
    cast_calls = []

    class _Format:
        FRACTAL_NZ = "FRACTAL_NZ"

    def npu_format_cast(tensor, fmt):
        cast_calls.append((tensor, fmt))
        return f"NZ({tensor})"

    # torch is touched for the ``torch.nn.Module`` annotation and the
    # float16/bfloat16 dtype guard in the cast condition.
    _module(
        "torch",
        nn=types.SimpleNamespace(Module=object),
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )
    _module(
        "torch_npu",
        npu_format_cast=npu_format_cast,
        Format=_Format,
    )

    for pkg in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "omni_npu",
        "omni_npu.model_config",
        "omni_npu.model_config.config_loader",
        "omni_npu.vllm_patches",
    ):
        _module(pkg)

    _module("vllm.logger", init_logger=lambda name: logging.getLogger(name))

    class UnquantizedLinearMethod:
        def process_weights_after_loading(self, layer):
            orig_calls.append((self, layer))

    _module(
        "vllm.model_executor.layers.linear",
        UnquantizedLinearMethod=UnquantizedLinearMethod,
    )

    def set_weight_attrs(param, attrs):
        for key, value in attrs.items():
            setattr(param, key, value)

    _module("vllm.model_executor.utils", set_weight_attrs=set_weight_attrs)

    class _OperatorOptConfig:
        def __init__(self):
            self.unquant_bmm_nz = True

    class _ModelExtraConfig:
        def __init__(self):
            self.operator_opt_config = _OperatorOptConfig()

    mec = _ModelExtraConfig()
    _module(
        "omni_npu.model_config.config_loader.loader",
        model_extra_config=mec,
    )

    class VLLMPatch:
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

    def register_patch(name, target):
        def decorator(cls):
            cls._target = target
            registered.append((name, target, cls))
            return cls

        return decorator

    _module(
        "omni_npu.vllm_patches.core",
        VLLMPatch=VLLMPatch,
        register_patch=register_patch,
    )

    return types.SimpleNamespace(
        registered=registered,
        orig_calls=orig_calls,
        cast_calls=cast_calls,
        mec=mec,
        linear_cls=UnquantizedLinearMethod,
    )


@contextlib.contextmanager
def _loaded_patch(module_name="patch_unq_nz_test"):
    """Load the real patch file under controlled stubs; restore sys.modules."""
    snapshot = {name: sys.modules.get(name, _MISSING) for name in _STUB_NAMES}
    try:
        state = _install_stubs()
        spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        yield mod, state
    finally:
        for name, value in snapshot.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        sys.modules.pop(module_name, None)


class _FakeWeight:
    # bfloat16 by default: the dtype NZ cast actually targets.
    def __init__(self, dtype="bfloat16"):
        self.data = "RAW"
        self.dtype = dtype


class _FakeLayer:
    def __init__(self, dtype="bfloat16"):
        self.weight = _FakeWeight(dtype)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_registers_against_upstream_class():
    with _loaded_patch() as (mod, st):
        assert len(st.registered) == 1
        name, target, cls = st.registered[0]
        assert name == "NPUUnquantizedLinearMethod"
        assert target is st.linear_cls
        assert cls._attr_names_to_apply == ["process_weights_after_loading"]
        assert (
            cls.process_weights_after_loading
            is mod._patched_process_weights_after_loading
        )


def test_apply_swaps_method_on_upstream_class():
    with _loaded_patch() as (mod, st):
        mod.NPUUnquantizedLinearMethodPatch.apply()
        assert (
            st.linear_cls.process_weights_after_loading
            is mod._patched_process_weights_after_loading
        )
        assert (
            st.linear_cls._omni_npu_applied_patches["process_weights_after_loading"]
            == "NPUUnquantizedLinearMethodPatch"
        )


def test_cast_and_mark_when_switch_on():
    with _loaded_patch() as (mod, st):
        mod.NPUUnquantizedLinearMethodPatch.apply()
        st.mec.operator_opt_config.unquant_bmm_nz = True
        method = st.linear_cls()
        layer = _FakeLayer()
        method.process_weights_after_loading(layer)
        assert st.orig_calls and st.orig_calls[-1][1] is layer
        assert st.cast_calls == [("RAW", "FRACTAL_NZ")]
        assert layer.weight.data == "NZ(RAW)"
        assert getattr(layer.weight, "is_weight_nz", False) is True


def test_no_cast_when_switch_off():
    with _loaded_patch() as (mod, st):
        mod.NPUUnquantizedLinearMethodPatch.apply()
        st.mec.operator_opt_config.unquant_bmm_nz = False
        method = st.linear_cls()
        layer = _FakeLayer()
        method.process_weights_after_loading(layer)
        assert st.orig_calls and st.orig_calls[-1][1] is layer
        assert st.cast_calls == []
        assert layer.weight.data == "RAW"
        assert not hasattr(layer.weight, "is_weight_nz")


def test_no_double_cast_when_already_nz():
    with _loaded_patch() as (mod, st):
        mod.NPUUnquantizedLinearMethodPatch.apply()
        st.mec.operator_opt_config.unquant_bmm_nz = True
        method = st.linear_cls()
        layer = _FakeLayer()
        layer.weight.is_weight_nz = True
        method.process_weights_after_loading(layer)
        assert st.cast_calls == []
        assert layer.weight.data == "RAW"


def test_no_cast_for_fp32_weight():
    # Regression guard (92B smoke): MHC phi is a fp32 ReplicatedLinear.
    # Casting it to FRACTAL_NZ breaks aclnnAiInfraManifoldConstrainedHyper
    # ConnectionPre, whose C++ layer requires "phi tensor format must be ND".
    with _loaded_patch() as (mod, st):
        mod.NPUUnquantizedLinearMethodPatch.apply()
        st.mec.operator_opt_config.unquant_bmm_nz = True
        method = st.linear_cls()
        layer = _FakeLayer(dtype="float32")
        method.process_weights_after_loading(layer)
        assert st.orig_calls and st.orig_calls[-1][1] is layer
        assert st.cast_calls == []
        assert layer.weight.data == "RAW"
        assert not hasattr(layer.weight, "is_weight_nz")


def test_fp16_weight_still_cast():
    # The dtype guard must not narrow the existing benefit: 16-bit float
    # weights (fp16 same as bf16) still take the NZ fast path.
    with _loaded_patch() as (mod, st):
        mod.NPUUnquantizedLinearMethodPatch.apply()
        st.mec.operator_opt_config.unquant_bmm_nz = True
        method = st.linear_cls()
        layer = _FakeLayer(dtype="float16")
        method.process_weights_after_loading(layer)
        assert st.cast_calls == [("RAW", "FRACTAL_NZ")]
        assert layer.weight.data == "NZ(RAW)"
        assert getattr(layer.weight, "is_weight_nz", False) is True


def test_header_declares_version_and_exit_condition():
    src = PATCH_PATH.read_text(encoding="utf-8")
    assert "0.25.1" in src, "docstring must name the target upstream version"
    assert "Remove once" in src, "docstring must state the removal condition"


if __name__ == "__main__":
    run_standalone_tests(globals())
