# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


pytestmark = pytest.mark.unit


class _FakeReplicatedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=False, **_kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size))


class _Linear(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


def _import_npu_mhc(monkeypatch, *, use_mega=True):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repo_root / "omni"))

    torch_npu_mod = types.ModuleType("torch_npu")
    torch_npu_mod.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.PretrainedConfig = SimpleNamespace
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    linear_mod = types.ModuleType("vllm.model_executor.layers.linear")
    linear_mod.ReplicatedLinear = _FakeReplicatedLinear
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name=None: SimpleNamespace(warning=lambda *_a: None)
    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.logger": logger_mod,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.linear": linear_mod,
    }.items():
        if name == "vllm":
            mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)

    loader_mod = types.ModuleType("omni_npu.model_config.config_loader.loader")
    loader_mod.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(use_mega_mhc_fusion_op=use_mega)
    )
    omni_pkg = types.ModuleType("omni_npu")
    omni_pkg.__path__ = [str(repo_root / "omni")]
    for name, mod in {
        "omni_npu": omni_pkg,
        "omni_npu.layers": types.ModuleType("omni_npu.layers"),
        "omni_npu.layers.mhc": types.ModuleType("omni_npu.layers.mhc"),
        "omni_npu.v1": types.ModuleType("omni_npu.v1"),
        "omni_npu.v1.utils": types.ModuleType("omni_npu.v1.utils"),
        "omni_npu.model_config": types.ModuleType("omni_npu.model_config"),
        "omni_npu.model_config.config_loader": types.ModuleType(
            "omni_npu.model_config.config_loader"
        ),
        "omni_npu.model_config.config_loader.loader": loader_mod,
    }.items():
        if name in ("omni_npu.layers", "omni_npu.layers.mhc", "omni_npu.v1"):
            mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["omni_npu.layers"].__path__ = [str(repo_root / "omni" / "layers")]
    sys.modules["omni_npu.layers.mhc"].__path__ = [
        str(repo_root / "omni" / "layers" / "mhc")
    ]
    sys.modules["omni_npu.v1.utils"].on_ascend950 = lambda: False

    monkeypatch.delitem(sys.modules, "omni_npu.layers.mhc.npu_mhc", raising=False)
    return importlib.import_module("omni_npu.layers.mhc.npu_mhc")


def _cfg():
    return SimpleNamespace(
        mhc_num_stream=4,
        hidden_size=2560,
        rms_norm_eps=1e-6,
        mhc_recur_norm=20,
        mhc_use_gamma=True,
    )


def test_pack_phi_3bf16_reconstructs_fp32(monkeypatch):
    module = _import_npu_mhc(monkeypatch)
    phi = torch.randn(24, 32)
    packed = module.pack_phi_3bf16(phi)
    assert packed.shape == (3, 24, 32)
    high, mid, low = packed.float().unbind(0)
    torch.testing.assert_close(
        high + 2**-8 * mid + 2**-16 * low, phi, atol=2e-5, rtol=0
    )


def test_process_weights_packs_3bf16_only_when_mega_enabled(monkeypatch):
    module = _import_npu_mhc(monkeypatch, use_mega=True)
    mhc = module.NPUmHC(_cfg(), pre_only=False)
    with torch.no_grad():
        mhc.phi.weight.fill_(0.3)
        mhc.norm_gamma.fill_(1.2)
    mhc.process_weights_after_loading()
    torch.testing.assert_close(mhc.phi_weight, mhc.phi.weight * mhc.norm_gamma)
    torch.testing.assert_close(
        mhc.phi_weight_3bf16, module.pack_phi_3bf16(mhc.phi_weight), atol=0, rtol=0
    )

    tail = module.NPUmHC(_cfg(), pre_only=True)
    with torch.no_grad():
        tail.phi.weight.fill_(0.3)
        tail.norm_gamma.fill_(1.2)
    tail.process_weights_after_loading()
    assert not hasattr(tail, "phi_weight_3bf16")


def test_process_weights_skips_3bf16_when_mega_disabled(monkeypatch):
    module = _import_npu_mhc(monkeypatch, use_mega=False)
    mhc = module.NPUmHC(_cfg(), pre_only=False)
    with torch.no_grad():
        mhc.phi.weight.fill_(0.3)
        mhc.norm_gamma.fill_(1.2)
    mhc.process_weights_after_loading()
    assert not hasattr(mhc, "phi_weight_3bf16")


def test_mhc_pre_pre_only_uses_hc_eps_for_norm_and_gate(monkeypatch):
    module = _import_npu_mhc(monkeypatch)
    mhc = module.NPUmHC.__new__(module.NPUmHC)
    nn.Module.__init__(mhc)
    mhc.num_stream = 2
    mhc.hidden_size = 2
    mhc.pre_only = True
    mhc.norm_eps = 1.0
    mhc.hc_eps = 0.25
    mhc.phi = _Linear(4, 2, bias=False)
    mhc.norm_gamma = nn.Parameter(torch.ones(4))
    mhc.branch_alpha_pre = nn.Parameter(torch.ones(1))
    mhc.branch_beta_pre = nn.Parameter(torch.zeros(2))

    with torch.no_grad():
        mhc.phi.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]))

    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    flattened = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + mhc.hc_eps)
    gates = F.sigmoid(F.linear(flattened * rsqrt * mhc.norm_gamma, mhc.phi.weight)) + mhc.hc_eps
    expected = torch.sum(gates.unsqueeze(-1) * hidden_states, dim=1)

    actual, h_post, h_res = mhc.mhc_pre(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert h_post is None
    assert h_res is None
