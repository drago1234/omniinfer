# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


class _IdentityAttn:
    prefix = "attn"

    def __call__(self, hs, _cos, _sin):
        return hs


def _identity(x):
    return x


def _return_last_arg(*args, **_kwargs):
    return args[-1]


def _bare_decoder_layer(*, use_mhc: bool = False):
    """Uninitialized OpenPanguV2DecoderLayer with identity attn/mlp/norm."""
    layer = model_mod.OpenPanguV2DecoderLayer.__new__(
        model_mod.OpenPanguV2DecoderLayer
    )
    layer.use_mhc = use_mhc
    layer.use_mhc_fusion_op = False
    layer.use_mega_mhc_fusion_op = False
    layer.engram = None
    layer.use_post_norm = False
    layer.side_stream = None
    layer.hidden_size = 4
    layer.mhc_num_stream = 1
    layer.input_layernorm = _identity
    layer.post_attention_layernorm = _identity
    layer.pre_mlp_layernorm = _identity
    layer.post_mlp_layernorm = _identity
    layer.self_attn = _IdentityAttn()
    layer.mlp = _identity
    return layer


def test_mhc_head_reuses_hidden_states_as_residual():
    """mhc_head residual is the input tensor, not a clone (line 1701)."""
    layer = _bare_decoder_layer(use_mhc=False)
    hidden_states = torch.randn(2, 4)

    out, residual, h_post, h_res, sk_event = layer.mhc_head(hidden_states)

    assert residual is hidden_states
    assert torch.equal(out, hidden_states)
    assert h_post is None
    assert h_res is None
    assert sk_event is None


def test_mhc_head_reuses_residual_when_mhc_enabled():
    layer = _bare_decoder_layer(use_mhc=True)
    hidden_states = torch.randn(2, 4)
    mixed = torch.randn(2, 4)
    layer.attn_mhc_module = MagicMock()
    layer.attn_mhc_module.mhc_pre.return_value = (mixed, "h_post", "h_res")
    layer.attn_mhc_module.mhc_sinkhorn.return_value = "h_res_sk"

    out, residual, h_post, h_res, sk_event = layer.mhc_head(hidden_states)

    assert residual is hidden_states
    assert torch.equal(out, mixed)
    assert h_post == "h_post"
    assert h_res == "h_res_sk"
    assert sk_event is None


def test_forward_naive_reuses_hidden_states_as_residual():
    """_forward_naive attn/FFN residuals alias the live tensors (lines 2074, 2106)."""
    layer = _bare_decoder_layer(use_mhc=False)
    hidden_states = torch.ones(2, 4)
    cos = torch.zeros(2, 2)
    sin = torch.zeros(2, 2)

    out = layer._forward_naive(hidden_states, cos, sin)

    # identity attn/mlp + residual add: hs -> 2*hs -> 4*hs
    torch.testing.assert_close(out, hidden_states * 4)


def test_forward_naive_passes_uncloned_residual_into_mhc_post():
    layer = _bare_decoder_layer(use_mhc=True)
    hidden_states = torch.randn(2, 4)
    attn_residuals = []
    mlp_residuals = []

    layer.attn_mhc_task_key = "attn_key"
    layer.mlp_mhc_task_key = "mlp_key"
    layer.attn_mhc_module = MagicMock()
    layer.mlp_mhc_module = MagicMock()
    layer.attn_mhc_module.mhc_pre.return_value = (hidden_states, "attn_post", "attn_res")
    layer.mlp_mhc_module.mhc_pre.return_value = (hidden_states, "mlp_post", "mlp_res")

    def attn_post(hs, h_post, residual, h_res):
        attn_residuals.append(residual)
        return hs

    def mlp_post(hs, h_post, residual, h_res):
        mlp_residuals.append(residual)
        return hs

    layer.attn_mhc_module.mhc_post.side_effect = attn_post
    layer.mlp_mhc_module.mhc_post.side_effect = mlp_post

    with patch.object(
        model_mod, "maybe_register_mhc_task", side_effect=_return_last_arg
    ), patch.object(
        model_mod, "resolve_mhc_h_res", side_effect=_return_last_arg
    ):
        out = layer._forward_naive(
            hidden_states,
            torch.zeros(2, 2),
            torch.zeros(2, 2),
        )

    assert attn_residuals == [hidden_states]
    assert attn_residuals[0] is hidden_states
    assert mlp_residuals == [hidden_states]
    assert mlp_residuals[0] is hidden_states
    assert out is hidden_states


def _fusion_layer(*, use_mega=False, use_old_fusion=False):
    layer = _bare_decoder_layer(use_mhc=True)
    layer.use_mhc_fusion_op = use_old_fusion
    layer.use_mega_mhc_fusion_op = use_mega
    layer.mhc_num_stream = 4
    layer.hidden_size = 2560
    return layer


def test_sandwich_uses_mega_mhc_and_sinkhorn(monkeypatch):
    layer = _fusion_layer(use_mega=True)
    h_out = torch.randn(2, 2560, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 2560, dtype=torch.bfloat16)
    h_post = torch.randn(2, 4)
    h_res = torch.randn(2, 4, 4)
    h_in_prime = torch.randn_like(h_out)
    h_post_new = torch.randn_like(h_post)
    h_res_new = torch.randn_like(h_res)
    x_2_out = torch.randn_like(residual)
    sk_h_res = torch.randn_like(h_res)
    calls = {}

    def fake_mega(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return h_in_prime, h_post_new, h_res_new, x_2_out

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(npu_ai_infra_mega_mhc=fake_mega),
        raising=False,
    )
    pre_mhc = SimpleNamespace(
        phi_weight_3bf16=torch.randn(3, 24, 4 * 2560, dtype=torch.bfloat16),
        branch_alpha=torch.tensor([0.1, 0.2, 0.3]),
        branch_beta=torch.randn(24),
        norm_eps=1e-6,
        hc_eps=1e-6,
        mhc_sinkhorn=MagicMock(return_value=sk_h_res),
    )
    post_norm = SimpleNamespace(weight_fp32=torch.ones(2560))
    pre_norm = SimpleNamespace(weight_fp32=torch.ones(2560) * 2)
    block_norm = SimpleNamespace(weight_fp32=torch.ones(4 * 2560))

    out, residual_out, h_post_out, h_res_out, sk_event = (
        layer.mhc_sandwich_norm_post_pre(
            h_out, residual, h_post, h_res,
            post_norm, MagicMock(), block_norm, pre_mhc, pre_norm,
            return_h_in_f32=True,
        )
    )

    assert calls["args"][0] is h_out
    assert calls["args"][1].shape == (2, 4, 2560)
    assert calls["args"][2] is h_post
    assert calls["args"][3] is h_res
    assert calls["args"][4] is pre_mhc.phi_weight_3bf16
    assert calls["args"][5] is pre_mhc.branch_alpha
    assert calls["args"][6] is pre_mhc.branch_beta
    assert calls["args"][7] is pre_norm.weight_fp32
    assert calls["args"][8] is post_norm.weight_fp32
    assert calls["args"][9] is block_norm.weight_fp32
    assert calls["kwargs"]["norm_eps"] == pre_mhc.norm_eps
    assert calls["kwargs"]["mhc_pre_norm_eps"] == pre_mhc.hc_eps
    assert calls["kwargs"]["hc_eps"] == pre_mhc.hc_eps
    assert calls["kwargs"]["compute_mode"] == 2
    pre_mhc.mhc_sinkhorn.assert_called_once_with(h_res_new)
    assert out["hidden_states_bf16"] is h_in_prime
    assert residual_out is x_2_out
    assert h_post_out is h_post_new
    assert h_res_out is sk_h_res
    assert sk_event is None


def test_old_mhc_fusion_branch_still_used_without_mega(monkeypatch):
    layer = _fusion_layer(use_old_fusion=True)
    hs = torch.randn(2, 2560, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 2560, dtype=torch.bfloat16)
    called = []

    def fake_sandwich(*_args, **_kwargs):
        called.append("sandwich")
        return hs, residual, None

    def fake_split(*_args, **_kwargs):
        called.append("split")
        return torch.randn(2, 4), torch.randn(2, 4, 4)

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_ai_infra_mhc_sandwich_norm_post_preonly_v2=fake_sandwich,
            npu_ai_infra_mhc_pre_split_post_res=fake_split,
        ),
        raising=False,
    )
    pre_mhc = SimpleNamespace(
        phi_weight_pre=torch.randn(4, 4 * 2560),
        phi_weight_post_res=torch.randn(20, 4 * 2560),
        branch_alpha_pre=torch.tensor(0.1),
        branch_alpha_post_res=torch.randn(2),
        branch_beta_pre=torch.randn(4),
        branch_beta_post_res=torch.randn(20),
        norm_eps=1e-6,
        hc_eps=1e-6,
        mhc_sinkhorn=lambda x: x,
    )
    layer.mhc_sandwich_norm_post_pre(
        hs, residual, torch.randn(2, 4), torch.randn(2, 4, 4),
        SimpleNamespace(weight_fp32=torch.ones(2560)),
        MagicMock(), None, pre_mhc,
        SimpleNamespace(weight_fp32=torch.ones(2560)),
    )
    assert called == ["sandwich", "split"]


def test_sandwich_skips_mega_on_model_tail(monkeypatch):
    layer = _fusion_layer(use_mega=True, use_old_fusion=True)
    hs = torch.randn(2, 2560, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 2560, dtype=torch.bfloat16)
    called = []

    def fake_sandwich(*_args, **_kwargs):
        called.append("old")
        return hs, residual, None

    def fake_mega(*_args, **_kwargs):
        called.append("mega")
        return hs, torch.randn(2, 4), torch.randn(2, 4, 4), residual

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(
            npu_ai_infra_mega_mhc=fake_mega,
            npu_ai_infra_mhc_sandwich_norm_post_preonly_v2=fake_sandwich,
        ),
        raising=False,
    )
    layer.mhc_sandwich_norm_post_pre(
        hs, residual, torch.randn(2, 4), torch.randn(2, 4, 4),
        SimpleNamespace(weight_fp32=torch.ones(2560)),
        MagicMock(), None,
        SimpleNamespace(
            phi_weight_pre=torch.randn(4, 4 * 2560),
            branch_alpha_pre=torch.tensor(0.1),
            branch_beta_pre=torch.randn(4),
            norm_eps=1e-6,
            hc_eps=1e-6,
        ),
        SimpleNamespace(weight_fp32=torch.ones(2560)),
        is_model_tail=True,
    )
    assert called == ["old"]


def test_sandwich_skips_mega_when_mhc_disabled(monkeypatch):
    """MTP layers inherit the global mega flag but have no MHC modules."""
    layer = _fusion_layer(use_mega=True)
    layer.use_mhc = False
    hs = torch.randn(2, 2560, dtype=torch.bfloat16)
    residual = torch.randn(2, 2560, dtype=torch.bfloat16)
    called = []

    def fake_mega(*_args, **_kwargs):
        called.append("mega")
        return hs, None, None, residual

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(npu_ai_infra_mega_mhc=fake_mega),
        raising=False,
    )
    out, residual_out, *_ = layer.mhc_sandwich_norm_post_pre(
        hs, residual, None, None,
        _identity, None, None, None, _identity,
    )
    assert called == []
    assert out is not None
    assert residual_out is not None


def test_sandwich_mega_passes_none_mid_norm(monkeypatch):
    layer = _fusion_layer(use_mega=True)
    hs = torch.randn(2, 2560, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 2560, dtype=torch.bfloat16)
    captured = {}

    def fake_mega(*args, **_kwargs):
        captured["gamma_2"] = args[9]
        return hs, torch.randn(2, 4), torch.randn(2, 4, 4), residual

    monkeypatch.setattr(
        torch.ops,
        "custom",
        SimpleNamespace(npu_ai_infra_mega_mhc=fake_mega),
        raising=False,
    )
    pre_mhc = SimpleNamespace(
        phi_weight_3bf16=torch.randn(3, 24, 4 * 2560, dtype=torch.bfloat16),
        branch_alpha=torch.tensor([0.1, 0.2, 0.3]),
        branch_beta=torch.randn(24),
        norm_eps=1e-6,
        hc_eps=1e-6,
        mhc_sinkhorn=lambda x: x,
    )
    layer.mhc_sandwich_norm_post_pre(
        hs, residual, torch.randn(2, 4), torch.randn(2, 4, 4),
        SimpleNamespace(weight_fp32=torch.ones(2560)),
        MagicMock(), None, pre_mhc,
        SimpleNamespace(weight_fp32=torch.ones(2560)),
    )
    assert captured["gamma_2"] is None


def test_dense_mlp_sk_scope(monkeypatch):
    monkeypatch.setattr(model_mod, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(model_mod, "MergedColumnParallelFlashCommLinear", lambda *a, **k: _identity)
    monkeypatch.setattr(model_mod, "RowParallelFlashCommLinear", lambda *a, **k: _identity)
    monkeypatch.setattr(model_mod, "SiluAndMul", lambda: _identity)
    mlp = model_mod.OpenPanguV2MLP(4, 8, "silu", prefix="model.layers.0.mlp")
    x = torch.ones(2, 4)
    assert torch.equal(mlp.forward(x), x)
    assert mlp._fuse_dense_mlp is True
    assert mlp._sk_dense_name == "dense_mlp_0"


def test_mhc_post_pre_sk_scope():
    layer = _bare_decoder_layer(use_mhc=True)
    layer.use_mhc_fusion_op = False
    layer.layer_idx = 0
    hidden = torch.ones(2, 4)
    post_mhc = MagicMock()
    post_mhc.mhc_post.return_value = hidden
    pre_mhc = MagicMock()
    pre_mhc.mhc_pre.return_value = (hidden, "h_post", "h_res")
    pre_mhc.mhc_sinkhorn.return_value = "h_res"

    out, residual, h_post, h_res, sk_event = layer.mhc_sandwich_norm_post_pre(
        hidden, hidden, None, None, _identity, post_mhc, None, pre_mhc, _identity,
        is_model_tail=True,
    )

    post_mhc.mhc_post.assert_called_once()
    assert out is hidden
    assert residual is None
    assert h_post == "h_post"
    assert h_res == "h_res"
    assert sk_event is None
