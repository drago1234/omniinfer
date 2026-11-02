# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


pytestmark = pytest.mark.unit


@pytest.fixture
def q_norm(monkeypatch):
    from omni_npu.layers.quantization.hifloat8 import Hifloat8FCLinearMethod
    from omni_npu.layers.quantization.mxfp8 import Mxfp8FCLinearMethod
    from omni_npu.v1.layers.attention import npu_mla

    config = npu_mla.model_extra_config.operator_opt_config
    monkeypatch.setattr(config, "enable_rmsnorm_quant_fusion", True)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    quantized = torch.zeros_like(x, dtype=torch.uint8)
    scale = torch.ones(2, 2, dtype=torch.uint8)
    mx_op = MagicMock(return_value=(quantized, scale, None))
    hif8_op = MagicMock(return_value=(quantized, None))
    monkeypatch.setattr(npu_mla.torch_npu, "npu_rms_norm_dynamic_mx_quant", mx_op)
    monkeypatch.setattr(npu_mla.torch_npu, "npu_rms_norm_quant_v2", hif8_op)

    methods = {"mxfp8": Mxfp8FCLinearMethod, "hifloat8": Hifloat8FCLinearMethod}

    def make_attention(quant_format, split=False):
        attention = npu_mla.MomeAttentionMixin()
        attention.on_ascend950 = True
        attention.split_q_up_in_multistream = split
        attention.skip_topk = False
        attention.q_a_layernorm = MagicMock(
            return_value=torch.ones_like(x), weight=torch.ones(64, dtype=x.dtype),
            variance_epsilon=1e-6,
        )
        attention._q_norm_quant_scale = torch.ones(1, dtype=torch.float32)
        for name in ("q_b_proj", "q_b_nope_proj", "q_b_pe_proj"):
            setattr(attention, name, MagicMock(quant_method=methods[quant_format](None)))
        attention.indexer = SimpleNamespace(
            wq_b=MagicMock(quant_method=methods[quant_format](None))
        )
        return attention

    return SimpleNamespace(
        make=make_attention, x=x, quantized=quantized, scale=scale,
        mx_op=mx_op, hif8_op=hif8_op, config=config,
    )


@pytest.mark.parametrize("quant_format", ["mxfp8", "hifloat8"])
@pytest.mark.parametrize("split", [False, True])
def test_q_float8_shared_by_projections_and_indexer(q_norm, quant_format, split):
    attention = q_norm.make(quant_format, split)
    result = attention._normalize_q_maybe_quant(q_norm.x)

    attention.q_a_layernorm.assert_not_called()
    if quant_format == "hifloat8":
        q_norm.hif8_op.assert_called_once()
        q_norm.mx_op.assert_not_called()
        assert set(result) == {"x_hif8"}
        assert result["x_hif8"] is q_norm.quantized
    else:
        q_norm.mx_op.assert_called_once()
        q_norm.hif8_op.assert_not_called()
        assert set(result) == {"x_mxfp8", "pertoken_scale"}
        assert result["x_mxfp8"] is q_norm.quantized
        assert result["pertoken_scale"] is q_norm.scale


@pytest.mark.parametrize("quant_format", [None, "mxfp8", "hifloat8"])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("gather_output", [False, True])
def test_mla_q_absorb_accepts_normalized_input_formats(
    q_norm, quant_format, split, gather_output
):
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    attention = q_norm.make(quant_format or "mxfp8", split)
    attention.qk_nope_head_dim = 4
    attention.qk_rope_head_dim = 2
    attention._q_project = NPUDeepseekMLAAttention._q_project.__get__(attention)
    attention._apply_rope = MagicMock(side_effect=lambda q, *_: q + 1)
    attention._q_nope_absorb = MagicMock(side_effect=lambda q: q * 2)

    # A projection may gather rows; its output determines the token count.
    tokens = q_norm.x.size(0) * (2 if gather_output else 1)
    heads = 2
    q_nope = torch.arange(tokens * heads * 4, dtype=torch.float32).view(tokens, heads, 4)
    q_pe = torch.arange(tokens * heads * 2, dtype=torch.float32).view(tokens, heads, 2)
    attention.q_b_nope_proj.return_value = (q_nope.flatten(1),)
    attention.q_b_pe_proj.return_value = (q_pe.flatten(1),)
    attention.q_b_proj.return_value = (torch.cat((q_nope, q_pe), dim=-1).flatten(1),)

    normalized = attention._normalize_q_maybe_quant(
        q_norm.x, allow_fusion=quant_format is not None
    )
    cos, sin = torch.ones(tokens, 1, 1, 2), torch.zeros(tokens, 1, 1, 2)
    actual_nope, actual_pe = NPUDeepseekMLAAttention._q_absorb(
        attention, normalized, cos, sin
    )

    torch.testing.assert_close(actual_nope, q_nope * 2)
    torch.testing.assert_close(actual_pe, q_pe + 1)
    assert attention._apply_rope.call_args.args[1:] == (cos, sin)
    if split:
        consumers = [attention.q_b_nope_proj, attention.q_b_pe_proj]
        attention.q_b_proj.assert_not_called()
    else:
        consumers = [attention.q_b_proj]
        attention.q_b_nope_proj.assert_not_called()
        attention.q_b_pe_proj.assert_not_called()
    for consumer in consumers:
        consumer.assert_called_once()
        assert consumer.call_args.args[0] is normalized


@pytest.mark.parametrize("quant_format", ["mxfp8", "hifloat8"])
@pytest.mark.parametrize(
    "reason",
    ["disabled", "communication", "other_device", "mixed_q", "mixed_indexer",
     "int8_indexer", "unquantized_q", "int8_q"],
)
def test_q_float8_falls_back_for_incompatible_consumers(q_norm, quant_format, reason):
    attention = q_norm.make(quant_format, split=True)
    other_format = "hifloat8" if quant_format == "mxfp8" else "mxfp8"
    if reason == "disabled":
        q_norm.config.enable_rmsnorm_quant_fusion = False
    elif reason == "other_device":
        attention.on_ascend950 = False
    elif reason == "mixed_q":
        attention.q_b_pe_proj.quant_method = SimpleNamespace(input_quant_format=other_format)
    elif reason in ("mixed_indexer", "int8_indexer"):
        attention.indexer.wq_b.quant_method = SimpleNamespace(
            input_quant_format=other_format if reason == "mixed_indexer" else "int8"
        )
    elif reason == "unquantized_q":
        attention.q_b_nope_proj.quant_method = None
    elif reason == "int8_q":
        attention.q_b_nope_proj.quant_method = SimpleNamespace(input_quant_format="int8")

    result = attention._normalize_q_maybe_quant(
        q_norm.x, allow_fusion=reason != "communication"
    )
    assert result is attention.q_a_layernorm.return_value
    attention.q_a_layernorm.assert_called_once_with(q_norm.x)
    q_norm.mx_op.assert_not_called()
    q_norm.hif8_op.assert_not_called()


@pytest.mark.parametrize("quant_format", ["mxfp8", "hifloat8"])
def test_q_float8_ignores_unused_indexer(q_norm, quant_format):
    attention = q_norm.make(quant_format)
    attention.skip_topk = True
    del attention.indexer
    assert isinstance(attention._normalize_q_maybe_quant(q_norm.x), dict)
    attention.q_a_layernorm.assert_not_called()


@pytest.mark.parametrize("quant_format", ["mxfp8", "hifloat8"])
@pytest.mark.parametrize("fusion_enabled", [False, True])
def test_q_float8_gathers_tensors_before_fusion(q_norm, quant_format, fusion_enabled):
    q_norm.config.enable_rmsnorm_quant_fusion = fusion_enabled
    attention = q_norm.make(quant_format)
    gathered = torch.zeros(4, 64, dtype=q_norm.x.dtype)
    sp_manager = SimpleNamespace(ag_tokens=MagicMock(return_value=gathered))

    result = attention._normalize_q_after_gather(q_norm.x, sp_manager)

    if fusion_enabled:
        sp_manager.ag_tokens.assert_called_once_with(q_norm.x)
        op = q_norm.hif8_op if quant_format == "hifloat8" else q_norm.mx_op
        assert op.call_args.args[0] is gathered
        assert isinstance(result, dict)
        attention.q_a_layernorm.assert_not_called()
    else:
        attention.q_a_layernorm.assert_called_once_with(q_norm.x)
        sp_manager.ag_tokens.assert_called_once_with(attention.q_a_layernorm.return_value)
        assert result is gathered
        q_norm.mx_op.assert_not_called()
        q_norm.hif8_op.assert_not_called()
