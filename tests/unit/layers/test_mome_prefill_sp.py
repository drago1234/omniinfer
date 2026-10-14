# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MoME-SP prefill helpers shared by MLA and DSA (A5 TTFT path)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _new_attention(cls):
    attention = cls.__new__(cls)
    torch.nn.Module.__init__(attention)
    return attention


def _dsa_cls():
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    return NPUDeepseekSparseAttention


def _mla_cls():
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    return NPUDeepseekMLAAttention


def _install_mome_conv(attention, weights, caches):
    attention.conv = SimpleNamespace(
        prefix="layers.0.conv",
        qa_conv=SimpleNamespace(weight=weights[0]),
        compresskv_conv=SimpleNamespace(weight=weights[1]),
        o_conv=SimpleNamespace(weight=weights[2]),
        kv_cache=tuple(caches),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state_indice", "conv_name"),
    [(0, "qa_conv"), (1, "compresskv_conv"), (2, "o_conv")],
)
def test_mome_sp_args_select_conv_weight_and_cache(monkeypatch, state_indice, conv_name):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    attention = _new_attention(_dsa_cls())
    attention.enable_mome_sp = True
    attention.on_ascend950 = True
    attention.noncontiguous_kv = True
    weights = [object(), object(), object()]
    caches = [object(), object(), object()]
    conv_sp_meta = (object(), object(), object())
    _install_mome_conv(attention, weights, caches)
    metadata = SimpleNamespace(prefill=object(), conv_sp_meta=conv_sp_meta)
    context = SimpleNamespace(attn_metadata={attention.conv.prefix: metadata})
    monkeypatch.setattr(mla_mod, "get_forward_context", lambda: context)

    args = attention._get_mome_prefill_sp_args(state_indice)

    assert getattr(attention.conv, conv_name).weight is weights[state_indice]
    assert args == (weights[state_indice], caches[state_indice], *conv_sp_meta)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("enable_mome_sp", "on_a5", "noncontiguous_kv"),
    [(False, True, True), (True, False, True), (True, True, False)],
    ids=["mome_sp_off", "not_a5", "contiguous_kv"],
)
def test_mome_sp_args_gated_by_switch_hardware_and_kv_layout(
    monkeypatch, enable_mome_sp, on_a5, noncontiguous_kv
):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    attention = _new_attention(_mla_cls())
    attention.enable_mome_sp = enable_mome_sp
    attention.on_ascend950 = on_a5
    attention.noncontiguous_kv = noncontiguous_kv
    context = MagicMock(side_effect=AssertionError("context must not be read"))
    monkeypatch.setattr(mla_mod, "get_forward_context", context)

    assert attention._get_mome_prefill_sp_args(0) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "attn_metadata",
    [
        None,
        {"layers.0.conv": SimpleNamespace(prefill=None, conv_sp_meta=(1, 2, 3))},
        {"layers.0.conv": SimpleNamespace(prefill=object(), conv_sp_meta=None)},
    ],
    ids=["dummy_run", "decode_only", "no_sp_metadata"],
)
def test_mome_sp_args_none_without_prefill_sp_metadata(monkeypatch, attn_metadata):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    attention = _new_attention(_mla_cls())
    attention.enable_mome_sp = True
    attention.on_ascend950 = True
    attention.noncontiguous_kv = True
    _install_mome_conv(attention, [object()] * 3, [object()] * 3)
    context = SimpleNamespace(attn_metadata=attn_metadata)
    monkeypatch.setattr(mla_mod, "get_forward_context", lambda: context)

    assert attention._get_mome_prefill_sp_args(1) is None


@pytest.mark.unit
@pytest.mark.parametrize("cls_getter", [_mla_cls, _dsa_cls], ids=["mla", "dsa"])
def test_mome_prefill_sp_uses_inplace_conv(monkeypatch, cls_getter):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    attention = _new_attention(cls_getter())
    x = torch.randn(2, 8)
    mome_sp_args = tuple(object() for _ in range(5))
    attention._get_mome_prefill_sp_args = MagicMock(return_value=mome_sp_args)
    # The A5 inplace kernel may return None. Path selection must depend on
    # argument availability, not on the convolution operator's return value.
    conv_sp_mock = MagicMock(return_value=None)
    monkeypatch.setattr(mla_mod, "conv_sp", conv_sp_mock)

    for state_indice in (0, 1):
        assert attention._apply_mome_prefill_sp(x, state_indice, inplace=True)
        attention._get_mome_prefill_sp_args.assert_called_with(state_indice)
        conv_sp_mock.assert_called_with(x, *mome_sp_args, inplace=True)


@pytest.mark.unit
def test_mome_prefill_sp_unavailable_returns_false(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    attention = _new_attention(_dsa_cls())
    attention._get_mome_prefill_sp_args = MagicMock(return_value=None)
    conv_sp_mock = MagicMock()
    monkeypatch.setattr(mla_mod, "conv_sp", conv_sp_mock)

    assert not attention._apply_mome_prefill_sp(torch.randn(2, 8), 0, inplace=True)
    conv_sp_mock.assert_not_called()


@pytest.mark.unit
def test_mla_mome_out_sp_transposes_before_conv(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    attention = _new_attention(_mla_cls())
    attention.enable_o_conv_sp = True
    local_out = torch.randn(8, 4)
    aligned_out = torch.randn(8, 4)
    sp_out = torch.randn(2, 16)
    sp_manager = SimpleNamespace(
        align_tokens=MagicMock(return_value=aligned_out),
        tp_to_sp=MagicMock(return_value=sp_out),
    )
    mome_sp_args = tuple(object() for _ in range(5))
    attention._get_mome_prefill_sp_args = MagicMock(return_value=mome_sp_args)
    conv_sp_mock = MagicMock(return_value=None)
    monkeypatch.setattr(mla_mod, "conv_sp", conv_sp_mock)

    output = attention._apply_mome_out_prefill_sp(local_out, sp_manager)

    assert output is sp_out
    attention._get_mome_prefill_sp_args.assert_called_once_with(2)
    sp_manager.align_tokens.assert_called_once_with(local_out)
    sp_manager.tp_to_sp.assert_called_once_with(aligned_out)
    conv_sp_mock.assert_called_once_with(sp_out, *mome_sp_args, inplace=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("enable_o_conv_sp", "mome_sp_args"),
    [(False, (1, 2, 3, 4, 5)), (True, None)],
    ids=["o_conv_sp_off", "no_metadata"],
)
def test_mla_mome_out_sp_falls_back(enable_o_conv_sp, mome_sp_args):
    attention = _new_attention(_mla_cls())
    attention.enable_o_conv_sp = enable_o_conv_sp
    attention._get_mome_prefill_sp_args = MagicMock(return_value=mome_sp_args)
    sp_manager = SimpleNamespace(align_tokens=MagicMock(), tp_to_sp=MagicMock())

    assert attention._apply_mome_out_prefill_sp(torch.randn(2, 4), sp_manager) is None
    sp_manager.align_tokens.assert_not_called()
    sp_manager.tp_to_sp.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("requires_partition", [True, False])
def test_dsa_mome_sp_o_conv_keeps_local_tokens(monkeypatch, requires_partition):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod

    attention = _new_attention(_dsa_cls())
    cp_out = torch.randn(2, 8)
    sp_out = torch.randn(2, 8)
    conv_out = torch.randn(2, 8)
    tp_out = torch.randn(4, 4)
    projected = torch.randn(4 if requires_partition else 2, 6)
    sliced = torch.randn(2, 6)
    mome_sp_args = tuple(object() for _ in range(5))

    attention._get_mome_prefill_sp_args = MagicMock(return_value=mome_sp_args)
    attention.o_proj = MagicMock(return_value=(projected, None))
    attention.o_proj.tp_size = 1
    attention.o_proj.requires_input_partition.return_value = requires_partition
    sp_manager = SimpleNamespace(
        cp_to_sp=MagicMock(return_value=sp_out),
        sp_to_tp=MagicMock(return_value=tp_out),
        slice_tokens=MagicMock(return_value=sliced),
    )
    conv_sp_mock = MagicMock(return_value=conv_out)
    monkeypatch.setattr(dsa_mod, "conv_sp", conv_sp_mock)

    output = attention._apply_mome_out_prefill_sp(cp_out, sp_manager)

    attention._get_mome_prefill_sp_args.assert_called_once_with(2)
    sp_manager.cp_to_sp.assert_called_once_with(cp_out)
    conv_sp_mock.assert_called_once_with(sp_out, *mome_sp_args, inplace=False)
    if requires_partition:
        sp_manager.sp_to_tp.assert_called_once_with(conv_out)
        attention.o_proj.assert_called_once_with(tp_out)
        sp_manager.slice_tokens.assert_called_once_with(projected)
        assert output is sliced
    else:
        sp_manager.sp_to_tp.assert_not_called()
        attention.o_proj.assert_called_once_with(conv_out)
        sp_manager.slice_tokens.assert_not_called()
        assert output is projected


@pytest.mark.unit
def test_dsa_mome_sp_o_conv_unavailable_returns_none(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod

    attention = _new_attention(_dsa_cls())
    attention._get_mome_prefill_sp_args = MagicMock(return_value=None)
    conv_sp_mock = MagicMock()
    monkeypatch.setattr(dsa_mod, "conv_sp", conv_sp_mock)

    assert attention._apply_mome_out_prefill_sp(torch.randn(2, 8), SimpleNamespace()) is None
    conv_sp_mock.assert_not_called()
