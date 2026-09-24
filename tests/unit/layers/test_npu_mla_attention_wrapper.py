# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from omni_npu.v1.layers.attention import npu_mla as wrapper_mod
from omni_npu.v1.layers.attention.npu_mla import NPUMLAAttention
from omni_npu.v1.layers.attention.weight_utils import refresh_mla_wrappers
from omni_npu.v1.layers.linear import UnquantizedFlashCommLinearMethod
from vllm.model_executor.layers.attention.mla_attention import MLAAttention


pytestmark = pytest.mark.unit


def _flashcomm_layer(weight: torch.Tensor):
    return SimpleNamespace(
        weight=weight,
        quant_method=UnquantizedFlashCommLinearMethod(),
    )


def _flashcomm_wrapper(weight: torch.Tensor):
    return SimpleNamespace(
        kv_b_proj=_flashcomm_layer(weight),
        kv_lora_rank=8,
        num_heads=2,
        qk_nope_head_dim=4,
        v_head_dim=4,
        quant_config=None,
        layer_name="model.layers.0.self_attn.attn",
    )


def test_non_flashcomm_weight_processing_delegates_to_vllm(monkeypatch):
    wrapper = NPUMLAAttention.__new__(NPUMLAAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper.kv_b_proj = SimpleNamespace(quant_method=None)
    upstream = MagicMock()
    monkeypatch.setattr(
        MLAAttention,
        "process_weights_after_loading",
        upstream,
    )

    wrapper.process_weights_after_loading(torch.bfloat16)

    upstream.assert_called_once_with(torch.bfloat16)


def test_flashcomm_refresh_keeps_derived_weight_addresses(monkeypatch):
    logical = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
    wrapper = _flashcomm_wrapper(logical)
    monkeypatch.setattr(wrapper_mod, "set_default_quant_scales", MagicMock())

    NPUMLAAttention.process_weights_after_loading(wrapper, torch.float32)
    w_uk_t_ptr = wrapper.W_UK_T.data_ptr()
    w_uv_ptr = wrapper.W_UV.data_ptr()

    updated_logical = logical + 1000
    updated_compute = updated_logical.T.contiguous()
    updated_compute.is_weight_transposed = True
    wrapper.kv_b_proj.weight = updated_compute
    NPUMLAAttention.process_weights_after_loading(wrapper, torch.float32)

    expected = updated_logical.T.view(8, 2, 8)
    expected_uk, expected_uv = expected.split([4, 4], dim=-1)
    assert wrapper.W_UK_T.data_ptr() == w_uk_t_ptr
    assert wrapper.W_UV.data_ptr() == w_uv_ptr
    assert wrapper.W_UK_T.is_contiguous()
    assert wrapper.W_UV.is_contiguous()
    torch.testing.assert_close(wrapper.W_UK_T, expected_uk.permute(1, 2, 0))
    torch.testing.assert_close(wrapper.W_UV, expected_uv.transpose(0, 1))


def test_flashcomm_refresh_rejects_storage_contract_changes(monkeypatch):
    wrapper = _flashcomm_wrapper(torch.zeros(16, 8))
    wrapper.W_UK_T = torch.zeros(3, 2)
    monkeypatch.setattr(wrapper_mod, "set_default_quant_scales", MagicMock())

    with pytest.raises(RuntimeError, match="cannot refresh MLA W_UK_T"):
        NPUMLAAttention.process_weights_after_loading(wrapper, torch.float32)


def test_refresh_mla_wrappers_calls_npu_wrapper_once():
    wrapper = NPUMLAAttention.__new__(NPUMLAAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper.process_weights_after_loading = MagicMock()
    model = torch.nn.Module()
    model.add_module("mla", wrapper)
    model.add_module("linear", torch.nn.Linear(2, 2))

    refresh_mla_wrappers(model, torch.bfloat16)

    wrapper.process_weights_after_loading.assert_called_once_with(torch.bfloat16)
