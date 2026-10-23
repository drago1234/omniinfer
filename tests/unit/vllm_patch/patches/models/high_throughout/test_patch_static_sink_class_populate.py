# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for StaticSinkAttentionClassPatch.populate_sink_kv (kv_nz branch).

populate_sink_kv has two branches gated by
``model_extra_config.operator_opt_config.kv_nz``: ``True`` takes the single
fused ``torch_npu.npu_scatter_pa_kv_cache`` write (same writer as the forward
KV path), ``False`` keeps the legacy two ``npu_scatter_nd_update_`` calls.
Both must set ``sink_populated = True``. The torch_npu writers are mocked
(no NPU hardware required).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import omni_npu.vllm_patches.patches.models.high_throughout.patch_static_sink_attention as sink_mod
from omni_npu.vllm_patches.patches.models.high_throughout.patch_static_sink_attention import (
    StaticSinkAttentionClassPatch,
)


pytestmark = pytest.mark.unit


BLOCK_SIZE = 16
SINK_LEN = 128
NUM_KV_HEADS = 1  # PanguEmbedded 2B TP=2 per-rank
HEAD_SIZE = 128


@pytest.fixture(autouse=True)
def _cpu_current_platform(monkeypatch):
    """populate_sink_kv builds slot mappings on current_platform.current_device().

    On a real box with the omni-npu plugin that is npu:0; keep these pure-CPU
    tests on the CPU so tensors are comparable regardless of environment.
    """
    monkeypatch.setattr(
        sink_mod.current_platform,
        "current_device",
        lambda: "cpu",
        raising=False,
    )


def _make_attn():
    """Build a minimal instance carrying only the fields populate_sink_kv reads."""
    attn = StaticSinkAttentionClassPatch.__new__(StaticSinkAttentionClassPatch)
    attn.block_size = BLOCK_SIZE
    attn.sink_len = SINK_LEN
    attn.sink_populated = False
    # sink_key/sink_value: (sink_len, num_kv_heads, head_size) per the class __init__
    attn.sink_key = torch.zeros(SINK_LEN, NUM_KV_HEADS, HEAD_SIZE)
    attn.sink_value = torch.zeros(SINK_LEN, NUM_KV_HEADS, HEAD_SIZE)
    return attn


def _make_cache(nz: bool):
    """KV caches in the layout the real path provides.

    ND: (block_num, block_size, num_kv_heads, head_size)
    NZ logical: (block_num, num_kv_heads, head_size//nz_dim, block_size, nz_dim)
    Both keep slot r = block r * block_size + offset for the first sink blocks.
    """
    block_num = 16
    if nz:
        nz_dim = 16
        shape = (block_num, NUM_KV_HEADS, HEAD_SIZE // nz_dim, BLOCK_SIZE, nz_dim)
    else:
        shape = (block_num, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return torch.zeros(shape), torch.zeros(shape)


def _expected_slots():
    # populate_sink_kv maps sink row i to slot block_size + i (skip block 0).
    return torch.arange(BLOCK_SIZE, SINK_LEN + BLOCK_SIZE, dtype=torch.long)


@pytest.mark.unit
def test_populate_sink_kv_nz_uses_fused_scatter_pa(monkeypatch):
    """kv_nz=True: one npu_scatter_pa_kv_cache call, flat slot mapping, flag set."""
    monkeypatch.setattr(
        sink_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(kv_nz=True)),
        raising=False,
    )
    attn = _make_attn()
    k_cache, v_cache = _make_cache(nz=True)

    with patch.object(sink_mod.torch_npu, "npu_scatter_pa_kv_cache") as mock_pa, patch.object(
        sink_mod.torch_npu, "npu_scatter_nd_update_"
    ) as mock_nd:
        attn.populate_sink_kv(k_cache, v_cache)

    mock_pa.assert_called_once()
    mock_nd.assert_not_called()

    args = mock_pa.call_args.args
    # (sink_key, sink_value, k_cache, v_cache, slot_mapping) passed verbatim.
    assert args[0] is attn.sink_key
    assert args[1] is attn.sink_value
    assert args[2] is k_cache
    assert args[3] is v_cache

    slots = args[4]
    expected = _expected_slots()
    assert slots.dtype == torch.long
    assert torch.equal(slots.cpu(), expected)

    assert attn.sink_populated is True


@pytest.mark.unit
def test_populate_sink_kv_nd_keeps_scatter_nd_path(monkeypatch):
    """kv_nz=False: legacy two npu_scatter_nd_update_ calls, (N,1) slots, flag set."""
    monkeypatch.setattr(
        sink_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(kv_nz=False)),
        raising=False,
    )
    attn = _make_attn()
    k_cache, v_cache = _make_cache(nz=False)

    with patch.object(sink_mod.torch_npu, "npu_scatter_pa_kv_cache") as mock_pa, patch.object(
        sink_mod.torch_npu, "npu_scatter_nd_update_"
    ) as mock_nd:
        attn.populate_sink_kv(k_cache, v_cache)

    mock_pa.assert_not_called()
    assert mock_nd.call_count == 2

    k_call, v_call = mock_nd.call_args_list
    # ND path signature: npu_scatter_nd_update_(cache_flat, slots, updates)
    # First call writes sink_key into the flattened k cache, second sink_value.
    assert k_call.args[2] is attn.sink_key
    assert v_call.args[2] is attn.sink_value
    # The flattened caches are views of the inputs.
    assert k_call.args[0].data_ptr() == k_cache.view(-1, k_cache.shape[-1]).data_ptr()
    # Slot mapping: same arange, reshaped to (sink_len, 1).
    slots = k_call.args[1]
    assert slots.shape == (SINK_LEN, 1)
    assert torch.equal(slots.flatten().cpu(), _expected_slots())

    assert attn.sink_populated is True


@pytest.mark.unit
def test_populate_sink_kv_nz_slot_mapping_skips_block_zero(monkeypatch):
    """The NZ slot mapping must start at block_size (block 0 reserved)."""
    monkeypatch.setattr(
        sink_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(kv_nz=True)),
        raising=False,
    )
    attn = _make_attn()
    k_cache, v_cache = _make_cache(nz=True)

    with patch.object(sink_mod.torch_npu, "npu_scatter_pa_kv_cache") as mock_pa:
        attn.populate_sink_kv(k_cache, v_cache)

    slots = mock_pa.call_args.args[4]
    assert slots.numel() == SINK_LEN
    assert int(slots.min()) == BLOCK_SIZE
    assert int(slots.max()) == SINK_LEN + BLOCK_SIZE - 1
    # Contiguous arange: sink rows land on consecutive slots.
    assert bool(torch.equal(slots.cpu().diff() == 1, torch.ones(SINK_LEN - 1, dtype=torch.bool)))
