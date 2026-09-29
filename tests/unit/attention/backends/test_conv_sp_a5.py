# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A5 (Ascend950) branches of the MoME-SP convolution helpers."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

MODULE = "omni_npu.attention.backends.utils"

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_bf16 = {"device": "cpu", "dtype": torch.bfloat16}


@pytest.fixture
def utils():
    return importlib.import_module(MODULE)


def _conv_inputs(state_rows):
    x = torch.randn(8, 16, **cfg_bf16)
    w = torch.randn(3, 16, **cfg_bf16)
    states = torch.zeros(2, state_rows, 16, **cfg_bf16)
    prefix = torch.zeros(2, **cfg_i32)
    cumlens = torch.tensor([0, 4, 8], **cfg_i32)
    return x, w, states, prefix, cumlens


@pytest.mark.unit
class TestSimpleConv:
    def test_a3_uses_custom_op_with_kernel_size_states(self, utils):
        x, w, states, prefix, cumlens = _conv_inputs(state_rows=3)
        custom_op = MagicMock(return_value=x.clone())
        with (
            patch.object(utils, "on_ascend950", return_value=False),
            patch("torch.ops.custom.npu_ai_infra_fused_causal_conv1d", custom_op, create=True),
        ):
            utils.simple_conv(x, w, states, prefix, cumlens, inplace=False)

        custom_op.assert_called_once()
        args, kwargs = custom_op.call_args
        assert args[0] is x and args[1] is w and args[2] is states
        assert kwargs["query_start_loc"] is cumlens
        assert kwargs["num_computed_tokens"] is prefix
        assert kwargs["inplace"] is False

    def test_a3_rejects_short_history(self, utils):
        x, w, states, prefix, cumlens = _conv_inputs(state_rows=2)
        with patch.object(utils, "on_ascend950", return_value=False):
            with pytest.raises(AssertionError):
                utils.simple_conv(x, w, states, prefix, cumlens)

    @pytest.mark.parametrize(
        ("inplace", "op_name"),
        [(False, "npu_fused_causal_conv1d"), (True, "npu_fused_causal_conv1d_v2")],
    )
    def test_a5_uses_torch_npu_fused_conv(self, utils, inplace, op_name):
        # The native A5 cache only keeps the K - 1 causal history rows.
        x, w, states, prefix, cumlens = _conv_inputs(state_rows=2)
        fused = MagicMock(return_value=x.clone())
        with (
            patch.object(utils, "on_ascend950", return_value=True),
            patch.object(utils.torch_npu, op_name, fused, create=True),
        ):
            utils.simple_conv(x, w, states, prefix, cumlens, inplace=inplace, max_query_len=4)

        fused.assert_called_once()
        args, kwargs = fused.call_args
        assert args[0] is x and args[1] is w and args[2] is states
        assert kwargs["query_start_loc"] is cumlens
        assert kwargs["num_computed_tokens"] is prefix
        assert kwargs["max_query_len"] == 4
        assert kwargs["residual_connection"] == 1
        assert kwargs["conv_mode"] == "pangu"

    def test_a5_requires_max_query_len(self, utils):
        x, w, states, prefix, cumlens = _conv_inputs(state_rows=2)
        with patch.object(utils, "on_ascend950", return_value=True):
            with pytest.raises(AssertionError):
                utils.simple_conv(x, w, states, prefix, cumlens)


@pytest.mark.unit
class TestSaveStates:
    def test_a5_uses_scatter_update(self, utils):
        cache = torch.zeros(8, 3, 16, **cfg_bf16)
        index = torch.tensor([1, 5], **cfg_i32)
        states = torch.randn(2, 3, 16, **cfg_bf16)
        scatter = MagicMock()
        with (
            patch.object(utils, "on_ascend950", return_value=True),
            patch.object(utils.torch_npu, "npu_scatter_nd_update_", scatter, create=True),
        ):
            utils.save_states(cache, index, states)

        scatter.assert_called_once()
        args = scatter.call_args.args
        assert args[0] is cache and args[2] is states
        assert args[1].dtype == torch.int64
        assert args[1].shape == (2, 1)

    def test_a3_uses_custom_op(self, utils):
        cache = torch.zeros(8, 3, 16, **cfg_bf16)
        index = torch.tensor([1, 5], **cfg_i32)
        states = torch.randn(2, 3, 16, **cfg_bf16)
        custom_op = MagicMock()
        with (
            patch.object(utils, "on_ascend950", return_value=False),
            patch("torch.ops.custom.npu_ai_infra_fused_causal_conv1d", custom_op, create=True),
        ):
            utils.save_states(cache, index, states)

        custom_op.assert_called_once()
        assert custom_op.call_args.kwargs["cache_indices"] is index
        assert custom_op.call_args.kwargs["inplace"] is True


def _scheme(utils, *, on_a5, state_len, kernel_size=3, cumlens=(0, 5, 9), computed=(0, 0)):
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    with patch.object(utils, "on_ascend950", return_value=on_a5):
        return utils.scheme_conv_sp(
            group,
            np.asarray(cumlens, dtype=np.int64),
            np.asarray(computed, dtype=np.int64),
            like=None,
            block_size=4,
            state_len=state_len,
            kernel_size=kernel_size,
        )


@pytest.mark.unit
class TestSchemeConvSpStateLen:
    @pytest.mark.parametrize(
        ("on_a5", "state_len", "expected_conv_state_len"),
        [(False, 3, 3), (True, 2, 2), (True, 5, 2)],
        ids=["a3_kernel_rows", "a5_minimum_native_state", "a5_mtp_cache"],
    )
    def test_conv_reads_only_causal_history_on_a5(
        self, utils, on_a5, state_len, expected_conv_state_len
    ):
        a2a_meta, conv_meta, batch_meta, _ = _scheme(utils, on_a5=on_a5, state_len=state_len)

        prefix, cumlens, reorg_idx, max_query_len = conv_meta
        bs, m, k, conv_state_len, sp_len, _ = batch_meta
        assert (bs, m, k, sp_len) == (2, state_len, 3, 9)
        assert conv_state_len == expected_conv_state_len
        # MTP slots are still saved in full; only the conv load is trimmed.
        num_saves = (reorg_idx.numel() - prefix.numel() * conv_state_len) // m
        assert reorg_idx.numel() == prefix.numel() * conv_state_len + num_saves * m
        assert max_query_len == int(torch.diff(cumlens).max())

    def test_a3_still_requires_kernel_size_rows(self, utils):
        with pytest.raises(AssertionError):
            _scheme(utils, on_a5=False, state_len=2)


@pytest.mark.unit
def test_conv_sp_a5_loads_kernel_minus_one_rows(utils):
    """Single-rank conv_sp hands simple_conv K - 1 history rows on A5."""
    metadata = _scheme(utils, on_a5=True, state_len=5, cumlens=(0, 8), computed=(0,))
    _, conv_meta, batch_meta, _ = metadata
    bs, m, k, conv_state_len, sp_len, _ = batch_meta
    dim = 4
    x = torch.randn(sp_len, dim)
    w = torch.randn(k, dim)
    cache = torch.zeros(6, m, dim)
    init_idx = torch.zeros(bs, dtype=torch.int64)
    save_idx = torch.arange(4, dtype=torch.int64)

    def fake_a2a(out, inp, *_args, **_kwargs):
        out.copy_(inp)

    captured = {}

    def fake_conv(x_, w_, loads, prefix, cumlens, inplace, max_query_len=None):
        captured["loads"] = loads
        captured["max_query_len"] = max_query_len
        return x_

    sp_group = SimpleNamespace(device_group=None, all_gather=lambda t, dim=0: t)
    batch_meta = (bs, m, k, conv_state_len, sp_len, sp_group)
    stream = MagicMock()
    with (
        patch("torch.npu", SimpleNamespace(current_stream=lambda: stream, stream=lambda _s: nullcontext()), create=True),
        patch.object(utils, "named_stream", return_value=stream),
        patch.object(utils, "select_dim0", side_effect=lambda t, i: t[i]),
        patch.object(utils, "simple_conv", side_effect=fake_conv),
        patch.object(utils, "save_states"),
        patch("torch.distributed.all_to_all_single", side_effect=fake_a2a),
    ):
        utils.conv_sp(x, w, cache, init_idx, save_idx, (metadata[0], conv_meta, batch_meta, metadata[3]))

    assert captured["loads"].shape[1:] == (k - 1, dim)
    assert captured["max_query_len"] == conv_meta[3]
