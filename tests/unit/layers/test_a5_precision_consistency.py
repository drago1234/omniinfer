# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A5 (Ascend950) branches of enable_precision_strong_consistency in MLA/DSA."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _consistency_config(*, strong=True, noncontiguous=True, batch_invariant=False):
    return SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            use_noncontiguous_kv=noncontiguous,
            enable_precision_strong_consistency=strong,
            use_batch_invariant_op=batch_invariant,
        )
    )


def _pioneer_outputs(T, N, L, value):
    return (
        torch.full((T, N, L), float(value)),
        None,
        torch.zeros(T, N, 1),
        torch.ones(T, N, 1),
    )


@pytest.mark.unit
def test_mla_consistency_core_a5_runs_normal_and_sink_pioneer_then_merges(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    T, valid, N, L, R, S, BLK = 3, 2, 2, 4, 2, 5, 8
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.param_sink_number = S
    fake.scaling = 0.5
    fake.sliding_window = None
    # StaticSinkMLAAttention keeps the 2-D sink tensors; the impl holds the 3-D ones.
    fake.attn = SimpleNamespace(
        sink_compressed_kv=torch.zeros(S, L),
        sink_k_pe=torch.zeros(S, R),
    )
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", "mask", raising=False)
    monkeypatch.setattr(mla_mod.NPUMLAImpl, "MAX_WINDOW_SIZE", 100, raising=False)
    monkeypatch.setattr(mla_mod, "model_extra_config", _consistency_config())

    meta = MagicMock(side_effect=["meta_normal", "meta_sink"])
    pioneer = MagicMock(
        side_effect=[_pioneer_outputs(valid, N, L, 1), _pioneer_outputs(valid, N, L, 2)]
    )
    merged = torch.full((valid, N, L), 9.0)
    rescale = MagicMock(return_value=(merged, None, None))
    monkeypatch.setattr(mla_mod.ops, "apply_FA_rescale_forward", rescale)
    monkeypatch.setattr(
        torch.ops.custom, "npu_ai_infra_attention_pioneer_metadata", meta, raising=False
    )
    monkeypatch.setattr(
        torch.ops.custom, "npu_ai_infra_attention_pioneer_v2", pioneer, raising=False
    )
    kv_cache = (torch.zeros(1, BLK, L), torch.zeros(1, BLK, R))
    block_table = torch.zeros(1, 2, dtype=torch.int32)

    out = NPUDeepseekMLAAttention._apply_sink_attention_consistency_core(
        fake,
        query=torch.zeros(T, N, L),
        query_rope=torch.zeros(T, N, R),
        kv_cache=kv_cache,
        q_cumlens=torch.tensor([valid], dtype=torch.int32),
        kv_lens=torch.tensor([BLK], dtype=torch.int32),
        block_table=block_table,
        num_actual_tokens=valid,
        on_a5=True,
    )

    # Padding rows beyond the valid tokens stay zero; valid rows get the merge.
    assert out.shape == (T, N, L)
    torch.testing.assert_close(out[:valid], merged)
    assert torch.count_nonzero(out[valid:]) == 0
    assert pioneer.call_count == 2

    normal, sink = pioneer.call_args_list
    assert normal.args[1] is kv_cache[0]
    assert normal.args[3] == "meta_normal"
    assert normal.kwargs["key_rope"] is kv_cache[1]
    assert normal.kwargs["block_table"] is block_table
    assert normal.kwargs["block_size"] == BLK
    assert normal.kwargs["pre_tokens"] == 100
    assert normal.args[0].shape[0] == valid

    assert sink.args[3] == "meta_sink"
    assert sink.args[1].shape == (1, S, L)
    assert sink.kwargs["key_rope"].shape == (1, S, R)
    assert sink.kwargs["block_size"] == S
    assert torch.equal(sink.kwargs["actual_seq_lengths_kv"], torch.tensor([S]))
    assert torch.count_nonzero(sink.kwargs["block_table"]) == 0
    assert meta.call_args_list[1].kwargs["actual_seq_lengths_kv"].tolist() == [S]
    assert meta.call_args_list[0].kwargs["block_size"] == normal.kwargs["block_size"]
    assert meta.call_args_list[1].kwargs["block_size"] == sink.kwargs["block_size"]
    rescale.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("strong, noncontiguous", [(False, True), (True, False)])
def test_mla_consistency_core_a5_requires_both_flags(monkeypatch, strong, noncontiguous):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(
        mla_mod,
        "model_extra_config",
        _consistency_config(strong=strong, noncontiguous=noncontiguous),
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    with pytest.raises(AssertionError, match="must be both True"):
        NPUDeepseekMLAAttention._apply_sink_attention_consistency_core(
            fake,
            query=torch.zeros(1, 1, 1),
            query_rope=torch.zeros(1, 1, 1),
            kv_cache=(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)),
            q_cumlens=torch.tensor([1]),
            kv_lens=torch.tensor([1]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            num_actual_tokens=1,
            on_a5=True,
        )


@pytest.mark.unit
def test_mla_sink_attention_a5_dispatches_to_consistency_core(monkeypatch):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(mla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock())
    monkeypatch.setattr(mla_mod, "model_extra_config", _consistency_config())
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.sliding_window = None
    fake.attn = SimpleNamespace(kv_cache=("k", "pe"))
    tnd = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    fake._apply_sink_attention_consistency_core = MagicMock(return_value=tnd)
    q_cumlens = torch.tensor([3])
    kv_lens = torch.tensor([7])
    block_table = torch.zeros(1, 1, dtype=torch.int32)

    out = NPUDeepseekMLAAttention._apply_sink_attention_a5(
        fake,
        q_nope=torch.zeros(3, 2, 4),
        q_pe=torch.zeros(3, 2, 2),
        keys=None,
        values=None,
        q_cumlens=q_cumlens,
        kv_lens=kv_lens,
        block_table=block_table,
        num_tokens=4,
        valid_tok=3,
        q_heads=2,
        metadata_caller="decode",
    )

    kwargs = fake._apply_sink_attention_consistency_core.call_args.kwargs
    assert kwargs["kv_cache"] == ("k", "pe")
    assert kwargs["num_actual_tokens"] == 3
    assert kwargs["on_a5"] is True
    assert kwargs["block_table"] is block_table
    assert "metadata_caller" not in kwargs
    assert torch.equal(out, tnd.transpose(0, 1))


@pytest.mark.unit
def test_mome_kv_skips_inplace_update_under_strong_consistency(monkeypatch):
    # The A5 kernel only adds the residual itself when it is not in consistency
    # mode, so the in-place (residual-less) update must not be used there.
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import MomeAttentionMixin

    monkeypatch.setattr(mla_mod, "model_extra_config", _consistency_config())
    fake = MomeAttentionMixin()
    fake.use_mome = True
    fake.noncontiguous_kv = True
    fake.use_mome_inplace_update = True
    fake.kv_lora_rank = 4
    fake.qk_rope_head_dim = 2
    calls = []

    def conv(x, state_indice, inplace=False, **kwargs):
        calls.append((state_indice, inplace))
        return x + 1

    fake.conv = conv
    kv = torch.zeros(3, 6)
    kv[:, 4:] = 7

    out = MomeAttentionMixin._maybe_mome_kv(fake, kv, lambda: {})

    assert calls == [(1, False)]
    assert torch.equal(out[:, :4], torch.ones(3, 4))
    assert torch.equal(out[:, 4:], torch.full((3, 2), 7.0))
    assert torch.count_nonzero(kv[:, :4]) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "attention_cls_name", ["NPUDeepseekMLAAttention", "NPUDeepseekSparseAttention"]
)
def test_mome_out_gathers_tokens_when_o_proj_tp_exceeds_attention_tp(
    monkeypatch, attention_cls_name
):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention import npu_mla as mla_mod

    cls = getattr(mla_mod if "MLA" in attention_cls_name else dsa_mod, attention_cls_name)
    attention = cls.__new__(cls)
    torch.nn.Module.__init__(attention)
    local = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    gathered = torch.cat([local, local + 100])
    gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(mla_mod, "layer_parallel_all_gather", gather)
    attention.use_mome = True
    attention.tp_size = 1
    attention.kv_b_proj = SimpleNamespace(tp_size=1)
    attention.o_proj = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        layer_name_inside_block="self_attn.o_proj",
        requires_input_partition=MagicMock(return_value=True),
    )
    attention._apply_mome = MagicMock(return_value=local)

    out = attention._maybe_mome_out(torch.zeros_like(local), lambda: {})

    gather.assert_called_once_with(local, "self_attn.o_proj", "x", dim=0)
    torch.testing.assert_close(out, gathered[:, 2:])
    assert out.is_contiguous()


@pytest.mark.unit
@pytest.mark.parametrize(
    "on_a5, sliding_window, batch_invariant, expect_absorb",
    [
        (True, 128, False, False),
        (True, 128, True, True),
        (True, None, False, True),
        (False, 128, False, True),
    ],
)
def test_mla_prefill_absorb_routing_with_batch_invariant(
    monkeypatch, on_a5, sliding_window, batch_invariant, expect_absorb
):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention

    monkeypatch.setattr(
        mla_mod, "model_extra_config", _consistency_config(batch_invariant=batch_invariant)
    )
    fake = NPUDeepseekMLAAttention.__new__(NPUDeepseekMLAAttention)
    fake.ena_sp = False
    fake.mla_absorb = True
    fake.on_ascend950 = on_a5
    fake.sliding_window = sliding_window
    fake.noncontiguous_kv = True
    fake._maybe_quant = lambda x: x
    fake._forward_prefill_absorb_pa = MagicMock(return_value="absorb")
    fake._forward_prefill_standard = MagicMock(return_value="standard")

    # A5 SWA prefill routes by scheduled tokens (swa_prefill_absorb_max_tokens,
    # default 8192); 16384 tokens select the standard path unless batch
    # invariance forces absorb.
    out = NPUDeepseekMLAAttention._forward_prefill(
        fake,
        torch.zeros(2, 4),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        SimpleNamespace(query_cumlens_list=[16384]),
    )

    assert out == ("absorb" if expect_absorb else "standard")


@pytest.mark.unit
def test_dsa_rescale_pioneer_a5_uses_pioneer_for_sink(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    T, N, L, R, S = 2, 2, 4, 2, 3
    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.on_ascend950 = True
    fake.scaling = 0.5
    fake.param_sink_number = S
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.dummy_value_cache = torch.zeros(1)
    fake.atten_mask = "mask"
    fake.attn = SimpleNamespace(
        impl=SimpleNamespace(
            sink_k_nope=torch.zeros(S, 1, L),
            sink_kv=torch.zeros(S, 1, L + R),
        )
    )
    sparse = MagicMock(
        return_value=(torch.zeros(T, N, L), torch.zeros(1, T, N), torch.ones(1, T, N))
    )
    meta = MagicMock(return_value="meta")
    pioneer_v2 = MagicMock(return_value=_pioneer_outputs(T, N, L, 1))
    legacy_meta = MagicMock()
    legacy_sink = MagicMock()
    merged = torch.full((T, N, L), 7.0)
    rescale = MagicMock(return_value=(merged, None, None))
    monkeypatch.setattr(dsa_mod.ops, "apply_FA_rescale_forward", rescale)
    for name, value in (
        ("npu_ai_infra_sparse_flash_attention_pioneer", sparse),
        ("npu_ai_infra_attention_pioneer_metadata", meta),
        ("npu_ai_infra_attention_pioneer_v2", pioneer_v2),
        ("_npu_fused_infer_attention_sink_metadata", legacy_meta),
        ("npu_fused_infer_attention_sink_v2", legacy_sink),
    ):
        monkeypatch.setattr(torch.ops.custom, name, value, raising=False)

    out = NPUDeepseekSparseAttention._apply_attention_rescale_pioneer(
        fake,
        q_nope=torch.zeros(T, N, L),
        q_pe=torch.zeros(T, N, R),
        q_cumlens=torch.tensor([T], dtype=torch.int32),
        kv_lens=torch.tensor([T], dtype=torch.int32),
        topk_idx=torch.zeros(T, 1, 2, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        kv_cache=(torch.zeros(4, 8, L),),
    )

    assert out is merged
    legacy_meta.assert_not_called()
    legacy_sink.assert_not_called()
    kwargs = pioneer_v2.call_args.kwargs
    assert kwargs["atten_mask"] == "mask"
    assert kwargs["block_size"] == S
    assert kwargs["key_rope"].shape == (1, S, R)
    assert torch.equal(kwargs["actual_seq_lengths_kv"], torch.tensor([S]))
    assert pioneer_v2.call_args.args[1].shape == (1, S, L)
    assert meta.call_args.kwargs["block_size"] == S
    rescale.assert_called_once()
