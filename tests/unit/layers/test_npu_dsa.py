# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import pytest

from tests.unit.layers.mome_out_test_utils import (
    configure_multistream_npu,
    make_marked_constant,
    make_marked_input,
    make_recording_event_type,
)

DSA_MODULE = "omni_npu.v1.layers.attention.npu_dsa"

cfg_i32 = {"device": "cpu", "dtype": torch.int32}
cfg_i64 = {"device": "cpu", "dtype": torch.int64}
cfg_bf16 = {"device": "cpu", "dtype": torch.bfloat16}


def _return_query_key(query, key, *_args, **_kwargs):
    return query, key


def _short_prefill_args():
    return {"short_prefill": True}


@pytest.mark.unit
def test_indexer_rope_fusion_uses_apply_rotary_pos_emb():
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    indexer = Indexer.__new__(Indexer)
    indexer.rope_dim = 2
    indexer.head_dim = 4
    indexer.use_rope_fusion_op = True
    x = torch.zeros(2, 3, 4)
    cos = torch.zeros(2, 1, 1, 2)
    sin = torch.zeros_like(cos)
    fused = MagicMock(side_effect=_return_query_key)

    with patch.object(dsa_mod.torch_npu, "npu_apply_rotary_pos_emb", fused):
        output = Indexer._apply_rope(indexer, x, cos, sin)

    assert output.shape == x.shape
    fused.assert_called_once()


@pytest.mark.unit
def test_forward_decode_routes_to_a5_multistream_when_eligible():
    from omni_npu.v1.layers.attention.npu_dsa import (
        NPUDeepseekSparseAttention,
    )

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.ena_sp = False
    fake.sharded_o_proj = False
    fake.noncontiguous_kv = True
    fake.side_stream = object()
    fake.li_stream = object()
    fake.skip_topk = False
    fake.ena_kvsp = False
    fake.use_mlaprolog = False
    fake.is_sfa_c8 = False
    fake.li_quant_mode = None
    fake.use_mome = True
    fake.param_sink_number = 0
    fake._forward_decode_multistream_a5 = MagicMock(return_value="a5")
    kv_cache = (MagicMock(), MagicMock(), MagicMock())
    metadata = MagicMock()
    x = MagicMock()
    cos = MagicMock()
    sin = MagicMock()

    output = NPUDeepseekSparseAttention._forward_decode(
        fake,
        x=x,
        cos=cos,
        sin=sin,
        attn_metadata=metadata,
        kv_cache=kv_cache,
        pd_mixed_flag=2,
    )

    assert output == "a5"
    fake._forward_decode_multistream_a5.assert_called_once()
    args = fake._forward_decode_multistream_a5.call_args.args
    assert args[:5] == (x, cos, sin, metadata, kv_cache)
    assert args[5]() == {}


@pytest.mark.unit
def test_shared_indexer_decode_uses_common_prolog_and_reuses_topk():
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.ena_sp = fake.sharded_o_proj = fake.ena_kvsp = False
    fake.use_mlaprolog = False
    fake.noncontiguous_kv = fake.use_mome = True
    fake.skip_topk = True
    fake.indexer = None
    fake.shared_indexer_kv_stream = object()
    fake._get_li_caches = MagicMock(return_value=(None, None))
    q_nope, q_pe = torch.ones(2, 1, 4), torch.ones(2, 1, 2)
    fake._decode_prolog_multistream_a5 = MagicMock(return_value=(q_nope, q_pe))
    fake._finish_multistream_decode = MagicMock(return_value="shared")
    metadata = SimpleNamespace(
        query_cumlens=torch.tensor([2]), seq_lens=torch.tensor([16]),
        block_table=torch.tensor([[0]]),
    )
    topk = torch.arange(24, dtype=torch.int32).view(3, 1, 8)
    original = topk.clone()
    output = fake._forward_decode(
        torch.zeros(2, 4), torch.zeros(2), torch.zeros(2), metadata,
        (None, None, None), topk,
    )
    assert output == "shared"
    assert fake._decode_prolog_multistream_a5.call_args.kwargs == {
        "kv_stream": fake.shared_indexer_kv_stream,
    }
    args = fake._finish_multistream_decode.call_args.args
    assert args[0] is q_nope and args[1] is q_pe
    assert torch.equal(args[2], topk[:2])
    assert torch.equal(topk, original)


@pytest.mark.unit
@pytest.mark.parametrize("split_q_up_in_multistream", [True, False])
def test_forward_decode_a5_multistream_runs_three_stream_pipeline(
    monkeypatch, split_q_up_in_multistream
):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import (
        NPUDeepseekSparseAttention,
    )

    submitted = []
    core_limits = []
    main_stream = object()
    side_stream = object()
    li_stream = object()

    configure_multistream_npu(monkeypatch, submitted, core_limits, main_stream)
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                enable_precision_strong_consistency=True,
            ),
        ),
    )

    def mark(name, result):
        submitted.append(name)
        return result

    indexer = SimpleNamespace(
        n_head=1,
        head_dim=3,
        wq_b=make_marked_constant(mark, "indexer_q", (torch.zeros(2, 3),)),
        wk=make_marked_constant(mark, "indexer_k", (torch.zeros(2, 3),)),
        k_norm=make_marked_input(mark, "indexer_k_norm"),
        _apply_rope=make_marked_input(mark, "indexer_rope"),
        _update_cache=make_marked_constant(mark, "indexer_cache", None),
        weights_proj=make_marked_constant(
            mark, "indexer_weights", (torch.ones(2, 1),)
        ),
        weights_scale=2.0,
        softmax_scale=0.5,
        _apply_lightning_indexer=make_marked_constant(
            mark,
            "lightning_indexer",
            torch.zeros(2, 1, 2, dtype=torch.int32),
        ),
    )

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.on_ascend950 = True
    fake.use_mome = False
    fake.split_q_up_in_multistream = split_q_up_in_multistream
    fake.noncontiguous_kv = False
    fake.side_stream = side_stream
    fake.li_stream = li_stream
    fake.num_local_heads = 1
    fake.qk_nope_head_dim = 4
    fake.qk_rope_head_dim = 2
    fake.indexer = indexer
    fake.q_a_proj = make_marked_constant(mark, "q_a", (torch.zeros(2, 3),))
    fake._maybe_mome_q = make_marked_input(mark, "q_mome")
    fake.q_a_layernorm = make_marked_input(mark, "q_norm")
    fake.q_b_nope_proj = make_marked_constant(
        mark, "q_b_nope", (torch.zeros(2, 4),)
    )
    fake.q_b_proj = make_marked_constant(mark, "q_b", (torch.zeros(2, 6),))
    fake.qk_head_dim = 6
    fake._q_nope_absorb = make_marked_input(mark, "q_absorb")
    fake.q_b_pe_proj = make_marked_constant(
        mark, "q_b_pe", (torch.zeros(2, 2),)
    )
    fake._apply_rope = make_marked_input(mark, "q_rope")
    fake.kv_a_proj_with_mqa = make_marked_constant(
        mark, "kv_a", (torch.zeros(2, 6),)
    )
    fake._maybe_mome_kv = make_marked_input(mark, "kv_mome")
    fake._kv_norm_rope_cache = make_marked_constant(
        mark, "kv_rope_cache", None
    )
    fake._apply_sink_offset = make_marked_input(mark, "sink_offset")
    fake._apply_attn_absorb = make_marked_constant(
        mark,
        "attention",
        torch.ones(2, 1, 4),
    )
    fake._decode_attn_epilog = make_marked_input(mark, "epilog")

    metadata = SimpleNamespace(
        query_cumlens=torch.tensor([0, 1, 2], dtype=torch.int64),
        seq_lens=torch.tensor([8, 9], dtype=torch.int64),
        block_table=torch.zeros(2, 1, dtype=torch.int32),
    )
    kv_cache = (object(), object(), object())

    raw = torch.zeros(2, 4)
    quantize = MagicMock(side_effect=AssertionError("prolog must pass raw input"))
    for name in ("npu_dynamic_mx_quant", "npu_dynamic_quant", "npu_dtype_cast"):
        monkeypatch.setattr(dsa_mod.torch_npu, name, quantize)
    linear_inputs = {}
    def capture_input(name, original):
        def project(value):
            linear_inputs[name] = value
            return original(value)
        return project
    fake.q_a_proj = capture_input("q", fake.q_a_proj)
    fake.kv_a_proj_with_mqa = capture_input("kv", fake.kv_a_proj_with_mqa)
    indexer.wk = capture_input("wk", indexer.wk)
    indexer.weights_proj = capture_input("weights", indexer.weights_proj)

    with patch.object(torch.Tensor, "record_stream", return_value=None):
        output, next_topk_indices = (
            NPUDeepseekSparseAttention._forward_decode_multistream_a5(
                fake,
                x=raw,
                cos=torch.zeros(2, 1, 1, 2),
                sin=torch.zeros(2, 1, 1, 2),
                attn_metadata=metadata,
                kv_cache=kv_cache,
                get_mome_args=_short_prefill_args,
            )
        )

    assert linear_inputs["q"] is linear_inputs["kv"]
    assert linear_inputs["wk"] is raw and linear_inputs["weights"] is raw
    assert linear_inputs["q"] is raw
    quantize.assert_not_called()

    assert torch.equal(output, torch.ones(2, 1, 4))
    assert torch.equal(
        next_topk_indices,
        torch.zeros(2, 1, 2, dtype=torch.int32),
    )
    q_up_ops = (
        ["record", "indexer_q", "record", "q_b_nope"]
        if split_q_up_in_multistream
        else ["indexer_q", "record", "q_b", "record"]
    )
    q_pe_ops = (
        ["q_b_pe", "indexer_rope", "q_rope"]
        if split_q_up_in_multistream
        else ["indexer_rope", "q_rope"]
    )
    assert [entry for entry in submitted if isinstance(entry, str)] == [
        "record",
        "q_a",
        "record",
        "q_mome",
        "q_norm",
        *q_up_ops,
        "q_absorb",
        "indexer_k",
        "record",
        "indexer_k_norm",
        "indexer_rope",
        "indexer_cache",
        *q_pe_ops,
        "record",
        "indexer_weights",
        "kv_a",
        "kv_mome",
        "kv_rope_cache",
        "record",
        "lightning_indexer",
        "sink_offset",
        "attention",
        "epilog",
    ]
    indexer_cache_pos = submitted.index("indexer_cache")
    li_sync_ops = [("wait", li_stream)]
    if split_q_up_in_multistream:
        li_sync_ops.append("q_b_pe")
    li_sync_ops.append(("wait", li_stream))
    assert submitted[
        indexer_cache_pos + 1:indexer_cache_pos + 1 + len(li_sync_ops)
    ] == li_sync_ops
    q_pe_core_limit = (
        [(8, 16)] if split_q_up_in_multistream else []
    )
    assert core_limits == [
        (24, 48),
        (8, 32),
        (24, 48),
        (32, 32),
        (8, 32),
        *q_pe_core_limit,
        (16, 32),
        (32, 32),
        (16, 32),
    ]


@pytest.mark.unit
def test_indexer_apply_rope_fusion_op(monkeypatch):
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    calls = {}

    def fake_apply(q, k, cos, sin, layout="BSND", rotary_mode="half"):
        calls["layout"] = layout
        calls["rotary_mode"] = rotary_mode
        return q, k

    monkeypatch.setattr("torch_npu.npu_apply_rotary_pos_emb", fake_apply, raising=False)
    idx = Indexer.__new__(Indexer)
    idx.use_rope_fusion_op = True
    idx.rope_dim = 4
    idx.head_dim = 8
    x = torch.ones(2, 3, 8)
    cos = torch.ones(2, 1, 1, 4)
    sin = torch.zeros(2, 1, 1, 4)
    out = idx._apply_rope(x, cos, sin)
    assert out.shape == x.shape
    assert calls == {"layout": "BSND", "rotary_mode": "half"}


@pytest.mark.unit
def test_indexer_apply_rope_interleave_and_mul(monkeypatch):
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    interleave = MagicMock(side_effect=lambda pe, cos, sin: pe + 1)
    rotary_mul = MagicMock(side_effect=lambda pe, cos, sin: pe + 2)
    monkeypatch.setattr("torch_npu.npu_interleave_rope", interleave, raising=False)
    monkeypatch.setattr("torch_npu.npu_rotary_mul", rotary_mul, raising=False)

    idx = Indexer.__new__(Indexer)
    idx.use_rope_fusion_op = False
    idx.rope_dim = 4
    idx.head_dim = 8
    idx.config = SimpleNamespace(indexer_rope_interleave=True)
    x = torch.zeros(2, 1, 8)
    cos = torch.ones(2, 1, 1, 4)
    sin = torch.zeros(2, 1, 1, 4)
    out = idx._apply_rope(x, cos, sin)
    interleave.assert_called_once()
    rotary_mul.assert_not_called()
    assert out.shape == x.shape

    idx.config = SimpleNamespace(indexer_rope_interleave=False)
    out = idx._apply_rope(x, cos, sin)
    rotary_mul.assert_called_once()
    assert out.shape == x.shape


def _stream_as_nullcontext(_stream):
    return nullcontext()


@pytest.mark.unit
@pytest.mark.parametrize("requires_partition", [True, False])
def test_pangu_mla_epilog_partitions_only_when_o_proj_requires_it(
    monkeypatch, requires_partition
):
    from omni_npu.v1.layers.attention import npu_pangu as pangu_mod
    from omni_npu.v1.layers.attention.npu_pangu import NPUPanguSparseAttention

    attention = NPUPanguSparseAttention.__new__(NPUPanguSparseAttention)
    torch.nn.Module.__init__(attention)
    mome_output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    split_output = (mome_output[:, :2], mome_output[:, 2:])
    split = MagicMock(return_value=split_output)
    monkeypatch.setattr(pangu_mod, "split_tensor_along_last_dim", split)
    attention.use_mome = True
    attention.disable_o_conv_tp = True
    attention.o_conv = SimpleNamespace(dim=4)
    attention.o_proj = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        requires_input_partition=MagicMock(return_value=requires_partition),
    )
    attention._apply_MOME = MagicMock(return_value=mome_output)
    attention._apply_o_proj = MagicMock(side_effect=lambda output: output)

    output = attention._mla_epilog(torch.zeros_like(mome_output))

    attention.o_proj.requires_input_partition.assert_called_once_with()
    if requires_partition:
        split.assert_called_once_with(mome_output, num_partitions=2)
        torch.testing.assert_close(output, split_output[1])
        assert output.is_contiguous()
    else:
        split.assert_not_called()
        assert output is mome_output


@pytest.mark.unit
@pytest.mark.parametrize("requires_partition", [True, False])
def test_dsa_mome_out_partitions_only_when_o_proj_requires_it(
    monkeypatch, requires_partition
):
    from omni_npu.v1.layers.attention import npu_mla as mla_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention
    from tests.unit.layers.mome_out_test_utils import run_mome_out_partition_case

    run_mome_out_partition_case(
        NPUDeepseekSparseAttention,
        mla_mod,
        monkeypatch,
        requires_partition,
        MagicMock(),
    )


# =========================
# 1. 无效果或简单mock
# =========================

@contextmanager
def _mock_torch_npu_stream():
    mock_npu = MagicMock()
    mock_npu.current_stream.return_value = MagicMock()
    mock_npu.Stream.return_value = MagicMock()
    mock_npu.stream.side_effect = _stream_as_nullcontext
    with patch("torch.npu", mock_npu):
        yield


@contextmanager
def _mock_misc(yarn_get_mscale_ret: float = 1.0):
    with (
        patch("vllm.logger.init_logger", return_value=MagicMock()),
        patch(f"{DSA_MODULE}.current_platform", MagicMock(device_type="cpu")),
        patch(f"{DSA_MODULE}.current_stream", MagicMock(), create=True),
        patch(f"{DSA_MODULE}.extract_layer_index", return_value=0),
        patch(f"{DSA_MODULE}.get_rope", return_value=None),
        patch(f"{DSA_MODULE}.yarn_get_mscale", return_value=yarn_get_mscale_ret),
        patch(
            "omni_npu.v1.layers.attention.npu_mla.get_layer_parallel_group",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.layers.rotary_embedding.get_rope_wrapper",
            MagicMock(return_value=None),
            create=True,
        ),
    ):
        yield


# =========================
# 2. 功能模块 mock
# =========================

@contextmanager
def _mock_torch_npu():
    def fused_infer_attention_sink(query, query_rope, key, value, key_rope,
                                   num_query_heads=0, num_key_value_heads=0,
                                   input_layout="TND", softmax_scale=1.0,
                                   sparse_mode=4, atten_mask=None,
                                   actual_seq_qlen=None, actual_seq_kvlen=None,
                                   pre_tokens=0, next_tokens=0, sink_number=0,
                                   key_sink=None, value_sink=None, key_rope_sink=None,
                                   block_table=None, block_size=0, **kwargs):
        assert input_layout in ["BSND", "TND", "TND_NTD"]
        assert query.dim() == 3
        assert value.dim() in [3, 4]
        T, N, _ = query.shape
        D = value.size(-1)
        if input_layout == "TND_NTD":
            return query.new_zeros(N, T, D), None
        return query.new_zeros(T, N, D), None

    def kv_rmsnorm_rope_cache_v2(latent_kv, weight, cos, sin, slot_mapping,
                                 k_cache, ckv_cache, k_rope_scale=None, k_rope_offset=None,
                                 epsilon=1e-6, cache_mode="PA", rotary_mode="interleave",
                                 quant_mode="none", is_output_kv=True):
        # latent_kv: [T, 1, 1, L+R]
        T = latent_kv.size(0)
        R = cos.size(-1)
        L = latent_kv.size(-1) - R
        k_pe = latent_kv.new_zeros(T, 1, 1, R)
        k_nope = latent_kv.new_zeros(T, 1, 1, L)
        return k_pe, k_nope

    def sparse_flash_attention_pioneer(
        query, key, value, **kw,
    ):
        return query.new_zeros(*query.shape[:2], value.size(-1)), None

    def fused_causal_conv1d(x, *_, **kw):
        return x

    def lightning_indexer(query, key, weights, sparse_count=2048, **kw):
        assert query.dim() == 3
        assert key.shape[-1] == query.shape[-1]
        assert weights.dim() >= 2
        T, N, D = query.shape
        assert sparse_count > 0
        K = sparse_count
        indices = torch.zeros(T, 1, K, **cfg_i32)
        attn_out = torch.zeros(T, N, K, **cfg_bf16)
        return (indices, attn_out)

    def mla_prolog_v3(token_x, weight_uk, kv_cache, rope_sin, rope_cos, rmsnorm_gamma_cq, **kw):
        assert token_x.dim() == 3
        assert kv_cache.dim() == 4
        assert rope_sin.shape[-1] == rope_cos.shape[-1]
        assert token_x.shape[0] == rope_sin.shape[0]
        bs, _, _ = token_x.shape
        kv_lora_rank = kv_cache.shape[-1]
        rope_dim = rope_sin.shape[-1]
        q_lora_rank = rmsnorm_gamma_cq.shape[0]
        n_head = weight_uk.shape[0]  # W_UK_T: (num_heads, qk_nope_head_dim, kv_lora_rank)
        assert n_head > 0
        q_nope = torch.zeros(bs, n_head, kv_lora_rank, **cfg_bf16)
        q_pe = torch.zeros(bs, n_head, rope_dim, **cfg_bf16)
        q_norm = torch.zeros(bs, q_lora_rank, **cfg_bf16)
        return (q_nope, q_pe, None, q_norm, None)

    def scatter_nd_update_(x, indices, updates):
        return x

    def rotary_mul(x, cos, sin):
        return x

    def interleave_rope(x, cos, sin):
        return x

    def apply_rotary_pos_emb(q, k, cos, sin, **_kw):
        return q, k

    def transpose_batchmatmul(input, weight=None, perm_x1=None, perm_x2=None, perm_y=None):
        if weight is not None:
            assert input.dim() == 3
            assert weight.dim() == 3
            x_perm = input.permute(*perm_x1) if perm_x1 else input
            weight_perm = weight.permute(*perm_x2) if perm_x2 else weight
            out = torch.bmm(x_perm, weight_perm)
            if perm_y is not None:
                out = out.permute(*perm_y)
            return out
        else:
            return input

    def sparse_flash_attention(query, key, value, query_rope, key_rope, sparse_indices, **kw):
        # only for absorbed PA
        assert query.dim() == 3
        assert key.dim() == 4
        assert value.shape == key.shape
        assert query_rope.shape[0] == query.shape[0]
        assert query_rope.shape[1] == query.shape[1]
        assert sparse_indices.dim() == 3
        assert sparse_indices.shape[0] == query.shape[0]
        T, N, D = query.shape
        attn_out = query.new_zeros(T, N, D)
        return attn_out, None

    def kv_rmsnorm_rope_cache(
        latent_kv, weight, cos, sin, slot_mapping, rope_cache, nope_cache, is_output_kv, **kw
    ):
        assert latent_kv.dim() == 4
        assert weight.dim() == 1
        assert cos.shape == sin.shape
        assert rope_cache.dim() == 4
        assert nope_cache.dim() == 4
        assert latent_kv.shape[-1] == rope_cache.shape[-1] + nope_cache.shape[-1]
        assert slot_mapping.dim() == 1
        k_pe, k_nope = None, None
        if is_output_kv:
            T = latent_kv.size(0)
            D, R = latent_kv.size(-1), cos.size(-1)
            k_pe = latent_kv.new_zeros(T, 1, 1, R)
            k_nope = latent_kv.new_zeros(T, 1, 1, D - R)
        return rope_cache, nope_cache, k_pe, k_nope

    with (
        patch.multiple(
            "torch_npu",
            npu_lightning_indexer=MagicMock(side_effect=lightning_indexer),
            npu_mla_prolog_v3=MagicMock(side_effect=mla_prolog_v3),
            npu_scatter_nd_update_=MagicMock(side_effect=scatter_nd_update_),
            npu_rotary_mul=MagicMock(side_effect=rotary_mul),
            npu_interleave_rope=MagicMock(side_effect=interleave_rope),
            npu_apply_rotary_pos_emb=MagicMock(side_effect=apply_rotary_pos_emb),
            npu_transpose_batchmatmul=MagicMock(side_effect=transpose_batchmatmul),
            npu_kv_rmsnorm_rope_cache=MagicMock(side_effect=kv_rmsnorm_rope_cache),
            npu_sparse_flash_attention=MagicMock(side_effect=sparse_flash_attention),
        ),
        patch(
            "torch.ops.custom",
            npu_sparse_flash_attention_enhance=MagicMock(side_effect=sparse_flash_attention),
            npu_lightning_indexer_enhance=MagicMock(side_effect=lightning_indexer),
            npu_fused_infer_attention_sink=MagicMock(side_effect=fused_infer_attention_sink),
            npu_ai_infra_kv_rmsnorm_rope_cache_v2=MagicMock(side_effect=kv_rmsnorm_rope_cache_v2),
            npu_ai_infra_sparse_flash_attention_pioneer=MagicMock(side_effect=sparse_flash_attention_pioneer),
            npu_ai_infra_fused_causal_conv1d=MagicMock(side_effect=fused_causal_conv1d),
        ),
    ):
        yield


@contextmanager
def _mock_vllm_distributed(cards: int = 4, rank: int = 0, dcp: bool = False):
    class MockGroupCoordinator:
        def __init__(self, world_size: int, rank_in_group: int):
            self.world_size = world_size
            self.rank_in_group = rank_in_group
            self.device_group = None

        def all_reduce(self, x):
            return x

        def all_gather(self, x, dim=-1):
            return torch.cat([x] * self.world_size, dim=dim)

        def reduce_scatter(self, x, dim=-1):
            return torch.chunk(x, self.world_size, dim=dim)[self.rank_in_group]

    def all_to_all_single(output, input, output_split_sizes, input_split_sizes, **kw):
        assert output.size(0) == sum(output_split_sizes)
        assert input.size(0) == sum(input_split_sizes)
    def all_reduce(x, *_, **kw):
        return x
    def all_gather_into_tensor(output_tensor, input_tensor, **kw):
        assert output_tensor.dim() == input_tensor.dim()

    coord_0 = MockGroupCoordinator(world_size=1, rank_in_group=0)
    coord = MockGroupCoordinator(world_size=cards, rank_in_group=rank)
    with (
        patch.multiple(
            "vllm.distributed.parallel_state",
            _WORLD=coord,
            _DP=coord_0,
            _TP=coord,
            _DCP=coord if dcp else coord_0,
        ),
        patch("torch.distributed.all_to_all_single", side_effect=all_to_all_single),
        patch("torch.distributed.all_reduce", side_effect=all_reduce),
        patch("torch.distributed.all_gather_into_tensor", side_effect=all_gather_into_tensor),
    ):
        yield


@contextmanager
def _mock_omni_cache(
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    index_head_dim: int = 128,
    num_slots: int = 16,
    pg: int = 128,
):
    fake_omni = MagicMock()
    kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank, **cfg_bf16)
    kv1 = torch.zeros(num_slots, pg, 1, qk_rope_head_dim, **cfg_bf16)
    kv2 = torch.zeros(num_slots, pg, 1, index_head_dim, **cfg_bf16)
    fake_omni_npu.device_cache = (kv0, kv1, kv2)
    fake_omni_npu.synchronize_d2h = MagicMock()
    fake_omni_npu.synchronize_h2d = MagicMock()

    cache_mod = MagicMock()
    cache_mod.omni_cache = fake_omni
    omni_parent = MagicMock()
    omni_parent.cache = cache_mod

    with patch.dict(
        sys.modules,
        {"omni_cache": omni_parent, "omni_cache.cache": cache_mod},
        clear=False,
    ):
        yield


@contextmanager
def _mock_mla_attention(
    num_slots: int = 64,
    pg: int = 128,
    use_omni_cache: bool = False,
    cont_kv: bool = False,
):
    class MockMLAAttention(torch.nn.Module):
        def __init__(
            self,
            num_heads: int,
            scale: float,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int | None,
            kv_lora_rank: int,
            kv_b_proj: torch.nn.Module,
            cache_config=None,
            quant_config=None,
            prefix: str = "",
            use_sparse: bool = True,
            indexer=None,
            **kwargs,
        ):
            super().__init__()
            self.layer_name = prefix
            impl = SimpleNamespace()
            self.W_UK_T = torch.zeros(
                num_heads,
                qk_nope_head_dim,
                kv_lora_rank,
                **cfg_bf16,
            )
            self.W_UV = torch.zeros(
                num_heads,
                kv_lora_rank,
                v_head_dim,
                **cfg_bf16,
            )
            impl.sink_kv = None
            impl.sink_k_nope = None
            self.impl = impl
            self.sink_k_pe = torch.zeros((128, qk_rope_head_dim), dtype=torch.bfloat16)
            self.sink_compressed_kv = torch.zeros((128, kv_lora_rank), dtype=torch.bfloat16)
            self.sink_populated = False

            # 0.25.1 binds the cache tuple itself (no per-virtual-engine list).
            if use_omni_cache:
                self.kv_cache = None
            elif cont_kv:
                idx_dim = getattr(indexer, "head_dim", 128) if indexer else 128
                kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank + qk_rope_head_dim, **cfg_bf16)
                kv1 = torch.zeros(num_slots, pg, 1, idx_dim, **cfg_bf16)
                self.kv_cache = (kv0, kv1)
            else:
                idx_dim = getattr(indexer, "head_dim", 128) if indexer else 128
                kv0 = torch.zeros(num_slots, pg, 1, kv_lora_rank, **cfg_bf16)
                kv1 = torch.zeros(num_slots, pg, 1, qk_rope_head_dim, **cfg_bf16)
                kv2 = torch.zeros(num_slots, pg, 1, idx_dim, **cfg_bf16)
                self.kv_cache = (kv0, kv1, kv2)

        def populate_sink_kv(self, k_nope_cache: torch.Tensor, k_pe_cache: torch.Tensor):
            self.sink_populated = True

        def update_sink_kv(self, k_pe: torch.Tensor, compressed_kv: torch.Tensor):
            self.sink_k_pe = k_pe
            self.sink_compressed_kv = compressed_kv
            self.sink_populated = True

    with (
        # A sink-less DSA layer builds NPUDSAAttention so its page covers the
        # indexer key.
        patch(f"{DSA_MODULE}.NPUDSAAttention", MockMLAAttention),
        patch(
            "vllm.model_executor.layers.attention.static_sink_attention."
            "StaticSinkMLAAttention",
            MockMLAAttention,
            create=True,
        ),
    ):
        yield


@contextmanager
def _mock_flash_comm_linear(init_comm=None): # callback

    def flash_comm(linear, x, op: str):
        if op in ["NoOp", "AllReduce"]:
            return x
        elif op == "ReduceScatter":
            T, tp = x.size(0), linear.tp_size
            assert T % tp == 0
            return x.new_zeros(T // tp, *x.shape[1:])
        else:
            raise ValueError(f"flash_comm err op {op}")

    def init_comm_0(linear):
        from vllm.distributed import get_tp_group
        linear.x_transform = "NoOp"
        linear.y_transform = "NoOp"
        linear.tp_rank = get_tp_group().rank_in_group
        linear.tp_size = get_tp_group().world_size
        linear.layer_name_inside_block = getattr(linear, "prefix", "") or "o_proj"
        if init_comm is not None:
            init_comm(linear)

    class MockReplicatedFlashCommLinear(torch.nn.Module):

        def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            quant_config=None,
            prefix: str = "",
            **kwargs,
        ):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.weight = torch.zeros(out_features, in_features, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = flash_comm(self, x, self.x_transform)
            assert x.size(-1) == self.in_features
            y = x.new_zeros(*x.shape[:-1], self.out_features)
            return flash_comm(self, y, self.y_transform), None

    class MockColumnParallelFlashCommLinear(torch.nn.Module):

        def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            quant_config=None,
            prefix: str = "",
            **kwargs,
        ):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.out_per_part = max(1, out_features // self.tp_size)
            self.weight = torch.zeros(self.out_per_part, in_features, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = flash_comm(self, x, self.x_transform)
            assert x.size(-1) == self.in_features
            y = x.new_zeros(*x.shape[:-1], self.out_per_part)
            return flash_comm(self, y, self.y_transform), None

    class MockRowParallelFlashCommLinear(torch.nn.Module):

        def __init__(self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            quant_config=None,
            prefix: str = "",
        ):
            super().__init__()
            self.prefix = prefix
            self.in_features = in_features
            self.out_features = out_features
            init_comm_0(self)
            self.in_per_part = max(1, in_features // self.tp_size)
            self.weight = torch.zeros(out_features, self.in_per_part, **cfg_bf16)

        def forward(self, x: torch.Tensor):
            assert x.dim() >= 2
            x = flash_comm(self, x, self.x_transform)
            assert x.size(-1) == self.in_per_part
            y = x.new_zeros(*x.shape[:-1], self.out_features)
            return flash_comm(self, y, self.y_transform), None

        def requires_input_partition(self):
            return self.tp_size > 1 and self.x_transform != "DP2TPAll2All"

    with (
        patch(f"{DSA_MODULE}.ReplicatedFlashCommLinear", MockReplicatedFlashCommLinear),
        patch(f"{DSA_MODULE}.ColumnParallelFlashCommLinear", MockColumnParallelFlashCommLinear),
        patch(f"{DSA_MODULE}.RowParallelFlashCommLinear", MockRowParallelFlashCommLinear),
    ):
        yield


@contextmanager
def _mock_layernorm_rmsnorm():
    class MockLayerNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.ones(dim, **cfg_bf16),
            )
            self.eps = eps

        def forward(self, x: torch.Tensor):
            assert x.size(-1) == self.weight.size(0)
            return x

    class MockRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6, dtype=None):
            super().__init__()
            weight_cfg = dict(cfg_bf16)
            if dtype is not None:
                weight_cfg["dtype"] = dtype
            self.weight = torch.nn.Parameter(
                torch.ones(dim, **weight_cfg),
            )
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor):
            assert x.size(-1) == self.weight.size(0)
            return x

    with (
        patch(f"{DSA_MODULE}.LayerNorm", MockLayerNorm),
        patch(f"{DSA_MODULE}.RMSNorm", MockRMSNorm),
    ):
        yield


@contextmanager
def _mock_mome():
    class MockAggregateConv:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, x, **kwargs):
            return x

    class MockMomeAttention:

        def __init__(self, *args, state_shapes, **kwargs):
            class MOME:
                def __init__(self, *args, **kwargs):
                    self.weight = None
                def __call__(self, x, *args, **kwargs):
                    return x
            self.qa_conv = MOME()
            self.compresskv_conv = MOME()
            self.o_conv = MOME()

            # caches = [torch.zeros(16, 48, *size) for size in state_shapes]
            # self.kv_cache = [(*caches,)]
            self.prefix = ".conv"
            self.kv_cache = MagicMock()

        def __call__(self, x, *args, **kwargs):
            return x

    import omni_npu.v1.layers.attention.npu_dsa as dsa_mod
    import omni_npu.v1.layers.attention.npu_mla as mla_mod
    mome_module = SimpleNamespace(MomeAttention=MockMomeAttention)
    with patch.dict(
        sys.modules,
        {"vllm.model_executor.layers.npumome": mome_module},
    ):
        dsa_mod.AggregateConv = MockAggregateConv
        mla_mod.AggregateConv = MockAggregateConv
        yield
        dsa_mod.AggregateConv = None
        mla_mod.AggregateConv = None


# =========================
# 3. 配置相关 mock
# =========================

def _make_dsa_config(
    index_topk=2048,
    index_n_heads=64,
    index_head_dim=128,
    qk_rope_head_dim=64,
    rms_norm_eps=1e-6,
    rope_type="default",
    factor=2.0,
    mscale_all_dim=False,
    apply_yarn_scaling=True,
    indexer_rope_interleave=False,
    kv_lora_rank=512,
    use_mome=False,
    param_sink_number=0,
    rope_scaling=None,
    num_hidden_layers=49,
    is_mtp_layer=False,
    num_nextn_predict_layers=0,
    index_topk_freq=1,
    indexer_types=None,
    index_topk_pattern=None,
    index_skip_topk_offset=2,
):
    return SimpleNamespace(
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_parameters={
            "rope_type": rope_type,
            "factor": factor,
            "mscale_all_dim": mscale_all_dim,
            "apply_yarn_scaling": apply_yarn_scaling,
            "rope_theta": 10000.0,
        },
        rope_interleave=True,
        indexer_rope_interleave=indexer_rope_interleave,
        kv_lora_rank=kv_lora_rank,
        use_mome=use_mome,
        router_sliding_window=3,
        param_sink_number=param_sink_number,
        param_sink_with_value=True,
        torch_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        rope_scaling=rope_scaling,
        num_hidden_layers=num_hidden_layers,
        is_mtp_layer=is_mtp_layer,
        num_nextn_predict_layers=num_nextn_predict_layers,
        index_topk_freq=index_topk_freq,
        indexer_types=indexer_types,
        index_topk_pattern=index_topk_pattern,
        index_skip_topk_offset=index_skip_topk_offset,
    )


def _make_vllm_config(
    speculative_config=None,
    cudagraph_capture_sizes=None,
):
    return SimpleNamespace(
        speculative_config=speculative_config,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            static_forward_context={},
        ),
        cache_config=SimpleNamespace(block_size=128),
    )


@contextmanager
def _mock_model_extra_config(
    seq_parallel=False,
    dsa_seq_parallel=False,
    omni_cache=False,
    mlaprolog_op=False,
    mtp_remove_redundant_kv=False,
    enable_dsa=False,
    sharded_o_proj=False,
    use_noncontiguous_kv=False,
    use_batch_invariant_op=False,
    enable_precision_strong_consistency=False,
    dtype=torch.bfloat16,
):
    with patch(
        f"{DSA_MODULE}.model_extra_config",
        MagicMock(
            dtype=dtype,
            parall_config=MagicMock(
                ena_seq_parallel=seq_parallel,
                ena_context_parallel=dsa_seq_parallel,
                sharded_o_proj=sharded_o_proj,
            ),
            operator_opt_config=MagicMock(
                use_omni_cache=omni_cache,
                enable_mlaprolog=mlaprolog_op,
                mtp_remove_redundant_kv=mtp_remove_redundant_kv,
                enable_dsa=enable_dsa,
                use_noncontiguous_kv=use_noncontiguous_kv,
                merge_q_kv_conv=False,
                use_batch_invariant_op=use_batch_invariant_op,
                enable_precision_strong_consistency=enable_precision_strong_consistency,
                use_rope_fusion_op=False,
                enable_multi_stream=False,
                split_q_up_in_multistream=False,
                enable_mome_sp=False,
                enable_prefill_graph_token_parallel=False,
            ),
        ),
    ):
        yield


@contextmanager
def _mock_forward_context(seq_lens: list, mode: str="prefill", use_mome=False):
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        virtual_engine=0,
        capturing=False,
        no_compile_layers={},
    )

    class _StageMetadata:
        def __init__(self, q_lens: list, kv_lens: list, pg: int = 128):
            cu_q_lens = torch.cumsum(torch.tensor(q_lens, **cfg_i32), dim=0)
            self.query_cumlens = cu_q_lens
            self.query_start_loc = torch.tensor([0] + cu_q_lens.tolist(), **cfg_i32)
            self.seq_lens = torch.tensor(kv_lens, **cfg_i32)
            blocks = max(32, sum(kv_lens) // pg + 1)
            self.block_table = torch.zeros(len(q_lens), blocks, **cfg_i32)
            self.prefix_meta = None
            self.slot_mapping_2d = None

    if mode == "pd_mixed":
        ctx.attn_metadata.prefill = _StageMetadata(seq_lens, seq_lens)
        ctx.attn_metadata.decode = _StageMetadata([1] * len(seq_lens), seq_lens)
        ctx.attn_metadata.slot_mapping = torch.arange(len(seq_lens) + sum(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = len(seq_lens)
        ctx.attn_metadata.num_prefills = len(seq_lens)
        ctx.attn_metadata.num_actual_tokens = len(seq_lens) + sum(seq_lens)
        ctx.attn_metadata.num_decode_tokens = len(seq_lens)
    elif mode == "prefill":
        ctx.attn_metadata.prefill = _StageMetadata(seq_lens, seq_lens)
        ctx.attn_metadata.decode = None
        ctx.attn_metadata.slot_mapping = torch.arange(sum(seq_lens), **cfg_i64)
        ctx.attn_metadata.num_decodes = 0
        ctx.attn_metadata.num_prefills = len(seq_lens)
        ctx.attn_metadata.num_actual_tokens = sum(seq_lens)
        ctx.attn_metadata.num_decode_tokens = 0

        from omni_npu.v1.layers.attention.npu_dsa import (
            get_tp_group,
            get_dcp_group,
            model_extra_config,
            SPManager,
            KVSPMaganer,
        )

        from omni_npu.attention.backends.utils import paged_cache
        if model_extra_config.parall_config.ena_seq_parallel:
            metadata = ctx.attn_metadata.prefill
            computed_lens = metadata.seq_lens - (metadata.query_start_loc[1:] - metadata.query_start_loc[:-1])
            metadata.sp_manager = SPManager.init_cp(
                sp_group=get_tp_group(),
                cumlens=metadata.query_start_loc,
                computed_lens=computed_lens,
                block_table_ref=metadata.block_table,
                table_size=metadata.block_table.size(1),
            )
            metadata.cache_fn = paged_cache(
                ctx.attn_metadata.slot_mapping,
                metadata.query_start_loc, # [B + 1]
            )
            if get_dcp_group().world_size > 1:
                metadata.kvsp_manager = KVSPMaganer(
                    q_cumlens=metadata.query_start_loc, # [B + 1]
                    kv_lens=metadata.seq_lens,
                    blk_table=metadata.block_table,
                )

    elif mode == "decode":
        ctx.attn_metadata.decode = _StageMetadata([1] * len(seq_lens), seq_lens)
        ctx.attn_metadata.prefill = None
        ctx.attn_metadata.slot_mapping = torch.arange(len(seq_lens), **cfg_i64)
        ctx.attn_metadata.decode.slot_mapping = ctx.attn_metadata.slot_mapping
        ctx.attn_metadata.num_decodes = 2
        ctx.attn_metadata.num_prefills = 0
        ctx.attn_metadata.num_actual_tokens = 2
        ctx.attn_metadata.num_decode_tokens = 2

        from omni_npu.v1.layers.attention.npu_dsa import (
            get_dcp_group,
            KVSPMaganer,
        )
        if get_dcp_group().world_size > 1:
            metadata = ctx.attn_metadata.decode
            metadata.kvsp_manager = KVSPMaganer(
                q_cumlens=metadata.query_start_loc,
                kv_lens=metadata.seq_lens,
                blk_table=metadata.block_table,
            )
    elif mode == "dummy_run":
        ctx.attn_metadata = None
    else:
        raise ValueError(f"error mode: {mode}")

    if ctx.attn_metadata is not None:
        slots = ctx.attn_metadata.slot_mapping
        ctx.attn_metadata.get_slot_mapping_2d = lambda _layer_idx=-1: torch.stack(
            [slots, slots], dim=-1
        )

    if use_mome:
        ctx.attn_metadata = {
            ".attn": ctx.attn_metadata,
            ".conv": MagicMock(),
        }

    with patch(f"{DSA_MODULE}.get_forward_context", return_value=ctx):
        yield


# =========================
# patch_and_gen_configs
# =========================

@contextmanager
def _patch_and_gen_configs(
    mode: str = "prefill",
    ena_seq_parallel: bool = False,
    ena_context_parallel: bool = False,
    use_omni_cache: bool = False,
    enable_mlaprolog: bool = False,
    use_noncontiguous_kv: bool = False,
    use_batch_invariant_op: bool = False,
    enable_precision_strong_consistency: bool = False,
    use_mome: bool = False,
    init_flash_comm = None,
    seq_lens: list = [32, 47],
    hidden_size: int = 2048,
    q_lora_rank: int = 1536,
    index_topk: int = 2048,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    rms_norm_eps: float = 1e-6,
    num_heads: int = 128,
    qk_nope_head_dim: int = 64,
    v_head_dim: int = 128,
    kv_lora_rank: int = 512,
    tp_size: int = 4,
    tp_rank: int = 0,
    pg: int = 128,
    rope_type: str = "default",
    indexer_rope_interleave: bool = False,
    factor: float = 2.0,
    mscale_all_dim: bool = False,
    apply_yarn_scaling: bool = True,
    param_sink_number: int = 0,
    ena_kvsp: bool = False,
    rope_scaling=None,
    num_hidden_layers: int = 49,
    is_mtp_layer: bool = False,
    num_nextn_predict_layers: int = 0,
    index_topk_freq: int = 1,
    indexer_types: list | None = None,
    index_topk_pattern: str | None = None,
    index_skip_topk_offset: int = 2,
    layer_idx: int | None = None,
):
    env = SimpleNamespace(
        seq_lens=seq_lens,
        tp_size=tp_size,
        cfg_bf16=dict(**cfg_bf16),
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        pg=pg,
        layer_idx=layer_idx,
    )
    cfg = _make_dsa_config(
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_type=rope_type,
        factor=factor,
        mscale_all_dim=mscale_all_dim,
        apply_yarn_scaling=apply_yarn_scaling,
        indexer_rope_interleave=indexer_rope_interleave,
        kv_lora_rank=kv_lora_rank,
        use_mome=use_mome,
        param_sink_number=param_sink_number,
        rope_scaling=rope_scaling,
        num_hidden_layers=num_hidden_layers,
        is_mtp_layer=is_mtp_layer,
        num_nextn_predict_layers=num_nextn_predict_layers,
        index_topk_freq=index_topk_freq,
        indexer_types=indexer_types,
        index_topk_pattern=index_topk_pattern,
        index_skip_topk_offset=index_skip_topk_offset,
    )
    vllm_cfg = _make_vllm_config(
        speculative_config=None,
    )
    mock_omni = _mock_omni_cache(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        index_head_dim=index_head_dim,
        num_slots=16, pg=pg,
    ) if use_omni_cache else nullcontext()

    # patch讲究顺序
    with (
        # 基础模块
        _mock_torch_npu_stream(),
        _mock_torch_npu(),
        _mock_misc(),
        # 配置项
        _mock_vllm_distributed(cards=tp_size, rank=tp_rank, dcp=ena_kvsp),
        _mock_model_extra_config(
            seq_parallel=ena_seq_parallel,
            dsa_seq_parallel=ena_context_parallel,
            omni_cache=use_omni_cache,
            mlaprolog_op=enable_mlaprolog,
            enable_dsa=True,
            use_noncontiguous_kv=use_noncontiguous_kv,
            use_batch_invariant_op=use_batch_invariant_op,
            enable_precision_strong_consistency=enable_precision_strong_consistency,
        ),
        patch(f"{DSA_MODULE}.get_current_vllm_config", return_value=vllm_cfg, create=True),
        # 功能模块
        _mock_flash_comm_linear(init_comm=init_flash_comm),
        _mock_layernorm_rmsnorm(),
        _mock_mome(),
        _mock_mla_attention(
            num_slots=64, pg=pg,
            use_omni_cache=use_omni_cache,
            cont_kv=use_noncontiguous_kv,
        ),
        mock_omni,
        # 运行时
        _mock_forward_context(seq_lens=seq_lens, mode=mode, use_mome=use_mome),
    ):
        yield cfg, vllm_cfg, env


# =========================
# Indexer 用例
# =========================

class _ConstProj(torch.nn.Module):
    def __init__(self, out: torch.Tensor):
        super().__init__()
        self._out = out

    def forward(self, _x: torch.Tensor):
        return self._out, None


class _IdentityNorm(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x


def _keep_tensor(x, *_args):
    return x


def _install_indexer_li_prolog_fakes(
    idx,
    *,
    wi_raw: torch.Tensor,
    qi_flat: torch.Tensor,
    ki_flat: torch.Tensor,
):
    idx.weights_proj = _ConstProj(wi_raw.clone())
    idx.wq_b = _ConstProj(qi_flat)
    idx.wk = _ConstProj(ki_flat)
    idx.k_norm = _IdentityNorm()
    idx._apply_rope = _keep_tensor


def _make_indexer(cfg, vllm_cfg, env, **kwargs):
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    return Indexer(
        vllm_config=vllm_cfg,
        config=cfg,
        hidden_size=env.hidden_size,
        q_lora_rank=env.q_lora_rank,
        quant_config=None,
        cache_config=vllm_cfg.cache_config,
        **kwargs,
    )


def _forward_indexer(idx, cfg, env):
    from omni_npu.v1.layers.attention.npu_dsa import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    meta = attn_metadata.prefill
    meta.slot_mapping = attn_metadata.slot_mapping
    meta.slot_mapping_2d = attn_metadata.get_slot_mapping_2d()
    token_count = meta.query_cumlens.flatten()[-1].item()
    hidden = env.hidden_size
    rope_dim = env.qk_rope_head_dim
    q_lora = env.q_lora_rank
    ki_dim = cfg.index_head_dim
    tok_idx, ki = idx.forward(
        x=torch.zeros(token_count, hidden, **env.cfg_bf16),
        qr=torch.zeros(token_count, q_lora, **env.cfg_bf16),
        cos=torch.zeros(token_count, 1, 1, rope_dim, **env.cfg_bf16),
        sin=torch.zeros(token_count, 1, 1, rope_dim, **env.cfg_bf16),
        attn_metadata=meta,
        ki_cache=torch.zeros(16, env.pg, 1, ki_dim, **env.cfg_bf16),
    )
    return tok_idx, ki, token_count, cfg.index_topk


def _run_default_indexer_forward(sink_len=128):
    with _patch_and_gen_configs(
        ena_seq_parallel=False,
        use_omni_cache=False,
        indexer_rope_interleave=False,
        rope_type="default",
    ) as (cfg, vllm_cfg, env):
        idx = _make_indexer(cfg, vllm_cfg, env, sink_len=sink_len)
        tok_idx, ki, token_count, topk = _forward_indexer(idx, cfg, env)
        return tok_idx, ki, token_count, topk, cfg


def _run_li_prolog_with_raw_wi(*, enable_precision_strong_consistency: bool):
    with _patch_and_gen_configs(
        ena_seq_parallel=False,
        use_omni_cache=False,
        enable_precision_strong_consistency=enable_precision_strong_consistency,
        index_n_heads=4,
        index_head_dim=8,
        hidden_size=16,
        q_lora_rank=16,
    ) as (cfg, vllm_cfg, env):
        idx = _make_indexer(cfg, vllm_cfg, env, sink_len=128)
        token_count, n_heads, head_dim = 3, cfg.index_n_heads, cfg.index_head_dim
        wi_raw = torch.full((token_count, n_heads), 2.0, dtype=torch.float32)
        _install_indexer_li_prolog_fakes(
            idx,
            wi_raw=wi_raw,
            qi_flat=torch.zeros(token_count, n_heads * head_dim, dtype=torch.float32),
            ki_flat=torch.zeros(token_count, head_dim, dtype=torch.float32),
        )
        wi, _qi, _ki = idx._li_prolog_ext(
            wx=torch.zeros(token_count, env.hidden_size, dtype=torch.float32),
            qr=torch.zeros(token_count, env.q_lora_rank, dtype=torch.float32),
            kx=torch.zeros(token_count, env.hidden_size, dtype=torch.float32),
            q_cos_sin=(torch.zeros(1), torch.zeros(1)),
            k_cos_sin=(torch.zeros(1), torch.zeros(1)),
        )
        return wi, wi_raw, idx


class TestIndexer:

    def test_case_1(self):
        tok_idx, ki, token_count, topk, cfg = _run_default_indexer_forward()
        assert tok_idx.shape == (token_count, 1, topk)
        assert ki.shape == (token_count, 1, cfg.index_head_dim)

    def test_case_2(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            indexer_rope_interleave=True,
            rope_type="deepseek_yarn",
        ) as (cfg, vllm_cfg, env):
            idx = _make_indexer(cfg, vllm_cfg, env)
            tok_idx, ki, token_count, topk = _forward_indexer(idx, cfg, env)
            assert tok_idx.shape == (token_count, 1, topk)
            assert ki.shape == (token_count, 1, cfg.index_head_dim)

    def test_precision_strong_consistency_sets_scales(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            enable_precision_strong_consistency=True,
        ) as (cfg, vllm_cfg, env):
            idx = _make_indexer(cfg, vllm_cfg, env, sink_len=128)
            assert idx.softmax_scale == pytest.approx(cfg.index_head_dim ** -0.5)
            assert idx.weights_scale == pytest.approx(cfg.index_n_heads ** -0.5)

    def test_precision_strong_consistency_disabled_skips_scales(self):
        with _patch_and_gen_configs(
            ena_seq_parallel=False,
            use_omni_cache=False,
            enable_precision_strong_consistency=False,
        ) as (cfg, vllm_cfg, env):
            idx = _make_indexer(cfg, vllm_cfg, env, sink_len=128)
            assert not hasattr(idx, "softmax_scale")
            assert not hasattr(idx, "weights_scale")

    def test_precision_strong_consistency_scales_wi(self):
        wi, wi_raw, idx = _run_li_prolog_with_raw_wi(
            enable_precision_strong_consistency=True,
        )
        expected = wi_raw * idx.weights_scale * idx.softmax_scale
        torch.testing.assert_close(wi, expected)

    def test_precision_strong_consistency_disabled_keeps_raw_wi(self):
        wi, wi_raw, _idx = _run_li_prolog_with_raw_wi(
            enable_precision_strong_consistency=False,
        )
        torch.testing.assert_close(wi, wi_raw)

    def test_forward_with_buffer(self):
        """Indexer.forward returns (tok_idx, ki); top-k is no longer written in-place."""
        tok_idx, ki, token_count, topk, cfg = _run_default_indexer_forward()
        assert tok_idx.shape == (token_count, 1, topk)
        assert ki.shape == (token_count, 1, cfg.index_head_dim)


# =========================
# NPUDeepseekSparseAttention 用例
# =========================

class TestNPUDeepseekSparseAttention:

    def _test_with_cfg(self,
        mode: str = "prefill",
        ena_seq_parallel: bool = False,
        ena_context_parallel: bool = False,
        use_noncontiguous_kv: bool = False,
        use_batch_invariant_op: bool = False,
        use_mome: bool = False,
        use_omni_cache: bool = False,
        enable_mlaprolog: bool = False,
        qk_nope_head_dim: int = 512,
        rope_type: str = "deepseek_yarn",
        param_sink_number: int = 0,
        ena_kvsp: bool = False,
        seq_lens: list | None = None,
        o_proj_unit_tp: bool = False,
        rope_scaling=None,
        num_hidden_layers: int = 49,
        is_mtp_layer: bool = False,
        num_nextn_predict_layers: int = 0,
        index_topk_freq: int = 1,
        indexer_types: list | None = None,
        index_topk_pattern: str | None = None,
        index_skip_topk_offset: int = 2,
        prefix: str | None = None,
        use_topk_indices_buffer: bool = False,
    ):
        def init_flash_comm(linear):
            if ena_seq_parallel:
                if ena_context_parallel:
                    if "q_b_proj" in linear.prefix:
                        linear.tp_size = 1
                    if "kv_b_proj" in linear.prefix:
                        linear.tp_size = 1
                if "o_proj" in linear.prefix:
                    if o_proj_unit_tp:
                        linear.tp_size = 1
                        linear.y_transform = "NoOp"
                    else:
                        linear.y_transform = "ReduceScatter"

        with _patch_and_gen_configs(
            mode=mode,
            ena_seq_parallel=ena_seq_parallel,
            ena_context_parallel=ena_context_parallel,
            use_noncontiguous_kv=use_noncontiguous_kv,
            use_batch_invariant_op=use_batch_invariant_op,
            use_mome=use_mome,
            use_omni_cache=use_omni_cache,
            enable_mlaprolog=enable_mlaprolog,
            qk_nope_head_dim=qk_nope_head_dim,
            rope_type=rope_type,
            init_flash_comm=init_flash_comm,
            param_sink_number=param_sink_number,
            ena_kvsp=ena_kvsp,
            seq_lens=seq_lens if seq_lens is not None else [32, 47],
            rope_scaling=rope_scaling,
            num_hidden_layers=num_hidden_layers,
            is_mtp_layer=is_mtp_layer,
            num_nextn_predict_layers=num_nextn_predict_layers,
            index_topk_freq=index_topk_freq,
            indexer_types=indexer_types,
            index_topk_pattern=index_topk_pattern,
            index_skip_topk_offset=index_skip_topk_offset,
        ) as (cfg, vllm_cfg, env):
            from omni_npu.v1.layers.attention.npu_dsa import (
                NPUDeepseekSparseAttention,
                get_forward_context,
            )
            import omni_npu.v1.layers.attention.npu_dsa as npu_dsa_mod
            if hasattr(npu_dsa_mod, "npu_dsa_forward"):
                torch.ops.vllm.npu_dsa_forward = npu_dsa_mod.npu_dsa_forward

            attn_kwargs = dict(
                vllm_config=vllm_cfg,
                config=cfg,
                hidden_size=env.hidden_size,
                num_heads=env.num_heads,
                qk_nope_head_dim=env.qk_nope_head_dim,
                qk_rope_head_dim=env.qk_rope_head_dim,
                v_head_dim=env.v_head_dim,
                q_lora_rank=env.q_lora_rank,
                kv_lora_rank=env.kv_lora_rank,
                cache_config=vllm_cfg.cache_config,
                quant_config=None,
            )
            if prefix is not None:
                attn_kwargs["prefix"] = prefix

            m = NPUDeepseekSparseAttention(**attn_kwargs)
            if not hasattr(m, "conv"):
                m.conv = None
            D, R = env.hidden_size, env.qk_rope_head_dim
            forward_ctx = get_forward_context()
            forward_ctx.no_compile_layers = {m.prefix: m}
            attn_metadata = forward_ctx.attn_metadata
            if type(attn_metadata) is dict:
                attn_metadata = attn_metadata[".attn"]
            if attn_metadata:
                T = 0
                for meta in [attn_metadata.prefill, attn_metadata.decode]:
                    if meta is not None:
                        T += meta.query_cumlens.flatten()[-1].item()
            else:
                T = 65536

            T0 = T # for cos, sin
            if ena_seq_parallel:
                T = -(-T // env.tp_size) # ceil_div
            # Decode-only + SP all-gathers tokens but not the topk buffer.
            topk_tokens = T0 if (ena_seq_parallel and mode == "decode") else T
            topk_indices_buffer = torch.empty(
                topk_tokens, 1, cfg.index_topk, dtype=torch.int32, device="cpu"
            )
            out, next_topk = m.forward(
                torch.zeros(T, D, **env.cfg_bf16),
                cos=torch.zeros(T0, 1, 1, R, **env.cfg_bf16),
                sin=torch.zeros(T0, 1, 1, R, **env.cfg_bf16),
                topk_indices_buffer=topk_indices_buffer,
            )
            assert out.shape == (T, D)
            assert next_topk is not None
            assert next_topk.shape[-1] == cfg.index_topk

    def test_dummy_run(self):
        self._test_with_cfg(mode="dummy_run")

    def test_prefill(self):
        self._test_with_cfg(mode="prefill")

    def test_decode(self):
        self._test_with_cfg(mode="decode")

    def test_pd_mixed(self):
        self._test_with_cfg(mode="pd_mixed")

    def test_decode_op_mlaprolog(self):
        self._test_with_cfg(
            mode="decode",
            enable_mlaprolog=True,
            rope_type="default"
        )

    def test_prefill_sp(self):
        self._test_with_cfg(
            mode="prefill",
            qk_nope_head_dim=128,
            ena_seq_parallel=True,
        )

    def test_prefill_cp(self):
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
        )

    def test_prefill_kvsp(self):
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            ena_kvsp=True,
        )

    def test_decode_sp(self):
        self._test_with_cfg(
            mode="decode",
            ena_seq_parallel=True,
        )

    def test_pd_mixed_sp(self):
        self._test_with_cfg(
            mode="pd_mixed",
            ena_seq_parallel=True,
        )

    def test_prefill_sink(self):
        self._test_with_cfg(
            mode="prefill",
            param_sink_number=128,
        )

    def test_decode_sink(self):
        self._test_with_cfg(
            mode="decode",
            param_sink_number=128,
        )

    def test_prefill_mome(self):
        self._test_with_cfg(
            mode="prefill",
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
        )

    def test_prefill_mome_cp(self):
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
        )

    def test_prefill_mome_cp_short_seq_no_mome_query_start_o_proj_tp1(self):
        """
        _forward_prefill_cp: use_mome but SPManager.init_cp skips _scheme_cp_mome when
        any seq len < 4*sp_size (e.g. [8,8] on sp_size=4), so hasattr cp_mome_query_start_loc
        is false. With o_proj.tp_size==1, the early return after _maybe_mome_out is taken.
        """
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
            seq_lens=[8, 8],
            o_proj_unit_tp=True,
        )

    def test_decode_mome(self):
        self._test_with_cfg(
            mode="decode",
            use_mome=True,
            use_noncontiguous_kv=True,
            param_sink_number=128,
        )

    def test_prefill_noncontiguous_batch_invariant(self):
        self._test_with_cfg(
            mode="prefill",
            use_noncontiguous_kv=True,
            use_batch_invariant_op=True,
            param_sink_number=128,
        )

    def test_mrope_without_rope_scaling(self):
        self._test_with_cfg(rope_scaling=None)

    def test_mrope_rope_scaling_no_mrope_section(self):
        self._test_with_cfg(rope_scaling={"factor": 2.0})

    def test_mrope_with_mrope_section(self):
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [12, 10, 10]},
            num_hidden_layers=49,
        )

    def test_mrope_with_mrope_section_mtp_layer(self):
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [12, 10, 10]},
            num_hidden_layers=49,
            is_mtp_layer=True,
            num_nextn_predict_layers=3,
        )

    def test_mrope_with_mrope_section_not_mtp_layer(self):
        self._test_with_cfg(
            rope_scaling={"factor": 2.0, "mrope_section": [12, 10, 10]},
            num_hidden_layers=49,
            is_mtp_layer=False,
            num_nextn_predict_layers=3,
        )

    # =========================
    # topk_indices_buffer is not None 分支覆盖
    # =========================

    def test_prefill_full_layer_with_buffer(self):
        """Full layer + prefill + buffer → 覆盖 _forward_prefill 写 buffer (line 1029)"""
        self._test_with_cfg(mode="prefill", use_topk_indices_buffer=True)

    def test_decode_full_layer_with_buffer(self):
        """Full layer + decode + buffer → 覆盖 _forward_decode 写 buffer (line 1273)"""
        self._test_with_cfg(mode="decode", use_topk_indices_buffer=True)

    def test_prefill_cp_full_layer_with_buffer(self):
        """Full layer + CP + buffer → 覆盖 _forward_prefill_cp 写 buffer (line 1173)"""
        self._test_with_cfg(
            mode="prefill",
            ena_seq_parallel=True,
            ena_context_parallel=True,
            use_topk_indices_buffer=True,
        )

    def test_prefill_shared_layer_with_buffer(self):
        """Shared layer + prefill + buffer → 覆盖 index_topk_pattern (line 450),
        self.indexer=None (line 471), _forward_prefill 读 buffer (lines 1031-1032)"""
        self._test_with_cfg(
            mode="prefill",
            index_topk_pattern="S",
            use_topk_indices_buffer=True,
        )

    def test_decode_shared_layer_with_buffer(self):
        """Shared layer + decode + buffer → 覆盖 _forward_decode 读 buffer (line 1275)"""
        self._test_with_cfg(
            mode="decode",
            index_topk_pattern="S",
            use_topk_indices_buffer=True,
        )

    def test_dummy_run_shared_layer(self):
        """Shared layer + dummy_run → 覆盖 _forward_prefill else 分支 dummy path (line 1034)"""
        self._test_with_cfg(mode="dummy_run", index_topk_pattern="S")

    def test_decode_kvsp_with_buffer(self):
        """Decode + KVSP + buffer → 覆盖 KVSP 路径 (lines 1255-1260, 1262) 及写 buffer (line 1273)"""
        self._test_with_cfg(
            mode="decode",
            ena_kvsp=True,
            use_topk_indices_buffer=True,
        )

    def test_prefill_indexer_types_shared(self):
        """Shared layer via indexer_types → 覆盖 indexer_types 分支 (line 452)"""
        self._test_with_cfg(
            mode="prefill",
            indexer_types=["shared"],
            use_topk_indices_buffer=True,
        )

@pytest.mark.unit
def test_indexer_update_cache_uses_a5_scatter(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.on_ascend950 = True
    scatter_mock = MagicMock()
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    ki = torch.zeros(3, 1, 8)
    slots_2d = torch.arange(3, dtype=torch.int64).view(3, 1)
    ki_cache = torch.zeros(4, 8, 1, 8)

    Indexer._update_cache(idx, ki, slots_2d, ki_cache)

    scatter_mock.assert_called_once()
    assert scatter_mock.call_args.args[0] is ki_cache
    assert torch.equal(scatter_mock.call_args.args[1], slots_2d)


@pytest.mark.unit
def test_indexer_noncontiguous_lightning_uses_a5_op(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.on_ascend950 = True
    idx.topk_tokens = 7
    expected = torch.ones(2, 1, 7, dtype=torch.int32)
    lightning_mock = MagicMock(return_value=(expected,))
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_lightning_indexer", lightning_mock)
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(use_noncontiguous_kv=True)),
    )

    out = Indexer._apply_lightning_indexer(
        idx,
        wi=torch.zeros(2, 4),
        qi=torch.zeros(2, 3, 8),
        ki_cache=torch.zeros(5, 8),
        q_cumlens=torch.tensor([2], dtype=torch.int32),
        kv_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )

    assert out is expected
    lightning_mock.assert_called_once()
    assert lightning_mock.call_args.kwargs["sparse_count"] == 7
    assert "sparse_block_size" not in lightning_mock.call_args.kwargs


@pytest.mark.unit
def test_sparse_kv_norm_rope_cache_a5_updates_contiguous_parts_separately(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.qk_rope_head_dim = 2
    fake.kv_lora_rank = 4
    fake.on_ascend950 = True
    fake.noncontiguous_kv = False
    fake.rope_interleaved = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda x, cos, sin: x

    scatter_mock = MagicMock()
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    slots = SimpleNamespace(
        slot_mapping=torch.arange(3, dtype=torch.int64),
        slot_mapping_2d=torch.arange(3, dtype=torch.int64).view(3, 1),
    )
    kv_cache = (torch.zeros(4, 8, 1, 4), torch.zeros(4, 8, 1, 2))

    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(3, 6),
        cos=torch.zeros(3, 1, 1, 2),
        sin=torch.zeros(3, 1, 1, 2),
        slots=slots,
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (3, 1, 4)
    assert k_pe.shape == (3, 1, 2)
    assert scatter_mock.call_count == 2
    assert scatter_mock.call_args_list[0].args[2].shape == (3, 1, 4)
    assert scatter_mock.call_args_list[1].args[2].shape == (3, 1, 2)


@pytest.mark.unit
def test_sparse_kv_norm_rope_cache_a5_uses_v2_for_noncontiguous_kv(monkeypatch):
    """A5 + noncontiguous_kv: npu_ai_infra_kv_rmsnorm_rope_cache_v2 is used instead of scatter."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    R, L, T = 2, 4, 3

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.qk_rope_head_dim = R
    fake.kv_lora_rank = L
    fake.on_ascend950 = True
    fake.noncontiguous_kv = True
    fake.rope_interleaved = False
    fake.kv_nz = False
    fake.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(L),
        variance_epsilon=1e-6,
    )
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_kv_rmsnorm_rope_cache=True)
        ),
    )
    combined_cache = torch.zeros(4, 8, 1, L + R)
    kv_cache = (combined_cache, torch.zeros(4, 8, 1, R))

    v2_mock = MagicMock(
        return_value=(
            torch.zeros(T, 1, 1, R),
            torch.zeros(T, 1, 1, L),
        )
    )
    scatter_mock = MagicMock()
    monkeypatch.setattr(
        "torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2",
        v2_mock,
    )
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter_mock)

    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(T, L + R),
        cos=torch.zeros(T, 1, 1, R),
        sin=torch.zeros(T, 1, 1, R),
        slots=torch.arange(T, dtype=torch.int64),
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (T, 1, L)
    assert k_pe.shape == (T, 1, R)
    v2_mock.assert_called_once()
    scatter_mock.assert_not_called()


@pytest.mark.unit
def test_post_attn_absorb_a5_uses_transposed_input(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.on_ascend950 = True
    fake.attn = SimpleNamespace(W_UV=torch.zeros(2, 4, 5))
    captured = {}

    def fake_bmm(**kwargs):
        captured.update(kwargs)
        return torch.zeros(2, 3, 5)

    monkeypatch.setattr(
        dsa_mod.torch_npu,
        "npu_transpose_batchmatmul",
        MagicMock(side_effect=fake_bmm),
    )

    out = NPUDeepseekSparseAttention._post_attn_absorb(
        fake,
        torch.zeros(3, 2, 4),
    )

    assert out.shape == (3, 10)
    assert captured["input"].shape == (3, 2, 4)
    assert captured["perm_x1"] == (1, 0, 2)


@pytest.mark.unit
def test_kv_norm_rope_cache_noncontiguous_batch_invariant_uses_scatter_block_update(
    monkeypatch,
):
    """When enable_kv_rmsnorm_rope_cache is enabled, fused_op is disabled and the non-contiguous
    cache update falls back to npu_ai_infra_scatter_block_update_."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.qk_rope_head_dim = 2
    fake.kv_lora_rank = 4
    fake.on_ascend950 = False
    fake.noncontiguous_kv = True
    fake.kv_nz = False
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda x, cos, sin: x

    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_kv_rmsnorm_rope_cache=False)
        ),
    )

    block_calls = []
    nd_calls = []

    def fake_scatter_block_update(cache, indices, updates):
        block_calls.append((cache, indices, updates))
        return cache

    def fake_scatter_nd_update(cache, indices, updates):
        nd_calls.append((cache, indices, updates))
        return cache

    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_scatter_block_update_",
        MagicMock(side_effect=fake_scatter_block_update),
        raising=False,
    )
    monkeypatch.setattr(
        dsa_mod.torch_npu,
        "npu_scatter_nd_update_",
        MagicMock(side_effect=fake_scatter_nd_update),
    )

    slots = SimpleNamespace(
        slot_mapping=torch.arange(3, dtype=torch.int64),
        slot_mapping_2d=torch.arange(3, dtype=torch.int64).view(3, 1),
    )
    kv_cache = (torch.zeros(4, 8, 1, 6), torch.zeros(4, 8, 1, 2))

    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(3, 6),
        cos=torch.zeros(3, 1, 1, 2),
        sin=torch.zeros(3, 1, 1, 2),
        slots=slots,
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (3, 1, 4)
    assert k_pe.shape == (3, 1, 2)
    assert len(block_calls) == 1
    assert len(nd_calls) == 0
    cache_arg, indices_arg, data_arg = block_calls[0]
    assert cache_arg is kv_cache[0]
    assert torch.equal(indices_arg, slots.slot_mapping_2d)
    assert data_arg.shape == (3, 6)


@pytest.mark.unit
def test_apply_attention_rescale_pioneer_calls_custom_ops(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    T, N, L, R, S, B = 2, 2, 4, 2, 3, 1
    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.on_ascend950 = False
    fake.scaling = 0.5
    fake.param_sink_number = S
    fake.num_local_heads = N
    fake.kv_lora_rank = L
    fake.dummy_value_cache = torch.zeros(1)
    fake.attn = SimpleNamespace(
        impl=SimpleNamespace(
            sink_k_nope=torch.zeros(S, 1, L),
            sink_kv=torch.zeros(S, 1, L + R),
        )
    )

    expected = torch.full((T, N, L), 7.0)
    pioneer = MagicMock(
        return_value=(
            torch.zeros(T, N, L),
            torch.zeros(1, T, N),
            torch.ones(1, T, N),
        )
    )
    meta = MagicMock(return_value="meta")
    sink_v2 = MagicMock(
        return_value=(
            torch.ones(T, N, L),
            None,
            torch.zeros(T, N, 1),
            torch.ones(T, N, 1),
        )
    )
    rescale = MagicMock(return_value=(expected, None, None))
    monkeypatch.setattr(dsa_mod.ops, "apply_FA_rescale_forward", rescale)
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_sparse_flash_attention_pioneer",
        pioneer,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "_npu_fused_infer_attention_sink_metadata",
        meta,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_fused_infer_attention_sink_v2",
        sink_v2,
        raising=False,
    )

    out = NPUDeepseekSparseAttention._apply_attention_rescale_pioneer(
        fake,
        q_nope=torch.zeros(T, N, L),
        q_pe=torch.zeros(T, N, R),
        q_cumlens=torch.tensor([T], dtype=torch.int32),
        kv_lens=torch.tensor([T], dtype=torch.int32),
        topk_idx=torch.zeros(T, 1, 2, dtype=torch.int32),
        block_table=torch.zeros(B, 4, dtype=torch.int32),
        kv_cache=(torch.zeros(4, 8, L),),
    )

    assert out is expected
    pioneer.assert_called_once()
    meta.assert_called_once()
    sink_v2.assert_called_once()
    rescale.assert_called_once()


@pytest.mark.unit
def test_apply_attn_absorb_precision_path_delegates_to_rescale_pioneer(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.noncontiguous_kv = True
    fake.param_sink_number = 128
    expected = torch.full((2, 2, 4), 3.0)
    pioneer = MagicMock(return_value=expected)
    fake._apply_attention_rescale_pioneer = pioneer
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                enable_precision_strong_consistency=True,
            )
        ),
    )

    out = NPUDeepseekSparseAttention._apply_attn_absorb(
        fake,
        q_nope=torch.zeros(2, 2, 4),
        q_pe=torch.zeros(2, 2, 2),
        q_cumlens=torch.tensor([2], dtype=torch.int32),
        kv_lens=torch.tensor([2], dtype=torch.int32),
        topk_idx=torch.zeros(2, 1, 2, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        kv_cache=(torch.zeros(4, 8, 4),),
    )

    assert out is expected
    pioneer.assert_called_once()


@pytest.mark.unit
def test_npu_dsa_forward_fake_returns_matching_hidden_and_topk_shapes():
    from omni_npu.v1.layers.attention.npu_dsa import npu_dsa_forward_fake

    hidden = torch.zeros(3, 16)
    topk = torch.zeros(3, 1, 8, dtype=torch.int32)
    out, out_topk = npu_dsa_forward_fake(
        hidden, torch.zeros(3, 1, 64), torch.zeros(3, 1, 64), "layer", topk, "layer.attn"
    )
    assert out.shape == hidden.shape
    assert out.dtype == hidden.dtype
    assert out_topk.shape == topk.shape
    assert out_topk.dtype == topk.dtype


@pytest.mark.unit
def test_prefill_cp_allows_omnicache_before_building_cp_metadata():
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.q_b_proj = SimpleNamespace(tp_size=1)
    fake.kv_b_proj = SimpleNamespace(tp_size=1)
    fake.ena_sp = True
    fake.noncontiguous_kv = False
    fake.use_omni_cache = True

    sp_manager = SimpleNamespace(
        cp_attn_meta=MagicMock(side_effect=RuntimeError("entered cp path"))
    )
    attn_metadata = SimpleNamespace(sp_manager=sp_manager)

    with (
        patch.object(dsa_mod, "DummySPManager", return_value=object()),
        pytest.raises(RuntimeError, match="entered cp path"),
    ):
        NPUDeepseekSparseAttention._forward_prefill_cp(
            fake,
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            attn_metadata,
            (None, None, None),
        )

    sp_manager.cp_attn_meta.assert_called_once()


def _cp_helper_host(use_mome=False, partition=False):
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    calls = []
    host = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    host.use_mome = use_mome
    host.q_a_proj = lambda x: (calls.append("q_a_proj") or x,)
    host._maybe_mome_q = lambda q, args: calls.append(("mome_q", args)) or q
    host.q_a_layernorm = lambda q: calls.append("q_a_layernorm") or q
    host.o_proj = SimpleNamespace(requires_input_partition=lambda: partition)
    host._apply_o_proj = lambda out: calls.append("o_proj") or out
    sp = SimpleNamespace(**{
        name: (lambda name: lambda t: calls.append(name) or t)(name)
        for name in ("ag_tokens", "cp_slice", "sp_to_cp", "sp_to_tp", "cp_to_sp")
    })
    return host, sp, calls


@pytest.mark.unit
@pytest.mark.parametrize("use_mome, mome_sp", ((False, False), (True, False), (True, True)))
def test_cp_q_lora_moves_the_down_projection_to_the_cp_layout(use_mome, mome_sp):
    host, sp, calls = _cp_helper_host(use_mome=use_mome)
    host._apply_mome_prefill_sp = MagicMock(return_value=mome_sp)
    x = torch.zeros(3, 4)
    assert host._cp_q_lora(x, sp, "mome args") is x
    middle = ["ag_tokens", ("mome_q", "mome args"), "cp_slice"] if use_mome and not mome_sp else ["sp_to_cp"]
    assert calls == ["q_a_proj", *middle, "q_a_layernorm"]
    if use_mome:
        host._apply_mome_prefill_sp.assert_called_once_with(x, 0, inplace=True)
    else:
        host._apply_mome_prefill_sp.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("partition", (False, True))
def test_cp_o_proj_returns_this_ranks_sp_tokens(partition):
    host, sp, calls = _cp_helper_host(partition=partition)
    x = torch.zeros(4, 6)
    assert host._cp_o_proj(x, sp) is x
    assert calls == [*(["sp_to_tp"] if partition else []), "o_proj", "cp_to_sp"]


@pytest.mark.unit
@pytest.mark.parametrize("with_topk", (False, True))
def test_cp_mome_sp_output_keeps_the_topk_return_contract(with_topk):
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    x = torch.zeros(2, 4)
    cp_out, sp_out = torch.ones(2, 6), torch.full((2, 4), 2.0)
    topk = torch.tensor([[[7]], [[9]]], dtype=torch.int32) if with_topk else None
    sp = SimpleNamespace(slice_tokens=lambda t, cached: t, ag_tokens=lambda t: t)
    host = SimpleNamespace(
        q_b_proj=SimpleNamespace(tp_size=1), kv_b_proj=SimpleNamespace(tp_size=1),
        ena_sp=True, ena_kvsp=False, noncontiguous_kv=True, use_mome=True,
        skip_topk=True, sharded_o_proj=False, kv_lora_rank=2,
        _cp_attention_metadata=lambda manager: (None, None, None),
        _get_li_caches=lambda cache: (cache[1], None),
        _cp_rotary_rows=lambda manager, cos, sin: (cos, sin),
        _cp_q_lora=lambda x, manager, args: x,
        _q_absorb=lambda q, cos, sin: (q, q),
        kv_a_proj_with_mqa=lambda x: (x,),
        _apply_mome_prefill_sp=MagicMock(return_value=True),
        _kv_norm_rope_cache=MagicMock(),
        _apply_sink_offset=lambda indices: indices,
        _apply_attn_absorb=MagicMock(return_value=cp_out),
        _post_attn_absorb=lambda out: out,
        _apply_mome_out_prefill_sp=MagicMock(return_value=sp_out),
    )
    with patch.object(torch.npu, "current_stream", return_value="main"), \
         patch(DSA_MODULE + ".named_stream", return_value="side"), \
         patch(DSA_MODULE + ".DummySPManager", return_value=sp):
        output, next_topk = NPUDeepseekSparseAttention._forward_prefill_cp(
            host, x, x, x, SimpleNamespace(sp_manager=sp), (None, None), topk,
        )
    assert output is sp_out and next_topk is topk
    host._apply_attn_absorb.assert_called_once()
    assert host._apply_attn_absorb.call_args.args[4] is topk
    host._apply_mome_out_prefill_sp.assert_called_once_with(cp_out, sp)


# Allow running directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _absorb_host(use_sink, rescale):
    """A sparse attention stripped down to what _apply_attn_absorb reads."""
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    host = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    host.noncontiguous_kv = True
    host.param_sink_number = 0
    host.kv_lora_rank = 4
    host.scaling = 0.5
    host.use_gpt_oss_sink = use_sink
    host.use_gpt_oss_sink_rescale = rescale
    host.gpt_oss_sink = torch.zeros(2)
    host.num_local_heads = 2
    return host


def _absorb_args():
    return dict(
        q_nope=torch.zeros(3, 2, 4), q_pe=torch.zeros(3, 2, 2),
        q_cumlens=torch.tensor([3], **cfg_i32), kv_lens=torch.tensor([8], **cfg_i32),
        topk_idx=torch.zeros(3, 1, 8, **cfg_i32),
        block_table=torch.zeros(1, 4, **cfg_i32),
        kv_cache=(torch.zeros(1, 8, 6),),
        attn_metadata=SimpleNamespace(),
    )


@pytest.mark.unit
@pytest.mark.parametrize("batch_invariant", (False, True))
def test_sinkless_sparse_attention_hands_the_sink_to_the_kernel(batch_invariant):
    """Without rescale the sparse kernel takes the sink logit itself."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    plain = MagicMock(return_value=(torch.ones(3, 2, 4),))
    with patch.object(dsa_mod.model_extra_config.operator_opt_config, "use_batch_invariant_op", batch_invariant), \
         patch.object(torch.ops.custom, "npu_ai_infra_sparse_flash_attention", plain, create=True):
        NPUDeepseekSparseAttention._apply_attn_absorb(
            _absorb_host(True, False), **_absorb_args()
        )
    assert torch.equal(plain.call_args.kwargs["sinks"], torch.zeros(2))
    assert plain.call_args.kwargs["batch_invariant"] is batch_invariant


@pytest.mark.unit
@pytest.mark.parametrize("batch_invariant", (False, True))
def test_sinkless_sparse_attention_rebuilds_the_sink_from_the_lse(batch_invariant):
    """With rescale the pioneer kernel returns softmax max/sum instead."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    pioneer = MagicMock(
        return_value=(torch.ones(3, 2, 4), torch.zeros(1, 3, 2), torch.ones(1, 3, 2))
    )
    host = _absorb_host(True, True)
    rescaled = []
    host._rescale_gpt_oss_sink_output = lambda out, lse, sink, layout: rescaled.append(layout) or out
    with patch.object(dsa_mod.model_extra_config.operator_opt_config, "use_batch_invariant_op", batch_invariant), patch.object(
        torch.ops.custom, "npu_ai_infra_sparse_flash_attention_pioneer", pioneer, create=True
    ):
        NPUDeepseekSparseAttention._apply_attn_absorb(host, **_absorb_args())
    assert pioneer.call_args.kwargs["return_softmax_lse"] is True
    assert "sinks" not in pioneer.call_args.kwargs  # this kernel has none
    assert rescaled == ["TND"]
    assert pioneer.call_args.kwargs["batch_invariant"] is batch_invariant


@pytest.mark.unit
def test_npu_dsa_attention_takes_the_dsa_page_and_the_npu_absorb_weights():
    """Swapping the bases would silently drop one of the two.

    The patch machinery can hand out more than one NPUMLAAttention object, so
    compare where each method resolves from rather than the objects themselves.
    """
    from omni_npu.v1.layers.attention.npu_dsa import NPUDSAAttention

    mro = [c.__name__ for c in NPUDSAAttention.__mro__]
    assert mro.index("NPUMLAAttention") < mro.index("DSAAttention")
    assert NPUDSAAttention.process_weights_after_loading.__qualname__ == (
        "NPUMLAAttention.process_weights_after_loading"
    )
    assert NPUDSAAttention.get_kv_cache_spec.__qualname__ == "DSAAttention.get_kv_cache_spec"
    assert NPUDSAAttention.get_attn_backend.__qualname__ == "DSAAttention.get_attn_backend"


@pytest.mark.unit
@pytest.mark.parametrize("rescale", (False, True))
def test_enhance_fallback_refuses_a_sink_in_either_mode(rescale):
    """npu_sparse_flash_attention_enhance takes no sink, so say so loudly."""
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    host = _absorb_host(True, rescale)
    host.noncontiguous_kv = False  # the fallback path
    with pytest.raises(NotImplementedError, match="takes no sink logit"):
        NPUDeepseekSparseAttention._apply_attn_absorb(host, **_absorb_args())


@pytest.mark.unit
@pytest.mark.parametrize("on_a5", (False, True))
@pytest.mark.parametrize("c8_global_cache", (False, True))
def test_unpacked_local_cache_writes_its_two_parts_separately(on_a5, c8_global_cache):
    """A CLA layer's local cache keeps nope and RoPE apart while the model's are packed."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    host = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    host.qk_rope_head_dim, host.kv_lora_rank = 2, 4
    host.noncontiguous_kv = True
    host.on_ascend950 = on_a5
    host.is_sfa_c8 = c8_global_cache
    host.cache_dtype = "hif8_ds_mla" if c8_global_cache else "auto"
    host.rope_interleaved = False
    host.kv_a_layernorm = lambda x: x
    host._apply_rope = lambda x, _c, _s: x
    slots = SimpleNamespace(
        slot_mapping=torch.zeros(3, **cfg_i32), slot_mapping_2d=torch.zeros(3, 2, **cfg_i32)
    )
    kv_cache = (torch.zeros(1, 8, 4), torch.zeros(1, 8, 2))
    update = MagicMock()
    opt = SimpleNamespace(enable_kv_rmsnorm_rope_cache=False)
    with patch.object(dsa_mod, "model_extra_config", SimpleNamespace(operator_opt_config=opt)), \
         patch.object(dsa_mod.torch_npu, "npu_scatter_nd_update_", update), \
         patch.object(torch.ops.custom, "npu_ai_infra_scatter_block_update_", update, create=True):
        host._kv_norm_rope_cache(
            torch.zeros(3, 6), torch.zeros(3, 2), torch.zeros(3, 2), slots, kv_cache,
            packed_kv=False,
        )
    assert update.call_count == 2  # one call per part, not one concatenated write
    assert update.call_args_list[0].args[0] is kv_cache[0]
    assert update.call_args_list[1].args[0] is kv_cache[1]
    assert update.call_args_list[0].args[2].shape[-1] == 4
    assert update.call_args_list[1].args[2].shape[-1] == 2


@pytest.mark.unit
def test_dsa_absorb_helpers_take_an_injected_weight():
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    host = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    host.attn = SimpleNamespace(W_UK_T=torch.zeros(256, 256), W_UV=torch.zeros(1))
    host.num_local_heads, host.kv_lora_rank, host.v_head_dim = 2, 3, 4
    seen = {}

    def record_kw(**kwargs):
        seen["w_uk_t"] = kwargs["weight"]
        return torch.zeros(1, 2, 3)

    with patch.object(dsa_mod.torch_npu, "npu_transpose_batchmatmul", record_kw):
        host._q_nope_absorb(torch.zeros(1, 2, 3), w_uk_t=torch.ones(128, 128))
    assert seen["w_uk_t"].shape == (128, 128)  # the injected one, not impl's

    def record_pos(**kwargs):
        seen["w_uv"] = kwargs["weight"]
        return torch.zeros(1, 2, 4)

    with patch.object(dsa_mod.torch_npu, "npu_transpose_batchmatmul", record_pos):
        host._post_attn_absorb(torch.zeros(1, 2, 3), w_uv=torch.ones(1))
    assert torch.equal(seen["w_uv"], torch.ones(1))


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_type, sink_len, expect_rms",
    (
        ("openpangu_v3", 0, True),
        ("openpangu_v2", 128, True),
        ("pangu_ultra_moe", 0, True),  # the name does not start with openpangu
        ("deepseek_v3", 128, False),  # the family decides, not the sink count
    ),
)
def test_indexer_key_norm_follows_the_model_family(model_type, sink_len, expect_rms):
    """param_sink_number stood in for "is Pangu" until V3 set it to zero."""
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    config = SimpleNamespace(
        model_type=model_type, index_topk=2048, index_n_heads=4, index_head_dim=128,
        qk_rope_head_dim=64, rms_norm_eps=1e-5,
    )
    opt = SimpleNamespace(enable_precision_strong_consistency=False)
    rms, layer = MagicMock(), MagicMock()
    with patch.object(dsa_mod, "model_extra_config", SimpleNamespace(operator_opt_config=opt)), \
         patch.object(dsa_mod, "ReplicatedFlashCommLinear", MagicMock()), \
         patch.object(dsa_mod, "RMSNorm", rms), patch.object(dsa_mod, "LayerNorm", layer), \
         patch.object(dsa_mod, "on_ascend950", lambda: False), \
         patch.object(Indexer, "_build_index_cache", MagicMock(), create=True):
        try:
            Indexer(None, config, 1536, 384, None, None, sink_len, "p")
        except Exception:
            pass  # the rest of __init__ needs a real vllm config; the norm choice is made first
    assert rms.called is expect_rms
    assert layer.called is (not expect_rms)


@pytest.mark.unit
def test_sinkless_sparse_attention_leaves_the_sink_out_entirely():
    """A model without sinks calls the same op with no sinks argument."""
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    host = _absorb_host(False, False)
    op = MagicMock(return_value=(torch.zeros(3, 2, 4),))
    with patch.object(torch.ops.custom, "npu_ai_infra_sparse_flash_attention", op, create=True):
        NPUDeepseekSparseAttention._apply_attn_absorb(host, **_absorb_args())

    assert "sinks" not in op.call_args.kwargs
    assert "return_softmax_lse" not in op.call_args.kwargs


@pytest.mark.unit
@pytest.mark.parametrize("dummy", (False, True))
def test_cp_page_gather_keeps_both_gathers_on_the_cache_stream(dummy):
    from contextlib import nullcontext
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    cache = None if dummy else ("latent", "rope", "indexer")
    calls = []
    manager = SimpleNamespace(ag_pages=lambda c, seperate: (lambda: calls.append(c) or c,))
    sub_stream = MagicMock()
    with patch.object(torch.npu, "stream", side_effect=lambda stream: nullcontext()) as stream:
        out = NPUDeepseekSparseAttention._cp_gather_cache_pages(manager, cache, "main", sub_stream)
    expected = (None, None) if dummy else ("latent", "rope")
    assert out == expected and calls == list(expected)
    sub_stream.wait_stream.assert_called_once_with("main")
    stream.assert_called_once_with(sub_stream)


@pytest.mark.unit
def test_cp_attention_metadata_uses_one_layout_for_lengths_pages_and_rope():
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    manager = SimpleNamespace(cp_attn_meta=lambda: ("q", "kv", "packed", "pages"),
                              cp_slice=MagicMock(side_effect=lambda t, cached: (t, cached)))
    assert NPUDeepseekSparseAttention._cp_attention_metadata(manager) == ("q", "kv", "pages")
    assert not manager.cp_slice.called  # metadata must not move rotary work earlier
    assert NPUDeepseekSparseAttention._cp_rotary_rows(manager, "cos rows", "sin rows") == (
        ("cos rows", "cos"), ("sin rows", "sin"))


@pytest.mark.unit
def test_dsa_cp_prepares_sp_rotary_rows_before_cp_rotary_rows():
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    calls = []
    host = SimpleNamespace(
        q_b_proj=SimpleNamespace(tp_size=1), kv_b_proj=SimpleNamespace(tp_size=1),
        ena_sp=True, ena_kvsp=False, noncontiguous_kv=True,
        _cp_attention_metadata=lambda sp: calls.append("metadata") or (None, None, None),
        _get_li_caches=lambda cache: (cache[1], None),
        _cp_rotary_rows=lambda sp, cos, sin: calls.append("cp rotary") or (cos, sin),
        _cp_q_lora=MagicMock(side_effect=RuntimeError("stop before projection")),
    )
    manager = SimpleNamespace(slice_tokens=lambda x, cached: calls.append("sp " + cached) or x)
    meta = SimpleNamespace(sp_manager=manager)
    with patch.object(torch.npu, "current_stream", return_value="main"), \
         patch(DSA_MODULE + ".named_stream", return_value="side"), \
         patch(DSA_MODULE + ".DummySPManager", return_value=manager):
        with pytest.raises(RuntimeError, match="stop before projection"):
            NPUDeepseekSparseAttention._forward_prefill_cp(host, None, "cos", "sin", meta, (None, None))
    assert calls == ["metadata", "sp cos", "sp sin", "cp rotary"]


@pytest.mark.unit
def test_dsa_c8_dtype_helpers():
    from omni_npu.v1.layers.attention.npu_dsa import (
        get_li_quant_mode,
        is_sfa_c8_dtype,
    )

    assert is_sfa_c8_dtype("fp8_ds_mla")
    assert is_sfa_c8_dtype("hif8_ds_mla")
    assert not is_sfa_c8_dtype("int8_ds_mla")
    assert not is_sfa_c8_dtype("li_int8_ds_mla")
    assert not is_sfa_c8_dtype(None)

    assert get_li_quant_mode("fp8_ds_mla") == "fp8"
    assert get_li_quant_mode("hif8_ds_mla") == "hif8"
    with pytest.raises(ValueError, match="INT8 C8 is not supported"):
        get_li_quant_mode("int8_ds_mla")
    assert get_li_quant_mode("li_int8_ds_mla") == "int8"
    assert get_li_quant_mode("bf16") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cache_dtype", "dynamic_calls"),
    [
        ("hif8_ds_mla", 0),
        ("fp8_ds_mla", 1),
    ],
)
def test_dsa_c8_pack_pertile128_kv(monkeypatch, cache_dtype, dynamic_calls):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.cache_dtype = cache_dtype
    fake.kv_lora_rank = 512
    fake.qk_rope_head_dim = 64
    fake.fp8_packed_kv_dim = 656
    dtype_cast = MagicMock(side_effect=lambda tensor, _dtype: torch.zeros_like(tensor, dtype=torch.uint8))

    def fake_dynamic_quant(tensor, **_kwargs):
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.ones(tensor.shape[:-1])

    dynamic_quant = MagicMock(side_effect=fake_dynamic_quant)
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_dtype_cast", dtype_cast)
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_dynamic_quant", dynamic_quant)
    monkeypatch.setattr(dsa_mod.torch_npu, "hifloat8", torch.uint8, raising=False)

    packed = NPUDeepseekSparseAttention._pack_pertile128_kv(
        fake,
        torch.zeros(2, 1, 512, dtype=torch.float16),
        torch.zeros(2, 1, 64, dtype=torch.float16),
    )

    assert packed.shape == (2, 1, 656)
    assert dtype_cast.call_count == int(cache_dtype == "hif8_ds_mla")
    assert dynamic_quant.call_count == dynamic_calls


@pytest.mark.unit
def test_dsa_c8_pack_rejects_bad_dims_and_dtype():
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.cache_dtype = "hif8_ds_mla"
    fake.kv_lora_rank = 512
    fake.qk_rope_head_dim = 64
    fake.fp8_packed_kv_dim = 656
    with pytest.raises(ValueError, match="Unexpected dims"):
        NPUDeepseekSparseAttention._pack_pertile128_kv(
            fake,
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 64),
        )

    fake.cache_dtype = "bad_dtype"
    with pytest.raises(ValueError, match="Unsupported C8 dtype"):
        NPUDeepseekSparseAttention._pack_pertile128_kv(
            fake,
            torch.zeros(1, 1, 512),
            torch.zeros(1, 1, 64),
        )


@pytest.mark.unit
def test_dsa_c8_scatter_cache_supports_slots_and_slots_2d(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.fp8_packed_kv_dim = 656
    packed = torch.zeros(3, 656, dtype=torch.uint8)
    fake._pack_pertile128_kv = MagicMock(return_value=packed)
    scatter = MagicMock()
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter)
    kv_cache = (torch.zeros(4, 8, 656, dtype=torch.uint8),)

    NPUDeepseekSparseAttention._scatter_c8_kv_cache(
        fake,
        torch.empty(0),
        torch.empty(0),
        kv_cache,
        slots=torch.tensor([0, 9, 17]),
        slots_2d=torch.arange(3).view(3, 1),
    )
    assert torch.equal(scatter.call_args.args[1], torch.arange(3).view(3, 1))

    NPUDeepseekSparseAttention._scatter_c8_kv_cache(
        fake,
        torch.empty(0),
        torch.empty(0),
        kv_cache,
        slots=torch.tensor([0, 9, 17]),
        slots_2d=None,
    )
    block_slots = scatter.call_args.args[1]
    assert torch.equal(block_slots, torch.tensor([[0, 0], [1, 1], [2, 1]]))


@pytest.mark.unit
def test_dsa_get_li_caches():
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    assert NPUDeepseekSparseAttention._get_li_caches(fake, None) == (None, None)

    kv_cache = ("mla", "li", "scale")
    fake.noncontiguous_kv = True
    fake.li_quant_mode = "int8"
    assert NPUDeepseekSparseAttention._get_li_caches(fake, kv_cache) == ("li", "scale")
    fake.li_quant_mode = None
    assert NPUDeepseekSparseAttention._get_li_caches(fake, kv_cache) == ("li", None)

    fake.noncontiguous_kv = False
    assert NPUDeepseekSparseAttention._get_li_caches(fake, kv_cache) == ("scale", None)


@pytest.mark.unit
@pytest.mark.parametrize("li_quant_mode", ["int8", "fp8", "hif8"])
def test_indexer_update_cache_quant_li(monkeypatch, li_quant_mode):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.li_quant_mode = li_quant_mode
    scatter = MagicMock()

    def fake_dynamic_quant(tensor, **_kwargs):
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.ones(tensor.size(0))

    monkeypatch.setattr(dsa_mod.torch_npu, "npu_dynamic_quant", MagicMock(side_effect=fake_dynamic_quant))
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_scatter_nd_update_", scatter)
    monkeypatch.setattr(dsa_mod.torch_npu, "hifloat8", torch.uint8, raising=False)

    Indexer._update_cache(
        idx,
        ki=torch.zeros(3, 1, 8),
        slots_2d=torch.arange(3).view(3, 1),
        ki_cache=torch.zeros(4, 8, 1, 8, dtype=torch.uint8),
        ki_scale_cache=torch.zeros(4, 8, 1),
    )

    assert scatter.call_count == 2


@pytest.mark.unit
def test_indexer_update_cache_quant_li_rejects_unknown_mode():
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.li_quant_mode = "bad"
    with pytest.raises(ValueError, match="Unsupported LI quant mode"):
        Indexer._update_cache(
            idx,
            ki=torch.zeros(1, 1, 8),
            slots_2d=torch.zeros(1, 1, dtype=torch.int64),
            ki_cache=torch.zeros(1, 1, 1, 8),
            ki_scale_cache=torch.zeros(1, 1, 1),
        )


@pytest.mark.unit
@pytest.mark.parametrize("li_quant_mode", ["int8", "fp8", "hif8"])
def test_indexer_apply_lightning_indexer_quant_li(monkeypatch, li_quant_mode):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.li_quant_mode = li_quant_mode
    idx.cache_dtype = f"{li_quant_mode}_ds_mla"
    idx.topk_tokens = 3
    expected = torch.ones(2, 1, 3, dtype=torch.int32)

    def fake_dynamic_quant(tensor, **_kwargs):
        scale_shape = tensor.shape[:-1]
        return torch.zeros_like(tensor, dtype=torch.uint8), torch.ones(scale_shape)

    quant_indexer = MagicMock(return_value=(expected,))
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_dynamic_quant", MagicMock(side_effect=fake_dynamic_quant))
    monkeypatch.setattr(dsa_mod.torch_npu, "npu_quant_lightning_indexer", quant_indexer, raising=False)
    monkeypatch.setattr(dsa_mod.torch_npu, "hifloat8", torch.uint8, raising=False)
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(use_noncontiguous_kv=True)),
    )

    out = Indexer._apply_lightning_indexer(
        idx,
        wi=torch.zeros(2, 4),
        qi=torch.zeros(2, 1, 8),
        ki_cache=torch.zeros(4, 8, 8, dtype=torch.uint8),
        q_cumlens=torch.tensor([2], dtype=torch.int32),
        kv_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        ki_scale_cache=torch.zeros(4, 8, 1),
    )

    assert out is expected
    kwargs = quant_indexer.call_args.kwargs
    assert kwargs["sparse_count"] == 3
    assert kwargs["key_dequant_scale"].shape == (4, 8, 1)
    assert kwargs["weights"].dtype == (torch.float16 if li_quant_mode == "int8" else torch.bfloat16)
    if li_quant_mode == "hif8":
        assert kwargs["query_dtype"] is torch.uint8
        assert kwargs["key_dtype"] is torch.uint8


@pytest.mark.unit
def test_indexer_apply_lightning_indexer_quant_li_requires_op(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.li_quant_mode = "int8"
    idx.cache_dtype = "int8_ds_mla"
    idx.topk_tokens = 3
    monkeypatch.delattr(dsa_mod.torch_npu, "npu_quant_lightning_indexer", raising=False)
    monkeypatch.setattr(
        dsa_mod.torch_npu,
        "npu_dynamic_quant",
        MagicMock(return_value=(torch.zeros(1, 1, 8), torch.ones(1, 1))),
    )
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(use_noncontiguous_kv=True)),
    )

    with pytest.raises(RuntimeError, match="npu_quant_lightning_indexer is required"):
        Indexer._apply_lightning_indexer(
            idx,
            wi=torch.zeros(1, 4),
            qi=torch.zeros(1, 1, 8),
            ki_cache=torch.zeros(1, 1, 8),
            q_cumlens=torch.tensor([1], dtype=torch.int32),
            kv_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            ki_scale_cache=torch.zeros(1, 1, 1),
        )


@pytest.mark.unit
def test_sparse_kv_norm_rope_cache_c8_delegates_to_scatter(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.qk_rope_head_dim = 64
    fake.kv_lora_rank = 512
    fake.is_sfa_c8 = True
    fake.cache_dtype = "hif8_ds_mla"
    fake.noncontiguous_kv = True
    fake.kv_a_layernorm = lambda x: x
    fake._apply_rope = lambda x, _cos, _sin: x
    fake._scatter_c8_kv_cache = MagicMock()
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(enable_kv_rmsnorm_rope_cache=False)),
    )

    slots = SimpleNamespace(
        slot_mapping=torch.arange(2, dtype=torch.int64),
        slot_mapping_2d=torch.arange(2, dtype=torch.int64).view(2, 1),
    )
    kv_cache = (torch.zeros(4, 8, 656, dtype=torch.uint8),)
    k_nope, k_pe = NPUDeepseekSparseAttention._kv_norm_rope_cache(
        fake,
        latent_kv=torch.zeros(2, 576),
        cos=torch.zeros(2, 1, 1, 64),
        sin=torch.zeros(2, 1, 1, 64),
        slots=slots,
        kv_cache=kv_cache,
    )

    assert k_nope.shape == (2, 1, 512)
    assert k_pe.shape == (2, 1, 64)
    fake._scatter_c8_kv_cache.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "hif8_ds_mla"])
def test_apply_attn_absorb_c8_quant_sfa(monkeypatch, cache_dtype):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.noncontiguous_kv = True
    fake.param_sink_number = 128
    fake.is_sfa_c8 = True
    fake.cache_dtype = cache_dtype
    fake.scaling = 0.5
    fake.dummy_value_cache_quant = torch.zeros(1, 8, 1, 656, dtype=torch.uint8)
    fake.attn = SimpleNamespace(
        impl=SimpleNamespace(
            sink_kv=torch.zeros(1, 1, 1),
            sink_k_nope=torch.zeros(1, 1, 1),
        )
    )
    monkeypatch.setattr(dsa_mod.torch_npu, "hifloat8", torch.uint8, raising=False)
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_precision_strong_consistency=False)
        ),
    )
    expected = torch.full((2, 1, 4), 9.0)
    custom_quant_sfa = MagicMock(return_value=expected)
    int8_quant_sfa = MagicMock(return_value=expected)
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_ai_infra_kv_quant_sparse_flash_attention",
        custom_quant_sfa,
        raising=False,
    )
    monkeypatch.setattr(
        dsa_mod.torch_npu,
        "npu_kv_quant_sparse_flash_attention",
        int8_quant_sfa,
        raising=False,
    )

    out = NPUDeepseekSparseAttention._apply_attn_absorb(
        fake,
        q_nope=torch.zeros(2, 1, 4),
        q_pe=torch.zeros(2, 1, 2),
        q_cumlens=torch.tensor([2], dtype=torch.int32),
        kv_lens=torch.tensor([2], dtype=torch.int32),
        topk_idx=torch.zeros(2, 1, 2048, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        kv_cache=(torch.zeros(1, 8, 656, dtype=torch.uint8),),
    )

    assert out is expected
    custom_quant_sfa.assert_called_once()
    int8_quant_sfa.assert_not_called()
    assert "key_sink" in custom_quant_sfa.call_args.kwargs
    if cache_dtype == "hif8_ds_mla":
        assert custom_quant_sfa.call_args.kwargs["key_dtype"] is torch.uint8



@pytest.mark.unit
def test_apply_attn_absorb_hif8_requires_sparse_count_2048(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.noncontiguous_kv = True
    fake.param_sink_number = 128
    fake.is_sfa_c8 = True
    fake.cache_dtype = "hif8_ds_mla"
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_precision_strong_consistency=False)
        ),
    )

    with pytest.raises(ValueError, match="HiFloat8 SFA requires"):
        NPUDeepseekSparseAttention._apply_attn_absorb(
            fake,
            q_nope=torch.zeros(2, 1, 4),
            q_pe=torch.zeros(2, 1, 2),
            q_cumlens=torch.tensor([2], dtype=torch.int32),
            kv_lens=torch.tensor([2], dtype=torch.int32),
            topk_idx=torch.zeros(2, 1, 7, dtype=torch.int32),
            block_table=torch.zeros(1, 4, dtype=torch.int32),
            kv_cache=(torch.zeros(1, 8, 656, dtype=torch.uint8),),
        )


@pytest.mark.unit
def test_indexer_update_cache_qli_v2_uses_mxfp8(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.li_quant_mode = "hif8"
    idx.use_qli_v2 = True
    cache_op = MagicMock()
    monkeypatch.setattr(
        dsa_mod.torch.ops.cann_ops_transformer,
        "indexer_quant_cache",
        cache_op,
        raising=False,
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 9, 18], dtype=torch.int64),
        slot_mapping_2d=torch.tensor([[0, 0], [1, 1], [2, 2]]),
    )
    key_cache = torch.zeros(4, 8, 1, 128, dtype=torch.float8_e4m3fn)
    scale_cache = torch.zeros(4, 8, 1, 4, dtype=torch.uint8)

    Indexer._update_cache(
        idx,
        ki=torch.zeros(3, 1, 128),
        slots_2d=metadata,
        ki_cache=key_cache,
        ki_scale_cache=scale_cache,
    )

    args = cache_op.call_args
    assert args.args[0] is key_cache
    assert args.args[1] is scale_cache
    assert args.args[2].shape == (3, 128)
    assert torch.equal(args.args[3], metadata.slot_mapping.to(torch.int32))
    assert args.args[3].dtype == torch.int32
    assert args.kwargs == {"quant_mode": "mxfp8", "round_scale": True}


@pytest.mark.unit
def test_indexer_apply_qli_v2_reuses_metadata(monkeypatch):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import Indexer

    idx = Indexer.__new__(Indexer)
    idx.li_quant_mode = "hif8"
    idx.use_qli_v2 = True
    idx.is_metadata_producer = False
    idx.n_head = 64
    idx.head_dim = 128
    idx.topk_tokens = 3
    metadata = torch.zeros(1024, dtype=torch.int32)
    idx.qli_v2_metadata_op = MagicMock(return_value=metadata)
    expected = torch.ones(2, 1, 3, dtype=torch.int32)
    qli_v2_op = MagicMock(return_value=(expected, torch.empty(0)))
    monkeypatch.setattr(
        dsa_mod.torch.ops.cann_ops_transformer,
        "quant_lightning_indexer",
        qli_v2_op,
        raising=False,
    )

    def fake_mx_quant(tensor, **_kwargs):
        return (
            torch.zeros_like(tensor, dtype=torch.float8_e4m3fn),
            torch.zeros(tensor.shape[0], 4, dtype=torch.uint8),
        )

    monkeypatch.setattr(
        dsa_mod.torch_npu,
        "npu_dynamic_mx_quant",
        MagicMock(side_effect=fake_mx_quant),
    )
    monkeypatch.setattr(
        dsa_mod,
        "model_extra_config",
        SimpleNamespace(operator_opt_config=SimpleNamespace(use_noncontiguous_kv=True)),
    )

    q_cumlens = torch.tensor([1, 2], dtype=torch.int32)
    kv_lens = torch.tensor([5, 9], dtype=torch.int32)
    block_table = torch.zeros(2, 1, dtype=torch.int32)
    key_scale = torch.zeros(4, 8, 1, 4, dtype=torch.uint8)
    out = Indexer._apply_lightning_indexer(
        idx,
        wi=torch.zeros(2, 64),
        qi=torch.zeros(2, 64, 128),
        ki_cache=torch.zeros(4, 8, 1, 128, dtype=torch.float8_e4m3fn),
        q_cumlens=q_cumlens,
        kv_lens=kv_lens,
        block_table=block_table,
        ki_scale_cache=key_scale,
        metadata_caller="decode",
    )

    assert out is expected
    metadata_args, recompute, caller = idx.qli_v2_metadata_op.call_args.args
    assert recompute is False
    assert caller == ("decode", 3)
    assert torch.equal(metadata_args["cu_seqlens_q"], torch.tensor([0, 1, 2]))
    assert metadata_args["seqused_k"] is kv_lens
    assert metadata_args["layout_k"] == "PA_BBND"
    main_args = qli_v2_op.call_args
    assert main_args.args[5:] == (3, 3)
    assert main_args.kwargs["metadata"] is metadata
    assert main_args.kwargs["block_table"] is block_table
    assert main_args.args[2].dtype == torch.float32
    assert main_args.args[3].dtype == torch.float8_e8m0fnu
    assert main_args.args[4].dtype == torch.float8_e8m0fnu
    assert main_args.args[4].shape == (4, 8, 1, 2, 2)


@pytest.mark.unit
@pytest.mark.parametrize("cache_dtype", ("auto", "fp8_ds_mla", "hif8_ds_mla", "li_int8_ds_mla"))
@pytest.mark.parametrize("enabled", (False, True))
def test_qli_v2_activation_matches_cache_layout(cache_dtype, enabled):
    from omni_npu.v1.layers.attention import npu_dsa as mod

    config = SimpleNamespace(
        model_type="openpangu_v2", index_topk=2048, index_n_heads=64,
        index_head_dim=128, qk_rope_head_dim=64, rms_norm_eps=1e-5,
    )
    opt = SimpleNamespace(
        enable_precision_strong_consistency=False, use_rope_fusion_op=False,
        use_noncontiguous_kv=True, use_quant_lightning_indexer_v2=enabled,
    )
    with patch.object(mod, "model_extra_config", SimpleNamespace(operator_opt_config=opt)), \
         patch.object(mod, "ReplicatedFlashCommLinear", MagicMock()), \
         patch.object(mod, "RMSNorm", MagicMock()), \
         patch.object(mod, "on_ascend950", return_value=True), \
         patch.object(mod.Indexer, "_init_qli_v2_ops") as init:
        indexer = mod.Indexer(None, config, 512, 512, None, SimpleNamespace(cache_dtype=cache_dtype))
    expected = enabled and cache_dtype == "hif8_ds_mla"
    assert indexer.use_qli_v2 is expected
    assert init.call_count == int(expected)


@pytest.mark.unit
@pytest.mark.parametrize("cache_dtype", ("hif8_ds_mla", "fp8_ds_mla"))
def test_c8_external_sinks_do_not_change_sparse_indices(cache_dtype):
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    host = SimpleNamespace(param_sink_number=128, noncontiguous_kv=True, cache_dtype=cache_dtype)
    indices = torch.tensor([[[0, 4, -1]]], dtype=torch.int32)
    assert NPUDeepseekSparseAttention._apply_sink_offset(host, indices) is indices


@pytest.mark.unit
@pytest.mark.parametrize(
    "cache_dtype,on_a5,packed,strong,dtype,latent_dim,message",
    [
        ("int8_ds_mla", True, True, False, torch.bfloat16, 512, "INT8 C8 is not supported"),
        ("hif8_ds_mla", False, True, False, torch.bfloat16, 512, "Ascend950/A5"),
        ("fp8_ds_mla", True, False, False, torch.bfloat16, 512, "use_noncontiguous_kv"),
        ("hif8_ds_mla", True, True, True, torch.bfloat16, 512, "precision strong consistency"),
        ("li_int8_ds_mla", True, True, True, torch.bfloat16, 512, "precision strong consistency"),
        ("hif8_ds_mla", True, True, False, torch.float16, 512, "BF16"),
        ("fp8_ds_mla", True, True, False, torch.bfloat16, 256, "kv_lora_rank=512"),
    ],
)
def test_dsa_rejects_unsupported_quantized_cache_before_allocating_weights(
    cache_dtype, on_a5, packed, strong, dtype, latent_dim, message,
):
    from omni_npu.v1.layers.attention import npu_dsa as mod

    opt = SimpleNamespace(
        kv_nz=False, use_noncontiguous_kv=packed, enable_multi_stream=False,
        split_q_up_in_multistream=False, use_mome_inplace_update=False,
        enable_precision_strong_consistency=strong,
    )
    with patch.object(mod, "model_extra_config", SimpleNamespace(dtype=dtype, operator_opt_config=opt)), \
         patch.object(mod, "get_tp_group", return_value=SimpleNamespace(world_size=1)), \
         patch.object(mod, "on_ascend950", return_value=on_a5), \
         pytest.raises(ValueError, match=message):
        mod.NPUDeepseekSparseAttention(
            vllm_config=SimpleNamespace(kv_transfer_config=None), config=SimpleNamespace(),
            hidden_size=512, num_heads=16, qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, q_lora_rank=512, kv_lora_rank=latent_dim,
            cache_config=SimpleNamespace(cache_dtype=cache_dtype), prefix="model.layers.0.self_attn",
        )


def _cpu_shared_op_registry():
    from omni_npu.attention.backends import utils as utils_mod

    saved = dict(utils_mod._CROSS_LAYER_SHARED_OPS)
    utils_mod._CROSS_LAYER_SHARED_OPS.clear()
    original_defaults = utils_mod.CrossLayerSharedOp.__init__.__defaults__
    utils_mod.CrossLayerSharedOp.__init__.__defaults__ = ("cpu",)
    try:
        yield utils_mod
    finally:
        utils_mod.CrossLayerSharedOp.__init__.__defaults__ = original_defaults
        utils_mod._CROSS_LAYER_SHARED_OPS.clear()
        utils_mod._CROSS_LAYER_SHARED_OPS.update(saved)


@pytest.fixture
def fia_metadata_registry():
    yield from _cpu_shared_op_registry()


def _dsa_sharing_fake(**kwargs):
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    defaults = dict(
        layer_idx=0,
        num_hidden_layers=8,
        dsa_layers=(0, 3, 6),
        num_local_heads=64,
        num_attention_heads=64,
        param_sink_number=128,
        tp_size=1,
    )
    defaults.update(kwargs)
    config = SimpleNamespace(
        model_type="openpangu_v2",
        num_hidden_layers=defaults["num_hidden_layers"],
        num_attention_heads=defaults["num_attention_heads"],
        dsa_layers=defaults["dsa_layers"],
    )
    fake = NPUDeepseekSparseAttention.__new__(NPUDeepseekSparseAttention)
    fake.layer_idx = defaults["layer_idx"]
    fake.param_sink_number = defaults["param_sink_number"]
    fake.num_local_heads = defaults["num_local_heads"]
    fake.kv_lora_rank = 512
    fake.qk_rope_head_dim = 64
    fake.tp_size = defaults["tp_size"]
    return fake, config


def _dsa_consistency_config(
    *,
    use_noncontiguous_kv=True,
    enable_precision_strong_consistency=True,
):
    return patch(
        DSA_MODULE + ".model_extra_config",
        SimpleNamespace(
            operator_opt_config=SimpleNamespace(
                use_noncontiguous_kv=use_noncontiguous_kv,
                enable_precision_strong_consistency=enable_precision_strong_consistency,
            ),
        ),
    )


@pytest.mark.unit
def test_dsa_metadata_sharing_disabled_without_gate(fia_metadata_registry):
    fake, config = _dsa_sharing_fake()
    with _dsa_consistency_config(enable_precision_strong_consistency=False):
        fake._init_fia_metadata_sharing(config)
    assert getattr(fake, "_share_fia_metadata", False) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer_idx", "decode_producer", "prefill_producer"),
    [(0, True, True), (1, False, False), (3, False, False)],
)
def test_dsa_metadata_sharing_selects_producers(
    fia_metadata_registry, layer_idx, decode_producer, prefill_producer
):
    fake, config = _dsa_sharing_fake(layer_idx=layer_idx)
    with _dsa_consistency_config():
        fake._init_fia_metadata_sharing(config)
    assert fake._share_fia_metadata is True
    assert fake._produce_sink_fia_metadata["decode"] is decode_producer
    assert fake._produce_sink_fia_metadata["prefill"] is prefill_producer


@pytest.mark.unit
def test_dsa_sink_metadata_reuses_one_buffer(fia_metadata_registry):
    producer, config = _dsa_sharing_fake(layer_idx=0)
    consumer, _ = _dsa_sharing_fake(layer_idx=3)
    expected = torch.arange(1024, dtype=torch.int32)
    direct = MagicMock(return_value=expected.clone())
    with patch.object(
        torch.ops.custom, "_npu_fused_infer_attention_sink_metadata", direct, create=True
    ), _dsa_consistency_config():
        producer._init_fia_metadata_sharing(config)
        consumer._init_fia_metadata_sharing(config)
        produced = producer._get_sink_fia_metadata({"num_heads_q": 64}, "decode")
        reused = consumer._get_sink_fia_metadata({"num_heads_q": 64}, "decode")
    assert produced.data_ptr() == reused.data_ptr()
    assert torch.equal(reused, expected)
    direct.assert_called_once()


@pytest.mark.unit
def test_dsa_rescale_pioneer_reads_shared_metadata(fia_metadata_registry):
    from omni_npu.v1.layers.attention import npu_dsa as dsa_mod
    from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention

    T, N, L, R, S, B = 2, 2, 4, 2, 3, 1
    producer, config = _dsa_sharing_fake(
        layer_idx=0, num_local_heads=N, param_sink_number=S
    )
    consumer, _ = _dsa_sharing_fake(
        layer_idx=3, num_local_heads=N, param_sink_number=S
    )
    for fake in (producer, consumer):
        fake.on_ascend950 = False
        fake.scaling = 0.5
        fake.kv_lora_rank = L
        fake.dummy_value_cache = torch.zeros(1)
        fake.attn = SimpleNamespace(
            impl=SimpleNamespace(
                sink_k_nope=torch.zeros(S, 1, L),
                sink_kv=torch.zeros(S, 1, L + R),
            )
        )
    expected = torch.full((T, N, L), 7.0)
    pioneer = MagicMock(return_value=(torch.zeros(T, N, L), torch.zeros(1, T, N), torch.ones(1, T, N)))
    direct_meta = MagicMock(return_value=torch.arange(1024, dtype=torch.int32))
    sink_v2 = MagicMock(return_value=(torch.ones(T, N, L), None, torch.zeros(T, N, 1), torch.ones(T, N, 1)))
    rescale = MagicMock(return_value=(expected, None, None))
    with patch.object(dsa_mod.ops, "apply_FA_rescale_forward", rescale), \
         patch.object(torch.ops.custom, "npu_ai_infra_sparse_flash_attention_pioneer", pioneer, create=True), \
         patch.object(torch.ops.custom, "_npu_fused_infer_attention_sink_metadata", direct_meta, create=True), \
         patch.object(torch.ops.custom, "npu_fused_infer_attention_sink_v2", sink_v2, create=True), \
         _dsa_consistency_config():
        producer._init_fia_metadata_sharing(config)
        consumer._init_fia_metadata_sharing(config)
        produced = producer._get_sink_fia_metadata({"num_heads_q": N}, "decode")
        out = NPUDeepseekSparseAttention._apply_attention_rescale_pioneer(
            consumer,
            q_nope=torch.zeros(T, N, L),
            q_pe=torch.zeros(T, N, R),
            q_cumlens=torch.tensor([T], dtype=torch.int32),
            kv_lens=torch.tensor([T], dtype=torch.int32),
            topk_idx=torch.zeros(T, 1, 2, dtype=torch.int32),
            block_table=torch.zeros(B, 4, dtype=torch.int32),
            kv_cache=(torch.zeros(4, 8, L),),
            metadata_caller="decode",
        )
    assert out is expected
    direct_meta.assert_called_once()
    assert sink_v2.call_args.kwargs["meta_data"].data_ptr() == produced.data_ptr()
