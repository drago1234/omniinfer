# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Engram residual and norm placement in mHC stream layouts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from omni_npu.layers import engram as engram_layer
from omni_npu.v1.models.pangu import pangu_v2_moe as pangu_moe
from omni_npu.v1.models.pangu import pangu_ultra_moe as ultra_moe

pytestmark = pytest.mark.unit

MODEL_VARIANTS = [
    pytest.param(pangu_moe.OpenPanguV2Model, id="low_latency"),
    pytest.param(ultra_moe.OpenPanguModel, id="high_throughout"),
]


@pytest.mark.parametrize("model_cls", MODEL_VARIANTS)
@pytest.mark.parametrize("ubatched", [False, True])
def test_engram_metadata_reads_current_context_and_ubatch(monkeypatch, ubatched, model_cls):
    from vllm.v1.worker import ubatching

    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.engram_hash = SimpleNamespace(cache_prefix="engram.test")
    first, second = object(), object()
    context = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(engram_layer, "get_forward_context", lambda: context)
    assert model._get_engram_metadata() is None
    if ubatched:
        context.attn_metadata = [{"engram.test": first}, {"engram.test": second}]
        for index, expected in enumerate((first, second)):
            monkeypatch.setattr(ubatching, "dbo_current_ubatch_id", lambda: index)
            assert model._get_engram_metadata() is expected
    else:
        for expected in (first, second):
            context.attn_metadata = {"engram.test": expected}
            assert model._get_engram_metadata() is expected
    context.attn_metadata = {}
    assert model._get_engram_metadata() is None


def _layer(num_streams=2, *, layer_cls=pangu_moe.OpenPanguV2DecoderLayer):
    layer = layer_cls.__new__(layer_cls)
    nn.Module.__init__(layer)
    layer.hidden_size = 4
    layer.layer_idx = 0
    layer.mhc_num_stream = num_streams
    layer.use_mhc = True
    layer.use_mhc_fusion_op = True
    layer.use_mega_mhc_fusion_op = True
    layer.side_stream = None

    class FakeEngram(nn.Module):
        layer_hash_index = 0

        def forward(self, hidden, hashes):
            # Test the model residual/norm placement independently of NPU lookup.
            return torch.ones_like(hidden) * hashes[:, 0].to(hidden.dtype).reshape(
                hidden.shape[0], *([1] * (hidden.ndim - 1))
            )

    layer.engram = FakeEngram()

    class ReferenceNorm(nn.Module):
        eps = 1e-5

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def forward(self, value):
            return _reference_post_norm(value, self)

    layer.post_self_attn_engram_layernorm = ReferenceNorm()
    return layer


def _reference_post_norm(value, norm):
    normalized = value.float()
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + norm.eps)
    return normalized.to(value.dtype) * norm.weight


@pytest.mark.parametrize("flattened", [False, True])
@pytest.mark.parametrize("layer_cls", [
    pytest.param(pangu_moe.OpenPanguV2DecoderLayer, id="low_latency"),
    pytest.param(ultra_moe.OpenPanguDecoderLayer, id="high_throughout"),
])
def test_engram_post_norm_preserves_stream_layout_and_residual(flattened, layer_cls):
    dtype = torch.bfloat16
    torch.manual_seed(31)
    layer = _layer(layer_cls=layer_cls).to(dtype)
    nn.init.normal_(layer.post_self_attn_engram_layernorm.weight)
    hidden = torch.randn(3, 2, 4, dtype=dtype)
    memory = torch.arange(12, dtype=torch.int32).reshape(3, 1, 4)
    expected_update = _reference_post_norm(
        layer.engram(hidden, memory[:, 0]),
        layer.post_self_attn_engram_layernorm,
    )
    if flattened:
        hidden = hidden.flatten(1)

    actual = layer._apply_engram(hidden, memory)

    torch.testing.assert_close(
        actual,
        hidden + expected_update.reshape_as(hidden),
        rtol=0,
        atol=0,
    )
    assert actual.shape == hidden.shape
    assert layer._apply_engram(hidden, None) is hidden


@pytest.mark.parametrize("skip_engram", [False, True])
def test_next_layer_engram_runs_between_mhc_post_and_pre(skip_engram):
    """The mHC fusion shortcut must not skip next-layer Engram or its norm."""
    dtype = torch.bfloat16
    torch.manual_seed(37)
    streams = 2
    layer = _layer(streams).to(dtype)
    nn.init.normal_(layer.post_self_attn_engram_layernorm.weight)
    hidden = torch.randn(3, 4, dtype=dtype)
    post_hidden = torch.randn(3, 8, dtype=dtype)
    memory = torch.arange(12, dtype=torch.int32).reshape(3, 1, 4)
    post_mhc = SimpleNamespace(mhc_post=MagicMock(return_value=post_hidden))
    pre_mhc = SimpleNamespace(
        mhc_pre=MagicMock(side_effect=lambda x: (x.reshape(3, streams, 4).sum(1), None, None)),
        mhc_sinkhorn=MagicMock(return_value=None),
    )
    expected_update = _reference_post_norm(
        layer.engram(post_hidden, memory[:, 0]).reshape(3, streams, 4),
        layer.post_self_attn_engram_layernorm,
    )
    expected_residual = post_hidden if skip_engram else post_hidden + expected_update.flatten(1)

    actual, residual, _, _, _ = layer.mhc_sandwich_norm_post_pre(
        hidden,
        torch.zeros_like(post_hidden),
        None,
        None,
        nn.Identity(),
        post_mhc,
        None,
        pre_mhc,
        nn.Identity(),
        pre_engram_module=layer.engram,
        pre_engram_norm=layer.post_self_attn_engram_layernorm,
        pre_engram_hash_ids=None if skip_engram else memory,
    )

    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(pre_mhc.mhc_pre.call_args.args[0], expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(
        actual,
        expected_residual.reshape(3, streams, 4).sum(1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("model_module,fused", [
    pytest.param(pangu_moe, False, id="low_latency"),
    pytest.param(ultra_moe, False, id="high_throughout"),
    pytest.param(ultra_moe, True, id="high_throughout_fused_mhc"),
])
def test_serial_lookup_prepares_all_layers_before_consumption(monkeypatch, model_module, fused):
    events = []
    low_latency = model_module is pangu_moe
    model_cls = pangu_moe.OpenPanguV2Model if low_latency else ultra_moe.OpenPanguModel
    hashes = torch.arange(6, dtype=torch.int32).reshape(3, 2, 1)

    class Embed(nn.Module):
        def forward(self, ids, **kwargs):
            return torch.zeros(ids.numel(), 2)

    class Layer(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.engram = None
            self.self_attn = SimpleNamespace(rotary_emb=SimpleNamespace(get_cos_sin=lambda pos: (pos, pos)))
            self.attn_mhc_module = SimpleNamespace(can_use_fusion=lambda hidden: True)

        def mhc_head(self, hidden, hashes):
            events.append(("head", self.index))
            return (hidden, None, None, None, None) if low_latency else (hidden, None, None, None)

        def forward(self, hidden, *args):
            assert args[-1] is hashes
            events.append(("layer", self.index))
            return (hidden, None, None, None, None) if low_latency else (hidden, None, None)

        def forward_mhc_fused(self, hidden, residual, h_post, h_res, cos, sin, **kwargs):
            assert kwargs["h_res_from_fused_split"] == (self.index == 3)
            events.append(("layer", self.index))
            return hidden, residual, h_post, h_res, None

    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.engram_hash = SimpleNamespace(has_bound_cache=lambda: True)
    model.engram_layout = SimpleNamespace(layer_ids=(1, 2), n_hash_cols=1)
    model.start_layer, model.end_layer = 0, 4
    model._prepare_key = id(model)
    model.layers = nn.ModuleList([Layer(i) for i in range(4)])
    model.embed_tokens = Embed()
    model.use_mhc = True
    model.use_mhc_fusion_op = fused
    model.num_stream = 2
    model.config = SimpleNamespace(index_topk=0)
    model.merge_mhc_module = SimpleNamespace(mhc_pre=lambda hidden: (hidden, None, None))
    model.norm = nn.Identity()
    model.hidden_size = model.mhc_num_stream = 2
    model.need_tp_padding = False
    model.cos_cached = model.sin_cached = None
    engrams = [nn.Module(), nn.Module()]
    for layer, index in zip(engrams, [1, 2]):
        layer.layer_idx = index
        layer.layer_hash_index = index - 1
        layer._lookup_key = id(layer)
    for layer, engram in zip(model.layers[1:], engrams):
        layer.engram = engram
    model._get_engram_metadata = lambda: True
    owners = {layer._lookup_key: layer for layer in engrams}

    def lookup(layer_hashes, owner):
        layer = owners[owner]
        torch.testing.assert_close(layer_hashes, hashes[:, layer.layer_hash_index])
        events.append(("prepare", layer.layer_idx))
        return torch.zeros(3, 1, 1)

    monkeypatch.setattr(pangu_moe, "high_throughout", lambda: False)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(
        model_module, "model_extra_config", SimpleNamespace(parall_config=SimpleNamespace(ena_seq_parallel=False))
    )
    monkeypatch.setattr(torch.ops.omni_npu, "engram_hash", lambda *args: hashes)
    monkeypatch.setattr(torch.ops.omni_npu, "engram_lookup_rows", lookup)

    with pytest.raises(ValueError, match="PanGu Engram requires raw token IDs"):
        model_cls.forward(model, None, torch.arange(3), None, inputs_embeds=torch.zeros(3, 2))
    assert events == []
    model_cls.forward(model, torch.arange(3), torch.arange(3), None)

    expected = [("prepare", 1), ("prepare", 2)]
    for index in range(4):
        if (fused and index < 3) or (low_latency and index == 0):
            expected.append(("head", index))
        expected.append(("layer", index))
    assert events == expected


@pytest.mark.parametrize("model_cls,gqa", [
    pytest.param(pangu_moe.OpenPanguV2Model, False, id="low_latency"),
    pytest.param(ultra_moe.OpenPanguModel, False, id="high_throughout"),
    pytest.param(ultra_moe.OpenPanguModel, True, id="high_throughout_gqa"),
])
def test_engram_hash_prepares_history_once(monkeypatch, model_cls, gqa):
    from omni_npu.layers import engram

    attention = nn.Module()
    attention.layer_name = "engram.test"
    attention.kv_cache = torch.empty(4, 4, 1)
    layout = SimpleNamespace(
        config=SimpleNamespace(seed=42, max_ngram_size=3, pad_token_id=0),
        layer_ids=(2,),
        head_sizes=(11, 13),
        primes=[[11, 13, 17, 19]],
        offsets=torch.tensor([[0, 11, 24, 41]]),
        max_ngram_size=3,
    )
    state = engram.NgramHashState(layout, torch.device("cpu"), attention)
    state.block_size = 4
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.engram_hash = state
    model.engram_layout = state.layout
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, -1, -1]),
        decode=None,
        prefill=SimpleNamespace(block_table=torch.tensor([[0]], dtype=torch.int32)),
    )
    if gqa:
        metadata.block_tables = metadata.prefill.block_table
        del metadata.decode
        del metadata.prefill
    monkeypatch.setattr(
        engram_layer, "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={attention.layer_name: metadata}),
    )
    prepare = MagicMock(wraps=state.ensure_cache)
    monkeypatch.setattr(state, "ensure_cache", prepare)
    expected = torch.ones(3, 1, 4, dtype=torch.int32)
    expected_dead = [False, True, True]

    def hash_forward(*args, **kwargs):
        assert state._cache.numel() == 16
        assert state.block_size == 4
        assert args[3].tolist() == expected_dead
        assert kwargs["block_table"].tolist() == [[0]]
        return expected

    monkeypatch.setattr(state, "forward", hash_forward)
    actual = model._compute_engram_hashes(torch.tensor([17, 23, 0]), torch.arange(3), None)
    assert actual is expected
    assert (actual[:, 0, 0] < 0).tolist() == expected_dead
    prepare.assert_called_once_with()


@pytest.mark.parametrize("model_cls", MODEL_VARIANTS)
def test_unbound_history_is_skipped_and_custom_hash_contract(monkeypatch, model_cls):
    from omni_npu.layers import engram
    from tests.unit.layers.test_engram import _config

    attention = nn.Module()
    attention.kv_cache = torch.empty(0)
    attention.layer_name = "history"
    state = engram.NgramHashState(engram.EngramLayout(_config()), torch.device("cpu"), attention)
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.engram_hash = state
    monkeypatch.setattr(engram_layer, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    ids = torch.tensor([1, 2])
    assert model.prepare_model_embeddings(ids, ids, None) is None
    with pytest.raises(RuntimeError, match="not been bound"):
        state.compute_hashes(None, ids, ids, None)
    attention.kv_cache = torch.ones(1, 4, 1)
    assert model.prepare_model_embeddings(ids, ids, None) is None
    with pytest.raises(RuntimeError, match="block size"):
        state.ensure_cache()
    expected = torch.ones(2, 1, 4, dtype=torch.int32)
    model._compute_engram_hashes = MagicMock(return_value=expected)
    key = id(model)
    engram._engram_model_owners[key] = model
    assert torch.equal(torch.ops.omni_npu.engram_hash(ids, ids, None, key, 1, 4), expected)
    model._compute_engram_hashes.assert_called_once_with(ids, ids, None)
    assert engram._engram_hash_fake(ids, ids, None, key, 1, 4).shape == expected.shape


def test_low_latency_model_binds_local_engram_and_loads_checkpoint_alias(monkeypatch):
    from tests.unit.models.test_pangu_ultra_moe import TestOpenPanguModelMHCWiring, _mhc_vllm_config

    cfg = TestOpenPanguModelMHCWiring._config()
    cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_vocab_size = [1], 2, 11
    cfg.engram_n_head_per_ngram, cfg.engram_n_embed_per_ngram, cfg.engram_pad_id = 2, 4, 0
    cfg.swa_layers, cfg.dsa_layers, cfg.param_sink_number, cfg.n_routed_experts = None, [], 1, 1
    layers = nn.ModuleList([TestOpenPanguModelMHCWiring._layer(i) for i in range(2)])
    for index, layer in enumerate(layers):
        attention = nn.Module()
        attention.layer_name, attention.kv_cache = f"layer{index}", torch.empty(0)
        layer.self_attn = SimpleNamespace(attn=attention, rotary_emb=SimpleNamespace(
            cos_cached=torch.zeros(8, 2), sin_cached=torch.zeros(8, 2)
        ))
        layer.post_self_attn_engram_layernorm = None
    memory = nn.Module()
    memory.use_sequence_parallel = False
    memory.wkv = nn.Linear(4, 48, bias=False)
    memory.embed_tokens = nn.Embedding(24, 2)
    layers[1].engram = memory
    for name in ("NPUmHC", "RMSNorm", "NPUVocabParallelEmbedding"):
        monkeypatch.setattr(pangu_moe, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(pangu_moe, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(pangu_moe, "get_engram_shared_group", lambda: object())
    monkeypatch.setattr(pangu_moe, "make_layers", lambda *args, **kwargs: (0, 2, layers))
    monkeypatch.setattr(pangu_moe.model_extra_config.operator_opt_config, "enable_multi_stream", False)
    vllm_config = _mhc_vllm_config(cfg)
    vllm_config.model_config.dtype = torch.float32
    model = pangu_moe.OpenPanguV2Model(vllm_config=vllm_config)
    assert model.engram_hash.cache_prefix == "layer0"
    assert layers[0]._tail_refs[3] is memory and layers[1]._tail_refs[3] is None
    owner = pangu_moe.OpenPanguV2ForCausalLM.__new__(pangu_moe.OpenPanguV2ForCausalLM)
    nn.Module.__init__(owner)
    owner.model, owner.config, owner.num_redundant_experts = model, cfg, 0
    assert owner.token_lookback_depth == 1
    monkeypatch.setattr(pangu_moe, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: [])
    monkeypatch.setattr(pangu_moe, "is_pp_missing_parameter", lambda *args: False)
    monkeypatch.setattr(pangu_moe, "high_throughout", lambda: False)
    weight = torch.arange(48, dtype=torch.float32).reshape(24, 2)
    loaded = owner.load_weights([("model.layers.1.engram.embed.weight", weight)])
    assert loaded == {"model.layers.1.engram.embed_tokens.weight"}
    torch.testing.assert_close(memory.embed_tokens.weight, weight)
    model.engram_hash = None
    assert owner.token_lookback_depth == 0
