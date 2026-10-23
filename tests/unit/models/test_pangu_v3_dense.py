# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Dense Pangu-V3 wiring, GQA numerics and checkpoint contracts on CPU."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import CompilationMode

from omni_npu.v1.models.pangu import pangu_v3_dense as dense
from omni_npu.v1.models.pangu import pangu_v3_dense_mtp as mtp

pytestmark = pytest.mark.unit


class Projection(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class Attention(nn.Module):
    def __init__(self, heads, dim, scale, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.kv_sharing_target_layer_name = kwargs['kv_sharing_target_layer_name']
        self.impl = NS()

    def forward(self, q, k, v):
        self.inputs = q, k, v
        return torch.ones_like(q).flatten(1)


@pytest.fixture
def cpu_model(monkeypatch, request):
    rng_state = torch.get_rng_state()
    request.addfinalizer(lambda: torch.set_rng_state(rng_state))
    torch.random.default_generator.manual_seed(17)
    group = NS(world_size=1, rank_in_group=0, all_gather=lambda x, dim: x)
    pp = NS(is_first_rank=True, is_last_rank=True)
    cfg = NS(hidden_size=16, intermediate_size=32, hidden_act='silu',
             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
             qk_rope_dim=4, rope_theta=10000., max_position_embeddings=64,
             rms_norm_eps=1e-5, num_hidden_layers=4, num_nextn_predict_layers=1,
             vocab_size=32, use_mhc=True, mhc_num_stream=2, sandwich_norm=True,
             block_post_layernorm_idx=[0], swa_layers=[0, 2, 4],
             sliding_window_list=[4, 4, 4, 4, 8], cla_explicit_mapping=[[3, 1]],
             use_gpt_oss_sink=True, tie_word_embeddings=False)
    vcfg = NS(model_config=NS(hf_config=cfg), quant_config=None, cache_config=None,
              compilation_config=NS(mode=CompilationMode.NONE),
              speculative_config=NS(draft_model_config=NS(hf_config=cfg)))

    class QKV(Projection):
        def __init__(self, hidden, dim, heads, kv_heads, **kwargs):
            self.num_heads, self.num_kv_heads = heads // group.world_size, kv_heads // group.world_size
            super().__init__(hidden, (self.num_heads + 2 * self.num_kv_heads) * dim, bias=False)
            self.weight.weight_loader = Mock()

    class MLP(nn.Module):
        def __init__(self, hidden, intermediate, *args, **kwargs):
            super().__init__()
            self.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.gate_up_proj.weight.weight_loader = Mock()

    def sink(host):
        param = nn.Parameter(torch.zeros(host.num_local_heads), requires_grad=False)
        param.weight_loader = host._load_gpt_oss_sink_weight
        return param

    monkeypatch.setattr(dense, 'get_tp_group', lambda: group)
    monkeypatch.setattr(dense, 'get_pp_group', lambda: pp)
    monkeypatch.setattr(dense, 'QKVParallelLinear', QKV)
    monkeypatch.setattr(dense, 'RowParallelLinear', lambda a, b, **kw: Projection(a, b, bias=False))
    monkeypatch.setattr(dense, 'RMSNorm', nn.RMSNorm)
    monkeypatch.setattr(dense, 'Attention', Attention)
    monkeypatch.setattr(dense, 'OpenPanguMLP', MLP)
    monkeypatch.setattr(dense, 'NPUmHCRL', lambda **kw: nn.Identity())
    monkeypatch.setattr(dense.PanGuEngramConfig, 'from_vllm_config', lambda _: None)
    monkeypatch.setattr(dense.NPUPanguV3GQAAttention, '_create_gpt_oss_sink', sink)
    monkeypatch.setattr(dense, 'get_rope', lambda dim, **kw: NS(
        get_cos_sin=lambda positions: (torch.ones(len(positions), 4), torch.zeros(len(positions), 4))))
    monkeypatch.setattr(dense.torch_npu, 'npu_rotary_mul', lambda x, c, s:
                        x * c + torch.cat((-x[..., 2:], x[..., :2]), dim=-1) * s)
    monkeypatch.setattr(dense, 'NPUVocabParallelEmbedding', lambda vocab, hidden, **kw: nn.Embedding(vocab, hidden))
    monkeypatch.setattr(dense, 'NPUParallelLMHead', lambda vocab, hidden, **kw: nn.Linear(hidden, vocab, bias=False))
    monkeypatch.setattr(dense, 'make_layers', lambda n, fn, prefix:
                        (0, n, nn.ModuleList([fn(f'{prefix}.{i}') for i in range(n)])))
    monkeypatch.setattr(dense.PanguV3DenseModel, '_init_engram_hash', lambda self: None)
    monkeypatch.setattr(dense, 'run_post_weight_load', Mock())
    monkeypatch.setattr(dense.envs, 'OMNI_PD_ROLE', 'decode')
    monkeypatch.setattr(dense.envs, 'OMNI_REUSE_PREFILLED_TOKENS', False)
    monkeypatch.setattr(dense.model_extra_config.parall_config, 'ena_seq_parallel', False)
    monkeypatch.setattr(dense.model_extra_config.operator_opt_config, 'use_mhc_fusion_op', False)
    monkeypatch.setattr(mtp, 'RMSNorm', nn.RMSNorm)
    monkeypatch.setattr(mtp, 'NPUVocabParallelEmbedding', lambda v, h: nn.Embedding(v, h))
    monkeypatch.setattr(mtp, 'SharedHead', lambda config, **kw: nn.RMSNorm(config.hidden_size, config.rms_norm_eps))
    return cfg, vcfg, group, pp


def test_gqa_sharing_rope_and_numerical_forward(cpu_model, monkeypatch):
    cfg, _, group, _ = cpu_model
    layers = [dense.NPUPanguV3GQAAttention(cfg, prefix=f'model.layers.{i}.self_attn') for i in range(5)]
    assert [i for i, l in enumerate(layers) if l.attn.kwargs['should_recompute_attn_meta']] == [0, 1, 4]
    assert layers[0].attn.sliding_window == 5 and layers[0].attn.impl.sliding_window == 4
    assert layers[1].attn.sliding_window is None
    assert layers[3].attn.kv_sharing_target_layer_name == 'model.layers.1.self_attn.attn'
    layer = layers[0]
    x, cos, sin = torch.randn(3, 16), torch.ones(3, 4), torch.zeros(3, 4)
    actual, topk = layer(x, cos, sin)
    q, k, v = F.linear(x, layer.qkv_proj.weight).split([32, 16, 16], dim=-1)
    q = F.rms_norm(q.reshape(3, 4, 8), (8,), eps=cfg.rms_norm_eps)
    k = F.rms_norm(k.reshape(3, 2, 8), (8,), eps=cfg.rms_norm_eps)
    for actual_input, expected in zip(layer.attn.inputs, (q, k, v)):
        torch.testing.assert_close(actual_input, expected)
    gate = torch.sigmoid(F.linear(x, layer.attention_gate.weight))
    torch.testing.assert_close(actual, F.linear(gate, layer.o_proj.weight))
    assert topk is None
    rotated = layer._apply_rope(torch.arange(32.).reshape(1, 4, 8), torch.zeros(1, 4), torch.ones(1, 4))
    torch.testing.assert_close(rotated[0, 0], torch.tensor([-2., -3., 0., 1., 4., 5., 6., 7.]))
    assert layers[3](x, cos, sin)[0].shape == (3, 16)
    assert layers[3].attn.inputs[1:] == (None, None)
    layer.ena_sp = True
    chunk = Mock(side_effect=lambda x: x)
    monkeypatch.setattr(dense, 'sequence_parallel_chunk', chunk)
    assert layer(x, cos, sin)[0].shape == (3, 16)
    chunk.assert_called_once()
    group.world_size, group.rank_in_group = 2, 1
    sharded = dense.NPUPanguV3GQAAttention(cfg, prefix='model.layers.0.self_attn')
    sharded._load_gpt_oss_sink_weight(sharded.gpt_oss_sink, torch.arange(4.))
    sharded._load_attention_gate_weight(sharded.attention_gate.weight, torch.arange(512.).reshape(32, 16))
    torch.testing.assert_close(sharded.gpt_oss_sink, torch.tensor([2., 3.]))
    torch.testing.assert_close(sharded.attention_gate.weight, torch.arange(512.).reshape(32, 16)[16:])


@pytest.mark.parametrize('draft', [False, True])
def test_main_and_mtp_checkpoint_loading(cpu_model, monkeypatch, draft):
    cfg, vcfg, _, _ = cpu_model
    model = mtp.PanguV3DenseMTP(vllm_config=vcfg) if draft else dense.PanguV3DenseForCausalLM(vllm_config=vcfg)
    idx = 4 if draft else 0
    prefix = f'model.layers.{idx}.'
    tensors = [(prefix + f'self_attn.{name}_proj.weight', torch.full((size, 16), float(i + 1)))
               for i, (name, size) in enumerate((('q', 32), ('k', 16), ('v', 16)))]
    tensors += [(prefix + f'mlp.{name}_proj.weight', torch.full((32, 16), float(i + 4)))
                for i, name in enumerate(('gate', 'up'))]
    tensors += [(prefix + 'input_layernorm.weight', torch.arange(16.))]
    tensors += [('model.layers.0.input_layernorm.weight' if draft else 'model.layers.4.hnorm.weight', torch.zeros(16))]
    if not draft:
        tensors += [('model.layers.0.rotary_emb.inv_freq', torch.zeros(1))]
        monkeypatch.setattr(dense, 'is_pp_missing_parameter', lambda n, m: n == 'missing.weight')
        tensors += [('missing.weight', torch.zeros(1))]
        cfg.tie_word_embeddings = True
        tensors += [('lm_head.weight', torch.zeros(32, 16))]
    post = Mock()
    monkeypatch.setattr(model, 'post_weight_load', post)
    loaded = model.load_weights(tensors)
    block = model.model.layers[str(idx)].mtp_block if draft else model.model.layers[idx]
    for param, weights, shards in (
        (block.self_attn.qkv_proj.weight, tensors[:3], ['q', 'k', 'v']),
        (block.mlp.gate_up_proj.weight, tensors[3:5], [0, 1]),
    ):
        calls = param.weight_loader.call_args_list
        assert [call.args[2] for call in calls] == shards
        assert all(call.args[0] is param and call.args[1] is weight for call, (_, weight) in zip(calls, weights))
    if draft:
        assert not block.use_mhc
    else:
        assert len(model.model.layers) == 4 and model.model.use_mhc
        assert not model.model.use_mhc_fusion_op
    torch.testing.assert_close(block.input_layernorm.weight, torch.arange(16.))
    assert len(loaded) == 3
    post.assert_called_once_with()
    if not draft:
        model.post_weight_load = lambda: dense.PanguV3DenseForCausalLM.post_weight_load(model)
        model.post_weight_load()
        dense.run_post_weight_load.assert_called_once_with(model)
