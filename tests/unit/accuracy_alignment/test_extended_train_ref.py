# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from omni_npu.diagnostics.accuracy_alignment.attention import AttentionTrainRefHooks
from omni_npu.diagnostics.accuracy_alignment.config import TrainRefConfig
from omni_npu.diagnostics.accuracy_alignment.dump_io import (
    LayerTrainingInputs,
    TrainRefDataError,
    dump_output,
    load_layer_training_inputs,
    load_reference_tensor,
    load_training_input,
    prepare_layer_mock_input,
    prepare_mock_input,
    primary_tensor,
)
from omni_npu.diagnostics.accuracy_alignment.hooks import (
    MethodHandle,
    ModuleTrainRefHooks,
    ModuleTrainRefTarget,
    gather_tp_output,
)
from omni_npu.diagnostics.accuracy_alignment.layer import collect_layer_targets
from omni_npu.diagnostics.accuracy_alignment.model_level import (
    ModelLevelTrainRefHooks,
)
from omni_npu.diagnostics.accuracy_alignment.moe import MoeTrainRefHooks
from torch import nn


def _config(tmp_path, *, custom=False, mock_shape=2):
    train_dir = tmp_path / "train"
    infer_dir = tmp_path / "infer"
    train_dir.mkdir()
    infer_dir.mkdir()
    custom_dir = None
    if custom:
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
    return TrainRefConfig(
        train_dump_dir=train_dir,
        infer_dump_dir=infer_dir,
        layer_id=0,
        role="P",
        mock_shape=mock_shape,
        mhc_custom_dump_dir=custom_dir,
    )


def _save_megatron(path, input_tensor, output_tensor=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tensors": {
                "input.0": input_tensor,
                "output.0": (input_tensor if output_tensor is None else output_tensor),
            }
        },
        path,
    )


def test_method_handle_restores_class_and_instance_methods():
    class Subject:
        def value(self):
            return "class"

    subject = Subject()
    handle = MethodHandle(subject, "value", lambda _self: "wrapped")
    assert subject.value() == "wrapped"
    handle.remove()
    handle.remove()
    assert subject.value() == "class"

    subject.value = lambda: "instance"
    handle = MethodHandle(subject, "value", lambda _self: "wrapped")
    handle.remove()
    assert subject.value() == "instance"


def test_gather_tp_output_keeps_local_and_replicated_values():
    value = torch.ones(2, 3)
    assert gather_tp_output(SimpleNamespace(tp_size=1), value) is value
    assert gather_tp_output(SimpleNamespace(tp_size=2), value.sum()) == value.sum()

    replicated_type = type("ReplicatedLinear", (), {"tp_size": 2})
    assert gather_tp_output(replicated_type(), value) is value
    assert (
        gather_tp_output(SimpleNamespace(tp_size=2, reduce_results=True), value)
        is value
    )
    assert (
        gather_tp_output(SimpleNamespace(tp_size=2, gather_output=True), value) is value
    )


def test_collect_layer_targets_skips_missing_modules_and_adds_mhc_alias(tmp_path):
    class PPMissingLayer(nn.Module):
        pass

    layer = SimpleNamespace(
        input_layernorm=nn.Identity(),
        self_attn=nn.Identity(),
        post_attention_layernorm=PPMissingLayer(),
        pre_mlp_layernorm=None,
        mlp=nn.Identity(),
        post_mlp_layernorm=nn.Identity(),
        block_post_layernorm=nn.Identity(),
        use_mhc=True,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])
    )

    targets = collect_layer_targets(model, _config(tmp_path))

    names = {target.module_name for target in targets}
    assert "decoder.layers.0.input_layernorm" in names
    assert "decoder.layers.0.post_self_attn_layernorm" not in names
    assert "pangu_transformer_mlp_1" in names
    mhc_target = next(
        target for target in targets if target.module_name == "pangu_transformer_mlp_1"
    )
    assert mhc_target.inject_input is False


def test_collect_layer_targets_respects_pipeline_ownership(tmp_path):
    config = replace(_config(tmp_path), layer_id=1)
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=[nn.Identity()])
    )
    assert collect_layer_targets(model, config) == []


def test_dump_loaders_accept_supported_payloads_and_reject_bad_data(tmp_path):
    output_path = tmp_path / "output.pt"
    tensor = torch.ones(2, 3)
    torch.save({"tensors": {"output.0": tensor}}, output_path)
    assert torch.equal(load_reference_tensor(output_path), tensor)

    raw_path = tmp_path / "raw.pt"
    torch.save(tensor, raw_path)
    assert torch.equal(load_reference_tensor(raw_path), tensor)

    invalid_path = tmp_path / "invalid.pt"
    torch.save({"unexpected": object()}, invalid_path)
    with pytest.raises(TrainRefDataError, match="must be a Tensor"):
        load_reference_tensor(invalid_path)
    with pytest.raises(TrainRefDataError, match="'tensors' dictionary"):
        load_training_input(invalid_path)

    scalar_path = tmp_path / "scalar.pt"
    torch.save({"tensors": {"input.0": torch.tensor(1.0)}}, scalar_path)
    with pytest.raises(TrainRefDataError, match="must not be scalar"):
        load_training_input(scalar_path)

    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not a torch payload")
    with pytest.raises(TrainRefDataError, match="failed to load"):
        load_reference_tensor(corrupt_path)


def test_layer_training_loader_reads_available_tp_shards(tmp_path):
    config = replace(_config(tmp_path), train_tp_size=3)
    for rank in (0, 1):
        _save_megatron(
            config.layer_training_dump_path("mlp.linear_fc2", rank),
            torch.full((2, 2), float(rank)),
        )

    loaded = load_layer_training_inputs(config, "mlp.linear_fc2")
    assert sorted(loaded.by_rank) == [0, 1]


def test_prepare_mock_input_validates_scalar_length_and_shape(tmp_path):
    config = _config(tmp_path, mock_shape=2)
    with pytest.raises(TrainRefDataError, match="runtime input must not be scalar"):
        prepare_mock_input(torch.ones(2, 3), torch.tensor(1.0), config, "embedding")
    with pytest.raises(TrainRefDataError, match="training sequence is too short"):
        prepare_mock_input(torch.ones(1, 3), torch.ones(2, 3), config, "embedding")
    with pytest.raises(TrainRefDataError, match="mock shape"):
        prepare_mock_input(torch.ones(2, 4), torch.ones(2, 3), config, "embedding")

    decode = replace(config, role="D")
    result = prepare_mock_input(
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.zeros(1, 3),
        decode,
        "embedding",
    )
    assert torch.equal(result, torch.tensor([[3.0, 4.0, 5.0]]))


def test_prepare_layer_mock_input_combines_or_selects_tp_shards(tmp_path):
    config = _config(tmp_path)
    combined = prepare_layer_mock_input(
        LayerTrainingInputs(
            "mlp.linear_fc1",
            {0: torch.ones(2, 2), 1: torch.full((2, 2), 2.0)},
        ),
        torch.zeros(2, 4),
        config,
        tp_rank=0,
    )
    assert combined.shape == (2, 4)

    selected = prepare_layer_mock_input(
        LayerTrainingInputs(
            "mlp.linear_fc1",
            {
                0: torch.arange(8, dtype=torch.float32).reshape(2, 4),
                1: torch.arange(8, dtype=torch.float32).reshape(2, 4) + 10,
            },
        ),
        torch.zeros(2, 2),
        config,
        tp_rank=1,
    )
    expected = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, 2:]
    assert torch.equal(selected, expected)

    with pytest.raises(TrainRefDataError, match="no training rank-0"):
        prepare_layer_mock_input(
            LayerTrainingInputs("mlp.linear_fc1", {1: torch.ones(2, 2)}),
            torch.zeros(2, 2),
            config,
            tp_rank=0,
        )
    with pytest.raises(TrainRefDataError, match="mock shape"):
        prepare_layer_mock_input(
            LayerTrainingInputs("mlp.linear_fc1", {0: torch.ones(2, 2)}),
            torch.zeros(2, 3),
            config,
            tp_rank=0,
        )


def test_primary_tensor_recurses_through_mapping_and_sequence():
    tensor = torch.ones(1, 2)
    assert primary_tensor({"first": object(), "second": (None, tensor)}) is tensor
    with pytest.raises(TrainRefDataError, match="does not contain"):
        primary_tensor({"bad": [None, "text"]})


def test_prepare_layer_mock_input_converts_legacy_ncl_layout(tmp_path):
    cfg = _config(tmp_path, mock_shape=3)
    training = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    runtime = torch.zeros(3, 2)

    result = prepare_layer_mock_input(
        LayerTrainingInputs("self_attention.qa_conv", {0: training}),
        runtime,
        cfg,
        tp_rank=0,
    )

    assert torch.equal(result, training.squeeze(0).transpose(0, 1)[:3])


def test_prepare_layer_mock_input_slices_all_mtp_decode_tokens(tmp_path):
    cfg = replace(_config(tmp_path, mock_shape=3), role="D")
    training = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    runtime = torch.zeros(2, 2)

    result = prepare_layer_mock_input(
        LayerTrainingInputs("self_attention.q_layernorm", {0: training}),
        runtime,
        cfg,
        tp_rank=0,
    )

    assert torch.equal(result, training[2:4])


def test_dump_output_keeps_all_mtp_decode_tokens(tmp_path):
    cfg = replace(_config(tmp_path, mock_shape=3), role="D")
    output = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    path = dump_output(output, "self_attention.q_layernorm", cfg, rank=0)

    assert path is not None
    assert path.parent.name == "decode"
    assert torch.equal(torch.load(path, weights_only=False), output)


class _MappingProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, payload):
        self.seen = payload
        return payload.get("hidden_states_bf16")


def test_module_hook_replaces_only_main_tensor_in_fused_mhc_mapping(tmp_path):
    cfg = _config(tmp_path)
    golden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _save_megatron(cfg.layer_training_dump_path("mlp"), golden)
    module = _MappingProbe()
    hooks = ModuleTrainRefHooks(cfg, rank=0)
    hooks.install(
        [
            ModuleTrainRefTarget(
                module_name="decoder.layers.0.mlp",
                training_name="mlp",
                module=module,
            )
        ]
    )
    fp32_copy = torch.full((2, 2), 9.0)

    result = module(
        {
            "hidden_states_bf16": torch.zeros(2, 2),
            "hidden_states_fp32": fp32_copy,
        }
    )

    assert torch.equal(result, golden[:2])
    assert module.seen.get("hidden_states_fp32") is fp32_copy
    hooks.close()


def test_contiguous_mome_o_conv_is_dumped(tmp_path):
    cfg = _config(tmp_path)
    attention = nn.Module()
    attention.o_conv = nn.Identity()
    hooks = AttentionTrainRefHooks(cfg, rank=2)
    hooks.install(attention)

    value = torch.ones(2, 3)
    assert torch.equal(attention.o_conv(value), value)
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "pangu_transformer_o_conv_1_prefill_train_ref_rank2.pt"
    ).is_file()
    hooks.close()


class _MomeSPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Module()
        self.q_a_layernorm = nn.Identity()
        self.kv_lora_rank = 2

    def _apply_mome_prefill_sp(self, value, state_indice, *, inplace):
        assert inplace is True
        value.add_(10 + state_indice)
        return True

    def _kv_norm_rope_cache(self, latent_kv, *args, **kwargs):
        return latent_kv[..., :2], latent_kv[..., 2:]

    def _apply_o_proj(self, value):
        return value + 1

    def _apply_mome_out_prefill_sp(self, value, _sp_manager):
        return self._apply_o_proj(value + 20)


def test_functional_mome_sp_path_restores_legacy_dump_boundaries(tmp_path):
    cfg = replace(_config(tmp_path), mode="real", mock_modules=())
    attention = _MomeSPAttention()
    hooks = AttentionTrainRefHooks(cfg, rank=0)
    hooks.install(attention)

    q = torch.zeros(2, 3)
    assert attention._apply_mome_prefill_sp(q, 0, inplace=True) is True
    attention.q_a_layernorm(q)

    kv = torch.zeros(2, 4)
    assert attention._apply_mome_prefill_sp(kv, 1, inplace=True) is True
    attention._kv_norm_rope_cache(kv, None)

    out = attention._apply_mome_out_prefill_sp(
        torch.zeros(2, 3),
        object(),
    )

    root = cfg.infer_dump_dir / "prefill"
    qa = torch.load(
        root / "decoder.layers.0.self_attention.qa_conv_prefill_real_rank0.pt",
        weights_only=False,
    )
    compressed = torch.load(
        root / "decoder.layers.0.self_attention.compresskv_conv_prefill_real_rank0.pt",
        weights_only=False,
    )
    o_conv = torch.load(
        root / "pangu_transformer_o_conv_1_prefill_real_rank0.pt",
        weights_only=False,
    )
    assert torch.equal(qa, torch.full((2, 3), 10.0))
    assert torch.equal(compressed, torch.full((2, 2), 11.0))
    assert torch.equal(o_conv, torch.full((2, 3), 20.0))
    assert torch.equal(out, torch.full((2, 3), 21.0))
    hooks.close()


class _Indexer(nn.Module):
    def _apply_lightning_indexer(self, wi, qi, ki_cache, *args, **kwargs):
        return qi[..., 0]


def test_absent_dsa_custom_files_do_not_block_other_custom_components(tmp_path):
    cfg = _config(tmp_path, custom=True)
    attention = nn.Module()
    attention.indexer = _Indexer()
    hooks = AttentionTrainRefHooks(cfg, rank=0)

    hooks.install(attention)

    assert hooks.installed_hook_count == 1
    output = attention.indexer._apply_lightning_indexer(
        torch.ones(2, 2),
        torch.ones(2, 2, 2),
        torch.ones(2, 2),
    )
    assert output.shape[0] == 2
    hooks.close()


def test_partial_dsa_custom_reference_set_is_rejected(tmp_path):
    cfg = _config(tmp_path, custom=True)
    torch.save(
        torch.ones(2, 2, 2),
        cfg.mhc_custom_dump_dir / "dsa_attention_index_query_all_heads_1_rank0.pt",
    )
    attention = nn.Module()
    attention.indexer = _Indexer()
    hooks = AttentionTrainRefHooks(cfg, rank=0)

    with pytest.raises(RuntimeError, match="incomplete DSA custom"):
        hooks.install(attention)


def test_moe_custom_directory_without_expert_files_uses_native_path(tmp_path):
    cfg = _config(tmp_path, custom=True)
    hooks = MoeTrainRefHooks(cfg, rank=0)

    hooks._load_custom_router_reference()

    assert hooks._custom == {}
    assert hooks._has_expert_reference is False


def test_partial_moe_expert_reference_set_is_rejected(tmp_path):
    cfg = _config(tmp_path, custom=True)
    torch.save(
        torch.ones(2, 2),
        cfg.mhc_custom_dump_dir / "moe_experts_up_permuted_1_rank0.pt",
    )
    hooks = MoeTrainRefHooks(cfg, rank=0)

    with pytest.raises(TrainRefDataError, match="incomplete custom expert"):
        hooks._load_custom_router_reference()


def test_moe_expert_ranks_fall_back_to_contiguous_auto_detection(tmp_path):
    cfg = _config(tmp_path, custom=True)
    prefixes = (
        "moe_experts_up_permuted",
        "moe_experts_up_tokens_per_expert",
        "moe_experts_up_weight1",
        "moe_experts_down_weight2",
    )
    for prefix in prefixes:
        torch.save(
            torch.ones(1, 2),
            cfg.mhc_custom_dump_dir / f"{prefix}_1_rank0.pt",
        )
    hooks = MoeTrainRefHooks(cfg, rank=0)

    hooks._load_custom_router_reference()

    assert hooks._has_expert_reference is True
    assert hooks._expert_ranks == (0,)


class _Projection(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, value):
        return value + self.delta, None


def test_q_and_kv_a_outputs_are_dumped_as_legacy_linear_qkv(tmp_path):
    cfg = _config(tmp_path)
    golden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _save_megatron(
        cfg.layer_training_dump_path("self_attention.linear_qkv"),
        golden,
    )
    attention = nn.Module()
    attention.q_a_proj = _Projection(1)
    attention.kv_a_proj_with_mqa = _Projection(2)
    hooks = AttentionTrainRefHooks(cfg, rank=0)
    hooks.install(attention)

    attention.q_a_proj(torch.zeros(2, 2))
    attention.kv_a_proj_with_mqa(torch.zeros(2, 2))

    dumped = torch.load(
        cfg.infer_dump_dir
        / "prefill"
        / "decoder.layers.0.self_attention.linear_qkv_prefill_train_ref_rank0.pt",
        weights_only=False,
    )
    assert torch.equal(dumped, torch.cat([golden[:2] + 1, golden[:2] + 2], -1))
    hooks.close()


class _SplitProjection(nn.Module):
    def __init__(self, width, offset):
        super().__init__()
        self.width = width
        self.offset = offset
        self.seen = None

    def forward(self, value):
        self.seen = value.clone()
        rows = torch.arange(
            self.width,
            dtype=value.dtype,
            device=value.device,
        ).unsqueeze(0)
        return rows.expand(value.shape[0], -1) + self.offset, None


def test_split_q_up_reconstructs_legacy_linear_qb_per_head(tmp_path):
    cfg = _config(tmp_path)
    golden = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _save_megatron(
        cfg.layer_training_dump_path("self_attention.linear_qb"),
        golden,
    )
    attention = nn.Module()
    attention.split_q_up_in_multistream = True
    attention.qk_nope_head_dim = 2
    attention.qk_rope_head_dim = 1
    attention.q_b_proj = nn.Identity()
    attention.q_b_nope_proj = _SplitProjection(4, 10)
    attention.q_b_pe_proj = _SplitProjection(2, 20)
    hooks = AttentionTrainRefHooks(cfg, rank=0)
    hooks.install(attention)

    runtime = torch.zeros(2, 3)
    nope, _ = attention.q_b_nope_proj(runtime)
    pe, _ = attention.q_b_pe_proj(runtime)

    expected = torch.cat(
        [nope.reshape(2, 2, 2), pe.reshape(2, 2, 1)],
        dim=-1,
    ).flatten(-2)
    dumped = torch.load(
        cfg.infer_dump_dir
        / "prefill"
        / "decoder.layers.0.self_attention.linear_qb_prefill_train_ref_rank0.pt",
        weights_only=False,
    )
    assert torch.equal(attention.q_b_nope_proj.seen, golden)
    assert torch.equal(attention.q_b_pe_proj.seen, golden)
    assert torch.equal(dumped, expected)
    hooks.close()


class _Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, input_ids):
        self.seen = input_ids.clone()
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 2)


class _Rotary:
    def __init__(self):
        self.seen = None

    def get_cos_sin(self, positions):
        self.seen = positions.clone()
        return positions + 10, positions + 20


class _LogitsModel:
    def __init__(self, pangu_model):
        self.model = pangu_model
        self.lm_head = nn.Identity()
        self.seen_logits_input = None

    def compute_logits(self, hidden_states):
        self.seen_logits_input = hidden_states.clone()
        return hidden_states + 1


def test_model_level_hooks_cover_embedding_rotary_and_compute_logits(tmp_path):
    cfg = _config(tmp_path)
    _save_megatron(
        cfg.training_dump_path("embedding"),
        torch.tensor([5, 6, 7]),
    )
    _save_megatron(
        cfg.model_training_dump_path("rotary_pos_emb"),
        torch.tensor([0, 1, 2]),
    )
    lm_input = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    _save_megatron(cfg.training_dump_path("lm_head"), lm_input)
    embedding = _Embedding()
    norm = nn.Identity()
    rotary = _Rotary()
    layer = SimpleNamespace(
        input_layernorm=norm,
        self_attn=SimpleNamespace(rotary_emb=rotary),
    )
    pangu = SimpleNamespace(
        embed_tokens=embedding,
        start_layer=0,
        layers=[layer],
    )
    model = _LogitsModel(pangu)
    hooks = ModelLevelTrainRefHooks(cfg, rank=0)
    hooks.install(model)

    embeddings = embedding(torch.zeros(2, dtype=torch.long))
    norm(embeddings)
    rotary.get_cos_sin(torch.full((2,), 9, dtype=torch.long))
    logits = model.compute_logits(torch.zeros(1, 4))

    assert torch.equal(embedding.seen, torch.tensor([5, 6]))
    assert torch.equal(rotary.seen, torch.tensor([0, 1]))
    assert torch.equal(model.seen_logits_input, lm_input[1, 0].unsqueeze(0))
    assert torch.equal(logits, model.seen_logits_input + 1)
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "embedding.word_embeddings_prefill_train_ref_rank0.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir / "prefill" / "rotary_pos_emb.cos_prefill_train_ref_rank0.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir / "prefill" / "lm_head_prefill_train_ref_rank0.pt"
    ).is_file()
    hooks.close()


def test_model_level_hooks_do_not_access_pp_missing_embedding_or_head(tmp_path):
    cfg = _config(tmp_path)
    _save_megatron(
        cfg.model_training_dump_path("rotary_pos_emb"),
        torch.tensor([0, 1, 2]),
    )

    class PPMissingLayer(nn.Module):
        pass

    rotary = _Rotary()
    pangu = SimpleNamespace(
        embed_tokens=PPMissingLayer(),
        start_layer=1,
        layers=[
            PPMissingLayer(),
            SimpleNamespace(
                input_layernorm=nn.Identity(),
                self_attn=SimpleNamespace(rotary_emb=rotary),
            ),
        ],
    )
    model = _LogitsModel(pangu)
    model.lm_head = PPMissingLayer()
    hooks = ModelLevelTrainRefHooks(cfg, rank=1)

    hooks.install(model)
    rotary.get_cos_sin(torch.zeros(2, dtype=torch.long))

    assert hooks.installed_hook_count == 1
    assert torch.equal(rotary.seen, torch.tensor([0, 1]))
    hooks.close()
