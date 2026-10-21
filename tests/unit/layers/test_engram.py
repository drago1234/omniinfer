# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU references and configuration contracts; hardware coverage is in integration/layers."""

from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from omni_npu.layers.engram import PanGuEngramConfig, EngramLayout

pytestmark = pytest.mark.unit


def test_sequence_parallel_rejects_head_sharding_before_table_allocation(monkeypatch):
    from unittest.mock import Mock
    from omni_npu.layers import engram

    allocate = Mock()
    monkeypatch.setattr(engram, "ParallelEngramEmbedding", allocate)
    layout = EngramLayout(_config(shard_heads=True))
    with pytest.raises(ValueError, match="sequence parallelism and TP head sharding cannot both be enabled"):
        engram.PanGuEngramFusion(layout, 2, None, True)
    allocate.assert_not_called()


def test_unsharded_sp_looks_up_only_local_tokens(monkeypatch):
    from omni_npu.layers import engram

    # Nonconstant rows expose wrong token offsets and stale padded entries.
    table = torch.arange(80, dtype=torch.float32).reshape(40, 2)
    layer = engram.PanGuEngramFusion.__new__(engram.PanGuEngramFusion)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(shard_heads=False)
    layer.layout = SimpleNamespace(n_hash_cols=2)
    calls = []

    def lookup(ids, output):
        calls.append(ids.shape[0])
        output.copy_(table[ids.long().clamp_min(0)].masked_fill(ids[:, :, None] < 0, 0))

    layer.embed_tokens = SimpleNamespace(model_tp_size=4, tp_size=1, lookup=lookup)
    for sequence in (False, True):
        layer.use_sequence_parallel = sequence
        for tokens in (0, 1, 7, 8):
            ids = torch.arange(tokens * 2, dtype=torch.int32).reshape(tokens, 2)
            if tokens:
                ids[-1, 0] = -1
            full = table[ids.long().clamp_min(0)].masked_fill(ids[:, :, None] < 0, 0)
            chunk = (tokens + 3) // 4 if sequence else tokens
            for rank in range(4):
                monkeypatch.setattr(engram, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
                layer.staged_rows = torch.full((chunk, 2, 2), float("nan"))
                layer.prepare_embeddings(ids)
                expected = full
                if sequence:
                    expected = F.pad(full, (0, 0, 0, 0, 0, chunk * 4 - tokens))
                    expected = expected[rank * chunk : (rank + 1) * chunk]
                assert calls[-1] == chunk
                torch.testing.assert_close(layer.embed(ids), expected, rtol=0, atol=0)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_model_hash_partition_follows_sequence_parallelism(monkeypatch, sequence_parallel):
    from omni_npu.layers import engram

    model = engram.PanGuEngramModelMixin()
    model.engram_layout = EngramLayout(_config())
    attention = torch.nn.Module()
    attention.kv_cache = torch.empty(0)
    fusion = SimpleNamespace(
        use_sequence_parallel=sequence_parallel,
        embed_tokens=SimpleNamespace(model_tp_size=4),
        wkv=SimpleNamespace(weight=torch.empty(1)),
    )
    model.layers = [SimpleNamespace(engram=fusion, self_attn=SimpleNamespace(attn=attention))]
    model.start_layer, model.end_layer = 0, 1
    monkeypatch.setattr(engram, "get_tp_group", lambda: SimpleNamespace(rank_in_group=2))
    model._init_engram_hash()
    assert model.engram_hash.sp_rank == (2 if sequence_parallel else 0)
    assert model.engram_hash.sp_size == (4 if sequence_parallel else 1)


def _config(**overrides) -> PanGuEngramConfig:
    values = {
        "layer_ids": (2,),
        "vocab_sizes": (11, 13),
        "max_ngram_size": 3,
        "num_heads_per_ngram": 2,
        "embedding_dim_per_ngram": 4,
        "pad_token_id": 99,
        "seed": 42,
        "hidden_size": 6,
        "num_streams": 2,
        "post_norm": True,
        "rms_norm_eps": 1e-5,
        "table_dtype": torch.float32,
    }
    values.update(overrides)
    return PanGuEngramConfig(**values)


def test_prime_61_is_preserved_in_head_layout():
    assert EngramLayout._is_prime(61)
    assert EngramLayout._find_next_prime(60, set()) == 61
    assert EngramLayout._find_next_prime(60, {61}) == 67
    layout = EngramLayout(_config(
        layer_ids=(2, 21), vocab_sizes=(11, 11),
        num_heads_per_ngram=8, embedding_dim_per_ngram=32,
    ))
    expected = ((11, 13, 17, 19, 23, 29, 31, 37), (41, 43, 47, 53, 59, 61, 67, 71))
    assert layout.primes == (expected, expected)
    assert layout.num_embeddings == (622, 622)
    torch.testing.assert_close(
        layout.offsets,
        torch.tensor([[0, 11, 24, 41, 60, 83, 112, 143, 180, 221, 264, 311, 364, 423, 484, 551]] * 2),
        rtol=0, atol=0,
    )


def test_config_rejects_unsupported_training_features():
    for field in (
        "engram_use_compressed_tokenizer",
        "engram_use_short_conv",
        "engram_enable_contextual_emb",
        "engram_enable_cross_head_mix",
    ):
        with pytest.raises(ValueError):
            PanGuEngramConfig.from_hf_config(_hf_config(**{field: True}))


def _hf_config(**overrides):
    return SimpleNamespace(
        **{
            "engram_layer_ids": [2],
            "engram_vocab_size": [11, 13],
            "engram_max_ngram_size": 3,
            "engram_n_head_per_ngram": 2,
            "engram_n_embed_per_ngram": 4,
            "engram_pad_id": 0,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "use_mhc": True,
            "mhc_num_stream": 4,
            **overrides,
        }
    )


@pytest.mark.parametrize("use_mhc,streams", [(None, 4), (False, 4), (True, 1)])
def test_config_requires_mhc_with_multiple_streams(use_mhc, streams):
    config = _hf_config(mhc_num_stream=streams)
    if use_mhc is None:
        del config.use_mhc
    else:
        config.use_mhc = use_mhc
    with pytest.raises(ValueError, match="Engram requires mHC"):
        PanGuEngramConfig.from_hf_config(config)


@pytest.mark.parametrize("sharding", ["none", "tp"])
def test_storage_modes_from_operator_json(monkeypatch, sharding):
    from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig, model_extra_config

    options = ModelOperatorOptConfig(
        engram_table_location="cpu_shared",
        engram_prefetch=False,
        engram_head_sharding=sharding,
    )
    monkeypatch.setattr(model_extra_config, 'operator_opt_config', options)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_hf_config(), dtype=torch.bfloat16),
        load_config=SimpleNamespace(load_format='dummy'),
    )
    parsed = PanGuEngramConfig.from_vllm_config(config)
    assert parsed.shard_heads == (sharding == 'tp')
    assert parsed.dummy_weights


@pytest.mark.parametrize(
    "field,value",
    [
        ("engram_table_location", "device"),
        ("engram_table_location", None),
        ("engram_table_location", ["cpu_shared"]),
        ("engram_prefetch", True),
        ("engram_prefetch", "false"),
        ("engram_prefetch", 0),
        ("engram_head_sharding", "typo"),
        ("engram_head_sharding", None),
        ("engram_head_sharding", ["tp"]),
    ],
)
def test_operator_config_rejects_unsupported_engram_options(field, value):
    from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig

    with pytest.raises(ValueError, match=field):
        ModelOperatorOptConfig(**{field: value})


def test_old_additional_config_is_not_silently_ignored():
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_hf_config(), dtype=torch.bfloat16),
        additional_config={'engram_config': {'cpu_offload': False}},
    )
    with pytest.raises(ValueError, match='operator_optimization_config'):
        PanGuEngramConfig.from_vllm_config(config)


@pytest.mark.parametrize("shared", [False, True])
def test_dummy_initialization_preserves_ordinary_weights_and_shared_storage(monkeypatch, shared):
    from unittest.mock import Mock
    from omni_npu.vllm_patches.patches.models.pangu_v2_base import patch_process_weights_after_loading as loading

    model = torch.nn.Module()
    model.ordinary = torch.nn.Parameter(torch.full((2, 3), 7.0))
    if shared:
        model.table = torch.nn.Parameter(torch.full((4, 3), 7.0), requires_grad=False)
        model.table.engram_shared_ptr = model.table.data_ptr()
    config = object()
    original = Mock()
    initialize = Mock(wraps=loading.initialize_single_dummy_weight)
    monkeypatch.setattr(loading, "_ORIGINAL_INITIALIZE_DUMMY_WEIGHTS", original)
    monkeypatch.setattr(loading, "initialize_single_dummy_weight", initialize)

    assert loading._initialize_dummy_weights(model, config, low=-0.2, high=0.3, seed=41) is None

    if shared:
        original.assert_not_called()
        initialize.assert_called_once_with(model.ordinary, -0.2, 0.3, 41)
        assert torch.all((model.ordinary >= -0.2) & (model.ordinary <= 0.3))
        assert model.table.data_ptr() == model.table.engram_shared_ptr
        torch.testing.assert_close(model.table, torch.full((4, 3), 7.0))
    else:
        original.assert_called_once_with(model, config, -0.2, 0.3, 41)
        initialize.assert_not_called()


@pytest.mark.parametrize("writer", [False, True])
def test_dummy_table_is_filled_only_by_writer_before_synchronization(writer):
    from unittest.mock import Mock
    from omni_npu.layers.engram import ParallelEngramEmbedding

    embedding = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.dummy_weights = True
    embedding.weight = torch.nn.Parameter(torch.full((3, 4), 7.0), requires_grad=False)
    embedding.weight.dummy_weight_value = 1.0
    expected = torch.full((3, 4), 1.0 if writer else 7.0)
    synchronize = Mock(side_effect=lambda: torch.testing.assert_close(embedding.weight, expected))
    embedding.storage = SimpleNamespace(is_writer=writer, synchronize=synchronize)
    pointer = embedding.weight.data_ptr()

    embedding.process_weights_after_loading()

    synchronize.assert_called_once_with()
    torch.testing.assert_close(embedding.weight, expected)
    assert embedding.weight.data_ptr() == pointer
    synchronize.reset_mock()
    embedding.dummy_weights = False
    embedding.process_weights_after_loading()
    synchronize.assert_not_called()
    torch.testing.assert_close(embedding.weight, expected)


def test_engram_group_lifecycle(monkeypatch):
    from unittest.mock import Mock
    from omni_npu.v1.distributed import parallel_state_ext as state

    shared = Mock()
    # This regression owns only Engram's group, not state left by A2 group tests.
    for name in ('_LOCAL_COMM_LIST', '_CROSS_COMM_LIST', '_CROSS_ROUND_COMM_LIST'):
        monkeypatch.setattr(state, name, None)
    monkeypatch.setattr(state, '_ENGRAM_SHARED', shared)
    assert state.get_engram_shared_group() is shared
    assert state.initialize_engram_shared_group(False) is shared
    state.destroy_parallel_state_ext_groups()
    state.destroy_parallel_state_ext_groups()
    shared.destroy.assert_called_once_with()
    assert state._ENGRAM_SHARED is None
    with pytest.raises(RuntimeError, match="not initialized"):
        state.get_engram_shared_group()


@pytest.mark.parametrize(
    'layers,sharding,shared',
    [
        ([], 'tp', None),
        ([2], 'tp', True),
        ([2], 'none', False),
    ],
)
def test_worker_initializes_only_selected_engram_groups(monkeypatch, layers, sharding, shared):
    from unittest.mock import Mock
    from omni_npu.v1.distributed import parallel_state_ext as state
    from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig, model_extra_config

    monkeypatch.setattr(
        model_extra_config, 'operator_opt_config', ModelOperatorOptConfig(engram_head_sharding=sharding)
    )
    initialize_shared = Mock()
    monkeypatch.setattr(state, 'initialize_engram_shared_group', initialize_shared)
    state.initialize_engram_model_parallel(
        SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(engram_layer_ids=layers)))
    )
    if shared is None:
        initialize_shared.assert_not_called()
    else:
        initialize_shared.assert_called_once_with(shared)


def test_history_cache_follows_bound_kv_allocation():
    from omni_npu.layers.engram import NgramHashState

    attention = torch.nn.Module()
    attention.layer_name = "model.layers.2.self_attn.attn"
    attention.kv_cache = torch.empty(0)
    state = NgramHashState(EngramLayout(_config()), torch.device("cpu"), attention)
    state.block_size = 16
    assert not state.has_bound_cache()
    assert not state.ensure_cache()
    attention.kv_cache = (torch.zeros(3, 16, 1), torch.zeros(3, 16, 2))
    assert state.has_bound_cache() and state._cache is None
    assert state.ensure_cache()
    assert state._cache.shape == (48,)
    history = state._cache
    history.fill_(17)
    assert state.ensure_cache() and state._cache is history
    attention.kv_cache = (torch.zeros(5, 16, 1),)
    assert state.ensure_cache() and state._cache is not history
    assert state._cache.shape == (80,) and not state._cache.any()
    attention.kv_cache = torch.empty(0)
    assert not state.has_bound_cache()
    assert not state.ensure_cache() and state._cache is None


@pytest.mark.parametrize("stage", ["capacity", "create"])
def test_shared_allocation_failure_is_broadcast_with_rank_and_stage(monkeypatch, stage):
    from unittest.mock import Mock
    from omni_npu.layers import engram
    from vllm.distributed.device_communicators import shm_broadcast

    monkeypatch.delattr(shm_broadcast, 'check_shm_free_space', raising=False)
    monkeypatch.setattr(
        engram.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0 if stage == "capacity" else 1024)
    )
    create = Mock(side_effect=OSError('injected failure'))
    broadcast = Mock(side_effect=lambda obj: obj)
    monkeypatch.setattr(engram.tempfile, 'NamedTemporaryFile', create)
    group = SimpleNamespace(rank_in_group=0, world_size=1, broadcast_object=broadcast)
    message = (
        'check shared-memory capacity:.*Insufficient space'
        if stage == 'capacity'
        else 'create backing file: OSError: injected failure'
    )
    with pytest.raises(RuntimeError, match=f'rank 0: {message}'):
        engram.DPSharedEngramStorage((2, 4), torch.bfloat16, group)
    broadcast.assert_called_once()
    if stage == 'capacity':
        create.assert_not_called()
    else:
        create.assert_called_once()


def test_shared_capacity_uses_vllm_when_available(monkeypatch):
    from unittest.mock import Mock
    from omni_npu.layers import engram
    from vllm.distributed.device_communicators import shm_broadcast

    check = Mock()
    monkeypatch.setattr(shm_broadcast, 'check_shm_free_space', check, raising=False)
    engram.DPSharedEngramStorage._check_shm_free_space(16)
    check.assert_called_once_with(16)


def test_registration_failure_reports_all_peer_ranks(monkeypatch):
    import sys
    from types import ModuleType
    from unittest.mock import Mock
    from omni_npu.layers import engram
    from omni_npu import allocator

    storage = engram.DPSharedEngramStorage.__new__(engram.DPSharedEngramStorage)
    storage._device_view = None
    storage.tensor = torch.zeros(2, 4)
    storage.world_size = 2
    storage.group = SimpleNamespace(cpu_group=object())
    monkeypatch.setattr(torch.npu, 'current_stream', lambda: None)
    monkeypatch.setattr(torch.npu, 'current_device', lambda: 0)

    def gather(errors, error, group):
        errors[:] = [error, 'host registration: peer failed']

    # Test error aggregation without importing or requiring a native extension.
    extension = ModuleType("omni_npu.allocator.npu_uva")
    extension.register_mapped_cpu_tensor = Mock(side_effect=RuntimeError('injected registration failure'))
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    monkeypatch.setattr(allocator, 'npu_uva', extension, raising=False)
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    with pytest.raises(
        RuntimeError, match='EDP rank 0: host registration:.*EDP rank 1: host registration: peer failed'
    ):
        storage.device_view()
    assert storage._device_view is None
    extension.register_mapped_cpu_tensor.assert_called_once_with(storage.tensor, 0)


@pytest.mark.parametrize("writer,padding", [(True, 1), (False, 0)])
def test_shared_weight_loading_writes_once_and_fences_peers(monkeypatch, writer, padding):
    from omni_npu.layers import engram

    storage = engram.DPSharedEngramStorage.__new__(engram.DPSharedEngramStorage)
    storage.is_writer = writer
    storage.group = SimpleNamespace(cpu_group=object())
    weight = torch.nn.Parameter(torch.zeros(3, 4, dtype=torch.bfloat16), requires_grad=False)
    weight.engram_vocab_start = 2
    full = torch.arange((7 + padding) * 4, dtype=torch.float32).reshape(7 + padding, 4)
    snapshots = []

    def barrier(*, group):
        assert group is storage.group.cpu_group
        snapshots.append(weight.detach().clone())

    monkeypatch.setattr(engram.torch.distributed, "barrier", barrier)
    pointer = weight.data_ptr()
    weight.engram_shared_ptr = pointer
    embedding = engram.ParallelEngramEmbedding.__new__(engram.ParallelEngramEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.num_embeddings, embedding.dim, embedding.storage = 7, 4, storage
    embedding.weight_loader(weight, full)
    assert weight.data_ptr() == pointer
    assert len(snapshots) == 1
    torch.testing.assert_close(snapshots[0], full[2:5].bfloat16() if writer else torch.zeros_like(weight))
    for invalid in (full[:6], full[:, :3], full.flatten()):
        with pytest.raises(ValueError, match="requires at least"):
            embedding.weight_loader(weight, invalid)
    assert len(snapshots) == 1


def test_hidden_stream_layouts_preserve_shape_validation():
    from omni_npu.layers.engram import PanGuEngramFusion

    layer = PanGuEngramFusion.__new__(PanGuEngramFusion)
    torch.nn.Module.__init__(layer)
    layer.num_streams, layer.hidden_size = 2, 4
    for tokens in (0, 3):
        streams = torch.arange(tokens * 8).reshape(tokens, 2, 4)
        for hidden in (streams, streams.flatten(-2)):
            actual = layer._reshape_hidden_streams(hidden)
            torch.testing.assert_close(actual, streams, rtol=0, atol=0)
    for shape, message in (
        ((3, 7), "flattened hidden-state width"),
        ((3, 3, 4), "hidden-state shape"),
        ((3, 2, 5), "hidden-state shape"),
        ((3,), "expects"),
    ):
        with pytest.raises(ValueError, match=message):
            layer._reshape_hidden_streams(torch.empty(shape))


@pytest.mark.parametrize("fusion,tp_size", [(True, 1), (False, 1), (True, 2)])
def test_projection_gate_switch_and_loaded_weight_layout(monkeypatch, fusion, tp_size):
    from unittest.mock import Mock
    from omni_npu.layers import engram
    from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig, model_extra_config

    assert not ModelOperatorOptConfig().use_engram_proj_gate_fusion_op
    monkeypatch.setattr(
        model_extra_config, "operator_opt_config", ModelOperatorOptConfig(use_engram_proj_gate_fusion_op=fusion)
    )
    monkeypatch.setattr(engram, "ParallelEngramEmbedding", lambda *args: SimpleNamespace(
        process_weights_after_loading=Mock()
    ))

    def linear(size, output, **kwargs):
        module = torch.nn.Linear(size, output // tp_size, bias=False, dtype=torch.bfloat16)
        module.tp_size = tp_size
        return module

    monkeypatch.setattr(engram, "ColumnParallelLinear", linear)
    layer = engram.PanGuEngramFusion(EngramLayout(_config()), 2, None, False)
    with torch.no_grad():
        layer.wkv.weight.copy_(torch.arange(layer.wkv.weight.numel()).reshape_as(layer.wkv.weight))
    layer.process_weights_after_loading()
    assert layer.use_engram_proj_gate_fusion_op == (fusion and tp_size == 1)
    if layer.use_engram_proj_gate_fusion_op:
        packed = layer.wkv.weight.reshape(3, 6, 8)
        torch.testing.assert_close(layer.proj_weights_key, packed[:2].transpose(1, 2))
        torch.testing.assert_close(layer.proj_weights_value, packed[2].t())
        assert layer.proj_weights_key.is_contiguous() and layer.proj_weights_value.is_contiguous()
    else:
        assert layer.proj_weights_key is layer.proj_weights_value is None


def test_projection_gate_custom_op_passes_model_eps_without_backward_caches(monkeypatch):
    import sys
    from unittest.mock import Mock
    from omni_npu.layers import engram

    hidden = torch.ones(3, 2, 4, dtype=torch.bfloat16)
    memory = torch.ones(3, 8, dtype=torch.bfloat16)
    key = torch.ones(2, 8, 4, dtype=torch.bfloat16)
    value = torch.ones(8, 4, dtype=torch.bfloat16)
    gamma = torch.ones(2, 4, dtype=torch.bfloat16)
    op = Mock(return_value=(hidden.unsqueeze(0).clone(), None, None, None, None))
    monkeypatch.setitem(
        sys.modules,
        "omni_npu.layers.ops.ai_infra_pypto_engram_proj_gate",
        SimpleNamespace(ai_infra_engram_proj_gate=op),
    )
    result = engram._engram_proj_gate(hidden, memory, key, value, gamma, gamma, 1e-6, 1e-5)
    torch.testing.assert_close(result, hidden)
    args, kwargs = op.call_args
    assert args[0].shape == (1, 3, 2, 4) and args[1].shape == (1, 3, 8)
    assert kwargs == {"clamp_value": 1e-6, "eps": 1e-5, "return_cache": False}
    op.reset_mock()
    assert engram._engram_proj_gate(hidden[:0], memory[:0], key, value, gamma, gamma, 1e-6, 1e-5).numel() == 0
    op.assert_not_called()
    fake = engram._engram_proj_gate_fake(hidden, memory, key, value, gamma, gamma, 1e-6, 1e-5)
    assert fake.shape == hidden.shape and fake.dtype == hidden.dtype


@pytest.mark.parametrize("overrides,message", [
    ({"engram_concat_fusion": False}, "concat fusion"),
    ({"engram_max_ngram_size": 1}, "at least 2"),
    ({"engram_vocab_size": [11, 13, 17]}, "one positive size"),
    ({"engram_n_head_per_ngram": 0}, "divisible"),
    ({"engram_layer_ids": [2, 2]}, "unique"),
    ({"engram_layer_ids": [4]}, "out-of-range"),
    ({"engram_pad_id": -1}, "non-negative"),
    ({"hidden_size": 0}, "positive"),
])
def test_invalid_checkpoint_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        PanGuEngramConfig.from_hf_config(_hf_config(**overrides))


def test_config_defaults_and_prime_search_boundaries():
    import math

    assert PanGuEngramConfig.from_hf_config(_hf_config(engram_layer_ids=[])) is None
    assert PanGuEngramConfig.from_vllm_config(SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace())
    )) is None
    for sizes in (11, [11]):
        assert PanGuEngramConfig.from_hf_config(_hf_config(engram_vocab_size=sizes)).vocab_sizes == (11, 11)
    for n in (0, 1, 2, 41, 341, 3215031751, 4294967291):
        expected = n >= 2 and all(n % divisor for divisor in range(2, math.isqrt(n) + 1))
        assert EngramLayout._is_prime(n) == expected
    with pytest.raises(ValueError, match="supported range"):
        EngramLayout._is_prime(2 ** 32)


@pytest.mark.parametrize("failure", [None, "allocate", "local_mapping", "peer_mapping"])
def test_shared_mapping_lifetime_cleanup_and_writer_load(monkeypatch, tmp_path, failure):
    import sys
    from types import ModuleType
    from unittest.mock import Mock
    from omni_npu.layers import engram
    from omni_npu import allocator

    temporary_file = engram.tempfile.NamedTemporaryFile
    monkeypatch.setattr(engram.tempfile, "NamedTemporaryFile", lambda **kwargs: temporary_file(
        **{**kwargs, "dir": tmp_path}
    ))
    if failure == "allocate":
        monkeypatch.setattr(engram.os, "posix_fallocate", Mock(side_effect=OSError("allocation failed")))
    original_mmap = engram.mmap.mmap
    mappings = []

    def mapping(*args, **kwargs):
        if failure == "local_mapping":
            raise OSError("mapping failed")
        result = original_mmap(*args, **kwargs)
        mappings.append(result)
        return result

    def gather(errors, error, group):
        errors[:] = [error, "peer failed" if failure == "peer_mapping" else None]

    monkeypatch.setattr(engram.mmap, "mmap", mapping)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    barrier = Mock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object(), broadcast_object=lambda obj: obj)
    if failure:
        with pytest.raises(RuntimeError, match="allocation failed|mapping failed|peer failed"):
            engram.DPSharedEngramStorage((3, 4), torch.bfloat16, group)
        assert all(m.closed for m in mappings)
    else:
        storage = engram.DPSharedEngramStorage((3, 4), torch.bfloat16, group)
        assert not list(tmp_path.iterdir())
        extension = ModuleType("omni_npu.allocator.npu_uva")
        extension.register_mapped_cpu_tensor = Mock(return_value=storage.tensor)
        monkeypatch.setitem(sys.modules, extension.__name__, extension)
        monkeypatch.setattr(allocator, "npu_uva", extension, raising=False)
        monkeypatch.setattr(torch.npu, "current_stream", lambda: None)
        monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
        assert storage.device_view() is storage.device_view() is storage.tensor
        extension.register_mapped_cpu_tensor.assert_called_once_with(storage.tensor, 0)
        weight = torch.nn.Parameter(storage.tensor, requires_grad=False)
        weight.engram_vocab_start = 0
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        storage.load_weight(weight, source)
        torch.testing.assert_close(storage.tensor, source.bfloat16())
        barrier.assert_called_once_with(group=group.cpu_group)
