# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Runner handoff: prompt lookbacks and device history cache binding."""

from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from omni_npu.worker.npu_model_runner import NPUModelRunner, GPUModelRunner

pytestmark = pytest.mark.unit


def test_gqa_pd_kv_allocation_alignment_preserves_data_and_aliases(monkeypatch):
    import omni_npu.worker.npu_model_runner as module

    backing = torch.arange(129, dtype=torch.int8)
    raw = backing[1:]
    expected = raw.clone()
    assert raw.data_ptr() % (2 * 1024 * 1024)
    tensors = {'layer0': raw, 'layer1': raw}
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=object())
    monkeypatch.setattr(module.model_extra_config.operator_opt_config, 'use_ai_infra_fa_for_gpt_oss_sink', True)
    monkeypatch.setattr(GPUModelRunner, '_allocate_kv_cache_tensors', lambda *args: tensors)
    result = runner._allocate_kv_cache_tensors(object())
    assert result is tensors and result['layer0'] is result['layer1']
    torch.testing.assert_close(result['layer0'], expected)
    assert result['layer0'].data_ptr() % (2 * 1024 * 1024) == 0


class CpuBuffer:
    def __init__(self, *shape):
        self.cpu = torch.empty(shape, dtype=torch.int64)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, size):
        return self.cpu[:size].clone()


def test_prompt_lookbacks_ignore_generated_placeholders_and_reorder(monkeypatch):
    event = Mock()
    monkeypatch.setattr(torch.npu, 'Event', Mock(return_value=event))
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner._lookback_copy_event = None
    runner.lookback_token_ids = CpuBuffer(4, 2)
    runner.max_num_reqs = 4
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0, 3, 4, 6]),
        num_prompt_tokens=np.array([3, 5, 3, 3]),
        token_ids_cpu=np.array(
            [[10, 11, 12, -1, -1, -1], [20, 21, 22, 23, 24, -1], [30, 31, 32, -1, -1, -1], [40, 41, 42, -1, -1, -1]]
        ),
    )
    actual = runner._prepare_lookback_token_ids(4)
    assert actual.tolist() == [[-1, -1], [22, 21], [-1, 32], [-1, -1]]
    order = [2, 0, 1, 3]
    for name in ('num_computed_tokens_cpu', 'num_prompt_tokens', 'token_ids_cpu'):
        setattr(runner.input_batch, name, getattr(runner.input_batch, name)[order])
    assert runner._prepare_lookback_token_ids(3)[:3].tolist() == actual[order][:3].tolist()
    event.synchronize.assert_called_once()


def test_history_binds_after_kv_initialization_and_uses_kernel_block_size(monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    import omni_npu.worker.npu_model_runner as runner_module

    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.engram_config = SimpleNamespace()
    runner.engram_hash_state = Mock(cache_prefix="layer")
    runner.get_model = lambda: SimpleNamespace(model=SimpleNamespace(engram_hash=runner.engram_hash_state))
    runner.input_batch = SimpleNamespace(block_table=[SimpleNamespace(block_size=16)])
    cache_config = SimpleNamespace(
        num_blocks=10, kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["layer"])]
    )
    cache = {'layer': object()}
    monkeypatch.setattr(GPUModelRunner, 'initialize_kv_cache_tensors', lambda *args: cache)
    monkeypatch.setattr(runner_module, 'has_kv_transfer_group', lambda: False)
    assert runner.initialize_kv_cache_tensors(cache_config, [16]) is cache
    assert runner.engram_hash_state.block_size == 16
    runner.engram_hash_state.ensure_cache.assert_called_once_with()


@pytest.mark.parametrize('engram_layers', [[], [2]])
def test_json_order_changes_only_for_engram(monkeypatch, engram_layers):
    from omni_npu.worker import npu_worker as worker_module

    events = []

    class InitializationObserved(Exception):
        pass

    def observe(name):
        def call(*args, **kwargs):
            events.append(name)
            if len(events) == 2:
                raise InitializationObserved

        return call

    worker = SimpleNamespace(
        device_config=SimpleNamespace(device=SimpleNamespace(type='npu')),
        parallel_config=SimpleNamespace(distributed_executor_backend='ray'),
        local_rank=0,
        rank=0,
        distributed_init_method='mock',
        world_ranks=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16, hf_config=SimpleNamespace(engram_layer_ids=engram_layers)),
        vllm_config=object(),
        scheduler_config=object(),
    )
    monkeypatch.setattr(
        worker_module,
        'current_platform',
        SimpleNamespace(
            device_type='npu',
            dist_backend='hccl',
            logical_device_id_to_visible_device_id=lambda x: x,
            check_if_supports_dtype=lambda x: None,
        ),
    )
    monkeypatch.setattr(torch.npu, 'set_device', lambda x: None)
    monkeypatch.setattr(worker_module, '_patch_npu_triton_capabilities', lambda: None)
    monkeypatch.setattr(worker_module, 'load_model_extra_config', observe('json'))
    monkeypatch.setattr(worker_module, 'init_worker_distributed_environment', observe('groups'))
    initialize = getattr(worker_module.NPUWorker.init_device, '__wrapped__', worker_module.NPUWorker.init_device)
    with pytest.raises(InitializationObserved):
        initialize(worker)
    assert events == (['json', 'groups'] if engram_layers else ['groups', 'json'])


def test_generic_lookbacks_use_explicit_or_current_request_count(monkeypatch):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.lookback_token_ids = object()
    runner.input_batch = SimpleNamespace(num_reqs=3)
    runner._prepare_lookback_token_ids = Mock(side_effect=lambda n: n)
    monkeypatch.setattr(GPUModelRunner, '_init_model_kwargs', lambda self: {'existing': True})
    assert runner._init_model_kwargs() == {'existing': True, 'lookback_token_ids': 3}
    assert runner._init_model_kwargs(0) == {'existing': True, 'lookback_token_ids': 0}
    runner.lookback_token_ids = None
    assert runner._init_model_kwargs() == {'existing': True}


@pytest.mark.parametrize("layers", [[], [2], [5]])
def test_runner_initialization_and_model_load_bind_only_local_history(monkeypatch, layers):
    import omni_npu.worker.npu_model_runner as module
    from tests.unit.layers.test_engram import _hf_config

    hf = _hf_config(engram_layer_ids=layers, num_hidden_layers=8)
    model_config = SimpleNamespace(hf_config=hf, hf_text_config=hf, dtype=torch.bfloat16,
                                   is_mm_prefix_lm=False, max_model_len=16, multimodal_config=None,
                                   get_layers_start_end_indices=lambda parallel: (0, 4))
    config = SimpleNamespace(model_config=model_config, load_config=SimpleNamespace(load_format="dummy"),
                             additional_config=None, cache_config=SimpleNamespace(block_size=4),
                             parallel_config=SimpleNamespace(data_parallel_rank=0))

    def initialize(self, vllm_config, device):
        self.vllm_config, self.model_config, self.device = vllm_config, model_config, device
        self.parallel_config, self.speculative_config, self.max_num_reqs = vllm_config.parallel_config, None, 4

    monkeypatch.setattr(GPUModelRunner, "__init__", initialize)
    monkeypatch.setattr(NPUModelRunner, "_make_buffer", lambda self, *shape, **kwargs: CpuBuffer(*shape))
    monkeypatch.setattr(module, "NPUSamplerV1", Mock())
    monkeypatch.setattr(module, "PIN_MEMORY", False)
    runner = NPUModelRunner(config, torch.device("cpu"))
    assert (runner.engram_config is not None) == (layers == [2])
    assert runner.lookback_token_ids is None and runner._lookback_copy_event is None
    target = SimpleNamespace(token_lookback_depth=2 if layers == [2] else 0)
    runner.get_model = lambda: target
    runner.get_drafter_model = lambda: None
    runner._hook_model_load_weights = Mock()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False))
    monkeypatch.setattr(GPUModelRunner, "load_model", lambda self, *args: setattr(self, "model", target))
    runner.load_model()
    if layers == [2]:
        assert runner.lookback_token_ids.np.shape == (4, 2)
    else:
        assert runner.lookback_token_ids is None


def test_missing_history_kv_group_is_rejected(monkeypatch):
    import omni_npu.worker.npu_model_runner as module

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.engram_config = object()
    runner.get_model = lambda: SimpleNamespace(
        model=SimpleNamespace(engram_hash=SimpleNamespace(cache_prefix="missing"))
    )
    monkeypatch.setattr(GPUModelRunner, "initialize_kv_cache_tensors", lambda *args: {})
    with pytest.raises(ValueError, match="requires its attention KV cache"):
        runner.initialize_kv_cache_tensors(SimpleNamespace(kv_cache_groups=[]), [])
