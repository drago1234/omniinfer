"""NPU execution scopes bypass NVML without changing its global policy."""
import os
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize('method', ['execute_model', 'sample_tokens', 'capture_model'])
@pytest.mark.parametrize('fail', [False, True])
def test_runtime_scope_restores_cuda_and_preserves_nvml_policy(monkeypatch, method, fail):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2
    from omni_npu.worker.npu import aclgraph_utils

    cuda = torch.cuda
    compiled = cuda._is_compiled
    monkeypatch.setenv('PYTORCH_NVML_BASED_CUDA_CHECK', '1')
    monkeypatch.setattr(cuda, '_device_count_nvml', lambda: pytest.fail('NVML probe'))
    monkeypatch.setattr(aclgraph_utils, 'ensure_graph_params', lambda cfg: None)
    monkeypatch.setattr(aclgraph_utils, 'assert_no_acl_tasks_registered', lambda: None)
    def upstream(self, *args, **kwargs):
        assert torch.cuda is torch.npu
        assert torch.cuda.is_available() == torch.npu.is_available()
        assert os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] == '1'
        if fail:
            raise ValueError('deliberate upstream failure')
        return 17
    monkeypatch.setattr(GPUModelRunner, method, upstream)
    runner = object.__new__(NPUModelRunnerV2)
    runner.vllm_config = SimpleNamespace(compilation_config=object())
    args = () if method == 'capture_model' else (None,)
    try:
        if fail:
            with pytest.raises(ValueError, match='deliberate upstream failure'):
                getattr(runner, method)(*args)
        else:
            assert getattr(runner, method)(*args) == 17
    finally:
        assert torch.cuda is cuda
        assert torch.cuda._is_compiled is compiled
        assert os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] == '1'


def test_alias_install_leaves_nvml_policy_and_compiler_interface(monkeypatch):
    from omni_npu.worker.npu.utils import (
        install_torch_cuda_aliases, restore_cuda_attrs, snapshot_cuda_attrs)
    monkeypatch.setenv('PYTORCH_NVML_BASED_CUDA_CHECK', '1')
    cuda, compiled = torch.cuda, torch.cuda._is_compiled
    saved = snapshot_cuda_attrs()
    try:
        install_torch_cuda_aliases()
        assert os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] == '1'
        assert torch.cuda is cuda and torch.cuda._is_compiled is compiled
    finally:
        restore_cuda_attrs(saved)


@pytest.mark.skipif(not torch.npu.is_available(), reason='requires NPU')
def test_real_acl_capture_replay_inside_switch(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import (
        BatchExecutionDescriptor, CudaGraphManager)
    from omni_npu.v1.utils import switch_torch_device
    from omni_npu.worker.npu import aclgraph_manager as ag
    from omni_npu.worker.npu.utils import (
        install_torch_cuda_aliases, restore_cuda_attrs, snapshot_cuda_attrs)

    cuda, saved = torch.cuda, snapshot_cuda_attrs()
    monkeypatch.setenv('PYTORCH_NVML_BASED_CUDA_CHECK', '1')
    # This test must not fabricate either missing CUDA attribute on torch.npu.
    assert not hasattr(torch.npu, 'CUDAGraph')
    assert not hasattr(torch.npu, '_is_compiled')
    try:
        install_torch_cuda_aliases()
        torch.npu.set_device(0)
        desc = BatchExecutionDescriptor(num_tokens=8, num_reqs=1, uniform_token_count=8,
            cg_mode=CUDAGraphMode.FULL)
        manager = object.__new__(ag.AclGraphManager)
        manager.device = torch.device('npu:0')
        manager.pool = torch.npu.graph_pool_handle()
        manager.graphs = {}
        manager._graphs_captured = False
        manager._capture_descs = {CUDAGraphMode.FULL: [desc]}
        stream = torch.npu.Stream(device=0)
        monkeypatch.setattr(ag, 'graph_capture', lambda **kw: torch.npu.stream(stream))
        monkeypatch.setattr(ag, 'is_global_first_rank', lambda: False)
        offloader = SimpleNamespace(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
        monkeypatch.setattr(ag, 'get_offloader', lambda: offloader)
        monkeypatch.setattr('vllm.v1.worker.gpu.cudagraph_utils.get_offloader', lambda: offloader)
        monkeypatch.setattr(ag, '_prepare_graph_params', lambda descs: None)
        monkeypatch.setattr(ag, 'ensure_weak_ref_graph_params', lambda: None)
        monkeypatch.setattr(ag, 'assert_no_acl_tasks_registered', lambda: None)
        x = torch.arange(8, dtype=torch.float32, device='npu')
        y = torch.empty_like(x)
        torch.npu.synchronize()
        monkeypatch.setattr(cuda, '_device_count_nvml', lambda: pytest.fail('NVML probe'))
        with switch_torch_device():
            manager.capture(lambda desc, warmup: (lambda mode: torch.mul(x, 3, out=y), None))
            for offset in (1, 7, 19):
                x.copy_(torch.arange(8, dtype=torch.float32, device='npu') + offset)
                CudaGraphManager.run_fullgraph(manager, desc)
                torch.npu.synchronize()
                torch.testing.assert_close(y.cpu(), (torch.arange(8) + offset).float() * 3,
                                           rtol=0, atol=0)
        assert torch.cuda is cuda
    finally:
        restore_cuda_attrs(saved)
