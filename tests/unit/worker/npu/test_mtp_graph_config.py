# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP configuration isolation, phase dispatch and producer execution policy."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode as Mode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from omni_npu.worker.npu import model_runner as runner_module
from omni_npu.worker.npu import spec_decode as spec_decode_module
from omni_npu.worker.npu.spec_decode import mtp_speculator as mtp_module


def config(role="kv_consumer", heads=1):
    return NS(
        speculative_config=NS(method="mtp", num_speculative_tokens=7,
            draft_model_config=NS(hf_config=NS(n_predict=heads)),
            uses_dynamic_speculative_decoding=lambda: False,
            use_gemma4_mtp=lambda: False),
        kv_transfer_config=NS(kv_role=role),
        scheduler_config=NS(max_num_seqs=3),
        compilation_config=NS(cudagraph_capture_sizes=[8, 16, 24],
            max_cudagraph_capture_size=24, compile_sizes=[8, 16, 24],
            static_forward_context={}), additional_config={},
    )


@pytest.fixture
def make_speculator(monkeypatch):
    def initialize(self, cfg, device):
        self.vllm_config, self.device = cfg, device
        self.num_speculative_steps = cfg.speculative_config.num_speculative_tokens
        self.draft_tokens = torch.zeros((3, self.num_speculative_steps), dtype=torch.int64)
    monkeypatch.setattr(mtp_module.MTPSpeculator, "__init__", initialize)
    return lambda cfg: mtp_module.NPUMTPSpeculator(cfg, "test-device")


def test_dummy_request_indices_reset_between_proposals(make_speculator, monkeypatch):
    """看护 backport 的 dummy 标记、真实请求恢复以及图 padding 索引清理。"""
    spec = make_speculator(config())
    assert spec._dummy_request_inputs is False
    spec.idx_mapping = torch.zeros(3, dtype=torch.int32)
    spec.temperature = torch.zeros(3)
    spec.seeds = torch.zeros(3, dtype=torch.int64)
    spec.draft_logits = None  # 无 logits buffer 时也必须清理 padding 行。
    indices = torch.tensor([1, 0], dtype=torch.int32)
    temperature, seeds = torch.tensor([0.2, 0.4, 0.6]), torch.arange(3)
    batch, result = object(), object()

    def propose(self, input_batch, *, dummy_run=False, skip_attn_for_dummy_run=False):
        assert input_batch is batch
        assert dummy_run is self._dummy_request_inputs
        assert skip_attn_for_dummy_run is True
        # 仅替代模型执行，使用真实上游 copy 和本仓 override。
        self._copy_request_inputs(2, indices, temperature, seeds)
        return result

    monkeypatch.setattr(mtp_module.MTPSpeculator, "propose", propose)
    ptr = spec.idx_mapping.data_ptr()
    for dummy in (True, False, True):
        spec.idx_mapping.fill_(2)  # 模拟上个 batch 留下的有效索引。
        kwargs = {"dummy_run": True} if dummy else {}
        assert spec.propose(batch, skip_attn_for_dummy_run=True, **kwargs) is result
        assert spec.idx_mapping.tolist() == ([-1, -1, -1] if dummy else [1, 0, -1])
        assert spec.idx_mapping.data_ptr() == ptr
        assert indices.tolist() == [1, 0]
        torch.testing.assert_close(spec.temperature, temperature)
        torch.testing.assert_close(spec.seeds, seeds)


def manager(cfg, q, mode):
    obj = object.__new__(CudaGraphManager)
    obj.vllm_config = cfg
    obj.compilation_config = cfg.compilation_config
    obj.max_num_reqs = cfg.scheduler_config.max_num_seqs
    obj.decode_query_len, obj.cudagraph_mode = q, mode
    obj.lora_capture_cases = [0]
    obj._lora_dispatch_map, obj._max_lora_case = {}, 0
    obj._candidates, obj._capture_descs = {}, {}
    obj._init_candidates()
    obj._graphs_captured = True
    return obj


@pytest.mark.parametrize("capture_sizes,compile_sizes,decode_sizes", [
    ([8, 16, 24], [8, 16, 24], [1, 2, 3]),
    ([1, 2, 4, 8, 16, 24, 32, 40, 48], [], [1, 2, 3, 4, 5, 6]),
    ([8, 16, 24], ["cudagraph_capture_sizes", 32], [1, 2, 3]),
])
def test_three_stage_configs_and_dispatch(
    make_speculator, monkeypatch, capture_sizes, compile_sizes, decode_sizes,
):
    cfg = config()
    cfg.compilation_config.cudagraph_capture_sizes = capture_sizes
    cfg.compilation_config.max_cudagraph_capture_size = max(capture_sizes)
    cfg.compilation_config.compile_sizes = compile_sizes
    spec = make_speculator(cfg)
    monkeypatch.setattr(mtp_module, "PrefillSpeculatorCudaGraphManager",
                        lambda cfg, device, mode, q: manager(cfg, q, mode))
    monkeypatch.setattr(mtp_module, "DecodeSpeculatorCudaGraphManager",
                        lambda cfg, device, mode, decode_query_len: manager(cfg, decode_query_len, mode))
    spec.init_cudagraph_manager(Mode.FULL)
    assert cfg.compilation_config.compile_sizes == compile_sizes
    assert cfg.compilation_config.cudagraph_capture_sizes == capture_sizes
    expected_compile = (set(compile_sizes) - {"cudagraph_capture_sizes"}) | set(capture_sizes) | set(decode_sizes)
    assert spec.vllm_config.compilation_config.compile_sizes == sorted(expected_compile)
    assert spec.vllm_config.compilation_config.static_forward_context is cfg.compilation_config.static_forward_context
    # 配置桶按 ceil(target_tokens / 8) 转换；max_num_seqs 只限制实际捕获。
    assert spec.decode_config.compilation_config.cudagraph_capture_sizes == decode_sizes
    assert spec.decode_config.compilation_config.max_cudagraph_capture_size == max(decode_sizes)
    total = 0
    for mgr, q, sizes in (
        (manager(cfg, 8, Mode.FULL), 8, capture_sizes),
        (spec.prefill_cudagraph_manager, 8, capture_sizes),
        (spec.decode_cudagraph_manager, 1, [1, 2, 3]),
    ):
        captured = sorted(d.num_tokens for ds in mgr._capture_descs.values() for d in ds)
        assert captured == sizes
        total += len(captured)
        for batch in (1, 2, 3):
            desc = mgr.dispatch(batch, batch*q, q, 0)
            assert desc.cg_mode == Mode.FULL and desc.num_tokens == batch*q
    assert total == 2 * len(capture_sizes) + 3


@pytest.mark.parametrize("sizes", [None, []])
def test_empty_capture_config_initializes_decode_config(make_speculator, sizes):
    cfg = config()
    cfg.compilation_config.cudagraph_capture_sizes = sizes
    spec = make_speculator(cfg)
    assert spec.decode_config.compilation_config.cudagraph_capture_sizes == sizes


@pytest.mark.parametrize("role,heads,single", [
    ("kv_producer", 1, True), ("kv_consumer", 1, False),
    (None, 1, False), ("kv_producer", 3, False),
])
def test_only_single_head_producer_skips_later_steps(make_speculator, monkeypatch, role, heads, single):
    cfg = config(role, heads)
    if role is None:
        cfg.kv_transfer_config = None
    spec = make_speculator(cfg)
    spec.draft_tokens[:, 0] = torch.tensor([11, 22, 33])
    # 上游返回 None，通过原地写入 draft_tokens 交付后续 proposal。
    def write_later_steps(num_reqs, skip_attn, batch_desc, num_tokens_across_dp):
        spec.draft_tokens[:num_reqs, 1:] = 77

    upstream = Mock(side_effect=write_later_steps)
    monkeypatch.setattr(mtp_module.MTPSpeculator, "_multi_step_decode", upstream)
    result = spec._multi_step_decode(2, False, "desc", None)
    assert result is None
    assert spec.prefill_single_step is single
    assert spec.num_speculative_steps == cfg.speculative_config.num_speculative_tokens == 7
    if single:
        upstream.assert_not_called()
        assert spec.draft_tokens[:2].tolist() == [[11]*7, [22]*7]
        assert spec.draft_tokens[2].tolist() == [33, 0, 0, 0, 0, 0, 0]
        assert cfg.compilation_config.compile_sizes == [8, 16, 24]
    else:
        upstream.assert_called_once_with(2, False, "desc", None)
        assert spec.draft_tokens[:2].tolist() == [[11] + [77]*6, [22] + [77]*6]
        assert spec.draft_tokens[2].tolist() == [33, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("case", ["no_speculator", "non_last_pp_rank"])
def test_runner_keeps_speculator_none_when_upstream_does(monkeypatch, case):
    from omni_npu.compilation import npugraph_ex_config
    cfg = config()
    if case == "no_speculator":
        cfg.speculative_config = None
    def upstream_init(self, config, device):
        self.vllm_config, self.device = config, device
        self.speculative_config = config.speculative_config
        self.is_last_pp_rank = case != "non_last_pp_rank"
        self.speculator = None
    factory = Mock()
    monkeypatch.setattr(runner_module, "init_speculator", factory)
    monkeypatch.setattr(runner_module, "install_torch_cuda_aliases", lambda: None)
    monkeypatch.setattr(npugraph_ex_config, "init_aclgraph_config", lambda config: None)
    monkeypatch.setattr(runner_module.GPUModelRunner, "__init__", upstream_init)
    runner = runner_module.NPUModelRunnerV2(cfg, "test-device")
    assert runner.speculator is None
    factory.assert_not_called()


@pytest.mark.parametrize("method,gemma4,custom", [
    ("mtp", False, True), ("mtp", True, False),
    ("eagle3", False, False),
])
def test_factory_routes_only_non_gemma4_mtp_to_npu(monkeypatch, method, gemma4, custom):
    cfg = config()
    cfg.speculative_config.method = method
    cfg.speculative_config.use_gemma4_mtp = Mock(return_value=gemma4)
    npu, upstream = Mock(), Mock()
    monkeypatch.setattr(mtp_module, "NPUMTPSpeculator", npu)
    monkeypatch.setattr(spec_decode_module, "vllm_init_speculator", upstream)
    result = spec_decode_module.init_speculator(cfg, "test-device")
    chosen, unused = (npu, upstream) if custom else (upstream, npu)
    chosen.assert_called_once_with(cfg, "test-device")
    unused.assert_not_called()
    assert result is chosen.return_value
    if method != "mtp":
        cfg.speculative_config.use_gemma4_mtp.assert_not_called()
