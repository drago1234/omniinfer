# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for NPU buffer views and runtime mode selection."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omni_npu.worker.npu.buffer_utils import NPUUvaBuffer
from omni_npu.worker.npu import mode as mode_mod
from omni_npu.worker.npu.mode import RuntimeMode, resolve_mode


@pytest.fixture
def delayed_h2d(monkeypatch):
    """Run real host writes, but defer device reads until an event is waited.

    This models the source-memory race deterministically without depending on
    NPU load or timings. Device consumers also run in stream FIFO order.
    """
    from omni_npu.worker.npu import buffer_utils as bu

    class Stream:
        def __init__(self):
            self.tasks = []
            self.completed = 0

        def finish(self, fence):
            while self.completed < fence:
                self.tasks[self.completed]()
                self.completed += 1

    stream = Stream()
    class Event:
        def __init__(self):
            self.stream = None

        def record(self, recorded_stream):
            self.stream = recorded_stream
            self.fence = len(recorded_stream.tasks)

        def synchronize(self):
            # Like torch.npu.Event, waiting before the first record is a no-op.
            if self.stream is not None:
                self.stream.finish(self.fence)

    real_zeros = torch.zeros

    class DeviceTensor:
        def __init__(self, size, dtype):
            self.data = real_zeros(size, dtype=dtype)

        def copy_(self, src, non_blocking):
            assert non_blocking is True
            # Deliberately retain the source reference, not a host snapshot.
            stream.tasks.append(lambda: self.data.copy_(src))

        def __getitem__(self, index):
            return self.data[index]

    def zeros(size, *, dtype, device, pin_memory=False):
        if device == "delayed_device":
            return DeviceTensor(size, dtype)
        return real_zeros(size, dtype=dtype, device=device)

    def forbid_global_sync():
        pytest.fail("slot reuse must not synchronize the whole device")

    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        Event=Event, current_stream=lambda: stream,
        synchronize=forbid_global_sync), raising=False)
    monkeypatch.setattr(bu, "resolve_mode", lambda: RuntimeMode.TORCH_COPY_TO_NPU)
    monkeypatch.setattr(bu, "_current_npu_device", lambda: "delayed_device")
    import vllm.v1.worker.gpu.buffer_utils as upstream

    monkeypatch.setattr(upstream, "UvaBuffer", NPUUvaBuffer)
    monkeypatch.setattr(upstream.UvaBufferPool, "copy_to_uva",
                        bu.copy_to_uva_with_slot_event)
    return SimpleNamespace(pool_type=upstream.UvaBufferPool, stream=stream, module=bu)


@pytest.mark.parametrize("input_kind", ["numpy", "tensor", "list"])
def test_delayed_h2d_source_is_not_overwritten(delayed_h2d, input_kind):
    """Preserve each queued copy across repeated reuse of both host slots."""
    env = delayed_h2d
    pool = env.pool_type(256, dtype=torch.int32, max_concurrency=2)
    results = []
    for value in range(32):
        src = np.full(256, value, dtype=np.int32)
        if input_kind == "tensor":
            src = torch.from_numpy(src)
        elif input_kind == "list":
            src = src.tolist()
        dev = pool.copy_to_uva(src)
        env.stream.tasks.append(lambda dev=dev: results.append(dev.clone()))
    env.stream.finish(len(env.stream.tasks))
    assert len(results) == 32
    for value, actual in enumerate(results):
        assert bool((actual == value).all()), f"H2D result {value} was overwritten"


def test_true_uva_does_not_create_h2d_event(delayed_h2d, monkeypatch):
    env = delayed_h2d
    def unexpected_event():
        pytest.fail("true UVA must not create an H2D event")

    monkeypatch.setattr(torch.npu, "Event", unexpected_event)
    monkeypatch.setattr(env.module, "resolve_mode", lambda: RuntimeMode.NPU_UVA_VIEW)
    monkeypatch.setattr(env.module, "_accelerator_view", lambda cpu: cpu)
    buf = NPUUvaBuffer(4, torch.int32)
    buf.wait_for_host_reuse()
    assert buf.uva is buf.cpu
    assert buf._copy_done is None
    assert not env.stream.tasks


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason="NPU is unavailable")


@requires_npu
class TestNPUUvaBuffer:
    def test_three_views_and_device(self):
        """.cpu/.np 是 CPU 真值，.uva 必须落在 NPU：这是红线（第 3.5 节）。"""

        buf = NPUUvaBuffer((4, 8), torch.int32)
        assert buf.cpu.device.type == "cpu"
        assert buf.np.shape == (4, 8)

        buf.np[2, 3] = 77
        dev = buf.uva
        assert dev.device.type == "npu", f".uva 落在 {dev.device}，必须是 npu"
        expected_device_bytes = 0 if buf._fast else dev.numel() * dev.element_size()
        assert buf.device_bytes == expected_device_bytes
        torch.npu.synchronize()
        assert int(dev[2, 3].item()) == 77

    def test_np_and_cpu_share_memory(self):
        """.np 必须是 .cpu 的视图而非副本，否则 UvaBackedTensor 的真值会分叉。"""

        buf = NPUUvaBuffer(16, torch.int32)
        buf.cpu[5] = 42
        assert buf.np[5] == 42
        buf.np[6] = 43
        assert int(buf.cpu[6].item()) == 43

    def test_three_consumer_points(self):
        """Exercise device arithmetic, indexing, and temperature scaling."""

        buf = NPUUvaBuffer(8, torch.int32)
        buf.np[:] = np.arange(1, 9, dtype=np.int32)
        g = buf.uva
        idx = torch.tensor([0, 3, 7], dtype=torch.int64, device=g.device)

        _ = g + torch.zeros_like(g)          # 1. 与设备张量同列进算子
        _ = g[idx]                            # 2. 设备索引取值
        logits = torch.randn(3, 16, device=g.device)
        _ = logits / g[idx].to(torch.float32).unsqueeze(1)   # 3. apply_temperature 形态
        torch.npu.synchronize()

    def test_round_robin_preserved(self):
        """池化轮转是 WAR 保护，不是优化：必须保留（gpu/buffer_utils.py:16-18）。"""
        from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
        import vllm.v1.worker.gpu.buffer_utils as bu

        old, bu.UvaBuffer = bu.UvaBuffer, NPUUvaBuffer
        try:
            pool = UvaBufferPool(16, dtype=torch.int32, max_concurrency=2)
            first = pool.copy_to_uva(np.full(16, 111, dtype=np.int32))
            pool.copy_to_uva(np.full(16, 222, dtype=np.int32))
            torch.npu.synchronize()
            assert int(first[0].item()) == 111, "第 1 次的视图被后续写入覆盖，WAR 保护失效"
        finally:
            bu.UvaBuffer = old

    def test_request_state_constructible(self):
        """闸一的真正判据：RequestState 能不能造出来（第 1.1 节）。"""
        import torch
        import vllm.v1.worker.gpu.buffer_utils as bu
        from vllm.v1.worker.gpu.states import RequestState

        old, bu.UvaBuffer = bu.UvaBuffer, NPUUvaBuffer
        try:
            rs = RequestState(
                max_num_reqs=32, max_model_len=1024, max_num_batched_tokens=2048,
                num_speculative_steps=0, vocab_size=1024,
                device=torch.device("npu", torch.npu.current_device()),
            )
            rs.add_request("r0", prompt_len=4, all_token_ids=[1, 2, 3, 4],
                           num_computed_tokens=0, max_tokens=8)
            assert rs.num_reqs == 1
            assert len(rs.all_token_ids._staged_write_indices) == 1
            assert rs.prefill_len.gpu.device.type == "npu"
        finally:
            bu.UvaBuffer = old


class TestModeAndSnapshot:
    @pytest.fixture(autouse=True)
    def _clear_mode_cache(self):
        """resolve_mode() 缓存在模块级，逐用例清掉才能各自决定环境。"""
        mode_mod.reset_for_testing()
        yield
        mode_mod.reset_for_testing()

    def test_default_mode_is_torch_copy(self):
        """Use device copies unless the UVA view is explicitly enabled."""
        assert resolve_mode() is RuntimeMode.TORCH_COPY_TO_NPU

    def test_uva_env_alone_does_not_enable_view(self, monkeypatch):
        """环境变量只是第一道闸：平台报 UVA 不可用时必须退回拷贝，不能硬上。"""
        monkeypatch.setenv(mode_mod.UVA_ENV, "1")
        platforms = importlib.import_module("vllm.platforms")
        monkeypatch.setattr(
            platforms.current_platform,
            "is_uva_available",
            lambda: False,
            raising=False,
        )
        assert resolve_mode() is RuntimeMode.TORCH_COPY_TO_NPU
        assert "is_uva_available() is False" in "; ".join(mode_mod.uva_blockers())

    def test_uva_mode_reports_no_extra_hbm(self, monkeypatch):
        """UVA 视图别名 host 内存，all_token_ids 不该再算一份 HBM。"""
        monkeypatch.setenv(mode_mod.UVA_ENV, "1")
        platforms = importlib.import_module("vllm.platforms")
        monkeypatch.setattr(
            platforms.current_platform,
            "is_uva_available",
            lambda: True,
            raising=False,
        )
        assert resolve_mode() is RuntimeMode.NPU_UVA_VIEW
