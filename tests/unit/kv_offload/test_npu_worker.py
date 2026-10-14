# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for NPU offload worker helpers and handler paths."""

import sys
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from omni_npu.v1.kv_offload.cpu import npu_worker as worker_mod
from omni_npu.v1.kv_offload.cpu.npu_worker import (
    NPUCPUOffloadingWorker,
    NpuSingleDirectionOffloadingHandler,
    Transfer,
    _block_slice,
    _get_swap_blocks_batch,
    pin_memory_region,
)


def test_block_slice_full_and_partial():
    # CI default device is NPU; keep helpers on CPU. int8 arange is not portable.
    t = torch.arange(12, dtype=torch.int32, device="cpu").to(torch.int8).reshape(3, 4)
    assert torch.equal(_block_slice(t, 1, 0, 4), t[1])
    assert torch.equal(_block_slice(t, 1, 1, 2), t[1, 1:3])


def test_group_block_indices_store_rotation_uses_local_world():
    handler = NpuSingleDirectionOffloadingHandler.__new__(
        NpuSingleDirectionOffloadingHandler
    )
    handler.npu_to_cpu = True
    handler._rotate_store_writers = True
    handler.src_block_size_factor = 1
    handler.dst_block_size_factor = 1
    # Gate uses local_world_size>1; mask is local world rank, not TP.
    handler._local_world_size = 4
    handler._local_world_rank = 1

    src = np.asarray([10, 11, 12, 13], dtype=np.int64)
    dst = np.asarray([0, 1, 2, 3], dtype=np.int64)
    _, _, src_blocks, dst_blocks = handler._group_block_indices(src, dst, 4, 0, 0)
    np.testing.assert_array_equal(src_blocks, [11])
    np.testing.assert_array_equal(dst_blocks, [1])

    handler._rotate_store_writers = False
    _, _, _, dst_blocks = handler._group_block_indices(src, dst, 4, 0, 0)
    np.testing.assert_array_equal(dst_blocks, [0, 1, 2, 3])

    handler._rotate_store_writers = True
    handler._local_world_size = 1
    _, _, _, dst_blocks = handler._group_block_indices(src, dst, 4, 0, 0)
    np.testing.assert_array_equal(dst_blocks, [0, 1, 2, 3])


def _skip_rotated_store_block(handler, dst_block: int) -> bool:
    """True when store rotation assigns this CPU block to another rank."""
    if not handler.npu_to_cpu or not handler._rotate_store_writers:
        return False
    if handler._local_world_size <= 1:
        return False
    return int(dst_block) % int(handler._local_world_size) != int(
        handler._local_world_rank
    )


def _ref_plan_group_copies(
    handler, group_src, group_dst, group_size, refs, src_skip, dst_skip
):
    """Old per-op Python loop; used as oracle for the numpy planner."""
    src_ptrs = []
    dst_ptrs = []
    sizes = []
    for data_ref in refs:
        src_t = handler.src_tensors[data_ref.tensor_idx]
        dst_t = handler.dst_tensors[data_ref.tensor_idx]
        page = data_ref.page_size_bytes
        for i in range(group_size):
            src_sub = src_skip + i
            dst_sub = dst_skip + i
            src_block = int(group_src[src_sub // handler.src_block_size_factor])
            dst_block = int(group_dst[dst_sub // handler.dst_block_size_factor])
            if _skip_rotated_store_block(handler, dst_block):
                continue
            src_page = (src_sub % handler.src_block_size_factor) * page
            dst_page = (dst_sub % handler.dst_block_size_factor) * page
            src_ptrs.append(
                int(src_t.data_ptr()) + src_block * int(src_t.stride(0)) + src_page
            )
            dst_ptrs.append(
                int(dst_t.data_ptr()) + dst_block * int(dst_t.stride(0)) + dst_page
            )
            sizes.append(page)
    return (
        np.asarray(src_ptrs, dtype=np.int64),
        np.asarray(dst_ptrs, dtype=np.int64),
        np.asarray(sizes, dtype=np.int64),
    )


def _fill_group_plan(handler, group_src, group_dst, group_size, refs, src_skip, dst_skip):
    group_src = np.asarray(group_src, dtype=np.int64)
    group_dst = np.asarray(group_dst, dtype=np.int64)
    max_ops = int(group_size) * len(refs)
    if max_ops <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    src_ptrs = np.empty(max_ops, dtype=np.int64)
    dst_ptrs = np.empty(max_ops, dtype=np.int64)
    sizes = np.empty(max_ops, dtype=np.int64)
    n = handler._fill_group_descriptors(
        src_ptrs,
        dst_ptrs,
        sizes,
        0,
        group_src,
        group_dst,
        group_size,
        refs,
        src_skip,
        dst_skip,
    )
    return src_ptrs[:n], dst_ptrs[:n], sizes[:n]


def _assert_plan_matches(handler, group_src, group_dst, group_size, refs, src_skip, dst_skip):
    got = _fill_group_plan(
        handler, group_src, group_dst, group_size, refs, src_skip, dst_skip
    )
    exp = _ref_plan_group_copies(
        handler, group_src, group_dst, group_size, refs, src_skip, dst_skip
    )
    np.testing.assert_array_equal(got[0], exp[0])
    np.testing.assert_array_equal(got[1], exp[1])
    np.testing.assert_array_equal(got[2], exp[2])


def test_plan_group_copies_matches_python_oracle():
    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    refs = handler.kv_cache_groups_data_refs[0]
    _assert_plan_matches(handler, [0, 1, 2], [4, 5, 6], 3, refs, 0, 0)

    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=2)
    refs = handler.kv_cache_groups_data_refs[0]
    # dst packed by factor=2; skip=1 shifts the first logical page.
    _assert_plan_matches(handler, [0, 1, 2, 3], [8, 9, 10], 3, refs, 0, 1)

    handler, npu, cpu = _make_handler(npu_to_cpu=False, factor=2)
    refs = handler.kv_cache_groups_data_refs[0]
    _assert_plan_matches(handler, [2, 3], [0, 1, 2, 3], 3, refs, 1, 0)


def test_plan_group_copies_multi_ref_and_rotation():
    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    extra_npu = torch.zeros((4, 4), dtype=torch.int8, device="cpu")
    extra_cpu = torch.zeros((4, 4), dtype=torch.int8, device="cpu")
    handler.src_tensors = [npu, extra_npu]
    handler.dst_tensors = [cpu, extra_cpu]
    refs = [
        SimpleNamespace(tensor_idx=0, page_size_bytes=8),
        SimpleNamespace(tensor_idx=1, page_size_bytes=4),
    ]
    _assert_plan_matches(handler, [0, 1], [2, 3], 2, refs, 0, 0)

    handler._rotate_store_writers = True
    handler._local_world_size = 2
    handler._local_world_rank = 0
    _assert_plan_matches(handler, [0, 1], [2, 3], 2, refs, 0, 0)

    empty = _fill_group_plan(
        handler,
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        0,
        refs,
        0,
        0,
    )
    assert empty[0].size == 0
    assert empty[1].size == 0
    assert empty[2].size == 0


def _two_group_plan_setup():
    """Two KV groups plus src/dst block ids shared by planner tests."""
    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    extra_npu = torch.zeros((4, 8), dtype=torch.int8, device="cpu")
    extra_cpu = torch.zeros((4, 8), dtype=torch.int8, device="cpu")
    handler.src_tensors = [npu, extra_npu]
    handler.dst_tensors = [cpu, extra_cpu]
    handler.kv_cache_groups_data_refs = [
        [SimpleNamespace(tensor_idx=0, page_size_bytes=8)],
        [SimpleNamespace(tensor_idx=1, page_size_bytes=8)],
    ]
    src = np.asarray([0, 1, 2], dtype=np.int64)
    dst = np.asarray([0, 1, 2], dtype=np.int64)
    return handler, src, dst


def test_plan_transfer_descriptors_two_groups():
    handler, src, dst = _two_group_plan_setup()
    src_np, dst_np, sz_np = handler._plan_transfer_descriptors(
        src, dst, [2, 1], [0, 0], 3, 3
    )
    g0 = _fill_group_plan(handler, src[:2], dst[:2], 2, handler.kv_cache_groups_data_refs[0], 0, 0)
    g1 = _fill_group_plan(handler, src[2:], dst[2:], 1, handler.kv_cache_groups_data_refs[1], 0, 0)
    np.testing.assert_array_equal(src_np, np.concatenate([g0[0], g1[0]]))
    np.testing.assert_array_equal(dst_np, np.concatenate([g0[1], g1[1]]))
    np.testing.assert_array_equal(sz_np, np.concatenate([g0[2], g1[2]]))


def _patch_local_world(monkeypatch, *, local_rank=0, world_size=1, node_count=1):
    monkeypatch.setattr(
        worker_mod,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=local_rank, world_size=world_size),
    )
    monkeypatch.setattr(worker_mod, "get_node_count", lambda: node_count)


def test_handler_init_reads_local_world_from_world_group(monkeypatch):
    _patch_local_world(monkeypatch, local_rank=3, world_size=16, node_count=2)
    npu = SimpleNamespace(
        dtype=torch.int8,
        ndim=2,
        device=SimpleNamespace(type="npu", index=0),
        shape=(4, 8),
    )
    cpu = SimpleNamespace(
        dtype=torch.int8,
        ndim=2,
        device=SimpleNamespace(type="cpu"),
        shape=(4, 8),
    )
    with patch.object(torch.npu, "Stream", return_value=MagicMock()), patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ):
        handler = NpuSingleDirectionOffloadingHandler(
            npu_tensors=[npu],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )
    assert handler._local_world_rank == 3
    assert handler._local_world_size == 8


def test_handler_init_rejects_cpu_npu_mismatch():
    cpu = torch.zeros((2, 8), dtype=torch.int8, device="cpu")
    with pytest.raises(ValueError, match="non-empty and same length"):
        NpuSingleDirectionOffloadingHandler(
            npu_tensors=[],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )
    with pytest.raises(ValueError, match="npu_tensor must be 2-D int8"):
        NpuSingleDirectionOffloadingHandler(
            npu_tensors=[torch.zeros(8, dtype=torch.int8, device="cpu")],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )
    with pytest.raises(ValueError, match="device must be npu"):
        NpuSingleDirectionOffloadingHandler(
            npu_tensors=[torch.zeros((2, 8), dtype=torch.int8, device="cpu")],
            cpu_tensors=[cpu],
            block_size_factor=1,
            kv_cache_groups_data_refs=[],
            npu_to_cpu=True,
        )


def test_get_swap_blocks_batch_cache_and_import_paths(monkeypatch):
    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", False)
    assert _get_swap_blocks_batch() is None

    cached = object()
    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", cached)
    assert _get_swap_blocks_batch() is cached

    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", None)
    fake_ext = SimpleNamespace(
        swap_blocks_batch=lambda *a, **k: None,
        host_location="host",
        cann_memcpy_batch=True,
    )
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._swap_blocks_batch": fake_ext},
    ):
        assert _get_swap_blocks_batch() is fake_ext.swap_blocks_batch

    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", None)
    stale = SimpleNamespace(
        swap_blocks_batch=lambda *a, **k: "fn",
        host_location="x",
        cann_memcpy_batch=False,
    )
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._swap_blocks_batch": stale},
    ):
        assert _get_swap_blocks_batch() is stale.swap_blocks_batch

    monkeypatch.setattr(worker_mod, "_SWAP_BLOCKS_BATCH", None)
    broken = SimpleNamespace()  # missing swap_blocks_batch → AttributeError
    with patch.dict(
        "sys.modules",
        {"omni_npu.v1.kv_offload.cpu._swap_blocks_batch": broken},
    ):
        assert _get_swap_blocks_batch() is None
        assert worker_mod._SWAP_BLOCKS_BATCH is False


def test_pin_memory_region_paths():
    region = MagicMock()
    region.is_pinned = True
    pin_memory_region(region)

    region = MagicMock()
    region.is_pinned = False
    region._base = torch.zeros(4096, dtype=torch.int8, device="cpu")
    with patch.object(region._base, "data_ptr", return_value=1), patch(
        "omni_npu.v1.kv_offload.cpu.host_register.pin_host_mmap"
    ) as pin:
        pin_memory_region(region)
        pin.assert_not_called()

    region = MagicMock()
    region.is_pinned = False
    region._base = torch.zeros(4096, dtype=torch.int8, device="cpu")
    region._hugepage = False
    aligned = (region._base.data_ptr() // 4096) * 4096
    with patch.object(region._base, "data_ptr", return_value=aligned), patch(
        "omni_npu.v1.kv_offload.cpu.host_register.pin_host_mmap",
        side_effect=RuntimeError("pin fail"),
    ), patch.object(
        torch.npu, "current_device", side_effect=RuntimeError("no npu")
    ):
        pin_memory_region(region)
        assert region.is_pinned is False

    region = MagicMock()
    region.is_pinned = False
    region._base = torch.zeros(4096, dtype=torch.int8, device="cpu")
    region._hugepage = True
    aligned = (region._base.data_ptr() // 4096) * 4096
    with patch.object(region._base, "data_ptr", return_value=aligned), patch(
        "omni_npu.v1.kv_offload.cpu.host_register.pin_host_mmap"
    ) as pin, patch.object(torch.npu, "current_device", return_value=0):
        pin_memory_region(region)
        pin.assert_called_once()
        assert region.is_pinned is True
        assert region.npu_base is None

    region = MagicMock()
    region.is_pinned = False
    import builtins

    real_import = builtins.__import__

    def selective_import(
        name, global_ns=None, local_ns=None, fromlist=(), level=0
    ):
        if name == "omni_npu.v1.kv_offload.cpu.host_register" or name.endswith(
            ".host_register"
        ):
            raise ImportError("unavailable")
        return real_import(name, global_ns, local_ns, fromlist, level)

    with patch("builtins.__import__", side_effect=selective_import):
        pin_memory_region(region)


def _make_handler(
    *,
    npu_to_cpu=True,
    factor=1,
    rotate=False,
    local_world_size=1,
    local_world_rank=0,
):
    npu = torch.zeros((4, 8), dtype=torch.int8, device="cpu")
    cpu = torch.zeros((4, 8 * factor), dtype=torch.int8, device="cpu")
    # Bypass device checks by constructing via __new__ after validating shapes.
    handler = NpuSingleDirectionOffloadingHandler.__new__(
        NpuSingleDirectionOffloadingHandler
    )
    handler.src_tensors = [npu] if npu_to_cpu else [cpu]
    handler.dst_tensors = [cpu] if npu_to_cpu else [npu]
    handler.npu_to_cpu = npu_to_cpu
    handler.kv_cache_groups_data_refs = [
        [SimpleNamespace(tensor_idx=0, page_size_bytes=8)]
    ]
    handler.src_block_size_factor = 1 if npu_to_cpu else factor
    handler.dst_block_size_factor = factor if npu_to_cpu else 1
    handler._mmap_region = MagicMock()
    handler._rotate_store_writers = rotate
    handler._local_world_rank = local_world_rank
    handler._local_world_size = local_world_size
    handler._device_index = 0
    handler._transfer_events = {}
    handler._transfers = deque()
    handler._transfers_by_id = {}
    handler._stream = MagicMock(name="stream")
    handler._event_pool = []
    handler._swap_blocks_batch = None
    handler._batch_direction = 1 if npu_to_cpu else 0
    handler._device_index = 0
    handler._sharded_executor = None
    handler.enable_sharded_load = False
    handler._shard_refusal_logged = False
    handler._init_async_submit_state()
    return handler, npu, cpu


class _FakeBlockSpec:
    def __init__(self, block_ids, group_sizes=None, block_indices=None, gpu=False):
        self.block_ids = np.asarray(block_ids, dtype=np.int64)
        self.group_sizes = group_sizes
        self.block_indices = block_indices
        if gpu:
            # isinstance checks against real classes — patch later
            pass


def _concrete_specs():
    """Concrete subclasses — abstract LoadStoreSpec cannot be __new__'d."""
    from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, GPULoadStoreSpec

    class ConcreteGPU(GPULoadStoreSpec):
        @property
        def medium(self):
            return "gpu"

    class ConcreteCPU(BlockIDsLoadStoreSpec):
        @property
        def medium(self):
            return "cpu"

    return ConcreteGPU, ConcreteCPU


def test_handler_transfer_get_finished_wait_shutdown():
    from vllm.v1.kv_offload.base import TransferResult

    handler, npu, cpu = _make_handler(npu_to_cpu=True, factor=1)
    ConcreteGPU, ConcreteCPU = _concrete_specs()

    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0, 1], dtype=np.int64)
    src.group_sizes = [2]
    src.block_indices = [0]

    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0, 1], dtype=np.int64)

    end_ev = MagicMock()
    end_ev.query.return_value = True
    end_ev.elapsed_time.return_value = 5.0  # milliseconds
    wait_event = MagicMock(name="compute_done")
    with patch.object(torch.npu, "Event", return_value=end_ev), patch.object(
        torch.npu, "set_device"
    ), patch.object(torch.npu, "set_stream"), patch.object(
        torch.npu, "current_stream", return_value=MagicMock()
    ):
        assert handler.transfer_async(7, src, dst, wait_event=wait_event) is True

    assert 7 in handler._transfers_by_id
    results = handler.get_finished()
    assert len(results) == 1
    assert isinstance(results[0], TransferResult)
    assert results[0].job_id == 7
    # OffloadingConnectorWorker only records Prometheus stats when both
    # transfer_size and transfer_time are set.
    assert results[0].transfer_size == 16  # 2 blocks * 8-byte pages
    assert results[0].transfer_time == pytest.approx(0.005)

    # Wait on missing + present job
    end2 = MagicMock()
    transfer = Transfer(
        job_id=9,
        stream=handler._stream,
        end_event=end2,
    )
    handler._transfers_by_id[9] = transfer
    handler.wait({9, 99})
    end2.synchronize.assert_called_once()

    # Shutdown with pending + mmap cleanup
    end3 = MagicMock()
    pending = Transfer(
        job_id=10,
        stream=handler._stream,
        end_event=end3,
    )
    handler._transfers.append(pending)
    handler._transfers_by_id[10] = pending
    mmap = handler._mmap_region
    handler.shutdown()
    end3.synchronize.assert_called()
    mmap.cleanup.assert_called_once()
    assert handler._mmap_region is None


def test_transfer_time_falls_back_to_monotonic_when_event_fails():
    start = MagicMock()
    start.elapsed_time.side_effect = RuntimeError("no timing")
    transfer = Transfer(
        job_id=1,
        stream=MagicMock(),
        end_event=MagicMock(),
        start_event=start,
        start_mono=time.monotonic() - 0.02,
    )
    elapsed = NpuSingleDirectionOffloadingHandler._transfer_time_s(transfer)
    assert elapsed >= 0.01
    assert NpuSingleDirectionOffloadingHandler._transfer_time_s(
        Transfer(job_id=2, stream=MagicMock(), end_event=MagicMock())
    ) == 0.0


def test_recycle_transfer_reports_size_and_time_for_prometheus():
    handler = NpuSingleDirectionOffloadingHandler.__new__(
        NpuSingleDirectionOffloadingHandler
    )
    handler._event_pool = []
    handler._transfer_events = {3: object()}
    handler._transfers_by_id = {3: object()}
    start = MagicMock()
    start.elapsed_time.return_value = 8.0  # milliseconds
    end = MagicMock()
    result = handler._recycle_transfer(
        Transfer(
            job_id=3,
            stream=MagicMock(),
            end_event=end,
            start_event=start,
            num_bytes=4096,
        )
    )
    assert result.job_id == 3
    assert result.success is True
    assert result.transfer_size == 4096
    assert result.transfer_time == pytest.approx(0.008)
    start.elapsed_time.assert_called_once_with(end)
    assert handler._event_pool == [end, start]
    assert 3 not in handler._transfer_events
    assert 3 not in handler._transfers_by_id


def test_handler_transfer_batch_path_and_rotation_skip():
    # local_world size 2 / rank 1 keeps odd dst blocks.
    handler, npu, cpu = _make_handler(
        npu_to_cpu=True,
        factor=1,
        rotate=True,
        local_world_size=2,
        local_world_rank=1,
    )
    swap_fn = MagicMock()
    handler._swap_blocks_batch = swap_fn
    # reuse events from pool
    pooled_start = MagicMock()
    pooled_end = MagicMock()
    handler._event_pool = [pooled_start, pooled_end]

    ConcreteGPU, ConcreteCPU = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)
    src.group_sizes = [4]
    src.block_indices = [0]
    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)

    wait_event = MagicMock(name="compute_done")
    with patch.object(torch.npu, "set_device"), patch.object(
        torch.npu, "set_stream"
    ), patch.object(torch.npu, "current_stream", return_value=MagicMock()):
        assert handler.transfer_async(1, src, dst, wait_event=wait_event) is True
    assert swap_fn.called
    batched = swap_fn.call_args[0]
    assert int(batched[2].numel()) == 2
    assert int(batched[2][0]) == 8
    assert int(batched[2][1]) == 8


def test_transfer_async_avoids_npu_stream_queries():
    """The submit thread must not construct a Stream in a forked worker."""
    handler, _, _ = _make_handler(npu_to_cpu=True, factor=1)
    handler._device_index = 7
    ConcreteGPU, ConcreteCPU = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0], dtype=np.int64)
    src.group_sizes = [1]
    src.block_indices = [0]
    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0], dtype=np.int64)

    wait_event = MagicMock(name="compute_done")
    end_ev = MagicMock()
    end_ev.query.return_value = True

    def _forbid_stream_ctx(*_args, **_kwargs):
        raise AssertionError("torch.npu.stream() must not be used")

    def _forbid_current_stream(*_args, **_kwargs):
        raise AssertionError("torch.npu.current_stream() must not be used")

    with patch.object(torch.npu, "Event", return_value=end_ev), patch.object(
        torch.npu, "stream", side_effect=_forbid_stream_ctx
    ), patch.object(torch.npu, "set_device") as set_device, patch.object(
        torch.npu, "set_stream"
    ) as set_stream, patch.object(
        torch.npu, "current_stream", side_effect=_forbid_current_stream
    ):
        assert handler.transfer_async(3, src, dst, wait_event=wait_event) is True

    set_device.assert_called_with(7)
    handler._stream.wait_event.assert_called_once_with(wait_event)
    set_stream.assert_called_once_with(handler._stream)
    assert handler._transfers_by_id[3].wait_event is wait_event


def test_current_npu_stream_uses_explicit_device():
    stream = MagicMock(name="compute_stream")
    with patch.object(
        worker_mod.torch_npu._C,
        "_npu_getCurrentStream",
        return_value=(17, 7, 20),
    ) as get_raw_stream, patch.object(
        worker_mod.torch_npu.npu, "Stream", return_value=stream
    ) as stream_cls:
        assert worker_mod._current_npu_stream(7) is stream

    get_raw_stream.assert_called_once_with(7)
    stream_cls.assert_called_once_with(
        device=7, stream_id=17, device_index=7, device_type=20
    )


def test_store_submit_records_compute_event_without_stream_query():
    handler, _, _ = _make_handler(npu_to_cpu=True, factor=1)
    compute_stream = MagicMock(name="compute_stream")
    compute_done = MagicMock(name="compute_done")

    with patch.object(
        worker_mod, "_current_npu_stream", return_value=compute_stream
    ) as get_stream, patch.object(
        torch.npu, "Event", return_value=compute_done
    ), patch.object(
        torch.npu,
        "current_stream",
        side_effect=AssertionError("current_stream must not be queried"),
    ), patch.object(handler, "_ensure_submit_thread"):
        assert handler.submit(11, MagicMock(), MagicMock()) is True

    get_stream.assert_called_once_with(handler._device_index)
    compute_done.record.assert_called_once_with(compute_stream)
    queued = handler._submit_queue.get_nowait()
    assert queued.wait_event is compute_done


def test_store_transfer_requires_compute_event():
    handler, _, _ = _make_handler(npu_to_cpu=True, factor=1)
    ConcreteGPU, ConcreteCPU = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0], dtype=np.int64)
    src.group_sizes = [1]
    src.block_indices = [0]
    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0], dtype=np.int64)

    with pytest.raises(RuntimeError, match="requires a compute event"):
        handler.transfer_async(3, src, dst)


def test_handler_transfer_rejects_bad_specs():
    handler, _, _ = _make_handler()
    with pytest.raises(TypeError, match="src_spec"):
        handler.transfer_async(1, object(), object())

    ConcreteGPU, ConcreteCPU = _concrete_specs()
    src = ConcreteCPU.__new__(ConcreteCPU)
    src.block_ids = np.asarray([0], dtype=np.int64)
    dst = ConcreteCPU.__new__(ConcreteCPU)
    dst.block_ids = np.asarray([0], dtype=np.int64)
    with pytest.raises(TypeError, match="npu_spec"):
        handler.transfer_async(1, src, dst)

    src = ConcreteGPU.__new__(ConcreteGPU)
    src.block_ids = np.asarray([0], dtype=np.int64)
    src.group_sizes = [1, 1]
    src.block_indices = [0, 0]
    with pytest.raises(ValueError, match="group_sizes length"):
        handler.transfer_async(1, src, dst)


def test_worker_init_submit_finished_wait_shutdown(monkeypatch):
    _patch_local_world(monkeypatch, local_rank=3, world_size=16, node_count=2)
    kv_tensor = MagicMock()
    kv_tensor.page_size_bytes = 8
    kv_tensor.tensor = torch.zeros(16, dtype=torch.int8, device="cpu")
    kv_caches = SimpleNamespace(
        tensors=[kv_tensor],
        group_data_refs=[[SimpleNamespace(tensor_idx=0, page_size_bytes=8)]],
    )
    mmap_region = MagicMock()
    mmap_region._hugepage = False
    mmap_region.is_pinned = False
    mmap_region.npu_base = None
    mmap_region.create_next_worker_view.return_value = torch.zeros(
        (2, 8), dtype=torch.int8, device="cpu"
    )

    fake_handler = MagicMock()
    fake_handler.transfer_async.return_value = True
    fake_handler.submit.return_value = True
    fake_handler.get_finished.return_value = []
    with patch.object(worker_mod, "_make_sharded_load", return_value=None), patch.object(
        worker_mod, "pin_memory_region"
    ), patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ), patch.object(
        worker_mod, "NpuSingleDirectionOffloadingHandler", return_value=fake_handler
    ), patch(
        "omni_npu.v1.kv_offload.cpu.npu_worker.PIN_MEMORY", False
    ):
        worker = NPUCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_cpu_blocks=2,
            mmap_region=mmap_region,
            rotate_store_writers=True,
        )
    assert worker._local_world_rank == 3
    assert worker._local_world_size == 8

    fake_handler.submit.return_value = True
    ConcreteGPU, _ = _concrete_specs()
    src = ConcreteGPU.__new__(ConcreteGPU)
    assert worker.submit_store(1, src, MagicMock()) is True
    assert worker.submit_load(2, MagicMock(), src) is True
    assert fake_handler.submit.call_count == 2
    assert worker.get_finished() == []
    worker.wait({1, 2})
    worker.shutdown()
    assert fake_handler.shutdown.call_count == 2


def test_worker_init_without_mmap_allocates_cpu(monkeypatch):
    _patch_local_world(monkeypatch, local_rank=1, world_size=8, node_count=2)
    kv_tensor = MagicMock()
    kv_tensor.page_size_bytes = 4
    kv_tensor.tensor = torch.zeros(8, dtype=torch.int8, device="cpu")
    kv_caches = SimpleNamespace(tensors=[kv_tensor], group_data_refs=[[]])
    fake_handler = MagicMock()
    with patch.object(worker_mod, "_make_sharded_load", return_value=None), patch.object(
        worker_mod,
        "NpuSingleDirectionOffloadingHandler",
        return_value=fake_handler,
    ) as handler_cls, patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ), patch(
        "omni_npu.v1.kv_offload.cpu.npu_worker.PIN_MEMORY", False
    ):
        worker = NPUCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_cpu_blocks=2,
            mmap_region=None,
        )
    assert handler_cls.call_count == 2
    assert worker._local_world_rank == 1
    assert worker._local_world_size == 4


def test_new_timing_event_falls_back_without_enable_timing():
    fallback = object()
    with patch.object(
        torch.npu, "Event", side_effect=[TypeError("no timing"), fallback]
    ) as event_cls:
        assert NpuSingleDirectionOffloadingHandler._new_timing_event() is fallback
    event_cls.assert_any_call(enable_timing=True)
    event_cls.assert_any_call()


def _transfer_specs(npu_to_cpu: bool):
    ConcreteGPU, ConcreteCPU = _concrete_specs()
    gpu = ConcreteGPU.__new__(ConcreteGPU)
    gpu.block_ids = np.asarray([0, 1], dtype=np.int64)
    gpu.group_sizes = [2]
    gpu.block_indices = [0]
    cpu = ConcreteCPU.__new__(ConcreteCPU)
    cpu.block_ids = np.asarray([0, 1], dtype=np.int64)
    return (gpu, cpu) if npu_to_cpu else (cpu, gpu)


# ---------------------------------------------------------------------------
# Sharded load
# ---------------------------------------------------------------------------


def _load_specs(num_blocks):
    ConcreteGPU, ConcreteCPU = _concrete_specs()
    cpu = ConcreteCPU.__new__(ConcreteCPU)
    cpu.block_ids = np.arange(num_blocks, dtype=np.int64) + 1
    gpu = ConcreteGPU.__new__(ConcreteGPU)
    gpu.block_ids = np.arange(num_blocks, dtype=np.int64)
    gpu.group_sizes = [num_blocks]
    gpu.block_indices = [0]
    return cpu, gpu


def _submit_load(handler, num_blocks, job_id=5):
    src, dst = _load_specs(num_blocks)
    with patch.object(torch.npu, "Event", return_value=MagicMock()), patch.object(
        torch.npu, "set_device"
    ), patch.object(torch.npu, "set_stream"), patch.object(
        torch.npu, "current_stream", return_value=MagicMock()
    ):
        assert handler.transfer_async(job_id, src, dst) is True
    return handler._transfers_by_id[job_id]


def _sharded_load_handler(min_blocks, plan="plan"):
    handler, _, _ = _make_handler(npu_to_cpu=False)
    handler._swap_blocks_batch = MagicMock()
    sharded = MagicMock(min_blocks=min_blocks)
    sharded.plan.return_value = plan
    sharded.run.return_value = ["keep"]
    handler._sharded_executor = sharded
    handler.enable_sharded_load = True
    return handler, sharded


def test_load_at_min_blocks_runs_the_sharded_plan():
    handler, sharded = _sharded_load_handler(min_blocks=3)
    transfer = _submit_load(handler, 3)

    tensor_idx, page, cpu_block, npu_block, src_ptr, _ = sharded.plan.call_args.args
    np.testing.assert_array_equal(tensor_idx, [0, 0, 0])
    np.testing.assert_array_equal(page, [8, 8, 8])
    np.testing.assert_array_equal(cpu_block, [1, 2, 3])
    np.testing.assert_array_equal(npu_block, [0, 1, 2])
    cpu = handler.src_tensors[0]
    np.testing.assert_array_equal(
        src_ptr, cpu.data_ptr() + np.array([1, 2, 3]) * cpu.stride(0)
    )

    plan, run_src, run_npu, dst_tensors, h2d = sharded.run.call_args.args
    assert plan == "plan"
    np.testing.assert_array_equal(run_src, src_ptr)
    np.testing.assert_array_equal(run_npu, [0, 1, 2])
    assert dst_tensors is handler.dst_tensors
    assert h2d == handler._h2d
    handler._swap_blocks_batch.assert_not_called()
    assert transfer.shard_keepalive == ["keep"]
    assert transfer.num_bytes == 24


def test_load_below_min_blocks_keeps_the_plain_load():
    handler, sharded = _sharded_load_handler(min_blocks=4)
    transfer = _submit_load(handler, 3)
    sharded.plan.assert_not_called()
    sharded.run.assert_not_called()
    handler._swap_blocks_batch.assert_called_once()
    assert transfer.shard_keepalive is None


def test_refused_sharded_plan_falls_back_and_warns_once():
    handler, sharded = _sharded_load_handler(min_blocks=1, plan=None)

    def refuse(*args):
        args[5].append("two sources")
        return None

    sharded.plan.side_effect = refuse
    with patch.object(worker_mod, "logger") as log:
        _submit_load(handler, 2, job_id=5)
        _submit_load(handler, 2, job_id=6)
    assert handler._swap_blocks_batch.call_count == 2
    sharded.run.assert_not_called()
    log.warning.assert_called_once()
    assert log.warning.call_args.args[2] == "two sources"


def test_only_the_load_handler_keeps_the_sharded_executor(monkeypatch):
    _patch_local_world(monkeypatch)
    npu = SimpleNamespace(
        dtype=torch.int8, ndim=2, device=SimpleNamespace(type="npu", index=0), shape=(4, 8)
    )
    cpu = SimpleNamespace(
        dtype=torch.int8, ndim=2, device=SimpleNamespace(type="cpu"), shape=(4, 8)
    )
    executor = object()
    handlers = {}
    with patch.object(torch.npu, "Stream", return_value=MagicMock()), patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ):
        for npu_to_cpu in (True, False):
            handlers[npu_to_cpu] = NpuSingleDirectionOffloadingHandler(
                npu_tensors=[npu],
                cpu_tensors=[cpu],
                block_size_factor=1,
                kv_cache_groups_data_refs=[],
                npu_to_cpu=npu_to_cpu,
                sharded_executor=executor,
            )
    assert handlers[True]._sharded_executor is None
    assert handlers[True].enable_sharded_load is False
    assert handlers[False]._sharded_executor is executor
    assert handlers[False].enable_sharded_load is True


def test_plan_transfer_descriptors_returns_shard_meta():
    handler, _, _ = _make_handler(npu_to_cpu=False, factor=1)
    src = np.asarray([2, 3], dtype=np.int64)
    dst = np.asarray([0, 1], dtype=np.int64)
    plain = handler._plan_transfer_descriptors(src, dst, [2], [0], 2, 2)
    sharded = handler._plan_transfer_descriptors(
        src, dst, [2], [0], 2, 2, with_shard_meta=True
    )
    for a, b in zip(plain, sharded[:3]):
        np.testing.assert_array_equal(a, b)
    meta = sharded[3]
    np.testing.assert_array_equal(meta.tensor_idx, [0, 0])
    np.testing.assert_array_equal(meta.cpu_block, [2, 3])
    np.testing.assert_array_equal(meta.npu_block, [0, 1])


def test_h2d_hands_descriptors_to_the_batch_copy():
    handler, _, _ = _make_handler(npu_to_cpu=False)
    handler._swap_blocks_batch = MagicMock()
    handler._h2d(np.array([1, 2]), np.array([3, 4]), np.array([8, 8]))
    srcs, dsts, sizes, direction = handler._swap_blocks_batch.call_args.args
    assert srcs.dtype == torch.int64
    assert (srcs.tolist(), dsts.tolist(), sizes.tolist()) == ([1, 2], [3, 4], [8, 8])
    assert direction == worker_mod._DIRECTION_H2D


def _patch_shard_groups(monkeypatch, *, hosts=("h", "h"), reduced=1):
    monkeypatch.setenv("OMNI_KV_OFFLOAD_SHARDED_LOAD", "1")
    monkeypatch.setattr(
        worker_mod,
        "get_world_group",
        lambda: SimpleNamespace(world_size=2, cpu_group="cpu_group"),
    )
    monkeypatch.setattr(
        worker_mod,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0, 1], world_size=2, rank_in_group=1),
    )
    monkeypatch.setattr(worker_mod, "_get_swap_blocks_batch", lambda: object())
    monkeypatch.setattr(worker_mod, "_free_device_bytes", lambda device: 8 << 30)
    fake = MagicMock()
    fake.get_rank.return_value = 1

    def gather(out, obj, group):
        assert obj == ((0, 1), worker_mod.socket.gethostname())
        for r in range(2):
            out[r] = ((0, 1), hosts[r])

    def all_reduce(tensor, op, group):
        tensor[0] = min(int(tensor[0]), reduced)

    fake.all_gather_object.side_effect = gather
    fake.all_reduce.side_effect = all_reduce
    monkeypatch.setattr(worker_mod, "dist", fake)
    return fake


@pytest.mark.parametrize("value", [None, "0"])
def test_make_sharded_load_is_off_unless_the_env_turns_it_on(monkeypatch, value):
    """Off by default: unset reads the same as an explicit 0, and neither talks
    to the other ranks."""
    if value is None:
        monkeypatch.delenv("OMNI_KV_OFFLOAD_SHARDED_LOAD", raising=False)
    else:
        monkeypatch.setenv("OMNI_KV_OFFLOAD_SHARDED_LOAD", value)
    fake = MagicMock()
    monkeypatch.setattr(worker_mod, "dist", fake)
    assert worker_mod._make_sharded_load(True, torch.device("cpu")) is None
    fake.all_gather_object.assert_not_called()


@pytest.mark.parametrize(
    "replicated,hosts,reduced,reason",
    [
        (False, ("h", "h"), 1, "KV layout is not replicated"),
        (True, ("a", "b"), 1, "TP group spans hosts"),
        (True, ("h", "h"), 0, "refused on another rank"),
    ],
)
def test_make_sharded_load_refusals_create_no_group(
    monkeypatch, replicated, hosts, reduced, reason
):
    fake = _patch_shard_groups(monkeypatch, hosts=hosts, reduced=reduced)
    # The options build needs a real torch_npu, so stub it out: these cases are
    # about the other refusals, and the last one about this rank having none.
    with patch.object(worker_mod, "_sharded_hccl_options", return_value="opts"),             patch.object(worker_mod, "logger") as log:
        assert worker_mod._make_sharded_load(replicated, torch.device("cpu")) is None
    fake.all_reduce.assert_called_once()
    fake.new_group.assert_not_called()
    assert reason in log.warning.call_args.args[1]


def test_make_sharded_load_builds_the_group_and_warms_up(monkeypatch):
    fake = _patch_shard_groups(monkeypatch)
    fake.new_group.return_value = "shard_group"
    with patch.object(worker_mod, "_sharded_hccl_options", return_value="opts"), patch.object(
        worker_mod, "ShardedLoadExecutor"
    ) as executor_cls, patch.object(worker_mod, "logger"):
        executor = worker_mod._make_sharded_load(True, torch.device("cpu"))

    fake.new_group.assert_called_once_with([0, 1], backend="hccl", pg_options="opts")
    kwargs = executor_cls.call_args.kwargs
    assert (kwargs["rank"], kwargs["world"], kwargs["chunk_bytes"], kwargs["min_blocks"]) == (
        1,
        2,
        worker_mod._SHARDED_CHUNK_MB << 20,
        worker_mod._SHARDED_MIN_BLOCKS,
    )
    assert executor is executor_cls.return_value
    executor.allocate.assert_called_once()
    executor.probe.assert_called_once()
    kwargs["allgather_async"]("out", "inp")
    fake.all_gather_into_tensor.assert_called_once_with(
        "out", "inp", group="shard_group", async_op=True
    )


def test_sharded_hccl_options(monkeypatch):
    """The group carries its own name and AI_CPU expansion, whatever
    HCCL_OP_EXPANSION_MODE says for the other groups. It does not touch
    hccl_buffer_size: the scratch is inherited like every other group's."""
    class Options:
        hccl_config = None

    fake_torch_npu = SimpleNamespace(
        _C=SimpleNamespace(
            _distributed_c10d=SimpleNamespace(ProcessGroupHCCL=SimpleNamespace(Options=Options))
        )
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    assert worker_mod._sharded_hccl_options("g").hccl_config == {
        "group_name": "g",
        "hccl_op_expansion_mode": 2,
    }


def test_sharded_load_reserved_bytes(monkeypatch):
    """Staging + HCCL scratch + one chunk of scatter workspace, or nothing when
    the load will not be sharded."""
    monkeypatch.setenv("OMNI_KV_OFFLOAD_SHARDED_LOAD", "1")
    chunk = worker_mod._SHARDED_CHUNK_MB
    slots = (chunk // 16) * 2 * (1 + 16)  # two send and two receive slots per rank
    hccl = worker_mod._SHARDED_COMM_RESERVE_MB
    assert worker_mod.sharded_load_reserved_bytes(16) == (slots + hccl + chunk) << 20
    assert worker_mod.sharded_load_reserved_bytes(16) == 340 << 20  # tp16 today
    assert worker_mod.sharded_load_reserved_bytes(1) == 0
    monkeypatch.setenv("OMNI_KV_OFFLOAD_SHARDED_LOAD", "0")
    assert worker_mod.sharded_load_reserved_bytes(16) == 0


def test_worker_init_gives_the_sharded_load_to_the_load_handler(monkeypatch):
    _patch_local_world(monkeypatch, local_rank=0, world_size=16, node_count=1)
    kv_tensor = MagicMock()
    kv_tensor.page_size_bytes = 4
    kv_tensor.tensor = torch.zeros(8, dtype=torch.int8, device="cpu")
    kv_caches = SimpleNamespace(tensors=[kv_tensor], group_data_refs=[[]])
    with patch.object(
        worker_mod, "NpuSingleDirectionOffloadingHandler", return_value=MagicMock()
    ) as handler_cls, patch.object(
        worker_mod, "_get_swap_blocks_batch", return_value=None
    ), patch.object(
        worker_mod, "_make_sharded_load", return_value="executor"
    ) as make, patch(
        "omni_npu.v1.kv_offload.cpu.npu_worker.PIN_MEMORY", False
    ):
        NPUCPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=1,
            num_cpu_blocks=2,
            mmap_region=None,
            rotate_store_writers=True,
        )
    store_call, load_call = handler_cls.call_args_list
    assert "sharded_executor" not in store_call.kwargs
    assert load_call.kwargs["sharded_executor"] == "executor"
    assert make.call_args.args[0] is True


def test_make_sharded_load_skips_tp_size_one_before_any_collective(monkeypatch):
    """TP=1 never builds the shard domain: no gather, no group, no HBM paid."""
    fake = _patch_shard_groups(monkeypatch)
    monkeypatch.setattr(
        worker_mod,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0], world_size=1, rank_in_group=0),
    )
    assert worker_mod._make_sharded_load(True, torch.device("cpu")) is None
    fake.all_gather_object.assert_not_called()
    fake.all_reduce.assert_not_called()
    fake.new_group.assert_not_called()


@pytest.mark.parametrize(
    "tp_size,extension,reason",
    [
        (2, None, "_swap_blocks_batch extension missing"),
    ],
)
def test_make_sharded_load_refuses_without_extension(
    monkeypatch, tp_size, extension, reason
):
    fake = _patch_shard_groups(monkeypatch)
    monkeypatch.setattr(
        worker_mod,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=[0, 1], world_size=tp_size, rank_in_group=1),
    )
    monkeypatch.setattr(worker_mod, "_get_swap_blocks_batch", lambda: extension)
    with patch.object(worker_mod, "logger") as log:
        assert worker_mod._make_sharded_load(True, torch.device("cpu")) is None
    fake.new_group.assert_not_called()
    assert reason in log.warning.call_args.args[1]


def test_make_sharded_load_refuses_when_device_memory_is_tight(monkeypatch):
    """Not enough free memory for the staging buffers: every rank agrees to
    fall back before any group is created, and the service still starts."""
    fake = _patch_shard_groups(monkeypatch)
    need = worker_mod.sharded_load_reserved_bytes(2)
    monkeypatch.setattr(
        worker_mod, "_free_device_bytes", lambda device: int(need * 1.4)
    )
    with patch.object(worker_mod, "logger") as log:
        assert worker_mod._make_sharded_load(True, torch.device("cpu")) is None
    fake.new_group.assert_not_called()
    reason = log.warning.call_args.args[1]
    assert "sharded load needs" in reason and "is free" in reason


def test_make_sharded_load_accepts_memory_with_headroom(monkeypatch):
    fake = _patch_shard_groups(monkeypatch)
    need = worker_mod.sharded_load_reserved_bytes(2)
    monkeypatch.setattr(
        worker_mod, "_free_device_bytes", lambda device: int(need * 1.6)
    )
    with patch.object(worker_mod, "_sharded_hccl_options", return_value="opts"), patch.object(
        worker_mod, "ShardedLoadExecutor"
    ) as executor_cls, patch.object(worker_mod, "logger"):
        assert worker_mod._make_sharded_load(True, torch.device("cpu")) is not None
    executor_cls.return_value.allocate.assert_called_once()
    executor_cls.return_value.probe.assert_called_once()
    fake.new_group.assert_called_once()


def test_make_sharded_load_skips_check_when_free_memory_is_unknown(monkeypatch):
    _patch_shard_groups(monkeypatch)
    monkeypatch.setattr(worker_mod, "_free_device_bytes", lambda device: None)
    with patch.object(worker_mod, "_sharded_hccl_options", return_value="opts"), patch.object(
        worker_mod, "ShardedLoadExecutor"
    ), patch.object(worker_mod, "logger"):
        assert worker_mod._make_sharded_load(True, torch.device("cpu")) is not None


def test_make_sharded_load_falls_back_when_allocation_fails_on_any_rank(monkeypatch):
    """allocate() raising on this rank, or all_reduce reporting a failure on
    another rank, must leave nobody inside the probe collective."""
    fake = _patch_shard_groups(monkeypatch)
    seen = {"n": 0}

    def all_reduce(tensor, op, group):
        # 1st call: the refusal agreement. 2nd: the allocation agreement, where
        # another rank reports a failure.
        seen["n"] += 1
        if seen["n"] == 2:
            tensor[0] = 0

    fake.all_reduce.side_effect = all_reduce

    with patch.object(worker_mod, "_sharded_hccl_options", return_value="opts"), patch.object(
        worker_mod, "ShardedLoadExecutor"
    ) as executor_cls, patch.object(worker_mod, "logger") as log:
        executor_cls.return_value.allocate.side_effect = RuntimeError("NPU out of memory")
        assert worker_mod._make_sharded_load(True, torch.device("cpu")) is None
    executor_cls.return_value.probe.assert_not_called()
    assert "staging did not fit on every rank" in log.warning.call_args.args[0]


def test_make_sharded_load_falls_back_on_memory_error(monkeypatch):
    """Not only RuntimeError: a plain MemoryError also becomes a refusal,
    otherwise one rank dies while the others wait in the all_reduce."""
    fake = _patch_shard_groups(monkeypatch)

    with patch.object(worker_mod, "_sharded_hccl_options", return_value="opts"), patch.object(
        worker_mod, "ShardedLoadExecutor"
    ) as executor_cls, patch.object(worker_mod, "logger") as log:
        executor_cls.return_value.allocate.side_effect = MemoryError("npu oom")
        # Every rank fails, so the agreement below sees 0 from this rank itself.
        fake.all_reduce.side_effect = lambda tensor, op, group: None
        assert worker_mod._make_sharded_load(True, torch.device("cpu")) is None
    executor_cls.return_value.probe.assert_not_called()
    assert "did not fit" in log.warning.call_args_list[-2].args[0]


def test_free_device_bytes_reads_mem_get_info(monkeypatch):
    monkeypatch.setattr(
        torch.npu, "mem_get_info", lambda index: (123 << 20, 64 << 30), raising=False
    )
    assert worker_mod._free_device_bytes(SimpleNamespace(index=3)) == 123 << 20
    monkeypatch.setattr(
        torch.npu, "mem_get_info", MagicMock(side_effect=RuntimeError("no npu")), raising=False
    )
    with patch.object(worker_mod, "logger"):
        assert worker_mod._free_device_bytes(SimpleNamespace(index=0)) is None

def test_sharded_load_submits_on_the_caller_thread(monkeypatch):
    """A sharded load is enqueued by the calling (main) thread and never starts
    the background submit thread; a plain job still goes to the submit queue."""
    handler, _, _ = _make_handler(npu_to_cpu=False)
    handler.transfer_async = MagicMock(return_value=True)
    handler._ensure_submit_thread = MagicMock()

    handler._sharded_executor = MagicMock()
    handler.enable_sharded_load = True
    assert handler.submit(5, "src", "dst") is True
    handler.transfer_async.assert_called_once_with(5, "src", "dst", wait_event=None)
    handler._ensure_submit_thread.assert_not_called()
    assert handler._submit_queue.empty()

    handler._sharded_executor = None
    handler.enable_sharded_load = False
    assert handler.submit(6, "src", "dst") is True
    handler.transfer_async.assert_called_once()  # still just the sharded one
    handler._ensure_submit_thread.assert_called_once()
    assert handler._submit_queue.get_nowait().job_id == 6
