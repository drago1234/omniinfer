# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# UT for omni_npu.distributed.eplb_communicate：acl/hixl/torch_npu/vllm 全 mock（只用 CPU torch 张量），按文件路径 exec 加载被测模块

import copy
import ctypes
import importlib
import importlib.util
import json
import pathlib
import socket
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


_MODULE_PATH = (pathlib.Path(__file__).resolve().parents[3]
                / "omni" / "distributed" / "eplb_communicate.py")


# --------------------------------------------------------------------------- #
# 假 device / 厂商层：被测模块对外部的全部依赖都收在这里
# --------------------------------------------------------------------------- #

# 记录 acl.rt.* 调用，不真的下发 memcpy / 设 device
class FakeAclRT:
    def __init__(self):
        self.copies = []   # SimpleNamespace(dst, dst_nbytes, src, src_nbytes, kind, stream)
        self.devices = []  # set_device 的入参

    def memcpy_async(self, dst, dst_nbytes, src, src_nbytes, kind, stream):
        self.copies.append(SimpleNamespace(
            dst=dst, dst_nbytes=dst_nbytes,
            src=src, src_nbytes=src_nbytes,
            kind=kind, stream=stream,
        ))
        return 0

    def set_device(self, device):
        self.devices.append(device)
        return 0


# 假 HIXL engine：记录 initialize/register_mem/transfer_sync，类级 init_script 脚本化返回值
class FakeHixlEngine:
    instances: list = []
    init_script: list = []

    def __init__(self):
        self.init_calls = []   # [(addr, options)]
        self.registered = []   # [(desc, mem_type)]
        self.transfers = []    # [(remote_addr, op, descs, timeout)]
        self.reg_ret = 0
        self.transfer_ret = 0
        FakeHixlEngine.instances.append(self)

    def initialize(self, addr, options):
        self.init_calls.append((addr, options))
        if FakeHixlEngine.init_script:
            return FakeHixlEngine.init_script.pop(0)
        return 0  # hixl.SUCCESS

    def register_mem(self, desc, mem_type):
        self.registered.append((desc, mem_type))
        return self.reg_ret, len(self.registered)

    def transfer_sync(self, remote_addr, op, descs, timeout_in_millis=0):
        self.transfers.append((remote_addr, op, list(descs), timeout_in_millis))
        return self.transfer_ret


# raw stream 的占位对象（只要身份稳定即可）
class FakeRawStream:
    pass


# 假的 vllm.distributed.eplb.eplb_communicator.EplbCommunicator
class FakeEplbCommunicator:

    def _log_initialized(self):
        self._logged = True


def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


# 构造被测模块依赖的全部外部模块，返回 {模块名: 假模块}
def _build_vendor_fakes():
    acl = _fake_module("acl", rt=FakeAclRT())

    hixl = _fake_module(
        "hixl",
        SUCCESS=0,
        OPTION_AUTO_CONNECT="option_auto_connect",
        OPTION_GLOBAL_RESOURCE_CONFIG="option_global_resource_config",
        MemType=SimpleNamespace(MEM_DEVICE="mem_device"),
        TransferOp=SimpleNamespace(READ="READ"),
        MemDesc=lambda addr, nbytes: SimpleNamespace(addr=addr, nbytes=nbytes),
        TransferOpDesc=lambda **kw: SimpleNamespace(**kw),
        Hixl=FakeHixlEngine,
    )

    torch_npu = _fake_module(
        "torch_npu",
        npu=SimpleNamespace(current_device=lambda: 0),
        _C=SimpleNamespace(_npu_getCurrentRawStream=lambda device: FakeRawStream()),
    )

    vllm = _fake_module("vllm")
    vllm_distributed = _fake_module("vllm.distributed")
    vllm_eplb = _fake_module("vllm.distributed.eplb")
    vllm_eplb_communicator = _fake_module(
        "vllm.distributed.eplb.eplb_communicator",
        EplbCommunicator=FakeEplbCommunicator,
    )
    # 让 from a.b.c import X 这条链上的父包属性也齐备
    vllm.distributed = vllm_distributed
    vllm_distributed.eplb = vllm_eplb
    vllm_eplb.eplb_communicator = vllm_eplb_communicator

    return {
        "acl": acl,
        "hixl": hixl,
        "torch_npu": torch_npu,
        "vllm": vllm,
        "vllm.distributed": vllm_distributed,
        "vllm.distributed.eplb": vllm_eplb,
        "vllm.distributed.eplb.eplb_communicator": vllm_eplb_communicator,
    }


# 优先按包名 import：只有走正常 import 链路，coverage 才会把行覆盖统计到
# omni_npu/distributed/eplb_communicate.py（按文件路径 exec 的模块 coverage 统计不到，
# 会让这个文件的覆盖率一直是 0）。
# 环境里缺 vllm / NPU 依赖时（例如纯 CPU 开发机）再退回“按文件路径 exec + 注入 fake 依赖”。
def _load_module_under_test():
    try:
        return importlib.import_module("omni_npu.distributed.eplb_communicate")
    except ImportError:
        pass

    fakes = _build_vendor_fakes()
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        spec = importlib.util.spec_from_file_location(
            "_eplb_communicate_under_test", _MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


_MOD = _load_module_under_test()
RawTensor = _MOD.RawTensor
_PinnedH2D = _MOD._PinnedH2D
_hixl_engine = _MOD._hixl_engine
HixlEplbCommunicator = _MOD.HixlEplbCommunicator


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

# 每个用例一套全新的假 device 层，并 patch 到被测模块里
@pytest.fixture
def dev():
    fakes = _build_vendor_fakes()
    FakeHixlEngine.instances = []
    FakeHixlEngine.init_script = []
    _PinnedH2D._instances.clear()
    with patch.multiple(_MOD, acl=fakes["acl"], hixl=fakes["hixl"],
                        torch_npu=fakes["torch_npu"]), \
            patch.object(socket, "gethostbyname", return_value="127.0.0.1"):
        yield SimpleNamespace(acl=fakes["acl"], hixl=fakes["hixl"],
                              torch_npu=fakes["torch_npu"],
                              engines=FakeHixlEngine.instances)


# 屏蔽 pin_memory：无 CUDA/NPU 的环境里 torch.empty(pin_memory=True) 会失败
@pytest.fixture(autouse=True)
def no_pin(monkeypatch):
    real_empty = torch.empty

    def empty(*shape, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*shape, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    yield


class FakeEvent:
    def __init__(self):
        self.syncs = 0
        self.records = 0

    def synchronize(self):
        self.syncs += 1

    def record(self):
        self.records += 1


@pytest.fixture
def fake_event():
    created = []

    def factory():
        event = FakeEvent()
        created.append(event)
        return event

    with patch.object(torch.cuda, "Event", factory):
        yield created


def _fake_group(size: int):
    group = MagicMock()
    group.size.return_value = size
    return group


# 把每个 peer 都填成与本地一致（单机多 rank 的最简模型）
def _gather_identity(obj_list, obj, group=None):
    for i in range(len(obj_list)):
        obj_list[i] = obj if i == 0 else copy.deepcopy(obj)


_PEER_ADDR = "peer:9000"


# all_gather_object 的 side_effect：本地对象放 [0]，peer 放其余 rank；peer desc 用真实 data_ptr
def _peer_gather(peer_slots, peer_addr):
    def gather(obj_list, obj, group=None):
        if isinstance(obj, str):
            peer = peer_addr
        else:
            peer = {k: (peer_slots[k[1]].data_ptr(), unit)
                    for k, (_, unit) in obj.items()}
        for i in range(len(obj_list)):
            obj_list[i] = obj if i == 0 else copy.deepcopy(peer)
    return gather


# 记录 transfer；对 READ 额外用 ctypes 真搬一次字节
def _transfer_with_memcpy(self, remote_addr, op, descs, timeout_in_millis=0):
    self.transfers.append((remote_addr, op, list(descs), timeout_in_millis))
    if op == "READ":
        for d in descs:
            ctypes.memmove(d.local_addr, d.remote_addr, d.len)
    return self.transfer_ret


# --------------------------------------------------------------------------- #
# _hixl_engine
# --------------------------------------------------------------------------- #

class TestHixlEngine:
    def test_returns_engine_and_addr_on_first_success(self, dev):
        FakeHixlEngine.init_script = [0]
        engine, addr = _hixl_engine({"k": "v"})
        assert isinstance(engine, FakeHixlEngine)
        host, port = addr.split(":")
        assert host == "127.0.0.1"
        assert 0 < int(port) < 65536
        assert engine.init_calls == [(addr, {"k": "v"})]

    def test_retries_until_success(self, dev):
        FakeHixlEngine.init_script = [1, 1, 0]
        engine, _ = _hixl_engine({"k": "v"}, retries=8)
        assert len(FakeHixlEngine.instances) == 3
        assert engine is FakeHixlEngine.instances[-1]

    def test_all_retries_fail_raises(self, dev):
        FakeHixlEngine.init_script = [1, 1, 1, 1]
        with pytest.raises(RuntimeError):
            _hixl_engine({"k": "v"}, retries=4)
        assert len(FakeHixlEngine.instances) == 4


# --------------------------------------------------------------------------- #
# RawTensor
# --------------------------------------------------------------------------- #

class TestRawTensor:
    def test_wraps_contiguous_2d(self, dev):
        t = torch.zeros(3, 4, dtype=torch.float32, device="cpu")
        rt = RawTensor(t)
        assert rt.ptr == t.data_ptr()
        assert rt.nbytes == t.nbytes
        assert rt.row_bytes == t.nbytes // 3
        assert rt.s == [None]           # 不传 s 时新建独立槽位

    def test_wraps_1d(self, dev):
        t = torch.zeros(5, dtype=torch.float32, device="cpu")
        rt = RawTensor(t)
        assert rt.row_bytes == t.nbytes

    def test_non_contiguous_2d_rejected(self, dev):
        t = torch.zeros(4, 6, device="cpu").t()
        assert not t.is_contiguous()
        with pytest.raises(ValueError):
            RawTensor(t)

    def test_attributes_forwarded_to_real_tensor(self, dev):
        t = torch.zeros(2, 3, device="cpu")
        rt = RawTensor(t)
        assert rt.shape == t.shape
        assert rt.dim() == 2
        assert rt.numel() == 6

    def test_len_and_data_ptr(self, dev):
        rt = RawTensor(torch.zeros(7, device="cpu"))
        assert len(rt) == 7
        assert rt.data_ptr() == rt.ptr

    def test_shared_slot_is_shared(self, dev):
        slot = [None]
        a = RawTensor(torch.zeros(2, device="cpu"), s=slot)
        b = RawTensor(torch.zeros(2, device="cpu"), s=slot)
        assert a.s is b.s is slot
        a.set_stream(123)
        assert b.s[0] == 123      # 改一处，同组全生效

    def test_set_stream_none_queries_torch_npu(self, dev):
        rt = RawTensor(torch.zeros(1, device="cpu"))
        got = rt.set_stream()
        assert isinstance(got, FakeRawStream)
        assert rt.s[0] is got

    def test_getitem_returns_row_view_sharing_slot(self, dev):
        rt = RawTensor(torch.zeros(3, 4, dtype=torch.float32, device="cpu"))
        rt.set_stream(7)
        row = rt[2]
        assert row.ptr == rt.ptr + 2 * rt.row_bytes
        assert row.nbytes == rt.row_bytes
        assert row.s is rt.s

    def test_copy_issues_raw_memcpy(self, dev):
        dst = RawTensor(torch.zeros(8, dtype=torch.int32, device="cpu"))
        src = RawTensor(torch.ones(8, dtype=torch.int32, device="cpu"))
        dst.set_stream(5)
        ret = dst.copy_(src)
        assert ret is dst
        assert len(dev.acl.rt.copies) == 1
        c = dev.acl.rt.copies[0]
        assert (c.dst, c.src) == (dst.ptr, src.ptr)
        assert (c.dst_nbytes, c.src_nbytes) == (dst.nbytes, src.nbytes)
        assert (c.kind, c.stream) == (0x3, 5)

    def test_copy_requires_non_blocking(self, dev):
        with pytest.raises(ValueError):
            RawTensor(torch.zeros(1, device="cpu")).copy_(RawTensor(torch.zeros(1, device="cpu")),
                                            non_blocking=False)


# --------------------------------------------------------------------------- #
# _PinnedH2D
# --------------------------------------------------------------------------- #

class TestPinnedH2D:
    def test_init_applies_fill(self, dev, no_pin):
        pt = _PinnedH2D(2, 3, dtype=torch.int32, fill=-1)
        assert pt.cpu.shape == (2, 3)
        assert torch.equal(pt.cpu, torch.full((2, 3), -1, dtype=torch.int32, device="cpu"))

    def test_get_reuses_same_name_and_shape(self, dev, no_pin):
        a = _PinnedH2D.get("m", 2, 4, dtype=torch.int32, fill=-1)
        b = _PinnedH2D.get("m", 2, 4, dtype=torch.int32, fill=-1)
        assert a is b

    def test_get_rebuilds_on_shape_or_dtype_change(self, dev, no_pin):
        a = _PinnedH2D.get("m", 2, 4)
        b = _PinnedH2D.get("m", 3, 4)
        c = _PinnedH2D.get("m", 3, 4, dtype=torch.int64)
        assert a is not b
        assert b.cpu.shape == (3, 4)
        assert c is not b and c.cpu.dtype == torch.int64

    def test_get_keeps_distinct_names_separate(self, dev, no_pin):
        assert _PinnedH2D.get("x", 1) is not _PinnedH2D.get("y", 1)

    def test_h2d_uploads_and_returns_event(self, dev, no_pin, fake_event):
        pt = _PinnedH2D(2, 4, dtype=torch.int32, fill=0)
        src = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device="cpu")
        dst = torch.zeros(2, 4, dtype=torch.int32, device="cpu")
        event = pt.h2d(src, dst)
        assert torch.equal(dst[:, :3], src)                       # 有效部分
        assert torch.equal(dst[:, 3], torch.zeros(2, dtype=torch.int32, device="cpu"))  # fill 尾部
        assert event is pt._done and event.records == 1
        assert torch.equal(src, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device="cpu"))  # src 不被改动

    def test_h2d_waits_for_previous_event(self, dev, no_pin, fake_event):
        pt = _PinnedH2D(1, 2, dtype=torch.int32, fill=0)
        dst = torch.zeros(1, 2, dtype=torch.int32, device="cpu")
        pt.h2d(torch.tensor([[1, 1]], dtype=torch.int32, device="cpu"), dst)
        pt.h2d(torch.tensor([[2, 2]], dtype=torch.int32, device="cpu"), dst)
        assert pt._done.syncs == 1     # 第二次上传前先等上一次读完成
        assert torch.equal(dst, torch.tensor([[2, 2]], dtype=torch.int32, device="cpu"))

    def test_h2d_event_ordering_across_rounds(self, dev, no_pin, fake_event):
        # 跨轮：覆写 pinned 缓冲前先 synchronize 第一轮 event，两轮共用同一 Event（src 宽度须与 buffer 一致）
        pt = _PinnedH2D.get("logical_to_physical", 3, 4, dtype=torch.int32, fill=-1)
        assert pt is _PinnedH2D.get("logical_to_physical", 3, 4, dtype=torch.int32, fill=-1)
        dst = torch.full((3, 4), -9, dtype=torch.int32, device="cpu")

        src1 = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                            dtype=torch.int32, device="cpu")
        e1 = pt.h2d(src1, dst)
        assert e1.syncs == 0 and e1.records == 1
        assert torch.equal(dst, src1)

        src2 = torch.tensor([[7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
                            dtype=torch.int32, device="cpu")
        e2 = pt.h2d(src2, dst)
        assert e2 is e1                       # 同一个常驻事件，逐轮复用
        assert e2.syncs == 1                  # 第二轮覆写 pinned 缓冲前先等第一轮读完
        assert e2.records == 2
        assert torch.equal(dst, src2)
        assert torch.equal(src2, torch.tensor([[7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
                                              dtype=torch.int32, device="cpu"))


# --------------------------------------------------------------------------- #
# HixlEplbCommunicator
# --------------------------------------------------------------------------- #

class TestHixlEplbCommunicator:
    def _make(self, layers=1, slots=2, local=4, height=8, ranks=2):
        weights = [[torch.zeros(local, height, device="cpu") for _ in range(slots)]
                   for _ in range(layers)]
        buffers = [torch.zeros(local, height, device="cpu") for _ in range(slots)]
        comm = HixlEplbCommunicator(_fake_group(ranks), weights, buffers)
        return comm, weights, buffers

    def test_proc_buffers_wraps_weights_and_buffers(self, dev):
        comm, _, _ = self._make(layers=2, slots=3)
        assert comm._local == 4
        for layer in comm._weights:
            assert all(isinstance(t, RawTensor) for t in layer)
        assert all(isinstance(b, RawTensor) for b in comm._buffer)
        # 所有权重/缓冲区共用同一个 raw stream 槽位
        slots = {id(t.s) for layer in comm._weights for t in layer}
        slots |= {id(b.s) for b in comm._buffer}
        assert len(slots) == 1

    def test_proc_buffers_splits_large_slots_per_expert(self, dev):
        comm, _, _ = self._make()
        weights = [[torch.zeros(4, 8, device="cpu") for _ in range(2)]]
        buffers = [torch.zeros(4, 8, device="cpu") for _ in range(2)]
        comm._proc_buffers(weights, buffers, split_threshold=1)   # 强制走大权重分支
        for rows in comm._buffer:
            assert isinstance(rows, list) and len(rows) == 4
            assert all(isinstance(r, RawTensor) for r in rows)

    def test_init_engine_registers_every_tensor(self, dev):
        comm, _, _ = self._make(layers=1, slots=2)
        engine = dev.engines[-1]
        assert len(engine.registered) == 2 + 2        # 2 个 weight slot + 2 个 buffer
        assert len(comm._registered_handles) == 4
        assert engine.registered[0][1] == dev.hixl.MemType.MEM_DEVICE

    def test_init_engine_options(self, dev):
        self._make()
        engine = dev.engines[-1]
        _, options = engine.init_calls[0]
        assert options[dev.hixl.OPTION_AUTO_CONNECT] == "1"
        cfg = json.loads(options[dev.hixl.OPTION_GLOBAL_RESOURCE_CONFIG])
        assert cfg == {"comm_resource_config.protocol_desc": ["roce:device", "roce:host"]}

    def test_register_mem_failure_raises(self, dev):
        with patch.object(FakeHixlEngine, "register_mem", return_value=(1, 0)):
            with pytest.raises(RuntimeError):
                self._make()

    def test_needs_profile_buffer_reservation_is_false(self, dev):
        comm, _, _ = self._make()
        assert comm.needs_profile_buffer_reservation is False

    def test_handshake_gathers_addrs_and_descs(self, dev):
        comm, _, _ = self._make(slots=2, ranks=2)
        with patch.object(torch.distributed, "all_gather_object",
                          side_effect=_gather_identity) as gather:
            comm._handshake()
            comm._handshake()          # 第二次应直接返回，不再 gather
        assert gather.call_count == 2  # addrs + descs
        assert comm._addrs == [comm._addr, comm._addr]
        assert set(comm._descs[0].keys()) == {(0, 0), (0, 1)}

    def test_set_stream_stores_stream(self, dev):
        comm, _, _ = self._make()
        stream = object()
        comm.set_stream(stream)
        assert comm._cuda_stream is stream

    def test_set_stream_sets_device_when_index_present(self, dev):
        comm, _, _ = self._make()
        comm._weights[0][0] = SimpleNamespace(device=SimpleNamespace(index=3))
        comm.set_stream(object())
        assert dev.acl.rt.devices == [3]

    def test_set_transfer_context_builds_expert_to_src_row(self, dev):
        comm, _, _ = self._make(local=2, ranks=2)
        with patch.object(torch.distributed, "all_gather_object",
                          side_effect=_gather_identity):
            comm.set_transfer_context(np.array([0, 1, 2, 3]), layer_idx=3)
        assert comm._layer_idx == 3
        assert comm._expert_to_src_row == [{0: 0, 1: 1}, {2: 0, 3: 1}]

    def test_set_transfer_context_ignores_padding(self, dev):
        comm, _, _ = self._make(local=2, ranks=2)
        with patch.object(torch.distributed, "all_gather_object",
                          side_effect=_gather_identity):
            comm.set_transfer_context(np.array([-1, 1, 2, -1]), layer_idx=0)
        assert comm._expert_to_src_row == [{1: 1}, {2: 0}]

    def test_add_send_is_noop(self, dev):
        comm, _, _ = self._make()
        assert comm.add_send([], 1, 5) is None

    def test_add_recv_builds_read_ops(self, dev):
        comm, _, _ = self._make(local=2, height=8)
        unit = 32
        comm._layer_idx = 0
        comm._expert_to_src_row = [{0: 0}, {0: 1}]
        comm._descs = [{(0, 0): (1000, unit)}, {(0, 0): (2000, unit)}]
        tensor = SimpleNamespace(nbytes=unit, data_ptr=lambda: 500)

        comm.add_recv([tensor], src_rank=1, expert_id=0)

        assert comm._xfer == [(1, [(500, 2000 + unit, unit)])]

    def test_add_recv_rejects_size_mismatch(self, dev):
        comm, _, _ = self._make()
        comm._layer_idx = 0
        comm._expert_to_src_row = [{0: 0}]
        comm._descs = [{(0, 0): (1000, 32)}]
        with pytest.raises(RuntimeError):
            comm.add_recv([SimpleNamespace(nbytes=64, data_ptr=lambda: 0)],
                          src_rank=0, expert_id=0)

    def test_execute_reads_and_clears_state(self, dev):
        comm, _, _ = self._make()
        comm._addrs = ["addr0", "addr1"]
        comm._layer_idx = 0
        comm._xfer = [(1, [(500, 2000, 300)])]

        comm.execute()

        engine = dev.engines[-1]
        assert len(engine.transfers) == 1
        remote, op, descs, timeout = engine.transfers[0]
        assert remote == "addr1"
        assert op == dev.hixl.TransferOp.READ
        assert [(d.local_addr, d.remote_addr, d.len) for d in descs] == [(500, 2000, 300)]
        assert timeout == 30000
        assert comm._xfer == []
        assert comm._expert_to_src_row is None and comm._layer_idx is None

    def test_execute_chunks_large_reads(self, dev):
        comm, _, _ = self._make()
        comm._addrs = ["addr0", "addr1"]
        comm._layer_idx = 0
        comm._xfer = [(1, [(0, 0, 131072 * 2 + 1)])]

        comm.execute()

        descs = dev.engines[-1].transfers[0][2]
        assert [d.len for d in descs] == [131072, 131072, 1]
        assert [d.local_addr for d in descs] == [0, 131072, 262144]

    def test_execute_without_pending_transfer_is_ok(self, dev):
        # 一轮结束后 transfer context 已被清空（_layer_idx=None），此时再调 execute 应无副作用。
        comm, _, _ = self._make()
        comm._layer_idx = None
        comm._xfer = []
        comm.execute()
        assert dev.engines[-1].transfers == []

    def test_split_threshold_boundary_is_inclusive(self, dev):
        # numel(ws[0]) 恰好 == 阈值 时仍走"每专家独立分配"（实现是 >=，钉住这个语义）
        comm, _, _ = self._make(local=2, height=8)
        weights = [[torch.zeros(2, 8, device="cpu")]]
        buffers = [torch.zeros(2, 8, device="cpu")]
        comm._proc_buffers(weights, buffers, split_threshold=8)
        rows = comm._buffer[0]
        assert isinstance(rows, list) and len(rows) == 2        # local=2 -> 每行一个
        assert all(isinstance(r, RawTensor) for r in rows)
        assert all(isinstance(t, RawTensor) for t in comm._weights[0])

    def test_split_threshold_below_keeps_single_buffer(self, dev):
        # 小于阈值：整块权重一个 buffer（不拆）
        comm, _, _ = self._make(local=2, height=8)
        weights = [[torch.zeros(2, 7, device="cpu")]]
        buffers = [torch.zeros(2, 7, device="cpu")]
        comm._proc_buffers(weights, buffers, split_threshold=8)
        assert isinstance(comm._buffer[0], RawTensor)

    def test_multi_rank_end_to_end_read(self, dev):
        # 本 rank=0、对端 rank=1 走通 set_transfer_context -> add_recv -> execute，并用 ctypes 真搬字节校验落地
        comm, _, _ = self._make(local=2, height=4, ranks=2)
        peer_slots = {
            0: torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]], device="cpu"),
            1: torch.tensor([[11., 12., 13., 14.], [15., 16., 17., 18.]], device="cpu"),
        }
        # old_indices: rank0 段 [0,1]、rank1 段 [2,3] -> 从 rank1 取专家 2 = 其第 0 行
        old_indices = np.array([0, 1, 2, 3])

        dst_rows = {}
        with patch.object(torch.distributed, "all_gather_object",
                          side_effect=_peer_gather(peer_slots, _PEER_ADDR)), \
                patch.object(FakeHixlEngine, "transfer_sync", _transfer_with_memcpy):
            comm.set_transfer_context(old_indices, layer_idx=0)
            assert comm._addrs == [comm._addr, _PEER_ADDR]
            assert comm._expert_to_src_row == [{0: 0, 1: 1}, {2: 0, 3: 1}]

            recv = []
            for slot in (0, 1):
                row = torch.zeros(1, 4, device="cpu")
                dst_rows[slot] = row
                recv.append(RawTensor(row))
            comm.add_recv(recv, src_rank=1, expert_id=2)
            comm.execute()

        # READ 描述：从对端各自 slot 的第 0 行读 1 行
        remote_addr, op, descs, timeout = dev.engines[-1].transfers[0]
        assert (remote_addr, op) == (_PEER_ADDR, dev.hixl.TransferOp.READ)
        assert [d.remote_addr for d in descs] == [peer_slots[0].data_ptr(),
                                                  peer_slots[1].data_ptr()]
        assert [d.len for d in descs] == [16, 16]
        # 落地：目标行内容等于对端 slot 的第 0 行
        assert torch.equal(dst_rows[0], peer_slots[0][:1])
        assert torch.equal(dst_rows[1], peer_slots[1][:1])
        # 状态清理
        assert comm._xfer == [] and comm._layer_idx is None

