# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
import socket
import numpy as np

import acl
import torch
import torch_npu

from torch.distributed import ProcessGroup
from vllm.distributed.eplb.eplb_communicator import EplbCommunicator

# NPU-EPLB: 只有 hixl 是可选依赖——未开启 EPLB 的环境（例如未安装 hixl 的 CI/UT 机器）
# 不要求装它，但 import 本模块（以及依赖本模块的 patch）必须仍然成功。
try:
    import hixl
except ImportError:  # pragma: no cover - 未安装 hixl 的环境（例如未开启 EPLB 的 UT）
    hixl = None


def _hixl_engine(options: dict, retries: int = 8) -> tuple[hixl.Hixl, str]:
    # NPU-EPLB: 取本机 IP + 空闲端口并初始化 HIXL；端口 bind 后立即释放有被抢占窗口，失败就换地址重试
    for _ in range(retries):
        ip = socket.gethostbyname(socket.gethostname())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((ip, 0))
            addr = f"{ip}:{s.getsockname()[1]}"
        engine = hixl.Hixl()
        ret = engine.initialize(addr, options)
        if ret == hixl.SUCCESS:
            return engine, addr
    # NPU-EPLB: 循环外必须 raise 而非 assert——既满足“生产代码禁用 assert”，也避免函数存在隐式返回 None 的分支
    raise RuntimeError(f"hixl init failed: {ret=} {addr=}")


class RawTensor:
    # NPU-EPLB: 伪装成 torch.Tensor 的裸地址句柄（不继承 torch.Tensor）。
    # 只把 t[i] 和 copy_ 换成 acl 裸 memcpy，其余属性转发给真实张量，
    # 这样 move_from/to_buffer 的原有代码不用改。行字节数由 stride 推出。
    # s: raw stream 的共享槽位（单元素 list），不传则新建；传入别的 RawTensor 的 s 即共用，改一处全都生效。
    __slots__ = ("_t", "ptr", "nbytes", "row_bytes", "s")

    def __init__(self, t=None, *, ptr=None, nbytes=None, row_bytes=None, s=None):
        self._t = t
        self.s = [None] if s is None else s
        if ptr is None:
            ptr = int(t.data_ptr())
            nbytes = int(t.nbytes)
            if t.dim() >= 2:
                # 行首地址 = base + i * row_bytes 的前提：第 0 维是专家维且行连续
                if not t.is_contiguous():
                    raise ValueError(f"RawTensor requires a contiguous tensor, got shape {tuple(t.shape)}")
                row_bytes = nbytes // int(t.shape[0])
            else:
                row_bytes = nbytes
        self.ptr = ptr
        self.nbytes = nbytes
        self.row_bytes = row_bytes

    def __getattr__(self, name):
        # __getattr__ 仅在常规查找失败时触发：把 .shape/.device/.numel() 等转发给真实张量
        try:
            t = object.__getattribute__(self, "_t")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(t, name)

    def data_ptr(self):
        return self.ptr

    def __len__(self):
        return int(self._t.shape[0])

    def set_stream(self, s=None):
        if s is None: # NPU-EPLB: s=None 时自行取当前 raw stream；只改共享槽位，同组 RawTensor 一起生效
            s = torch_npu._C._npu_getCurrentRawStream(torch_npu.npu.current_device())
        self.s[0] = s
        return s

    def __getitem__(self, i):
        i = int(i)
        return RawTensor(
            ptr=self.ptr + i * self.row_bytes,
            nbytes=self.row_bytes,
            row_bytes=self.row_bytes,
            s=self.s,
        )

    def copy_(self, other, non_blocking=True):
        if not non_blocking:
            raise ValueError("RawTensor.copy_ only supports non_blocking=True")
        acl.rt.memcpy_async(
            self.ptr, self.nbytes, other.ptr, other.nbytes, 0x3, self.s[0]
        )
        return self


class _PinnedH2D:
    # NPU-EPLB: pinned 异步 H2D（PipeTensor 风格）：
    # 创建时定 shape/dtype，每个实例只维护一个 pinned 张量；h2d(src, dst) 把 src 拷进它再上传，不碰 src。
    _instances: dict = {}  # NPU-EPLB: 实例按名字挂在类上，同名复用（可能同时存在多个）

    def __init__(self, *shape, dtype=torch.int32, fill=None):
        self.cpu = torch.empty(*shape, dtype=dtype, pin_memory=True)
        if fill is not None:
            self.cpu.fill_(fill)
        self._done = None  # 上一次 H2D 的读完成事件

    @classmethod
    def get(cls, name, *shape, dtype=torch.int32, fill=None) -> "_PinnedH2D":
        # NPU-EPLB: 同名同 shape/dtype 复用，否则（首次或 dst shape 变了）重建
        pt = _PinnedH2D._instances.get(name)
        if pt is None or tuple(pt.cpu.shape) != tuple(shape) or pt.cpu.dtype != dtype:
            pt = _PinnedH2D._instances[name] = cls(*shape, dtype=dtype, fill=fill)
        return pt

    def h2d(self, src, dst) -> torch.cuda.Event:
        # 覆盖 pinned 缓冲前先等上一次 H2D 真正读完（跨轮/连续调用都安全）；返回本次 H2D 的事件
        if self._done is not None:
            self._done.synchronize()
        self.cpu[..., : src.shape[-1]].copy_(src)
        dst.copy_(self.cpu, non_blocking=True)
        if self._done is None:
            self._done = torch.cuda.Event()
        self._done.record()
        return self._done


class HixlEplbCommunicator(EplbCommunicator):

    def __init__(self,
        cpu_group: ProcessGroup,
        all_expert_weights: list[list[torch.Tensor]],
        expert_buffer: list[torch.Tensor],
    ):
        self._gloo = cpu_group
        self._ranks = int(cpu_group.size())
        self._proc_buffers(all_expert_weights, expert_buffer)
        self._init_engine()
        self._xfer = []
        self._log_initialized()

    def _proc_buffers(self,
        expert_weights: list[list[torch.Tensor]],
        expert_buffer: list[torch.Tensor],
        split_threshold: int = 1000000, # elements
    ):
        if not (expert_weights and expert_buffer):
            raise ValueError("expert_weights and expert_buffer must be non-empty")
        self._local = expert_weights[0][0].size(0)
        s: list = [None]  # NPU-EPLB: 所有权重/缓冲区共用同一个 raw stream 槽位，绑一次全生效
        # 大权重每专家独立分配缓冲区（HIXL 往大块分配的内部地址写会静默写错）——用原始张量判定
        for i, old in enumerate(expert_buffer):
            if not isinstance(old, torch.Tensor):
                raise TypeError(f"expert_buffer must be torch.Tensor, got {type(old)}")
            ws = expert_weights[0][i]
            if ws[0].numel() >= split_threshold:
                # NPU-EPLB: 只要同形状的空缓冲区；w4a8 权重 clone 会下发 TransData 算子导致 tiling 失败，故用 new_empty
                expert_buffer[i] = [RawTensor(w.new_empty(*w.shape), s=s) for w in ws]
            else:
                expert_buffer[i] = RawTensor(old, s=s)

        # 权重包成伪张量 RawTensor（只做裸 memcpy，避免 torch.Tensor 包装开销）
        for layer in expert_weights:
            for i, t in enumerate(layer):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"expert_weights must be torch.Tensor, got {type(t)}")
                layer[i] = RawTensor(t, s=s)

        self._buffer = expert_buffer
        self._weights = expert_weights

    def _init_engine(self):
        resource_config = {
            "comm_resource_config.protocol_desc": ["roce:device", "roce:host"]
        }
        options = {
            hixl.OPTION_AUTO_CONNECT: "1",
            hixl.OPTION_GLOBAL_RESOURCE_CONFIG: json.dumps(resource_config),
        }
        self._engine, self._addr = _hixl_engine(options)

        #register_buffers
        all_tensors: list[RawTensor] = [t for layer in self._weights for t in layer]
        for rows in self._buffer:
            all_tensors += rows if isinstance(rows, list) else [rows]

        self._registered_handles = []
        for tensor in all_tensors:
            ret, handle = self._engine.register_mem(
                hixl.MemDesc(int(tensor.data_ptr()), int(tensor.nbytes)),
                hixl.MemType.MEM_DEVICE,
            )
            if ret != hixl.SUCCESS:
                raise RuntimeError(f"hixl register_mem failed at {tensor.data_ptr()}, ret={ret}")
            self._registered_handles.append(handle)

    def _handshake(self):
        if hasattr(self, "_addrs"):
            return

        addrs = [None] * self._ranks
        torch.distributed.all_gather_object(
            addrs, self._addr, group=self._gloo)

        desc = {
            (layer_id, i): (x.data_ptr(), x.nbytes // self._local)
            for layer_id, layer in enumerate(self._weights)
            for i, x in enumerate(layer)
        }
        local_keys = set(desc.keys())
        descs = [None] * self._ranks
        torch.distributed.all_gather_object(
            descs, desc, group=self._gloo)

        for peer in descs:
            if peer is None:
                raise RuntimeError("hixl handshake: remote desc is None")
            if set(peer.keys()) != local_keys:
                raise RuntimeError(f"hixl handshake: desc keys mismatch, extra={set(peer.keys()) - local_keys}")
            for key in local_keys:
                _, unit = desc[key]
                _, _unit = peer[key]
                if unit != _unit:
                    raise RuntimeError(f"hixl handshake: per-expert size mismatch, {unit} != {_unit}")

        self._descs = descs
        self._addrs = addrs

    @property
    def needs_profile_buffer_reservation(self):
        return False # READ 由接收方发起，profile 无需 dummy 集合通信（与 NIXL 一致）。

    def set_stream(self, cuda_stream):
        # worker 线程里先设好 device（device context 是线程私有的）；transfer 是 host 阻塞调用，stream 只记下来
        self._cuda_stream = cuda_stream
        dev = self._weights[0][0].device.index
        if dev is not None:
            acl.rt.set_device(dev)

    def set_transfer_context(self, old_indices: np.ndarray, layer_idx: int):
        self._handshake()
        if self._xfer:
            raise RuntimeError("set_transfer_context called while a transfer is pending")
        self._layer_idx = layer_idx
        rank_experts = old_indices[: self._ranks * self._local].reshape(self._ranks, -1)
        self._expert_to_src_row = [
            {int(eid): i for i, eid in enumerate(row) if eid != -1}
            for row in rank_experts
        ]

    def add_send(self, tensors, dst_rank: int, expert_id: int):
        pass # No-op：READ 由接收方发起，发送方权重已 register 原地可读（同 NIXL）。

    def add_recv(self, tensors, src_rank: int, expert_id: int):
        if self._expert_to_src_row is None:
            raise RuntimeError("add_recv called before set_transfer_context: expert_to_src_row is None")
        if self._layer_idx is None:
            raise RuntimeError("add_recv called before set_transfer_context: layer_idx is None")
        src_row = self._expert_to_src_row[src_rank][expert_id]
        layer_idx = self._layer_idx

        ops: list[tuple[int, int, int]] = []
        for i, x in enumerate(tensors):
            ptr, unit = self._descs[src_rank][(layer_idx, i)]
            if x.nbytes != unit:
                raise RuntimeError(f"tensor {i}: local nbytes {x.nbytes} != remote {unit}")
            ops.append((x.data_ptr(), ptr + src_row * unit, x.nbytes))
        self._xfer.append((src_rank, ops))

    def execute(self):
        if self._layer_idx is None and self._xfer:
            raise RuntimeError("execute called with pending transfers but no transfer context")

        def read(
            rank: int,
            ops: list[tuple[int, int, int]],
            chunk_size: int = 131072,  # 128k
            xfer_timeout: int = 30000, # ms
        ):
            if not ops:
                raise ValueError("read: ops must be non-empty")
            descs: list = []
            for local_addr, remote_addr, nbytes in ops:
                off = 0
                while off < nbytes:
                    chunk = min(chunk_size, nbytes - off)
                    descs.append(
                        hixl.TransferOpDesc(
                            local_addr=local_addr + off,
                            remote_addr=remote_addr + off,
                            len=chunk,
                        )
                    )
                    off += chunk

            ret = self._engine.transfer_sync(
                self._addrs[rank],
                hixl.TransferOp.READ,
                descs,
                timeout_in_millis=xfer_timeout,
            )
            if ret != hixl.SUCCESS:
                raise RuntimeError(f"hixl read failed: {ret=} {rank=}")

        for rank, ops in self._xfer:
            read(rank, ops)
        self._gloo.barrier().wait()
        self._xfer.clear()
        self._expert_to_src_row = None
        self._layer_idx = None
