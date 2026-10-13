# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# EPLB 的 NPU / HIXL 适配：全部通过 omniinfer 的 VLLMPatch 机制实现，vllm 仓库零侵入。
#
# 一、为什么必须用 patch，而不是继承 / 子类化等更解耦的方式
#   我们要改的 EPLB 介入点分两类，继承都够不着：
#     1) 模块级函数：compute_logical_maps、_commit_eplb_maps(_for_layer)、move_to_buffer、
#        move_from_buffer、transfer_layer、create_eplb_communicator。它们是函数而非方法，没有继承点；
#        而且调用方都是 from ... import 早绑定，改定义处不够，必须就地替换模块属性；
#     2) vllm 内部直接构造的类：EplbState / EplbLayerState 由 gpu_model_runner、
#        MoERunner.set_eplb_state 等内部 new 出来，外部没有把子类塞进去的口子。
#   所以「模块属性替换」是唯一既能生效、又不改 vllm 源码的介入点，这正是 VLLMPatch 提供的能力。
#   换成 fork vllm / 在 vllm 里打源码补丁也能实现，但会把上游升级挡在门外，还要长期维护一份 vllm 分支。
#   同理，torch.cuda 的 stream/Event 与 EPLBConfig.communicator 的取值域也没有扩展点，
#   只能分别用 TorchCudaStreamPatch 和 EPLBCommunicatorBackendPatch 兜住。
#
# 二、适配的 vllm 版本
#   本文件针对 vllm 0.25.1 编写（omniinfer 当前依赖的版本），依赖以下内部符号的名字与签名：
#     vllm.distributed.eplb.{eplb_state, eplb_communicator, rebalance_execute, async_worker}
#     vllm.model_executor.layers.fused_moe.layer.EplbLayerState
#     vllm.config.parallel.EPLBConfig.communicator
#   上游一旦改名 / 改签名 / 挪模块，本文件必须同步适配，否则 NPU 上开启 EPLB 会直接初始化失败。
#
# 三、日落条件（满足任一条即可删除本文件）
#   - 上游 EplbState 不再要求 device.type == "cuda" 才设置 cuda_device_index；
#   - 上游 create_eplb_communicator 提供第三方后端注册机制（HIXL 可以直接注册，不必再 patch
#     EPLBCommunicatorBackend 的 Literal 并 rebuild_dataclass）；
#   - 上游 compute_logical_maps 与 commit maps 已足够快（不再需要向量化实现与 pinned H2D 替换）；
#   - 上游把 torch.cuda 的 stream / Event 依赖抽象掉（不再需要 TorchCudaStreamPatch）。
#   在此之前，NPU 上开启 EPLB 必须带着本文件。

import torch
import torch_npu  # noqa: F401  # 仅用于注册 torch.npu 命名空间，下面的 TorchCudaStreamPatch 依赖它
from typing import Literal, Optional
from pydantic.dataclasses import rebuild_dataclass

import vllm.config.parallel as _parallel
import vllm.distributed.eplb.eplb_state as _eplb_state
import vllm.distributed.eplb.eplb_communicator as _eplb_communicator
import vllm.distributed.eplb.rebalance_execute as _rebalance_execute
import vllm.distributed.eplb.async_worker as _async_worker
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer
from vllm.config.parallel import EPLBConfig
from vllm.distributed.parallel_state import get_ep_group

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.distributed.eplb_communicate import HixlEplbCommunicator, _PinnedH2D

# 上游引用必须在 patch apply 前快照（与 from-import 早绑定同理），否则调用时拿到被替换后的自己 -> 递归。
_upstream_create_eplb_communicator = _eplb_communicator.create_eplb_communicator
_upstream_move_to_buffer = _rebalance_execute.move_to_buffer
_upstream_move_from_buffer = _rebalance_execute.move_from_buffer
_upstream_transfer_layer = _rebalance_execute.transfer_layer
_UpstreamEplbState = _eplb_state.EplbState
_UpstreamEplbLayerState = _eplb_state.EplbLayerState


# =====================================================================================
# NPU 实现（由下面的 Patch 注册装到 vllm 上）
# =====================================================================================


# ---- EplbState / EplbLayerState ----

class NPUEplbState(_UpstreamEplbState):

    def __init__(self, parallel_config, device: torch.device):
        super().__init__(parallel_config, device)
        if device.type == "npu" and getattr(self, "cuda_device_index", None) is None:
            # 沿用官方字段名 cuda_device_index，避免改 async_worker / rebalance_execute
            self.cuda_device_index = (
                device.index if device.index is not None
                else torch.npu.current_device()  # type: ignore[attr-defined]
            )
        if self.cuda_device_index is None:
            raise ValueError(f"EPLB requires cuda_device_index, got None for device {device.type}.")


class NPUEplbLayerState(_UpstreamEplbLayerState):  # @dataclass
    replica_table: torch.Tensor = None  # [num_logical_experts] int32：逻辑专家 -> 物理专家（无冗余版，1D）

    def set_layer_state(self, *args, **kw):
        super().set_layer_state(*args, **kw)
        ep = get_ep_group()
        num_local = self.expert_load_view.numel() // ep.world_size
        start = ep.rank_in_group * num_local
        self.local_load = self.expert_load_view[start: start + num_local]  # 本 rank 物理专家切片，等价 physical_expert_start

    def refresh_replica_table(self):
        new_table = self.logical_to_physical_map[:, 0].int()
        if self.replica_table is not None and self.replica_table.shape == new_table.shape:
            self.replica_table.copy_(new_table)  # 原地改，保持张量 identity（ACL graph 稳定）
        else:
            self.replica_table = new_table

    def apply_mapping(self, topk_ids: torch.Tensor):
        return self.replica_table[topk_ids.flatten()].view(*topk_ids.shape)  # 一次 1D gather，替代官方 Triton 映射算子

    def record_load(self, expert_tokens: torch.Tensor):
        self.local_load.add_(expert_tokens.int())  # 累加本 rank 物理专家的 token 数


# ---- compute_logical_maps：向量化替代逐槽位 Python 循环 ----

def npu_compute_logical_maps(physical_to_logical_map: torch.Tensor, num_logical_experts: int):
    # NPU-EPLB: 向量化版 compute_logical_maps：逐槽位 Python 循环 -> 稳定排序 + bincount/cumsum
    device = physical_to_logical_map.device
    if physical_to_logical_map.device.type != "cpu":
        raise ValueError(f"physical_to_logical_map must be on CPU, got {physical_to_logical_map.device}")

    dtype = physical_to_logical_map.dtype

    per_layer = physical_to_logical_map.dim() == 1
    physical_to_logical_map_view = physical_to_logical_map
    if per_layer:
        physical_to_logical_map_view = physical_to_logical_map.unsqueeze(0)
    if len(physical_to_logical_map_view.shape) != 2:
        raise ValueError(f"physical_to_logical_map must be 1D or 2D, got {tuple(physical_to_logical_map_view.shape)}")
    num_layers, num_physical = physical_to_logical_map_view.shape

    valid_mask = physical_to_logical_map_view >= 0
    logical_replica_count = torch.zeros(
        num_layers,
        num_logical_experts,
        dtype=dtype,
        device=device,
    )
    logical_replica_count.scatter_add_(
        1,
        physical_to_logical_map_view.clamp(min=0),
        valid_mask.to(dtype),
    )

    max_replicas = int(logical_replica_count.max().item())
    logical_to_physical_map_out = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=dtype,
        device=device,
    )

    # 按 (layer, logical expert) 稳定排序，保证组内物理槽位升序 -> replica index 与原实现一致
    flat = physical_to_logical_map_view.reshape(-1)
    valid_idx = torch.nonzero(flat >= 0, as_tuple=False).squeeze(1)
    if valid_idx.numel() > 0:
        layer_ids = valid_idx // num_physical
        phys_ids = valid_idx % num_physical
        logical_ids = flat[valid_idx].long()
        group_key = layer_ids * num_logical_experts + logical_ids
        order = torch.argsort(group_key, stable=True)
        sorted_key = group_key[order]
        group_counts = torch.bincount(
            sorted_key, minlength=num_layers * num_logical_experts
        )
        group_start = torch.cumsum(group_counts, 0) - group_counts
        replica_idx = torch.arange(sorted_key.numel()) - group_start[sorted_key]
        logical_to_physical_map_out[
            layer_ids[order], logical_ids[order], replica_idx
        ] = phys_ids[order].to(dtype)

    if per_layer:
        return logical_to_physical_map_out.squeeze(0), logical_replica_count.squeeze(0)
    return logical_to_physical_map_out, logical_replica_count


# ---- pinned 异步 H2D 的 commit（逐层版 / 整模型版共用逻辑）----

def _h2d(name, src, dst, fill=None):
    # NPU-EPLB: 按 name 常驻一块 pinned 缓冲（shape/dtype 取自 dst），返回本次 H2D 的事件
    return _PinnedH2D.get(name, *dst.shape, dtype=dst.dtype, fill=fill).h2d(src, dst)


def _commit_logical_maps(src, num_logical_experts, dst_logical, dst_replica, tag=""):
    # NPU-EPLB: pinned H2D 的公共部分：commit logical_to_physical_map + logical_replica_count
    new_logical, new_replica_count = npu_compute_logical_maps(src, num_logical_experts)
    _h2d(f"logical_to_physical_map{tag}", new_logical, dst_logical, fill=-1)  # 尾部 padding 常驻 -1
    _h2d(f"logical_replica_count{tag}", new_replica_count, dst_replica)


def npu_commit_eplb_maps(model_state, new_physical_to_logical_map):
    # NPU-EPLB: pinned 异步 H2D 提交整模型 map
    src = new_physical_to_logical_map
    dst = model_state.physical_to_logical_map
    # Rare Case: 物理专家数变化时直接换整张 map（只在 vLLM 运行中 GPU 数变化时发生）
    if src.shape[1] != dst.shape[1]:
        model_state.physical_to_logical_map = src.to(dst.device)
    else:
        _h2d("physical_to_logical_map", src, dst)

    _commit_logical_maps(
        src,
        model_state.logical_to_physical_map.shape[1],
        model_state.logical_to_physical_map,
        model_state.logical_replica_count,
    )

    for layer in model_state.model.moe_layers:
        eplb_state: NPUEplbLayerState = layer.eplb_state
        if eplb_state is not None:
            eplb_state.refresh_replica_table()  # 重排提交后刷新缓存的 replica_table


def npu_commit_eplb_maps_for_layer(model_state, new_physical_to_logical_map, layer):
    # NPU-EPLB: pinned 异步 H2D 提交单层 map（同 npu_commit_eplb_maps，逐层版）
    src = new_physical_to_logical_map
    dst = model_state.physical_to_logical_map[layer]
    if src.shape != dst.shape:
        raise ValueError(
            "The number of physical experts must stay the same while running Async EPLB. "
            f"Current number of physical experts: {dst.shape[0]}. New number of physical "
            f"experts {src.shape[0]}."
        )
    # 逐层提交会连续调用，pinned 缓冲按 layer 分开（名字带 layer）
    _h2d(f"physical_to_logical_map:{layer}", src, dst)

    _commit_logical_maps(
        src,
        model_state.logical_to_physical_map.shape[1],
        model_state.logical_to_physical_map[layer],
        model_state.logical_replica_count[layer],
        tag=f":{layer}",
    )

    eplb_state: NPUEplbLayerState = model_state.model.moe_layers[layer].eplb_state
    if eplb_state is not None:
        eplb_state.refresh_replica_table()


# ---- RawTensor 的 raw stream 绑定 + move_* / transfer_layer 的 wrap ----

def _bind_raw_stream(expert_weights, cuda_stream=None) -> None:
    # NPU-EPLB: 绑定本层 RawTensor 的共享 raw stream；非 HIXL/非 NPU（普通张量）自动跳过
    if not expert_weights:
        return
    if not hasattr(expert_weights[0], "set_stream"):
        return
    if cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            expert_weights[0].set_stream()
    else:
        expert_weights[0].set_stream()


def _arg(name, idx, args, kwargs):
    # NPU-EPLB: 按上游参数位置取实参（move_* / transfer_layer 的签名在 patch 时已核对）
    if name in kwargs:
        return kwargs[name]
    return args[idx] if len(args) > idx else None


def npu_move_to_buffer(*args, **kwargs):
    _bind_raw_stream(
        _arg("expert_weights", 3, args, kwargs),
        _arg("cuda_stream", 5, args, kwargs),
    )
    return _upstream_move_to_buffer(*args, **kwargs)


def npu_move_from_buffer(*args, **kwargs):
    _bind_raw_stream(_arg("expert_weights", 0, args, kwargs))
    return _upstream_move_from_buffer(*args, **kwargs)


def npu_transfer_layer(*args, **kwargs):
    cuda_stream = _arg("cuda_stream", 7, args, kwargs)
    if cuda_stream is not None:
        cuda_stream.synchronize()  # 先等上一层读 expert_buffer 真正读完，再覆写
    return _upstream_transfer_layer(*args, **kwargs)


# ---- create_eplb_communicator：HIXL 走后端 NPU 实现 ----

def create_npu_eplb_communicator(
    group_coordinator,
    backend: str,
    expert_weights: list[list[torch.Tensor]],
    expert_buffer: list[torch.Tensor],
) -> _eplb_communicator.EplbCommunicator:
    if backend == "hixl":
        return HixlEplbCommunicator(  # NPU 版：每专家独立分配 + RawTensor
            cpu_group=group_coordinator.cpu_group,
            all_expert_weights=expert_weights,
            expert_buffer=expert_buffer,
        )
    return _upstream_create_eplb_communicator(  # 其余后端交给官方实现
        group_coordinator, backend, expert_weights, expert_buffer)


# =====================================================================================
# Patch 注册：把上面的 NPU 实现装到 vllm 上
# =====================================================================================


@register_patch("TorchCudaStreamPatch", torch.cuda)
class TorchCudaStreamPatch(VLLMPatch):  # torch.cuda 的 stream/Event 等透明换成 torch.npu
    _attr_names_to_apply = ["stream", "Stream", "Event", "current_stream"]
    stream = torch.npu.stream
    Stream = torch.npu.Stream
    Event = torch.npu.Event
    current_stream = torch.npu.current_stream


@register_patch("EplbStatePatch", _eplb_state)
class EplbStatePatch(VLLMPatch):  # eplb_state 模块内的类/函数
    # create_eplb_communicator / move_from_buffer 在 eplb_state 里是 from ... import 早绑定，
    # 只换 vllm 模块里的定义不够，必须在这里就地替换。
    _attr_names_to_apply = [
        "EplbState",
        "EplbLayerState",
        "create_eplb_communicator",
        "compute_logical_maps",
        "_commit_eplb_maps",
        "_commit_eplb_maps_for_layer",
        "move_from_buffer",
    ]
    EplbState = NPUEplbState
    EplbLayerState = NPUEplbLayerState
    create_eplb_communicator = create_npu_eplb_communicator
    compute_logical_maps = npu_compute_logical_maps
    _commit_eplb_maps = npu_commit_eplb_maps
    _commit_eplb_maps_for_layer = npu_commit_eplb_maps_for_layer
    move_from_buffer = npu_move_from_buffer


@register_patch("EplbCommunicatorPatch", _eplb_communicator)
class EplbCommunicatorPatch(VLLMPatch):
    # 换掉 eplb_communicator 模块里的工厂：晚于 patch 才 from ... import 它的模块（如 elastic_execute）会拿到 NPU 版。
    _attr_names_to_apply = ["create_eplb_communicator"]
    create_eplb_communicator = create_npu_eplb_communicator


@register_patch("EplbFusedMoELayerPatch", _fused_moe_layer)
class EplbFusedMoELayerPatch(VLLMPatch):  # fused_moe.layer 早于 patch 被 import，其 EplbLayerState 需就地替换
    _attr_names_to_apply = ["EplbLayerState"]
    EplbLayerState = NPUEplbLayerState


@register_patch("EplbRebalanceNpuPatch", _rebalance_execute)
class EplbRebalanceNpuPatch(VLLMPatch):  # rebalance_execute 里的 move_* / transfer_layer 用 wrapper 替换
    _attr_names_to_apply = ["transfer_layer", "move_to_buffer", "move_from_buffer"]
    transfer_layer = npu_transfer_layer
    move_to_buffer = npu_move_to_buffer
    move_from_buffer = npu_move_from_buffer


@register_patch("AsyncWorkerTransferLayerNpuPatch", _async_worker)
class AsyncWorkerTransferLayerNpuPatch(VLLMPatch):  # async_worker 里 transfer_layer 是早绑定（from ... import），要就地换
    _attr_names_to_apply = ["transfer_layer"]
    transfer_layer = npu_transfer_layer


_NEW_EPLB_BACKEND = Literal["torch_nccl", "torch_gloo", "nixl", "pynccl", "hixl"]
_NEW_EPLB_COMMUNICATOR_FIELD = Optional[_NEW_EPLB_BACKEND]


@register_patch("EPLBCommunicatorBackendPatch", _parallel)
class EPLBCommunicatorBackendPatch(VLLMPatch):  # Literal 加 hixl 后重建 EPLBConfig.communicator 字段
    _attr_names_to_apply = ["EPLBCommunicatorBackend"]
    EPLBCommunicatorBackend = _NEW_EPLB_BACKEND

    @classmethod
    def apply(cls):
        super().apply()
        EPLBConfig.__annotations__["communicator"] = _NEW_EPLB_COMMUNICATOR_FIELD
        EPLBConfig.__pydantic_fields__["communicator"].annotation = _NEW_EPLB_COMMUNICATOR_FIELD
        EPLBConfig.__dataclass_fields__["communicator"].type = _NEW_EPLB_COMMUNICATOR_FIELD
        rebuild_dataclass(EPLBConfig, force=True)
