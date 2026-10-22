# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import logging
import math
import mmap
import os
import shutil
import tempfile
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch_npu
from torch import nn
from vllm.triton_utils import tl, triton

from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mla.common import MLACommonMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PanGuEngramConfig:
    """Collect the configuration needed for PanGu Engram inference.

    Parse and validate the checkpoint's Engram fields, including layer indices,
    n-gram orders, head dimensions, and mHC settings. Read TP head sharding from
    the model JSON and expose derived dimensions used by hashing and lookup.
    """

    layer_ids: tuple[int, ...]  # Zero-based model layer indices using Engram.
    vocab_sizes: tuple[int, ...]  # Base capacity per n-gram order; heads use distinct prime capacities.
    max_ngram_size: int  # Maximum n-gram length; includes orders 2 through this value.
    num_heads_per_ngram: int  # Number of hash heads per n-gram order.
    embedding_dim_per_ngram: int  # Concatenated embedding width of all heads in one n-gram order.
    pad_token_id: int  # Padding ID for missing history tokens.
    seed: int  # Random seed for generating hash multipliers.
    hidden_size: int  # Hidden width of each mHC stream.
    num_streams: int  # Number of mHC streams.
    post_norm: bool  # Normalize the Engram output before adding it to the residual.
    rms_norm_eps: float  # RMSNorm epsilon.
    table_dtype: torch.dtype  # Engram table data type.
    dummy_weights: bool = False  # Use dummy weight loading.
    shard_heads: bool = True  # Shard heads across TP ranks; False disables head sharding.

    @property
    def num_ngram_orders(self) -> int:
        return self.max_ngram_size - 1

    @property
    def num_heads(self) -> int:
        return self.num_ngram_orders * self.num_heads_per_ngram

    @property
    def head_dim(self) -> int:
        return self.embedding_dim_per_ngram // self.num_heads_per_ngram

    @property
    def memory_dim(self) -> int:
        return self.num_ngram_orders * self.embedding_dim_per_ngram

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> PanGuEngramConfig | None:
        hf_config = vllm_config.model_config.hf_config
        layer_ids = getattr(hf_config, "engram_layer_ids", None)
        if not layer_ids:
            return None
        config = cls.from_hf_config(
            hf_config,
            table_dtype=vllm_config.model_config.dtype,
            dummy_weights=(getattr(getattr(vllm_config, "load_config", None), "load_format", None) == "dummy"),
        )
        from omni_npu.model_config.config_loader.loader import model_extra_config

        options = model_extra_config.operator_opt_config
        additional = getattr(vllm_config, "additional_config", None)
        if isinstance(additional, dict) and "engram_config" in additional:
            raise ValueError(
                "Move Engram options to model JSON operator_optimization_config: "
                "engram_table_location, engram_head_sharding"
            )
        return replace(
            config,
            shard_heads=options.engram_head_sharding != "none",
        )

    @classmethod
    def from_hf_config(
        cls,
        config: PretrainedConfig,
        *,
        table_dtype: torch.dtype = torch.bfloat16,
        dummy_weights: bool = False,
    ) -> PanGuEngramConfig | None:
        raw_layer_ids = getattr(config, "engram_layer_ids", None)
        if not raw_layer_ids:
            return None
        if getattr(config, "engram_use_compressed_tokenizer", False):
            raise ValueError("omni-npu Engram does not support compressed_tokenizer")
        if getattr(config, "engram_use_short_conv", False):
            raise ValueError("omni-npu Engram does not support short_conv")
        if getattr(config, "engram_enable_contextual_emb", False):
            raise ValueError("omni-npu Engram does not support contextual embeddings")
        if getattr(config, "engram_enable_cross_head_mix", False):
            raise ValueError("omni-npu Engram does not support cross-head mixing")
        if hasattr(config, "engram_concat_fusion") and not config.engram_concat_fusion:
            raise ValueError("omni-npu Engram only supports concat fusion")

        max_ngram_size = int(config.engram_max_ngram_size)
        if max_ngram_size < 2:
            raise ValueError("engram_max_ngram_size must be at least 2")
        num_orders = max_ngram_size - 1
        raw_vocab_sizes = config.engram_vocab_size
        if isinstance(raw_vocab_sizes, int):
            vocab_sizes = (int(raw_vocab_sizes),) * num_orders
        else:
            vocab_sizes = tuple(int(size) for size in raw_vocab_sizes)
            if len(vocab_sizes) == 1:
                vocab_sizes *= num_orders
        if len(vocab_sizes) != num_orders or any(size < 2 for size in vocab_sizes):
            raise ValueError("engram_vocab_size must contain one positive size per n-gram order")

        num_heads = int(config.engram_n_head_per_ngram)
        embed_dim = int(config.engram_n_embed_per_ngram)
        if num_heads < 1 or embed_dim < 1 or embed_dim % num_heads:
            raise ValueError("engram_n_embed_per_ngram must be divisible by " "engram_n_head_per_ngram")
        layer_ids = tuple(int(layer_id) for layer_id in raw_layer_ids)
        if len(set(layer_ids)) != len(layer_ids) or any(layer_id < 0 for layer_id in layer_ids):
            raise ValueError("engram_layer_ids must be unique non-negative indices")
        num_hidden_layers = int(getattr(config, "num_hidden_layers", max(layer_ids) + 1))
        if any(layer_id >= num_hidden_layers for layer_id in layer_ids):
            raise ValueError("engram_layer_ids contains an out-of-range layer")
        pad_token_id = int(config.engram_pad_id)
        if pad_token_id < 0:
            raise ValueError("engram_pad_id must be non-negative")
        hidden_size = int(config.hidden_size)
        num_streams = int(getattr(config, "mhc_num_stream", 1))
        if not getattr(config, "use_mhc", False) or num_streams <= 1:
            raise ValueError("PanGu Engram requires mHC enabled with mhc_num_stream > 1")
        if hidden_size < 1:
            raise ValueError("Engram hidden_size must be positive")

        return cls(
            layer_ids=layer_ids,
            vocab_sizes=vocab_sizes,
            max_ngram_size=max_ngram_size,
            num_heads_per_ngram=num_heads,
            embedding_dim_per_ngram=embed_dim,
            pad_token_id=pad_token_id,
            seed=int(getattr(config, "engram_seed", 42)),
            hidden_size=hidden_size,
            num_streams=num_streams,
            post_norm=bool(getattr(config, "sandwich_norm", False)),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-5)),
            table_dtype=table_dtype,
            dummy_weights=dummy_weights,
        )


class EngramLayout:
    """Describe how n-gram head tables are concatenated in each Engram layer.

    Assign a prime capacity to each (n-gram order, head) pair and calculate its
    starting row, the total table size, and embedding dimensions. Hashing uses
    the capacities and offsets; weight loading and lookup use the same layout.
    PanGu restarts prime assignment in each layer, so layers share this layout.
    """

    def __init__(self, config: PanGuEngramConfig) -> None:
        self.config = config
        self.layer_ids = config.layer_ids
        self.max_ngram_size = config.max_ngram_size
        self.n_heads = config.num_heads_per_ngram  # Hash heads per n-gram order.
        self.head_dim = config.head_dim
        self.pad_token_id = config.pad_token_id
        self.n_hash_cols = config.num_heads  # Total hash heads across all n-gram orders.
        seen: set[int] = set()
        primes = []
        for target in config.vocab_sizes:
            sizes, current = [], target - 1
            for _ in range(self.n_heads):
                current = self._find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            primes.append(tuple(sizes))
        # Pangu restarts prime assignment in each layer; DeepSeek keeps a
        # cross-layer `seen` set. Reuse this layout without changing Pangu hashes.
        # Shape of self.primes: [Engram layers, n-gram orders, heads per order].
        self.primes = tuple(tuple(primes) for _ in self.layer_ids)
        # Flattened prime capacities in n-gram/head order: embedding rows per hash head.
        # Shape of self.head_sizes: [total hash heads]; all Engram layers use the same capacities.
        self.head_sizes = tuple(p for order in primes for p in order)
        self.num_embeddings = (sum(self.head_sizes),) * len(self.layer_ids)
        offsets = np.cumsum([0, *self.head_sizes[:-1]])
        # Shape of self.offsets: [Engram layers, total hash heads]; each entry is a head's starting row.
        self.offsets = torch.from_numpy(np.tile(offsets, (len(self.layer_ids), 1)))

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
        if n >= 2**32:
            raise ValueError("Engram prime search exceeds the supported range n < 2**32")
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 7, 61):
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _find_next_prime(cls, start: int, seen_primes: set[int]) -> int:
        """The smallest prime above `start` that has not been handed out yet."""
        candidate = start + 1
        while not cls._is_prime(candidate) or candidate in seen_primes:
            candidate += 1
        return candidate


# Adapted from vLLM's DPSharedEngramStorage (Apache-2.0), commit 23cfaad49701:
# vllm/models/deepseek_v41/nvidia/engram.py. Registration uses the NPU backend.
class DPSharedEngramStorage:
    """Allocate and register an Engram table shared by ranks within one node.

    Map the same host backing file on participating ranks and register each
    mapping for direct NPU reads. Only the group's writer loads weights; peers
    synchronize before lookup. The group shares either one TP head shard across
    DP replicas or the full table across TP/DP ranks when heads are unsharded.
    """

    def __init__(self, shape: tuple[int, int], dtype: torch.dtype, group: GroupCoordinator) -> None:
        self.group = group
        self.world_size = group.world_size
        self.is_writer = group.rank_in_group == 0  # Creates shared storage and writes Engram table weights.
        self.tensor = self._allocate(shape, dtype)  # CPU tensor backed by the shared host mapping.
        self._device_view = None  # NPU view of the same host storage; registered lazily by device_view().

    def device_view(self) -> torch.Tensor:
        """Register the original mmap on every sharing rank without a private copy."""
        if self._device_view is not None:
            return self._device_view
        view, error = None, None
        try:
            from omni_npu.allocator import npu_uva

            torch.npu.current_stream()
            # C++ extension: register the shared host table and return an NPU view
            # of the same storage, keeping its registration alive without copying it.
            view = npu_uva.register_mapped_cpu_tensor(self.tensor, torch.npu.current_device())
        except Exception as exc:
            error = f"host registration: {type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        torch.distributed.all_gather_object(errors, error, group=self.group.cpu_group)
        failures = [f"EDP rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
        if failures:
            del view
            raise RuntimeError("Engram host registration failed: " + "; ".join(failures))
        self._device_view = view
        return view

    def _allocate(self, shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
        num_bytes = math.prod(shape) * torch.empty((), dtype=dtype, device="cpu").element_size()
        group = self.group
        with ExitStack() as stack:
            path, error = None, None
            if self.is_writer:
                stage = "check shared-memory capacity"
                try:
                    self._check_shm_free_space(num_bytes)
                    stage = "create backing file"
                    backing_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(prefix="vllm_engram_", dir="/dev/shm")
                    )
                    stage = "allocate shared memory"
                    backing_file.truncate(num_bytes)
                    os.posix_fallocate(backing_file.fileno(), 0, num_bytes)
                    path = backing_file.name
                except Exception as exc:
                    error = f"{stage}: {type(exc).__name__}: {exc}"
            path, error = group.broadcast_object((path, error))
            if error is not None:
                raise RuntimeError("Engram shared-memory creation failed on rank 0: " + error)
            mapping = None
            stage = "open backing file"
            try:
                with open(path, "r+b") as file:
                    stage = "mmap"
                    mapping = mmap.mmap(file.fileno(), num_bytes, flags=mmap.MAP_SHARED)
            except Exception as exc:
                error = f"{stage}: {type(exc).__name__}: {exc}"
            errors = [None] * self.world_size
            # Fence peer mappings before each host's writer unlinks its file.
            torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
            failures = [f"EDP rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
            if failures:
                if mapping is not None:
                    mapping.close()
                raise RuntimeError("Engram shared-memory mapping failed: " + "; ".join(failures))
            # Tensor storage retains the mmap after the temporary file is unlinked.
            return torch.frombuffer(mapping, dtype=dtype).view(shape)

    def synchronize(self) -> None:
        torch.distributed.barrier(group=self.group.cpu_group)

    def load_weight(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Only the writer loads; every peer waits before using shared rows."""
        if self.is_writer:
            source = loaded_weight.detach().narrow(0, param.engram_vocab_start, param.shape[0])
            rows_per_chunk = max(1, (256 << 20) // (param.shape[1] * max(source.element_size(), param.element_size())))
            for begin in range(0, param.shape[0], rows_per_chunk):
                end = min(begin + rows_per_chunk, param.shape[0])
                param.data[begin:end].copy_(source[begin:end].to(device=param.device, dtype=param.dtype))
        self.synchronize()

    @staticmethod
    def _check_shm_free_space(required_bytes: int) -> None:
        from vllm.distributed.device_communicators import shm_broadcast

        check = getattr(shm_broadcast, "check_shm_free_space", None)
        if check is not None:
            check(required_bytes)
            return
        # Backport the filesystem check from vLLM b68408043d80 for older runtimes.
        # Its newer cgroup headroom logging is not required for this precheck.
        shm_path = "/dev/shm"
        if os.path.isdir(shm_path):
            free_bytes = shutil.disk_usage(shm_path).free
            if required_bytes > free_bytes:
                mib = 1 << 20
                raise RuntimeError(
                    f"Insufficient space in {shm_path} for shared-memory allocation: "
                    f"{required_bytes / mib:.0f} MiB required, "
                    f"{free_bytes / mib:.0f} MiB free. Increase {shm_path} "
                    "(e.g. --shm-size or --ipc=host)."
                )


class ParallelEngramEmbedding(nn.Module):
    """Load and look up shared host embeddings, optionally sharded by TP head.

    Select this rank's complete head tables from EngramLayout and allocate them
    through DPSharedEngramStorage. Preserve host storage during module device
    moves, load the corresponding checkpoint rows, and launch NPU lookup into
    a supplied buffer. The direct forward path also gathers TP head results.
    """

    def __init__(self, layout: EngramLayout, layer_idx: int, group: GroupCoordinator) -> None:
        super().__init__()
        head_sizes = layout.head_sizes  # Number of embedding rows in each hash head's table.
        layer_hash_index = layout.layer_ids.index(layer_idx)  # Index among Engram layers.
        self.num_embeddings = layout.num_embeddings[layer_hash_index]  # Full table row count.
        self.dim = layout.head_dim  # Embedding width per row.
        self.n_hash_cols = layout.n_hash_cols  # Total hash heads across all n-gram orders.
        self.model_tp_size = get_tp_group().world_size  # Model TP size, also used for SP token slices.
        self.tp_size = self.model_tp_size if layout.config.shard_heads else 1  # Head shard count.
        shard_size = self.tp_size
        shard_rank = get_tp_group().rank_in_group if layout.config.shard_heads else 0
        # Heads per shard, including padding.
        self.part_n_hash_cols = (self.n_hash_cols + shard_size - 1) // shard_size
        if (shard_size - 1) * self.part_n_hash_cols >= self.n_hash_cols:
            raise ValueError(f"Engram sharding leaves ranks without hash heads: {self.n_hash_cols} / {shard_size}")
        self.head_start = shard_rank * self.part_n_hash_cols  # First head owned by this shard.
        head_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        self.vocab_start_idx = sum(head_sizes[: self.head_start])  # First row in the full table.
        self.vocab_end_idx = sum(head_sizes[:head_end])  # Exclusive row end in the full table.
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx  # Rows stored in this shard.
        self.dtype = layout.config.table_dtype  # Data type of the embedding table and lookup output.
        self.dummy_weights = layout.config.dummy_weights  # Fill constant rows instead of loading checkpoint data.
        # Allocate this shard's host table, shared by the ranks in group.
        self.storage = DPSharedEngramStorage((self.part_num_embeddings, self.dim), self.dtype, group)
        weight = self.storage.tensor
        self._mapped_weight = self.storage.device_view()  # NPU view of the shared host rows.
        # Register the host table as a model parameter without copying its storage.
        self.weight = nn.Parameter(weight, requires_grad=False)
        # Vector core count used to cap the lookup kernel's grid.
        self._num_cores = torch.npu.get_device_properties(torch.npu.current_device()).vector_core_num
        set_weight_attrs(
            self.weight,
            {
                "weight_loader": self.weight_loader,
                "engram_vocab_start": self.vocab_start_idx,
                "dummy_weight_value": 1.0,
                "engram_shared_ptr": self.weight.data_ptr(),
            },
        )
        logger.info(
            "Engram table allocated: layer=%d rank=%d table_location=cpu_shared "
            "execution=npu_npu head_sharding=%s prefetch=false "
            "shard_rank=%d shard_count=%d heads=[%d,%d)/%d "
            "shape=(%d,%d) dtype=%s mapped_size=%.3f GiB "
            "storage=shared shared_ranks=%s writer_rank=%s",
            layer_idx,
            torch.distributed.get_rank(),
            "tp" if layout.config.shard_heads else "none",
            shard_rank,
            shard_size,
            self.head_start,
            head_end,
            self.n_hash_cols,
            self.part_num_embeddings,
            self.dim,
            self.dtype,
            self.weight.numel() * self.weight.element_size() / 1024**3,
            group.ranks,
            group.ranks[0],
        )

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> ParallelEngramEmbedding:
        # Preserve the chosen storage and its registration when model.to() visits.
        return self

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # As in vLLM, load logical head rows and ignore any checkpoint tail padding.
        if (
            loaded_weight.ndim != 2
            or loaded_weight.shape[0] < self.num_embeddings
            or loaded_weight.shape[1] != self.dim
        ):
            raise ValueError(
                f"Engram table weight shape {tuple(loaded_weight.shape)} requires at least "
                f"{self.num_embeddings} rows and {self.dim} columns"
            )
        if param.data_ptr() != param.engram_shared_ptr:
            raise RuntimeError("Engram parameter storage must not be replaced")
        self.storage.load_weight(param, loaded_weight)

    def process_weights_after_loading(self) -> None:
        if self.dummy_weights:
            if self.storage.is_writer:
                self.weight.data.fill_(self.weight.dummy_weight_value)
            self.storage.synchronize()

    def lookup(self, indices: torch.Tensor, output: torch.Tensor) -> None:
        """Read local head rows from the shared host table on the NPU.

        Args:
            indices: Hash IDs including head offsets, shaped [tokens, total heads].
                Each valid ID is an absolute row index in this layer's full table;
                -1 marks an invalid lookup.
            output: NPU buffer shaped [tokens, local head slots, head dimension].
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        grid = min(triton.cdiv(rows, 16), max(1, self._num_cores))
        _engram_lookup_kernel[(grid,)](
            self._mapped_weight,
            indices,
            output,
            rows,
            self.vocab_start_idx,
            self.vocab_end_idx,
            indices.stride(0),
            indices.stride(1),
            grid,
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            BLOCK_D=triton.next_power_of_2(self.dim),
            BLOCK_R=16,
            num_warps=4,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up and gather all heads for the supplied tokens.

        Args:
            indices: Hash IDs including head offsets, shaped [tokens, total heads].
                Each valid ID is an absolute row index in this layer's full table;
                -1 marks an invalid lookup.
        """
        rows = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim), dtype=self.dtype, device=indices.device
        )
        self.lookup(indices, rows)
        if self.tp_size > 1:
            rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.n_hash_cols]


@triton.jit(do_not_specialize=["num_rows", "vocab_start", "vocab_end"])
def _engram_lookup_kernel(
    weight,
    ids,
    output,
    num_rows,
    vocab_start,
    vocab_end,
    stride_t,
    stride_h,
    grid,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Gather this rank's head embeddings from the shared host table on the NPU.

    Inputs: the registered host table view, absolute hash IDs [token, head],
    and this rank's head and vocabulary ranges.
    Output: write embeddings [token, local head, DIM] into the NPU output buffer.
    Steps: map each output row to a token/head, read its hash ID, subtract
    vocab_start, and copy the table row. Invalid IDs and padded heads write zeros.
    """
    # vLLM's persistent gather, with BF16 loads instead of FP8 dequantization.
    # All constexpr arguments remain last (Ascend launcher compatibility).
    cols = tl.arange(0, BLOCK_D)
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, grid * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        valid = (rows < num_rows) & (head < TOTAL_HEADS)
        index = tl.load(ids + token * stride_t + head * stride_h, valid, other=-1).to(tl.int64)
        owned = valid & (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :], owned[:, None] & (cols[None, :] < DIM), other=0
        )
        tl.store(
            output + rows[:, None].to(tl.int64) * DIM + cols[None, :],
            values,
            (rows[:, None] < num_rows) & (cols[None, :] < DIM),
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _write_hash_cache_kernel(
    input_ids,
    dead_mask,
    slot_mapping,
    cache,
    num_tokens,
    input_stride,
    mask_stride,
    slot_stride,
    dead_id,
    BLOCK_SIZE: tl.constexpr,
):
    """Save the current batch's token IDs in the device history cache.

    Inputs: token IDs, their DEAD mask, and physical slots from slot_mapping.
    Output: update cache in place; there is no separate output tensor.
    Steps: skip invalid slots, replace DEAD tokens with dead_id, and write
    each token to its slot. Complete these writes before the hash kernel reads history.
    """
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slot = tl.load(slot_mapping + token_idx * slot_stride, token_idx < num_tokens, other=-1).to(tl.int64)
    valid = (token_idx < num_tokens) & (slot >= 0)
    token = tl.load(input_ids + token_idx * input_stride, valid, other=0)
    value = token
    dead = tl.load(dead_mask + token_idx * mask_stride, valid, other=False)
    value = tl.where(dead.to(tl.int1), dead_id, value)
    tl.store(cache + slot, value, valid)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_query_rows",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    token_start,
    token_end,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_query_rows,
    num_table_rows,
    max_blocks,
    cache_block_size,
    num_heads,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
    SINGLE_QUERY: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    MAX_NGRAM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute absolute Engram table row IDs for each token, layer, and hash head.

    Inputs: current tokens, positions, request boundaries and DEAD masks;
    explicit lookbacks, optional slot history and block mappings; hash
    multipliers, per-head prime capacities, and table offsets.
    Output: write int32 hash IDs for [token_start, token_end), retaining the
    full [token, layer, (MAX_NGRAM - 1) * num_heads] output layout.
    Steps: locate the request, then read history from the current chunk,
    explicit lookbacks, or the slot cache, in that order. Pad positions before
    the request start and DEAD boundaries (including earlier history). Multiply
    token IDs by the layer's multipliers and XOR the products; for each n-gram
    order and head, take the remainder modulo its prime and add its table offset.
    """
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    if SINGLE_QUERY:
        single_chunk_idx = tl.minimum(tl.load(query_start_loc), num_tokens - 1).to(tl.int64)
        single_chunk_start = tl.load(positions + single_chunk_idx * position_stride)
    # Respect the current stream's program budget; each program walks
    # disjoint token tiles.
    for base in tl.range(token_start + tl.program_id(0) * BLOCK_T, token_end, tl.num_programs(0) * BLOCK_T):
        token = base + tl.arange(0, BLOCK_T)
        valid = token < token_end
        if SINGLE_QUERY:
            req = tl.full((), 0, tl.int64)
            chunk_start = single_chunk_start
        else:
            # Upper bound in query_start_loc[1:], including repeated padding rows.
            # Unroll the bounded search for Ascend; dynamic vector loops fail lowering.
            lo = tl.full((BLOCK_T,), 0, tl.int32)
            hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                active = lo < hi
                end = tl.load(query_start_loc + (mid + 1) * query_stride, active, other=0)
                right = token >= end
                lo = tl.where(active & right, mid + 1, lo)
                hi = tl.where(active & ~right, mid, hi)
            req = tl.minimum(lo, num_query_rows - 1).to(tl.int64)
            chunk_idx = tl.load(query_start_loc + req * query_stride)
            chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
            chunk_start = tl.load(positions + chunk_idx * position_stride)
        position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
        head = tl.arange(0, BLOCK_H)
        blocked = tl.full((BLOCK_T,), False, tl.int1)
        rolling = tl.full((BLOCK_T,), 0, tl.int64)
        for shift in tl.static_range(MAX_NGRAM):
            lookback = position - shift
            in_batch = lookback >= chunk_start
            batch_idx = tl.maximum(token - shift, 0)
            batch_token = tl.load(input_ids + batch_idx * input_stride, valid & in_batch, other=0)
            batch_source = batch_token
            batch_dead = tl.load(dead_mask + batch_idx * mask_stride, valid & in_batch, other=False)
            batch_source = tl.where(batch_dead.to(tl.int1), dead_id, batch_source)

            col = chunk_start - 1 - lookback
            in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
            col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
            window_token = tl.load(
                lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
                in_window,
                other=-1,
            )
            known = in_window & (window_token >= 0)
            window_source = window_token
            window_dead = tl.load(
                lookback_dead_mask + req * lookback_mask_row_stride + col * lookback_mask_col_stride,
                known,
                other=False,
            )
            window_source = tl.where(window_dead.to(tl.int1), dead_id, window_source)

            if cache is not None:
                clamped = tl.minimum(tl.maximum(lookback, 0), max_blocks * cache_block_size - 1)
                block_row = tl.minimum(req, num_table_rows - 1)
                needs_cache = valid & ~in_batch & ~known
                block = tl.load(
                    block_table + block_row * table_stride + (clamped // cache_block_size) * table_col_stride,
                    needs_cache,
                    other=0,
                ).to(tl.int64)
                slot = tl.minimum(
                    tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                    num_slots - 1,
                )
                fallback = tl.load(cache + slot, needs_cache, other=0)
            else:
                fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
            source = tl.where(in_batch, batch_source, tl.where(known, window_source, fallback)).to(tl.int64)
            blocked |= (lookback < 0) | (source == dead_id)
            value = tl.where(blocked, pad_id, source)
            multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
            rolling ^= value * multiplier
            if shift > 0:
                col = (shift - 1) * num_heads + head
                param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
                prime = tl.load(primes + param_offset, head < num_heads, other=1)
                offset = tl.load(offsets + param_offset, head < num_heads, other=0)
                hashed = rolling[:, None] % prime[None, :]
                # PanGu uses NumPy remainder, including signed int64 wraparound.
                hashed = tl.where(hashed < 0, hashed + prime[None, :], hashed) + offset[None, :]
                out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * ((MAX_NGRAM - 1) * num_heads) + col[
                    None, :
                ]
                tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class NgramHashState(nn.Module):
    """Manage PanGu NPU hashing parameters and token history across forwards.

    Keep multipliers, prime capacities, and table offsets as device buffers.
    Follow vLLM ModelRunner V1 by storing token IDs in a separate cache indexed
    by KV slots.
    Bind its size to the attention KV allocation and
    rebuild history when that allocation changes. The caller ensures the binding
    before forward(). This class does not own or look up the Engram embedding table.
    """

    def __init__(self, layout: EngramLayout, device: torch.device, cache_module: nn.Module) -> None:
        super().__init__()
        self.layout = layout
        self.config = layout.config
        self.register_buffer("multipliers", self._compute_hash_multipliers().to(device), persistent=False)
        self.register_buffer("primes", torch.tensor(layout.primes, device=device), persistent=False)
        self.register_buffer("offsets", layout.offsets.to(device), persistent=False)
        self.register_buffer("_cache", None, persistent=False)
        self.block_size = 0
        self.sp_rank = 0
        self.sp_size = 1
        self._cache_module_ref = weakref.ref(cache_module)
        self._kv_cache_ref = None
        self._kv_cache_ptr = None
        self.cache_prefix = getattr(cache_module, "layer_name", None)

    def has_bound_cache(self) -> bool:
        """Check whether the tracked attention module has nonempty KV cache storage.

        This traceable check does not allocate token history or verify that it
        matches the current KV allocation; ensure_cache() does that in the hash op.
        """
        kv = getattr(self._cache_module_ref(), "kv_cache", None)
        while isinstance(kv, (tuple, list)):
            kv = kv[0] if kv else None
        return kv is not None and kv.numel() > 0

    def ensure_cache(self) -> bool:
        """Follow vLLM V1: rebuild token history when the bound KV tensor changes.

        NPU KV storage may be a tuple (latent keys, indexer keys, scales).
        Its first tensor owns the block allocation. The runner supplies the
        backend's physical block size, which may differ from the JSON size.
        """
        module = self._cache_module_ref()
        kv = getattr(module, "kv_cache", None)
        while isinstance(kv, (tuple, list)):
            kv = kv[0] if kv else None
        if kv is None or not kv.numel():
            self._cache = None
            self._kv_cache_ref = None
            self._kv_cache_ptr = None
            return False
        if self.block_size <= 0:
            raise RuntimeError("Engram cache block size has not been bound")
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv and self._kv_cache_ptr == kv.data_ptr():
            return True
        self._cache = torch.zeros(
            kv.shape[0] * self.block_size, dtype=torch.int32, device=self.multipliers.device
        )
        self._kv_cache_ref = weakref.ref(kv)
        self._kv_cache_ptr = kv.data_ptr()
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        *,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hash device inputs into absolute int32 [token, layer, hash column] row IDs.

        The NPU kernel finds request boundaries itself. Positions within each
        chunk must be contiguous. Missing explicit lookbacks are -1; the
        separate dead mask terminates the n-gram prefix.

        Optional cache entries hold raw token IDs (or -1 for DEAD) indexed by KV
        slot. Write this batch's valid slots before hashing.
        Call ensure_cache() before forwarding when slot history is required.
        Callers supply valid block/slot mappings; cached history must never
        outlive the corresponding KV allocation.
        Explicit lookbacks take precedence, including after P/D or offload loads.

        This method performs no host transfers or synchronization. Supply `out`
        to reuse fixed storage, including inside an NPU graph.
        Under sequence parallelism, hash only this rank's token window and mark
        other rows -1. Every rank still updates the complete token history.
        """
        num_tokens = input_ids.numel()
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        n_cols = (max_ngram - 1) * num_heads
        expected = (num_tokens, num_layers, n_cols)
        if out is None:
            out = torch.empty(expected, dtype=torch.int32, device=input_ids.device)
        chunk = (num_tokens + self.sp_size - 1) // self.sp_size
        token_start = min(self.sp_rank * chunk, num_tokens)
        token_end = min(token_start + chunk, num_tokens)
        if self.sp_size > 1:
            # Preserve the full hash/mask shape; only this SP rank's window is consumed.
            out.fill_(-1)
        if not num_tokens:
            return out
        if self._cache is not None:
            if slot_mapping is None or block_table is None:
                raise ValueError("Engram slot cache requires slot_mapping and block_table")
            # Update the full token history on every rank because SP token partitions can change between batches.
            _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
                input_ids,
                dead_mask,
                slot_mapping,
                self._cache,
                num_tokens,
                input_ids.stride(0),
                dead_mask.stride(0),
                slot_mapping.stride(0),
                dead_id=-1,
                BLOCK_SIZE=256,
            )
        # Skip hashing when this rank's token partition is empty.
        if token_start == token_end:
            return out
        core_limit = torch.npu.get_stream_limit(torch.npu.current_stream())["vector_core_num"]
        # Smaller tiles expose more parallelism in small batches; wider tiles
        # amortize per-tile work in larger batches.
        block_t = 2 if num_tokens <= 256 else 32
        grid = min(triton.cdiv(token_end - token_start, block_t), max(1, core_limit // num_layers))
        _hash_ids_kernel[(grid, num_layers)](
            input_ids=input_ids,
            dead_mask=dead_mask,
            positions=positions,
            block_table=block_table,
            query_start_loc=query_start_loc,
            multipliers=self.multipliers,
            primes=self.primes,
            offsets=self.offsets,
            cache=self._cache,
            lookback_token_ids=lookback_token_ids,
            lookback_dead_mask=lookback_dead_mask,
            output=out,
            num_tokens=num_tokens,
            token_start=token_start,
            token_end=token_end,
            num_slots=self._cache.numel() if self._cache is not None else 0,
            pad_id=self.config.pad_token_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if self._cache is not None else 0,
            table_col_stride=block_table.stride(1) if self._cache is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_query_rows=query_start_loc.numel() - 1,
            num_table_rows=block_table.shape[0] if self._cache is not None else 0,
            max_blocks=block_table.shape[1] if self._cache is not None else 0,
            cache_block_size=self.block_size,
            num_heads=num_heads,
            dead_id=-1,
            lookback_depth=max_ngram - 1,
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            SINGLE_QUERY=query_start_loc.numel() == 2,
            SEARCH_STEPS=max(1, (query_start_loc.numel() - 1).bit_length()),
            MAX_NGRAM=max_ngram,
            BLOCK_T=block_t,
            BLOCK_H=triton.next_power_of_2(num_heads),
            num_warps=4,
        )
        return out

    def _compute_hash_multipliers(self) -> torch.Tensor:
        """Generate PanGu hash multipliers [layer, n-gram position] at initialization.

        Reproduce PanGu's seed and one-based training layer numbering to generate
        an odd int64 multiplier for each n-gram position. NgramHashState uploads
        these multipliers to the NPU; this helper does not compute runtime hashes.
        """

        layout = self.layout
        config = self.config
        max_long = np.iinfo(np.int64).max
        max_multiplier = int(max_long // max(layout.head_sizes))
        half_bound = max(1, max_multiplier // 2)
        rows = []
        for layer_idx in layout.layer_ids:
            # Training constructs TransformerLayer with a one-based layer_number,
            # while engram_layer_ids and HF layer indices are zero-based.
            layer_number = int(layer_idx) + 1
            rng = np.random.default_rng(int(config.seed + 10007 * layer_number))
            random_values = rng.integers(0, half_bound, size=(config.max_ngram_size,), dtype=np.int64)
            rows.append(torch.tensor(random_values * 2 + 1))
        return torch.stack(rows)

    def compute_hashes(
        self,
        metadata: MLACommonMetadata,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Hash current tokens and update their KV-slot history using attention metadata."""
        if not self.ensure_cache():
            raise RuntimeError("Engram slot history has not been bound to the KV cache")
        num_tokens = input_ids.numel()
        starts = metadata.query_start_loc
        num_reqs = starts.numel() - 1
        slots = metadata.slot_mapping[:num_tokens]
        if hasattr(metadata, "block_tables"):
            block_table = metadata.block_tables
        else:
            blocks = [part.block_table for part in (metadata.decode, metadata.prefill) if part is not None]
            block_table = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)
        if lookback_token_ids is None:
            lookback_token_ids = input_ids.new_full((num_reqs, self.layout.max_ngram_size - 1), -1)
        lookbacks = lookback_token_ids[:num_reqs]
        dead = slots < 0
        hashes = self(
            input_ids, positions, starts, dead, lookbacks,
            torch.zeros_like(lookbacks, dtype=torch.bool),
            slot_mapping=slots, block_table=block_table,
        )
        hashes.masked_fill_(dead[:, None, None], -1)
        return hashes


class PanGuEngramModelMixin:
    """Share Engram history binding, hashing and lookup preparation across PanGu models.

    Models provide engram_layout, layers and their local [start_layer, end_layer)
    range. This mixin preserves the existing model parameter hierarchy.
    """

    def _init_engram_hash(self) -> None:
        self.engram_hash = None
        local_engram = next(
            (
                layer.engram
                for layer in islice(self.layers, self.start_layer, self.end_layer)
                if getattr(layer, "engram", None) is not None
            ),
            None,
        )
        if local_engram is not None:
            self.engram_hash = NgramHashState(
                self.engram_layout,
                local_engram.wkv.weight.device,
                self.layers[self.start_layer].self_attn.attn,
            )
            if local_engram.use_sequence_parallel:
                self.engram_hash.sp_rank = get_tp_group().rank_in_group
                self.engram_hash.sp_size = local_engram.embed_tokens.model_tp_size
            self._prepare_key = id(self)
            _engram_model_owners[self._prepare_key] = self

    def _get_engram_metadata(self) -> "MLACommonMetadata | None":
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            from vllm.v1.worker.ubatching import dbo_current_ubatch_id

            metadata = metadata[dbo_current_ubatch_id()]
        return metadata.get(self.engram_hash.cache_prefix) if isinstance(metadata, dict) else None

    def _compute_engram_hashes(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        # Resolve metadata at eager/capture execution, as NPU attention custom
        # ops do. Device tensor reads and writes become nodes of every replay.
        return self.engram_hash.compute_hashes(
            self._get_engram_metadata(), input_ids, positions, lookback_token_ids
        )

    def prepare_model_embeddings(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Skip hashing and lookup until the attention KV cache and metadata are available.
        cache_ready = self.engram_hash.has_bound_cache()
        if not cache_ready and not torch.compiler.is_compiling():
            self.engram_hash.ensure_cache()  # Discard history from an unbound KV allocation.
        metadata = self._get_engram_metadata()
        if metadata is None or not cache_ready:
            return None
        hashes = torch.ops.omni_npu.engram_hash(
            input_ids,
            positions,
            lookback_token_ids,
            self._prepare_key,
            len(self.engram_layout.layer_ids),
            self.engram_layout.n_hash_cols,
        )
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                engram.staged_rows = torch.ops.omni_npu.engram_lookup_rows(
                    hashes[:, engram.layer_hash_index],
                    engram._lookup_key,
                )
        return hashes


# Map model IDs to models for the Engram hash custom op without keeping models alive.
_engram_model_owners = weakref.WeakValueDictionary()


@torch.library.custom_op("omni_npu::engram_hash", mutates_args=())
def _engram_hash(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    lookbacks: torch.Tensor | None,
    owner: int,
    num_layers: int,
    num_heads: int,
) -> torch.Tensor:
    # Keep hashing on the current stream, as measured in the prefill profiles.
    return _engram_model_owners[owner]._compute_engram_hashes(input_ids, positions, lookbacks)


@_engram_hash.register_fake
def _engram_hash_fake(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    lookbacks: torch.Tensor | None,
    owner: int,
    num_layers: int,
    num_heads: int,
) -> torch.Tensor:
    return input_ids.new_empty((input_ids.numel(), num_layers, num_heads), dtype=torch.int32)


class PanGuEngramPostNorm(nn.Module):
    """Normalize the Engram update before the model adds it to the residual.

    Apply the learned RMSNorm weight over the final hidden dimension using
    the fused NPU operator, preserving token and mHC stream dimensions.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(value, self.weight, epsilon=self.eps)[0]


_engram_embedding_owners = weakref.WeakValueDictionary()


@torch.library.custom_op("omni_npu::engram_lookup_rows", mutates_args=())
def _engram_lookup_rows(hash_ids: torch.Tensor, owner: int) -> torch.Tensor:
    layer = _engram_embedding_owners[owner]
    # Keep lookup inside the graph without mutating a persistent graph input.
    rows = hash_ids.new_empty(
        (layer._lookup_num_tokens(hash_ids.shape[0]), layer.embed_tokens.part_n_hash_cols, layer.layout.head_dim),
        dtype=layer.config.table_dtype,
    )
    layer.staged_rows = rows
    layer.prepare_embeddings(hash_ids)
    return rows


@_engram_lookup_rows.register_fake
def _engram_lookup_rows_fake(hash_ids: torch.Tensor, owner: int) -> torch.Tensor:
    layer = _engram_embedding_owners[owner]
    return hash_ids.new_empty(
        (layer._lookup_num_tokens(hash_ids.shape[0]), layer.embed_tokens.part_n_hash_cols, layer.layout.head_dim),
        dtype=layer.config.table_dtype,
    )


@torch.library.custom_op("omni_npu::engram_proj_gate", mutates_args=())
def _engram_proj_gate(
    hidden_states: torch.Tensor,
    embeddings: torch.Tensor,
    weight_key: torch.Tensor,
    weight_value: torch.Tensor,
    gamma_key: torch.Tensor,
    gamma_query: torch.Tensor,
    clamp_value: float,
    eps: float,
) -> torch.Tensor:
    # Keep PyPTO JIT and its Python validation outside Dynamo tracing.
    from omni_npu.layers.ops.ai_infra_pypto_engram_proj_gate import ai_infra_engram_proj_gate

    if hidden_states.numel() == 0:
        return torch.empty_like(hidden_states)
    output, *_ = ai_infra_engram_proj_gate(
        hidden_states.unsqueeze(0).contiguous(),
        embeddings.unsqueeze(0).contiguous(),
        weight_key,
        weight_value,
        gamma_key,
        gamma_query,
        clamp_value=clamp_value,
        eps=eps,
        return_cache=False,
    )
    return output.squeeze(0)


@_engram_proj_gate.register_fake
def _engram_proj_gate_fake(
    hidden_states: torch.Tensor,
    embeddings: torch.Tensor,
    weight_key: torch.Tensor,
    weight_value: torch.Tensor,
    gamma_key: torch.Tensor,
    gamma_query: torch.Tensor,
    clamp_value: float,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


class PanGuEngramFusion(nn.Module):
    """Turn shared host embeddings into a gated update for PanGu mHC streams.

    Stage NPU lookup results before layer consumption. With SP, look up only
    this rank's token window; otherwise optional TP head shards are gathered.
    SP and head sharding are mutually exclusive. Project embeddings into keys
    and a shared value with wkv, normalize keys and hidden-state queries, and
    use their dot products to gate the value for each mHC stream. Mask invalid
    tokens and preserve the input layout; the model handles optional post-norm
    and residual addition outside this module.
    """

    def __init__(
        self, layout: EngramLayout, layer_idx: int, group: GroupCoordinator, use_sequence_parallel: bool
    ) -> None:
        super().__init__()
        config = layout.config
        if use_sequence_parallel and config.shard_heads:
            raise ValueError("Engram sequence parallelism and TP head sharding cannot both be enabled")
        self.config = config
        self.layout = layout
        self.layer_hash_index = layout.layer_ids.index(layer_idx)
        self.use_sequence_parallel = use_sequence_parallel
        self.num_streams = config.num_streams
        self.hidden_size = config.hidden_size
        self.memory_dim = config.memory_dim
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.embed_tokens = ParallelEngramEmbedding(layout, layer_idx, group)

        # As in vLLM, shard projection columns only when tokens are replicated.
        self.wkv = ColumnParallelLinear(
            self.memory_dim,
            (self.num_streams + 1) * self.hidden_size,
            bias=False,
            gather_output=True,
            return_bias=False,
            disable_tp=use_sequence_parallel,
        )
        self.k_weight = nn.Parameter(torch.ones(self.num_streams, self.hidden_size))
        self.q_weight = nn.Parameter(torch.ones(self.num_streams, self.hidden_size))
        self.register_buffer("norm_unit_weight", torch.ones(self.hidden_size), persistent=False)
        from omni_npu.model_config.config_loader.loader import model_extra_config

        # PyPTO needs complete BF16 projections. Preserve TP projection and
        # its gather when output columns are sharded across multiple ranks.
        self.use_engram_proj_gate_fusion_op = (
            model_extra_config.operator_opt_config.use_engram_proj_gate_fusion_op
            and self.wkv.tp_size == 1
            and self.wkv.weight.dtype == torch.bfloat16
        )
        self.register_buffer("proj_weights_key", None, persistent=False)
        self.register_buffer("proj_weights_value", None, persistent=False)
        # NPU hashing uses graph-local rows allocated by engram_lookup_rows.
        self.staged_rows = None
        self._lookup_key = id(self)
        _engram_embedding_owners[self._lookup_key] = self

    def _lookup_num_tokens(self, num_tokens: int) -> int:
        if self.use_sequence_parallel:
            tp_size = self.embed_tokens.model_tp_size
            return (num_tokens + tp_size - 1) // tp_size
        return num_tokens

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Look up rows before the decoder layer consumes them."""
        if self.staged_rows is None:
            raise RuntimeError("Engram staging is not initialized")
        if self.use_sequence_parallel and self.embed_tokens.model_tp_size > 1:
            # Each SP rank consumes all heads for only its own token window.
            # Slice the small indices before lookup, rather than materializing
            # every token's embedding on every rank and slicing afterwards.
            chunk = self._lookup_num_tokens(hash_ids.shape[0])
            start = get_tp_group().rank_in_group * chunk
            hash_ids = hash_ids[start:start + chunk]
            if hash_ids.shape[0] < chunk:
                hash_ids = torch.nn.functional.pad(hash_ids, (0, 0, 0, chunk - hash_ids.shape[0]), value=-1)
        if hash_ids.shape[0] > self.staged_rows.shape[0]:
            raise ValueError("Engram staging buffer too small")
        self.embed_tokens.lookup(hash_ids, self.staged_rows[:hash_ids.shape[0]])

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather heads, returning only local tokens when SP is enabled."""
        rows = self.staged_rows[: self._lookup_num_tokens(hash_ids.shape[0])]
        tp_size = self.embed_tokens.tp_size
        if tp_size > 1:
            rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : self.layout.n_hash_cols]

    def _reshape_hidden_streams(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 2:
            if hidden_states.shape[1] != self.num_streams * self.hidden_size:
                raise ValueError(
                    "PanGu Engram flattened hidden-state width does not match "
                    f"mHC config: {tuple(hidden_states.shape)}, expected "
                    f"{self.num_streams * self.hidden_size}"
                )
            # The fused mHC post operator returns [tokens, streams * hidden],
            # while the unfused path returns [tokens, streams, hidden]. Both
            # layouts represent the same streams and must retain their layout
            # for the caller's residual addition.
            hidden_streams = hidden_states.reshape(-1, self.num_streams, self.hidden_size)
        elif hidden_states.ndim == 3:
            if hidden_states.shape[1:] != (self.num_streams, self.hidden_size):
                raise ValueError("PanGu Engram hidden-state shape does not match mHC config")
            hidden_streams = hidden_states
        else:
            raise ValueError("PanGu Engram expects [tokens, streams * hidden] or " "[tokens, streams, hidden]")
        return hidden_streams

    def forward(
        self, hidden_states: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden_streams = self._reshape_hidden_streams(hidden_states)
        memory = self.embed(hash_ids).flatten(-2)
        if memory.shape != (hidden_streams.shape[0], self.memory_dim):
            raise ValueError("PanGu Engram memory is not aligned with hidden states")

        dtype = self.wkv.weight.dtype
        memory = memory.to(dtype=dtype)
        hidden_streams = hidden_streams.to(dtype=dtype)
        if self.use_engram_proj_gate_fusion_op:
            output = torch.ops.omni_npu.engram_proj_gate(
                hidden_streams,
                memory,
                self.proj_weights_key,
                self.proj_weights_value,
                self.k_weight,
                self.q_weight,
                self.clamp_value,
                self.eps,
            )
        else:
            projected = self.wkv(memory).unflatten(-1, (self.num_streams + 1, self.hidden_size))
            # Normalize only the hidden dimension, then apply each stream's own weight.
            keys = torch_npu.npu_rms_norm(projected[:, : self.num_streams], self.norm_unit_weight, epsilon=self.eps)[0]
            keys = keys * self.k_weight
            values = projected[:, self.num_streams]
            queries = torch_npu.npu_rms_norm(hidden_streams, self.norm_unit_weight, epsilon=self.eps)[0]
            queries = queries * self.q_weight
            logits = (queries * keys).sum(dim=-1) / math.sqrt(self.hidden_size)
            logits = logits.abs().clamp_min(self.clamp_value).sqrt() * logits.sign()
            output = torch.sigmoid(logits).unsqueeze(-1) * values.unsqueeze(1)
        if token_mask is None:
            token_mask = hash_ids[:, 0] >= 0
        else:
            if token_mask.shape != (hash_ids.shape[0],):
                raise ValueError("Engram token_mask must match the full token batch")
            token_mask = token_mask & (hash_ids[:, 0] >= 0)
        if self.use_sequence_parallel and self.embed_tokens.model_tp_size > 1:
            chunk = hidden_streams.shape[0]
            padding = chunk * self.embed_tokens.model_tp_size - token_mask.shape[0]
            token_mask = torch.nn.functional.pad(token_mask, (0, padding), value=False)
            start = get_tp_group().rank_in_group * chunk
            token_mask = token_mask[start:start + chunk]
        output = output.masked_fill(~token_mask[:, None, None], 0)
        # Keep the token axis inferred for dynamic NPU graph buckets.
        return output.flatten(-2) if hidden_states.ndim == 2 else output

    def process_weights_after_loading(self) -> None:
        """Finish table initialization before the loader returns the model."""
        self.embed_tokens.process_weights_after_loading()
        if self.use_engram_proj_gate_fusion_op:
            # Convert the checkpoint's [output, input] WKV once after loading,
            # avoiding a full weight transpose on every inference step.
            keys, value = self.wkv.weight.unflatten(0, (self.num_streams + 1, self.hidden_size)).split(
                [self.num_streams, 1], dim=0
            )
            self.proj_weights_key = keys.transpose(1, 2).contiguous()
            self.proj_weights_value = value.squeeze(0).t().contiguous()
