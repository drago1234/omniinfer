# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Attach CPU-computed placeholder row indices to multimodal embeddings.

Target vLLM version: v0.25.1.

Why patch instead of plugin/inheritance: this optimization is enabled by the
openpangu_v1_vl patch group. Model plugins do not control the runner's gathered
embedding container, and there is no per-model gather hook. Using a dedicated
runner subclass would require model-specific worker selection changes. This
patch retains the upstream gather implementation and augments its result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.v1.worker import gpu_model_runner as _runner_module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.utils import CpuGpuBuffer

logger = logging.getLogger(__name__)

_original_gather_mm_embeddings = getattr(
    GPUModelRunner, "_gather_mm_embeddings", None
)


class _MultimodalBuffer:
    """Fallback CPU/device buffer with a shared NumPy view of CPU storage."""

    def __init__(self, runner: GPUModelRunner, dtype: torch.dtype) -> None:
        pin_memory = getattr(
            runner, "pin_memory", getattr(_runner_module, "PIN_MEMORY", False)
        )
        self.cpu = torch.empty(
            runner.max_num_tokens, dtype=dtype, device="cpu", pin_memory=pin_memory
        )
        self.np = self.cpu.numpy()
        self.gpu = torch.empty(
            runner.max_num_tokens, dtype=dtype, device=runner.device
        )

    def copy_to_gpu(self, count: int) -> torch.Tensor:
        """Copy the active prefix to the device with non_blocking=True."""
        self.gpu[:count].copy_(self.cpu[:count], non_blocking=True)
        return self.gpu[:count]


def _make_multimodal_buffer(
    runner: GPUModelRunner, dtype: torch.dtype
) -> CpuGpuBuffer | _MultimodalBuffer:
    """Prefer the runner's native buffer factory when available."""
    make_buffer = getattr(runner, "_make_buffer", None)
    if make_buffer is not None:
        return make_buffer(runner.max_num_tokens, dtype=dtype)
    return _MultimodalBuffer(runner, dtype)


class _MultimodalEmbeddingsWithIndices(list[torch.Tensor]):
    """Embedding list carrying int64 placeholder row indices on the device."""

    def __init__(
        self,
        embeddings: list[torch.Tensor],
        placeholder_indices: torch.Tensor,
    ) -> None:
        super().__init__(embeddings)
        self.placeholder_indices = placeholder_indices


def _attach_placeholder_indices(
    runner: GPUModelRunner,
    mm_embeds: list[torch.Tensor],
    is_mm_embed: torch.Tensor,
    total_num_scheduled_tokens: int,
) -> _MultimodalEmbeddingsWithIndices:
    """Extract True positions on CPU and copy their indices to the device.

    CPU masks are read directly. Device masks use the CPU mirror maintained by
    older upstream gathers. Row indices refer to the current scheduled batch.
    The caller initializes the reusable index buffers before calling this helper.
    The runner must order transfers and consumers before reusing a buffer slot.
    """
    if is_mm_embed.device.type == "cpu":
        cpu_mask = is_mm_embed[:total_num_scheduled_tokens].numpy()
    else:
        # Older upstream gathers return a device mask and fill this CPU buffer.
        cpu_mask = runner.is_mm_embed_buffers[runner.is_mm_embed_idx].np[
            :total_num_scheduled_tokens
        ]
    cpu_indices = np.flatnonzero(cpu_mask)
    num_mm_tokens = int(cpu_indices.size)

    # Rotate index buffers independently of the upstream mask buffers.
    buffer_idx = getattr(runner, "mm_embed_index_idx", 0)
    index_buf = runner.mm_embed_index_buffers[buffer_idx]
    if num_mm_tokens:
        index_buf.np[:num_mm_tokens] = cpu_indices
        placeholder_indices = index_buf.copy_to_gpu(num_mm_tokens)
    else:
        placeholder_indices = index_buf.gpu[:0]
    runner.mm_embed_index_idx = 1 - buffer_idx

    return _MultimodalEmbeddingsWithIndices(mm_embeds, placeholder_indices)


class GPUModelRunnerGatherMMEmbeddingsPatch(VLLMPatch):
    """Preserve upstream gathering and attach indices for embedding insertion."""

    _attr_names_to_apply = ["_gather_mm_embeddings"]

    def _gather_mm_embeddings(
        self,
        scheduler_output: SchedulerOutput,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Legacy gathers expect mask buffers to exist before they run.
        if not hasattr(self, "is_mm_embed_idx"):
            self.is_mm_embed_idx = 0
        if not hasattr(self, "is_mm_embed_buffers"):
            self.is_mm_embed_buffers = [
                _make_multimodal_buffer(self, torch.bool),
                _make_multimodal_buffer(self, torch.bool),
            ]
        if not hasattr(self, "mm_embed_index_buffers"):
            self.mm_embed_index_buffers = [
                _make_multimodal_buffer(self, torch.int64),
                _make_multimodal_buffer(self, torch.int64),
            ]
        mm_embeds, is_mm_embed = _original_gather_mm_embeddings(
            self,
            scheduler_output,
            shift_computed_tokens,
        )
        mm_embeds_with_indices = _attach_placeholder_indices(
            self,
            mm_embeds,
            is_mm_embed,
            scheduler_output.total_num_scheduled_tokens,
        )
        return mm_embeds_with_indices, is_mm_embed


if callable(_original_gather_mm_embeddings):
    register_patch("GPUModelRunnerGatherMMEmbeddings", GPUModelRunner)(
        GPUModelRunnerGatherMMEmbeddingsPatch
    )
else:
    logger.warning(
        "Skipping GPUModelRunnerGatherMMEmbeddings patch: "
        "GPUModelRunner._gather_mm_embeddings is missing or not callable. "
        "This patch targets vLLM 0.25.1; keeping the upstream runner unchanged."
    )
