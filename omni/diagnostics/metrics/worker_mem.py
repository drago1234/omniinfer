# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Worker-process memory sampling. kv_transfer ships results over KVConnectorStats."""

from typing import Optional

from vllm.logger import init_logger

from omni_npu import envs

logger = init_logger(__name__)

# Memory is slow-changing; sample the allocator only every N collect calls.
SAMPLE_EVERY_N_STEPS = envs.OMNI_METRICS_WORKER_MEM_EVERY

_step = 0


def maybe_sample(rank: int) -> Optional[dict]:
    # On sample ticks, read the torch allocator without a device sync.
    # Returns {rank: {alloc, reserved}} or None. Rank keys survive aggregate
    # union. Failures are isolated and never affect inference.
    global _step
    _step += 1
    if _step % SAMPLE_EVERY_N_STEPS != 0:
        return None
    try:
        import torch

        return {
            str(rank): {
                "alloc": float(torch.npu.memory_allocated()),
                "reserved": float(torch.npu.memory_reserved()),
            }
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("worker_mem sample failed on rank %s: %s", rank, e)
        return None
