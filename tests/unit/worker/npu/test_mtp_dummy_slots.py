# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""迁移 vLLM PR #56734 的 dummy slot 回归测试，仅适配 NPU 运行环境。"""

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu import block_table, buffer_utils
from vllm.v1.worker.gpu.block_table import BlockTables

from omni_npu.vllm_patches.patches.common.patch_mrv2_dummy_slots import (
    MRv2DummySlotKernelPatch,
)
from omni_npu.worker.npu.buffer_utils import NPUUvaBuffer


@pytest.mark.npu
def test_dummy_request_slot_mapping_is_pad(monkeypatch):
    """dummy/图 padding 请求不得通过残留源 block table 生成真实 KV slot。"""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("需要 NPU 执行真实 slot kernel")

    # 上游依赖 CUDA UVA；此处使用本仓的 NPU buffer 和 slot kernel patch。
    monkeypatch.setattr(buffer_utils, "UvaBuffer", NPUUvaBuffer)
    monkeypatch.setattr(
        block_table, "_compute_slot_mappings_kernel",
        MRv2DummySlotKernelPatch._compute_slot_mappings_kernel,
    )
    device = torch.device("npu", torch.npu.current_device())
    block_tables = BlockTables(
        block_sizes=[4],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[4],
    )
    block_tables.append_block_ids(req_index=0, new_block_ids=([7, 8],), overwrite=True)
    block_tables.apply_staged_writes()
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32, device=device)
    positions = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)

    real = block_tables.compute_slot_mappings(
        torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc,
        positions,
        num_tokens_padded=3,
    )
    assert real[0].tolist() == [7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3]

    dummy = block_tables.compute_slot_mappings(
        torch.tensor([-1], dtype=torch.int32, device=device),
        query_start_loc,
        positions,
        num_tokens_padded=3,
    )
    assert dummy[0].tolist() == [PAD_SLOT_ID] * 3
