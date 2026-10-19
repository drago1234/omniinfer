# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture-time graph signals for MRv2. Implementation: omni/worker/npu/aclgraph_utils.py.

Two patches, both serving P5 (see 2-ModelRunnerV2支持 Graph分析/ModelRunnerV2图模式适配设计.md):

- ForwardContext.capturing: omni's attention backends read this attribute, which
  upstream does not define. MRv1 assigns it per instance; V2 never does, so the
  descriptor supplies a thread-local default while an explicit assignment still wins.
- CudaGraphManager.capture: upstream calls the recorded pass with NONE because no
  inner wrapper takes over once V2 records from the outside, and omni reads that
  field to select its compiled model path. Both FULL warmup and recording use
  FULL so static-kernel compilation runs before the outer capture starts.
  The factory's `warmup` argument keeps the capturing flag confined to recording;
  warmup uses separate attention state and must not register ACL tasks.
  Neither signal is attached to the shared torch.cuda.graph API.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.aclgraph_utils import (
    _CapturingDescriptor,
    build_capture_with_full_mode,
    build_run_fullgraph_with_acl_refresh,
)

logger = init_logger(__name__)


try:
    from vllm.forward_context import ForwardContext
except ImportError:
    logger.error(
        "[omni-npu/mrv2] ForwardContext unavailable; capturing flag not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2CapturingFlagPatch", ForwardContext)
    class MRv2CapturingFlagPatch(VLLMPatch):
        _attr_names_to_apply = ["capturing"]

        capturing = _CapturingDescriptor()


try:
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
except ImportError:
    logger.error(
        "[omni-npu/mrv2] CudaGraphManager unavailable; capture mode not patched",
        exc_info=True,
    )
else:
    # The base method is what ModelCudaGraphManager and the speculator managers
    # all reach through super(), so one patch covers every capture path.
    @register_patch("MRv2CaptureModePatch", CudaGraphManager)
    class MRv2CaptureModePatch(VLLMPatch):
        _attr_names_to_apply = ["capture", "run_fullgraph"]

        capture = build_capture_with_full_mode(CudaGraphManager.capture)
        run_fullgraph = build_run_fullgraph_with_acl_refresh(
            CudaGraphManager.run_fullgraph
        )
