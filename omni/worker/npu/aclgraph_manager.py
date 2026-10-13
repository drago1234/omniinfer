# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU graph managers for the MRv2 target and autoregressive drafter.

Like vllm-ascend, retain the upstream managers' input preparation, dispatch
and output ownership. This vLLM revision has separate draft prefill/decode
managers. Their common capture implementation uses NPU APIs explicitly.

The attention contract remains device-side aicpu tiling. This change does
not enable attention backends requiring out-of-graph ACL task updates.
"""

from contextlib import contextmanager

import torch
from tqdm import tqdm

from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture, is_global_first_rank
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    CudaGraphManager,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    DecodeSpeculatorCudaGraphManager,
    PrefillSpeculatorCudaGraphManager,
)

from omni_npu.compilation.acl_graph import (
    ensure_weak_ref_graph_params,
    get_graph_params,
    set_graph_params,
)
from omni_npu.worker.npu.aclgraph_utils import (
    assert_no_acl_tasks_registered,
    build_capture_with_full_mode,
)

logger = init_logger(__name__)


def _prepare_graph_params(capture_descs):
    # Use resolved sizes: FULL_DECODE_ONLY can round raw sizes to a multiple
    # of decode_query_len, and later MTP steps use independent size buckets.
    sizes = {desc.num_tokens for descs in capture_descs.values() for desc in descs}
    params = get_graph_params()
    if params is None:
        set_graph_params(sizes)
    else:
        for size in sizes:
            params.task_entries.setdefault(size, {})
            params.workspaces.setdefault(size, {})


class AclGraphManager(CudaGraphManager):
    """Shared capture lifecycle; replay/dispatch remain upstream-owned."""

    @build_capture_with_full_mode
    def capture(self, create_forward_fn, progress_bar_desc="Capturing ACL graphs"):
        # Adapted from this revision's CudaGraphManager.capture: upstream has
        # no graph-object/context factory hook. Preserve offloader stream joins
        # and separate warmup/capture metadata when selecting the NPU backend.
        _prepare_graph_params(self._capture_descs)
        attn_states = {}
        with graph_capture(device=self.device):
            for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
                descs = self._capture_descs.get(mode, ())
                if is_global_first_rank() and descs:
                    descs = tqdm(descs, desc=f"{progress_bar_desc} ({mode.name})")
                for desc in descs:
                    forward_fn, warmup_state = create_forward_fn(desc, warmup=True)
                    forward_fn(CUDAGraphMode.NONE)
                    if mode == CUDAGraphMode.PIECEWISE:
                        attn_states[desc] = AttentionStatePair(warmup_state, warmup_state)
                        forward_fn(CUDAGraphMode.PIECEWISE)
                        continue

                    forward_fn, captured_state = create_forward_fn(desc, warmup=False)
                    attn_states[desc] = AttentionStatePair(warmup_state, captured_state)
                    assert desc not in self.graphs, f"Graph already captured for {desc}"
                    graph = torch.npu.NPUGraph()
                    get_offloader().sync_prev_onload()
                    with torch.npu.graph(graph, pool=self.pool):
                        forward_fn(CUDAGraphMode.NONE)
                        get_offloader().join_after_forward()
                    # Fail before exposing a replayable graph with an event
                    # wait that this branch cannot satisfy through task update.
                    assert_no_acl_tasks_registered()
                    ensure_weak_ref_graph_params()
                    self.graphs[desc] = graph
                    compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True
        logger.info("[omni-npu/mrv2] %s captured %d NPU graphs; token sizes=%s",
                    type(self).__name__, len(self.graphs),
                    sorted({desc.num_tokens for desc in self.graphs}))
        return attn_states


# The diamond intentionally puts AclGraphManager before CudaGraphManager:
# each specialized upstream capture() calls super().capture(factory), which
# enters the shared NPU implementation without copying its model/draft logic.
class ModelAclGraphManager(ModelCudaGraphManager, AclGraphManager):
    pass


class PrefillSpeculatorAclGraphManager(PrefillSpeculatorCudaGraphManager, AclGraphManager):
    pass


class DecodeSpeculatorAclGraphManager(DecodeSpeculatorCudaGraphManager, AclGraphManager):
    pass


@contextmanager
def graph_manager_wrapper():
    """Select NPU manager classes only while the NPU runner builds managers.

    Match vllm-ascend's scoped manager factory, including restoration if KV
    initialization fails. MRv1 does not import this module through its runner.
    """
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator

    bindings = (
        (model_runner, "ModelCudaGraphManager", ModelAclGraphManager),
        (speculator, "PrefillSpeculatorCudaGraphManager", PrefillSpeculatorAclGraphManager),
        (speculator, "DecodeSpeculatorCudaGraphManager", DecodeSpeculatorAclGraphManager),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in bindings]
    try:
        for module, name, manager in bindings:
            setattr(module, name, manager)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
