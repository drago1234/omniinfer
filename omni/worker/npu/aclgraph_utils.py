# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACL Graph capture support for the NPU V2 model runner.

Scoped to Pangu's default use_aicpu_fa_tiling=true: attention reads its
tiling from device tensors at execution time, so nothing registers an ACL
graph task and a replay needs no out-of-graph refresh. The refresh and the
attention-metadata handoff it consumes therefore live on
feature/mrv2_graph_full, not here; assert_no_acl_tasks_registered() fails
loudly if that assumption ever stops holding.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

import torch

from vllm.config import CUDAGraphMode


_CAPTURING = threading.local()
_CAPTURING_ATTR = "_omni_npu_capturing"


def _capturing_default() -> bool:
    return getattr(_CAPTURING, "value", False)


class _CapturingDescriptor:
    """ForwardContext.capturing with a thread-local V2 capture fallback."""

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return obj.__dict__.get(_CAPTURING_ATTR, _capturing_default())

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__[_CAPTURING_ATTR] = bool(value)


class CaptureWindow:
    """Raise the thread-local capturing flag for one recorded pass."""

    def __init__(self) -> None:
        # Placeholder; __enter__ records the value actually restored on exit.
        self._previous = False

    def __enter__(self) -> "CaptureWindow":
        self._previous = _capturing_default()
        _CAPTURING.value = True
        return self

    def __exit__(self, *exc: Any) -> None:
        _CAPTURING.value = self._previous


def _force_full_mode(forward_fn):
    """Run the recorded pass as FULL, with the capturing flag raised.

    Both signals are scoped to this one call rather than to torch.cuda.graph:
    the wrapper is only installed for the recorded pass, so the warmup pass --
    which runs outside the graph and must register nothing -- is untouched,
    and unrelated code entering torch.cuda.graph no longer trips the flag.
    """

    def forward_fn_full(cg_mode):
        with CaptureWindow():
            return forward_fn(
                CUDAGraphMode.FULL if cg_mode is CUDAGraphMode.NONE else cg_mode
            )

    return forward_fn_full


def build_capture_with_full_mode(original_capture):
    """Report FULL to the pass that upstream records with NONE.

    Upstream passes NONE because no inner wrapper takes over once V2 records
    from the outside, but omni reads that field as "am I in a graph" in three
    places. Rewriting the argument rather than the ForwardContext field also
    keeps set_forward_context synthesizing the BatchDescriptor that MoME
    dereferences. The warmup pass keeps NONE: it runs outside the graph.
    """

    @functools.wraps(original_capture)
    def capture(self, create_forward_fn, *args, **kwargs):
        from omni_npu.compilation.acl_graph import get_graph_params

        graph_params = get_graph_params()
        if graph_params is not None:
            for entries in graph_params.task_entries.values():
                entries.clear()
            for workspaces in graph_params.workspaces.values():
                workspaces.clear()

        def create_forward_fn_full(desc, warmup):
            forward_fn, attn_state = create_forward_fn(desc, warmup)
            # desc.cg_mode is already a resolved runtime mode: dual configs
            # such as FULL_DECODE_ONLY are split by decode_mode()/mixed_mode()
            # when the capture descs are built, so only PIECEWISE and FULL
            # reach here and a new dual config needs no change.
            if warmup or desc.cg_mode is not CUDAGraphMode.FULL:
                return forward_fn, attn_state
            return _force_full_mode(forward_fn), attn_state

        result = original_capture(self, create_forward_fn_full, *args, **kwargs)

        from omni_npu.compilation.acl_graph import ensure_weak_ref_graph_params

        if graph_params is not None:
            ensure_weak_ref_graph_params()
            self._omni_acl_task_entries = {
                size: dict(entries)
                for size, entries in graph_params.task_entries.items()
            }
            self._omni_acl_workspaces = {
                size: dict(workspaces)
                for size, workspaces in graph_params.workspaces.items()
            }
        return result

    return capture


def ensure_graph_params(compilation_config) -> None:
    """Give capture_graph_task a table to write into, even on this branch.

    Nothing should register while attention takes its tiling from the aicpu,
    but two sites ignore that flag. Reaching one with graph_params unset
    dereferences None inside _get_or_create_workspace -- loud yet
    undiagnosable. Let the registration succeed instead, so
    assert_no_acl_tasks_registered() can say what happened and where to
    look.
    """
    from omni_npu.compilation.acl_graph import get_graph_params, set_graph_params

    if get_graph_params() is not None:
        return  # set_graph_params raises when called twice
    set_graph_params(set(compilation_config.cudagraph_capture_sizes or []))


def assert_no_acl_tasks_registered() -> None:
    """Fail if capture registered tasks this branch cannot refresh.

    Two capture_graph_task call sites do not check use_aicpu_fa_tiling --
    NPUAttentionBackendImpl.forward's FIA_V2 branch and
    NPUDeepseekMLAAttention._apply_standard_attention -- so a model reaching
    either would need the replay-time refresh that only
    feature/mrv2_graph_full carries. capture_graph_task records event.wait
    into the graph, and nothing here ever records the event, so the first
    replay would hang rather than compute a wrong result. Raise at the end
    of capture instead.
    """
    from omni_npu.compilation.acl_graph import get_graph_params

    graph_params = get_graph_params()
    if graph_params is None:
        return
    registered = {
        size: len(entries)
        for size, entries in graph_params.task_entries.items()
        if entries
    }
    if registered:
        raise RuntimeError(
            "[omni-npu/mrv2] capture registered ACL graph tasks "
            f"({registered}), but this branch omits the replay-time refresh. "
            "Restore feature/mrv2_graph_full before running this model."
        )


def refresh_acl_graph_tasks(manager: Any, forward_context: Any, vllm_config: Any, update_stream: Any) -> None:
    """Refresh dynamic attention graph tasks after an outer V2 replay."""
    if forward_context.attn_metadata is None:
        return
    runtime_shape = forward_context.batch_descriptor.num_tokens
    task_entries = getattr(manager, "_omni_acl_task_entries", {}).get(runtime_shape, {})
    workspaces = getattr(manager, "_omni_acl_workspaces", {}).get(runtime_shape, {})
    if not task_entries:
        return
    with torch.npu.stream(update_stream):
        for layer_name, task_entry in task_entries.items():
            update_fn = task_entry.op_desc.compute_dynamic_kwargs
            dynamic_kwargs = None if update_fn is None else update_fn(
                forward_context, layer_name, vllm_config
            )
            if dynamic_kwargs is None:
                task_entry.event.record(update_stream)
                continue
            updated_kwargs = {**task_entry.captured_kwargs, **dynamic_kwargs}
            if (updated_kwargs.get("input_layout", "").startswith("BSND")
                    and "actual_seq_lengths" in updated_kwargs):
                updated_kwargs["actual_seq_lengths"] = None
            workspace = workspaces[task_entry.op_desc.workspace_fn]
            torch.npu.graph_task_update_begin(update_stream, task_entry.handle)
            task_entry.op_desc.op_out_fn(
                **updated_kwargs, workspace=workspace, out=task_entry.out_tensors
            )
            torch.npu.graph_task_update_end(update_stream)
            task_entry.event.record(update_stream)


def build_run_fullgraph_with_acl_refresh(original_run_fullgraph):
    """Add ACL task refresh to an upstream full-graph replay."""
    @functools.wraps(original_run_fullgraph)
    def run_fullgraph(self, desc, *args, **kwargs):
        result = original_run_fullgraph(self, desc, *args, **kwargs)
        context_provider = getattr(self, "_omni_acl_context_provider", None)
        update_stream = getattr(self, "_omni_acl_update_stream", None)
        if context_provider is not None and update_stream is not None:
            refresh_acl_graph_tasks(
                self, context_provider(desc), self.vllm_config, update_stream
            )
        return result
    return run_fullgraph


def reset_for_testing() -> None:
    # The thread-local is the only process state left: the capture wrapper
    # is applied as a patch and needs no per-test teardown.
    _CAPTURING.value = False
