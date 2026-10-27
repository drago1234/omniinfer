# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU layout and dispatch checks; these do not validate NPU attention kernels."""
import ast
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "dsa_local_query", ROOT / "omni/v1/layers/attention/dsa_local_query.py"
)
layout_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout_module)
ZigzagQueryLayout = layout_module.ZigzagQueryLayout


@pytest.mark.parametrize("width", [128, 256, 512, 1024, 2048])
def test_zigzag_roundtrip_and_request_metadata(width):
    x = torch.arange(width * 3).reshape(width, 3)
    layouts = [ZigzagQueryLayout(width, 4, rank) for rank in range(4)]
    gathered = torch.cat([layout.select(x) for layout in layouts])
    assert torch.equal(layouts[0].restore_rank_major(gathered), x)
    # Multiple requests, prefix histories, a short request and padding rows.
    ends = torch.tensor([7, width // 2, width], dtype=torch.int32)
    starts = [0, 7, width // 2]
    histories = [100, 200, 0]
    kv = ends - torch.tensor(starts) + torch.tensor(histories)
    blocks = torch.arange(12).reshape(3, 4)
    for layout in layouts:
        q_ends, kv_lens, local_blocks = layout.metadata(ends, kv, blocks)
        lengths, expected_kv = [], []
        for lo in layout.starts:
            hi = lo + layout.fragment
            for start, end, history in zip(starts, ends.tolist(), histories):
                lengths.append(max(0, min(end, hi) - max(start, lo)))
                expected_kv.append(history + max(0, min(end, hi) - start))
        assert q_ends.tolist() == torch.tensor(lengths).cumsum(0).tolist()
        assert kv_lens.tolist() == expected_kv
        assert torch.equal(local_blocks, torch.cat([blocks, blocks]))


def _source_functions(path, names, namespace, class_name=None):
    """Load actual routing code while replacing only device dependencies."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        # Ignore annotations and device registration decorators for this CPU check.
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("condition", ["valid", "non_a5", "no_sp", "no_cp", "tp1", "no_q_lora"])
def test_graph_token_parallel_constructor_validation(enabled, condition):
    path = ROOT / "omni/v1/layers/attention/npu_dsa.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "NPUDeepseekSparseAttention")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name == "__init__")
    guard = next(node for node in init.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.enable_prefill_graph_token_parallel")
    owner = SimpleNamespace(
        enable_prefill_graph_token_parallel=enabled, on_ascend950=condition != "non_a5",
        ena_sp=condition != "no_sp", ena_cp=condition != "no_cp",
        tp_size=1 if condition == "tp1" else 4,
        q_lora_rank=None if condition == "no_q_lora" else 16,
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(path), "exec")
    if enabled and condition != "valid":
        with pytest.raises(ValueError, match="DSA local query requires"):
            exec(code, {"self": owner})
    else:
        exec(code, {"self": owner})


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail_prefill", [False, True])
@pytest.mark.parametrize("width,graph", [(256, True), (260, True), (2048, True), (4096, True), (4096, False)])
def test_graph_dispatch_keeps_zigzag_topk_local_and_long_cp_unchanged(rank, enabled, fail_prefill, width, graph):
    group_calls = []
    local_rows = width // 4
    full_hidden = torch.arange(width * 2, dtype=torch.float32).reshape(width, 2)

    def all_gather(x, dim=0):
        group_calls.append(x.dtype)
        return full_hidden if x.dtype == torch.float32 else torch.cat([x] * 4)

    group = SimpleNamespace(rank_in_group=rank, world_size=4, all_gather=all_gather)
    meta = SimpleNamespace(
        num_actual_tokens=width, num_decodes=0, prefill=object(),
        prefill_graph_width=width if graph else None,
    )
    calls = []
    owner = SimpleNamespace(
        ena_sp=True, ena_cp=True, enable_prefill_graph_token_parallel=enabled, tp_size=4,
        param_sink_number=0, noncontiguous_kv=True, layer_idx=0,
        attn=SimpleNamespace(kv_cache=()), on_ascend950=True,
        o_proj=SimpleNamespace(tp_size=1, x_transform="NoOp", y_transform="NoOp"),
    )

    def prefill(x, cos, sin, metadata, cache, indices, **kwargs):
        local_query = kwargs.get("use_local_query", False)
        calls.append(("graph", local_query))
        assert not owner.ena_sp
        assert x.shape == full_hidden.shape
        assert indices.shape[0] == (local_rows if local_query else width)
        # Different rank-local indices expose an erroneous post-query-parallel split.
        result = torch.full_like(indices, rank + 10) if local_query else torch.arange(width).reshape(width, 1, 1)
        output_slice = kwargs.get("output_sp_slice")
        if output_slice is not None:
            assert enabled
            x = x[output_slice].contiguous()
        if fail_prefill:
            raise RuntimeError("intentional prefill failure")
        return x, result

    def prefill_cp(x, cos, sin, metadata, cache, indices):
        calls.append(("cp", False))
        assert owner.ena_sp
        return x, indices

    owner._forward_prefill, owner._forward_prefill_cp = prefill, prefill_cp
    namespace = {
        "torch": torch, "get_tp_group": lambda: group,
        "get_forward_context": lambda: SimpleNamespace(no_compile_layers={"dsa": owner}, attn_metadata=meta),
        "get_batch_desc": lambda *args: (slice(0, width), slice(0, 0), True, False),
        "model_extra_config": SimpleNamespace(operator_opt_config=SimpleNamespace(enable_prefill_graph_token_parallel=enabled)),
    }
    _source_functions("omni/v1/layers/attention/npu_mla.py", {"_prefill_o_proj_sp_slice"}, namespace)
    _source_functions("omni/attention/backends/utils.py", {"lazy_zero_like", "sp_disabled"}, namespace)
    namespace["sp_disabled"] = contextmanager(namespace["sp_disabled"])
    _source_functions("omni/v1/layers/attention/npu_dsa.py", {"npu_dsa_forward"}, namespace)
    indices = torch.full((local_rows, 1, 1), -1, dtype=torch.int64)
    hidden = full_hidden.chunk(4)[rank]
    if fail_prefill and graph:
        with pytest.raises(RuntimeError, match="intentional prefill failure"):
            namespace["npu_dsa_forward"](hidden, full_hidden, full_hidden, "dsa", indices, "attn")
        assert owner.ena_sp  # Both graph branches restore SP on exceptions.
        return
    output, topk = namespace["npu_dsa_forward"](hidden, full_hidden, full_hidden, "dsa", indices, "attn")
    assert owner.ena_sp  # PR context manager restores SP after graph execution.
    assert torch.equal(output, hidden)
    assert topk.shape == indices.shape
    if not graph:
        assert calls == [("cp", False)]
        assert group_calls == []
    elif enabled and width % 8 == 0:
        assert calls == [("graph", True)]
        assert torch.all(topk == rank + 10)
        assert group_calls == [torch.float32]  # Only the PR hidden AG remains.
    else:
        assert calls == [("graph", False)]
        assert topk.flatten().tolist() == list(range(rank * local_rows, (rank + 1) * local_rows))
        assert group_calls == [torch.int64, torch.float32]
