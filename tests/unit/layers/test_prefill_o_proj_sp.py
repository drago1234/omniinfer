# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU checks of projection row ownership and unchanged full-token cache calls."""
from contextlib import contextmanager
import ast
from types import SimpleNamespace

import pytest
import torch

from tests.unit.layers.test_dsa_local_query import ROOT, ZigzagQueryLayout, _source_functions


def test_mla_constructor_keeps_compilation_config_for_prefill_dispatch():
    tree = ast.parse((ROOT / "omni/v1/layers/attention/npu_mla.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUDeepseekMLAAttention")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(node for node in init.body if isinstance(node, ast.Assign)
                      and any(ast.unparse(target) == "self.compilation_config" for target in node.targets))
    owner, config = SimpleNamespace(), object()
    namespace = {"self": owner, "compilation_config": config}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "mla_init_config", "exec"), namespace)
    assert owner.compilation_config is config


def _namespace(rank, enabled=True):
    group = SimpleNamespace(world_size=4, rank_in_group=rank)
    return {
        "torch": torch,
        "get_tp_group": lambda: group,
        "model_extra_config": SimpleNamespace(operator_opt_config=SimpleNamespace(
            enable_prefill_graph_token_parallel=enabled, li_prolog_multi_stream=False,
        )),
    }


@pytest.mark.parametrize("condition", ["disabled", "non_a5", "non_sp", "tp2", "x_ag", "y_ar", "unaligned", "tp1"])
def test_projection_slice_guards(condition):
    namespace = _namespace(2, enabled=condition != "disabled")
    owner = SimpleNamespace(
        on_ascend950=condition != "non_a5", ena_sp=condition != "non_sp",
        o_proj=SimpleNamespace(tp_size=2 if condition == "tp2" else 1,
                              x_transform="AllGather" if condition == "x_ag" else "NoOp",
                              y_transform="AllReduce" if condition == "y_ar" else "NoOp"),
    )
    width = 258 if condition == "unaligned" else 256
    if condition == "tp1":
        namespace["get_tp_group"]().world_size = 1
    _source_functions("omni/v1/layers/attention/npu_mla.py", {"_prefill_o_proj_sp_slice"}, namespace)
    assert namespace["_prefill_o_proj_sp_slice"](owner, width) is None


def test_sp_wrapper_restores_flags_after_exception():
    namespace = _namespace(2)
    namespace["get_tp_group"]().all_gather = lambda x, dim=0: torch.cat([x] * 4)
    _source_functions("omni/attention/backends/utils.py", {"lazy_zero_like", "sp_disabled"}, namespace)
    wrapper = contextmanager(namespace["sp_disabled"])
    projection = SimpleNamespace(x_transform="NoOp", y_transform="ReduceScatter")
    owner = SimpleNamespace(ena_sp=True, o_proj=projection)
    with pytest.raises(RuntimeError, match="intentional"):
        with wrapper(owner, torch.zeros(64, 2)) as (x, y, out):
            assert x.size(0) == 256
            assert y.shape[0] == 256
            assert not owner.ena_sp and projection.y_transform == "AllReduce"
            raise RuntimeError("intentional")
    assert owner.ena_sp and projection.y_transform == "ReduceScatter"


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("mode", ["graph", "disabled", "eager", "mome_sp", "tp2", "non_sp"])
def test_mla_conv_precedes_slice_and_projection_keeps_local_output(rank, mode):
    namespace = _namespace(rank, enabled=mode != "disabled")
    namespace["graph_prefill_width"] = lambda *args: None if mode == "eager" else 256
    _source_functions("omni/v1/layers/attention/npu_mla.py", {"_prefill_o_proj_sp_slice"}, namespace)
    _source_functions("omni/v1/layers/attention/npu_mla.py",
                      {"_should_slice_prefill_output", "_finish_prefill"}, namespace,
                      class_name="NPUDeepseekMLAAttention")
    full = torch.arange(256 * 6, dtype=torch.float32).reshape(256, 6)
    rows = slice(rank * 64, (rank + 1) * 64)
    calls = []
    owner = SimpleNamespace(
        on_ascend950=True, ena_sp=mode != "non_sp", compilation_config=object(),
        o_proj=SimpleNamespace(tp_size=2 if mode == "tp2" else 1, x_transform="NoOp", y_transform="NoOp"),
    )
    owner._should_slice_prefill_output = lambda sp, mome: namespace["_should_slice_prefill_output"](owner, sp, mome)

    def conv(out, args):
        calls.append(("conv", out.size(0)))
        return out + 1

    def project(out):
        calls.append(("project", out.size(0)))
        return out[:, :2] * 3

    def slice_tokens(out):
        calls.append(("post_slice", out.size(0)))
        return out[rows]

    owner._maybe_mome_out = conv
    owner._apply_mome_out_prefill_sp = lambda out, sp: (out + 1)[rows] if mode == "mome_sp" else None
    owner._apply_o_proj = project
    sp = None if mode == "non_sp" else SimpleNamespace(tok=256, slice_tokens=slice_tokens, align_tokens=lambda out: out)
    x = torch.zeros(256 if sp is None else 64, 2)
    result = namespace["_finish_prefill"](owner, full, x, sp, lambda: {})
    expected = (full + 1)[:, :2] * 3
    torch.testing.assert_close(result, expected if sp is None else expected[rows])
    if mode != "mome_sp":
        assert calls[0] == ("conv", 256)
    projected_rows = 64 if mode in ("graph", "mome_sp") else 256
    assert ("project", projected_rows) in calls
    assert any(name == "post_slice" for name, _ in calls) == (mode in ("disabled", "eager", "tp2"))


@pytest.mark.parametrize("mode", ["disabled", "no_sp", "mome_sp", "eager", "graph"])
def test_prefill_output_slice_predicate_preserves_short_circuit(mode):
    namespace = _namespace(0, enabled=mode != "disabled")
    calls = []

    def graph_width(*args):
        calls.append(args)
        return 256 if mode == "graph" else None

    namespace["graph_prefill_width"] = graph_width
    _source_functions("omni/v1/layers/attention/npu_mla.py",
                      {"_should_slice_prefill_output"}, namespace,
                      class_name="NPUDeepseekMLAAttention")
    owner = SimpleNamespace(compilation_config=object())
    sp = None if mode == "no_sp" else SimpleNamespace(tok=256)
    mome = object() if mode == "mome_sp" else None
    result = namespace["_should_slice_prefill_output"](owner, sp, mome)
    assert result == (mode == "graph")
    assert len(calls) == (1 if mode in ("eager", "graph") else 0)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("query_parallel", [False, True])
@pytest.mark.parametrize("project_local", [False, True])
@pytest.mark.parametrize("reuse_topk", [False, True])
def test_dsa_full_conv_and_kv_calls_unchanged_before_local_projection(monkeypatch, rank, query_parallel, project_local, reuse_topk):
    namespace = _namespace(rank)
    namespace["DummySPManager"] = lambda: object()
    namespace["ZigzagQueryLayout"] = ZigzagQueryLayout
    namespace["torch"] = SimpleNamespace(int32=torch.int32, npu=SimpleNamespace(current_stream=lambda: None))
    _source_functions("omni/v1/layers/attention/npu_dsa.py", {"_forward_prefill"}, namespace,
                      class_name="NPUDeepseekSparseAttention")
    full = torch.arange(256 * 2, dtype=torch.float32).reshape(256, 2)
    rows = slice(rank * 64, (rank + 1) * 64)
    layout = ZigzagQueryLayout(256, 4, rank)
    calls = []

    def restore(layout_self, out):
        torch.testing.assert_close(out, layout_self.select(full))
        calls.append(("query_gather", out.size(0)))
        return full.clone()

    monkeypatch.setattr(ZigzagQueryLayout, "gather_full_tokens", restore)

    def conv(out, args):
        calls.append(("conv", out.size(0)))
        return out + 1

    def project(out):
        calls.append(("project", out.size(0)))
        return out * 3

    indexer = SimpleNamespace(
        _li_prolog_ext=lambda local_x, local_q, x, *args: (local_x, local_q, x),
        _li_prolog=lambda x, *args: (x, x, x),
        _update_cache=lambda ki, *args: calls.append(("index_cache", ki.size(0))),
        _apply_lightning_indexer=lambda wi, *args, **kwargs: torch.zeros(wi.size(0), 1, 2, dtype=torch.int32),
    )
    owner = SimpleNamespace(
        sharded_o_proj=False, ena_kvsp=False, noncontiguous_kv=True, ena_sp=False,
        use_mome=True, q_b_proj=SimpleNamespace(tp_size=1), skip_topk=reuse_topk,
        _get_li_caches=lambda cache: (None, None),
        q_a_proj=lambda x: (x, None), q_a_layernorm=lambda x: x,
        _maybe_mome_q=lambda q, args: q, _q_absorb=lambda q, *args: (q, q),
        kv_a_proj_with_mqa=lambda x: (x, None), _maybe_mome_kv=lambda kv, args: kv,
        _kv_norm_rope_cache=lambda kv, *args: calls.append(("kv_cache", kv.size(0))),
        indexer=indexer, _apply_sink_offset=lambda indices: indices,
        _apply_attn_absorb=lambda q, *args, **kwargs: q,
        _post_attn_absorb=lambda out: out, _maybe_mome_out=conv, _apply_o_proj=project,
    )
    metadata = SimpleNamespace(seq_lens=torch.tensor([107, 249]), query_cumlens=torch.tensor([7, 256]),
                               block_table=torch.zeros(2, 4, dtype=torch.int32))
    topk = torch.zeros(64 if query_parallel else 256, 1, 2, dtype=torch.int32)
    result, indices = namespace["_forward_prefill"](
        owner, full, full, full, metadata, (), topk,
        use_local_query=query_parallel, output_sp_slice=rows if project_local else None,
    )
    expected = (full + 1) * 3
    torch.testing.assert_close(result, expected[rows] if project_local else expected)
    assert indices.shape == topk.shape
    assert ("kv_cache", 256) in calls
    assert ("conv", 256) in calls
    assert ("project", 64 if project_local else 256) in calls
    assert calls.index(("conv", 256)) < calls.index(("project", 64 if project_local else 256))
    if not reuse_topk:
        assert ("index_cache", 256) in calls
