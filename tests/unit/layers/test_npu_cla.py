# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cross-layer attention for the high-throughput stack."""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import omni_npu.v1.layers.attention.npu_cla as cla_mod
from omni_npu.v1.layers.attention.npu_cla import (
    CLAOutputGate,
    CLAReuseLayerMixin,
    NPUDeepseekMLACLAAttention,
    NPUDeepseekSparseCLAAttention,
    npu_cla_forward,
)
from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention


@pytest.mark.unit
def test_both_cla_classes_provide_what_the_mixin_calls():
    """The mixin runs on a sparse host and a dense one; both must satisfy it.

    The two hosts do not share a base, so a helper that only one of them
    inherits raises AttributeError on the first real forward -- and a unit test
    that stubs the helper onto its fake host will not notice.
    """
    module = ast.parse(inspect.getsource(cla_mod))
    mixin = next(n for n in module.body
                 if isinstance(n, ast.ClassDef) and n.name == "CLAReuseLayerMixin")
    called = {n.func.attr for n in ast.walk(mixin)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"}
    defined = {n.name for n in mixin.body if isinstance(n, ast.FunctionDef)}
    # Attributes the mixin creates itself are not the host's to provide.
    assigned = {t.attr for n in ast.walk(module) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name) and t.value.id == "self"}
    # Layers the host builds in its own __init__, so they exist per instance
    # rather than on the class.
    host_attributes = {"q_a_proj", "q_a_layernorm"}
    required = called - defined - assigned - host_attributes
    assert required, "the walk found nothing, so it is no longer checking anything"
    for host in (NPUDeepseekSparseCLAAttention, NPUDeepseekMLACLAAttention):
        missing = sorted(n for n in required if not hasattr(host, n))
        assert not missing, f"{host.__name__} is missing {missing}"


@pytest.mark.unit
def test_cla_output_gate_narrows_an_unsharded_gate_to_local_heads():
    gate = CLAOutputGate.__new__(CLAOutputGate)
    torch.nn.Module.__init__(gate)
    gate.num_local_heads = 2
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]] * 3)  # 4 global heads, this rank owns 2
    gate.w_gate = lambda _x: logits
    with patch.object(cla_mod, "get_tp_group", return_value=SimpleNamespace(rank_in_group=1)):
        out = gate.forward(torch.zeros(3, 6), torch.ones(3, 2 * 5))
    assert out.shape == (3, 10)
    expected = torch.sigmoid(logits[:, 2:]).repeat_interleave(5, dim=1)  # heads 2 and 3
    torch.testing.assert_close(out, expected)


@pytest.mark.unit
def test_local_branch_attention_takes_the_window_page_and_the_npu_absorb_weights():
    """Swapping the bases would silently drop one of the two.

    The patch machinery can hand out more than one NPUMLAAttention object, so
    compare where each method resolves from rather than the objects themselves.
    """
    from omni_npu.v1.layers.attention.npu_cla import NPUMLASWAAttention

    mro = [c.__name__ for c in NPUMLASWAAttention.__mro__]
    assert mro.index("NPUMLAAttention") < mro.index("MLASWAAttention")
    assert NPUMLASWAAttention.process_weights_after_loading.__qualname__ == (
        "NPUMLAAttention.process_weights_after_loading"
    )
    assert NPUMLASWAAttention.get_kv_cache_spec.__qualname__ == "MLASWAAttention.get_kv_cache_spec"
    assert NPUMLASWAAttention.get_attn_backend.__qualname__ == "MLASWAAttention.get_attn_backend"


@pytest.mark.unit
def test_cla_swa_page_size_padded_matches_the_dsa_page():
    fn = CLAReuseLayerMixin._cla_swa_page_size_padded
    cache = SimpleNamespace(cache_dtype="auto", block_size=128)
    dense = SimpleNamespace(index_topk=0, kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128)
    assert fn(dense, cache) is None  # no DSA layer in the group, nothing to pad to

    sparse = SimpleNamespace(index_topk=2048, kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128)
    with patch.object(cla_mod, "model_extra_config", SimpleNamespace(dtype=torch.bfloat16)):
        assert fn(sparse, cache) == 128 * (512 + 64 + 128) * 2

    with pytest.raises(NotImplementedError, match="quantized KV"):
        fn(sparse, SimpleNamespace(cache_dtype="fp8_ds_mla", block_size=128))


@pytest.mark.unit
def test_only_the_sparse_cla_layer_declares_an_unpacked_local_cache():
    assert CLAReuseLayerMixin._local_kv_layout_kwargs(object()) == {}
    sparse = NPUDeepseekSparseCLAAttention.__new__(NPUDeepseekSparseCLAAttention)
    assert sparse._local_kv_layout_kwargs() == {"packed_kv": False}


@pytest.mark.unit
def test_combine_cla_branches_sums_both_gated_branches():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.swa_gate_global = lambda _x, out: out * 2
    host.swa_gate_local = lambda _x, out: out * 3
    combined = host._combine_cla_branches(
        torch.zeros(2, 4), torch.ones(2, 6), torch.full((2, 6), 2.0)
    )
    torch.testing.assert_close(combined, torch.full((2, 6), 8.0))  # 1*2 global + 2*3 local


@pytest.mark.unit
def test_npu_cla_forward_returns_zeros_on_a_dummy_run():
    layer = SimpleNamespace(cla_swa_attn_name="x", prefix="y", layer_idx=0)
    ctx = SimpleNamespace(no_compile_layers={"y": layer}, attn_metadata=None)
    with patch.object(cla_mod, "get_forward_context", return_value=ctx):
        out = npu_cla_forward(
            torch.ones(3, 4), torch.ones(3, 2), torch.ones(3, 2), "y", torch.zeros(3, 1, 8)
        )
    torch.testing.assert_close(out, torch.zeros(3, 4))


@pytest.mark.unit
def test_init_cla_reuse_rejects_a_plain_layer():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.layer_idx = 4
    config = SimpleNamespace(num_hidden_layers=32, cla_explicit_mapping=[[7, 3]])
    with pytest.raises(ValueError, match="not a CLA reuse layer"):
        host._init_cla_reuse(config, "p", None, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "q_tp, kv_tp, kvsp, error",
    ((8, 1, False, ValueError), (1, 8, False, ValueError), (1, 1, True, NotImplementedError)),
)
def test_init_cla_reuse_needs_every_head_and_no_kvsp_under_cp(q_tp, kv_tp, kvsp, error):
    """CP attends with every head on every rank, and pages are never gathered."""
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.layer_idx = 7
    host.ena_cp, host.ena_kvsp = True, kvsp
    host.q_b_proj, host.kv_b_proj = SimpleNamespace(tp_size=q_tp), SimpleNamespace(tp_size=kv_tp)
    config = SimpleNamespace(num_hidden_layers=32, cla_explicit_mapping=[[7, 3]])
    with pytest.raises(error):
        host._init_cla_reuse(config, "p", None, None)


@pytest.mark.unit
def test_forward_cla_gathers_heads_before_the_output_projection():
    """forward_cla does not go through the decode epilog, so it must gather here."""
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.ena_sp = False
    host.q_a_proj = lambda x: (x,)
    host.q_a_layernorm = lambda x: x
    host._q_project = lambda *_a: (torch.zeros(2, 2, 3), torch.zeros(2, 2, 2))
    host._write_local_kv = lambda *_a: None
    host._cla_global_attention = lambda *_a: torch.ones(2, 6)
    host._apply_local_swa = lambda *_a: torch.ones(2, 6)
    host._combine_cla_branches = lambda _x, g, l: g + l
    calls = []
    host._maybe_mome_out = lambda out, args: calls.append(("mome_out", args)) or out
    host._apply_o_proj = lambda out: calls.append(("o_proj",)) or out

    host.forward_cla(torch.zeros(2, 4), None, None, SimpleNamespace(num_tokens=2), None, "decode", None, None)
    assert [c[0] for c in calls] == ["mome_out", "o_proj"]  # gather first, then project


@pytest.mark.unit
@pytest.mark.parametrize("rescale", (True, False))
@pytest.mark.parametrize("batch_invariant", (False, True))
def test_local_branch_picks_the_sink_mode_the_config_asks_for(rescale, batch_invariant):
    """The local branch runs the same two kernels the plain layers choose between."""
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.scaling, host.kv_lora_rank = 0.5, 6
    host.cla_swa_gate_window = 512
    host.is_cla_fa_metadata_producer = True
    host.use_gpt_oss_sink = True
    host.use_gpt_oss_sink_rescale = rescale
    host.gpt_oss_sink_swa = torch.zeros(2)
    host.num_local_heads = 2
    host.attn_cla_swa = SimpleNamespace(
        kv_cache=(torch.zeros(4, 8, 6), torch.zeros(4, 8, 2)),
        W_UK_T=torch.zeros(1), W_UV=torch.zeros(1),
    )
    host._q_nope_absorb = lambda q, _w: q
    host._lengths_to_i64 = lambda lengths, _dev: lengths
    host._post_attn_absorb = lambda out, _w: out
    host._rescale_gpt_oss_sink_output = MagicMock(side_effect=lambda out, *_a: out)
    native = MagicMock(return_value=torch.zeros(2, 3, 6))
    host._apply_gpt_oss_fia_native = native
    meta = SimpleNamespace(
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        query_cumlens=torch.tensor([3]), seq_lens=torch.tensor([8]),
    )
    sink_op = MagicMock(return_value=(torch.zeros(2, 3, 6), torch.zeros(3, 2, 1)))
    tiling = MagicMock()
    with patch.object(cla_mod.model_extra_config.operator_opt_config, "use_batch_invariant_op", batch_invariant), \
         patch.object(cla_mod, "cache_fit_shape", lambda t, _l: t), \
         patch.object(cla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock()), \
         patch.object(cla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, create=True), \
         patch.object(cla_mod, "npu_fused_infer_attention_sink_metadata", tiling), \
         patch.object(torch.npu, "get_stream_limit",
                      return_value={"cube_core_num": 24, "vector_core_num": 48}), \
         patch.object(torch.ops.custom, "npu_fused_infer_attention_sink", sink_op, create=True):
        host._apply_local_swa(torch.zeros(3, 2, 6), torch.zeros(3, 2, 2), meta, "decode", 3)

    assert native.called is (not rescale)
    assert sink_op.called is rescale
    # The window is 511 for the plain layers too, so the key has to say cla_ or
    # this branch reads tiling built for the full-context cache.
    caller = (tiling if rescale else native).call_args.args[2]
    assert caller == ("cla_decode", 511)

    kwargs = sink_op.call_args.kwargs if rescale else native.call_args.args[0]
    assert kwargs["batch_invariant"] is batch_invariant
    if rescale:
        assert tiling.call_args.args[0]["batch_invariant"] is batch_invariant

@pytest.mark.unit
def test_write_local_kv_normalises_with_the_local_weights():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.kv_a_proj_with_mqa_swa = lambda x: (x * 2,)
    host.kv_a_layernorm_swa = "local norm"
    host.attn_cla_swa = SimpleNamespace(kv_cache=("nope", "rope"))
    host._local_kv_layout_kwargs = lambda: {"packed_kv": False}
    seen = {}
    host._kv_norm_rope_cache = lambda *args, **kw: seen.update(args=args, kw=kw)

    host._write_local_kv(torch.ones(2, 3), "cos", "sin", "slots")

    torch.testing.assert_close(seen["args"][0], torch.full((2, 3), 2.0))
    assert seen["args"][1:] == ("cos", "sin", "slots", ("nope", "rope"))
    assert seen["kw"] == {"kv_a_layernorm": "local norm", "packed_kv": False}


@pytest.mark.unit
def test_local_branch_pads_a_captured_tail_back_on():
    """A captured decode batch runs padded; the op only fills the valid tokens."""
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.scaling, host.kv_lora_rank = 0.5, 6
    host.cla_swa_gate_window = 512
    host.is_cla_fa_metadata_producer = True
    host.use_gpt_oss_sink = False
    host.use_gpt_oss_sink_rescale = False
    host.num_local_heads = 2
    host.attn_cla_swa = SimpleNamespace(
        kv_cache=(torch.zeros(4, 8, 6), torch.zeros(4, 8, 2)),
        W_UK_T=torch.zeros(1), W_UV=torch.zeros(1),
    )
    host._q_nope_absorb = lambda q, _w: q
    host._post_attn_absorb = lambda out, _w: out
    meta = SimpleNamespace(
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        query_cumlens=torch.tensor([2]), seq_lens=torch.tensor([8]),
    )
    # The op returns 2 valid tokens while the batch is padded to 3.
    sink_op = MagicMock(return_value=(torch.ones(2, 2, 6), torch.zeros(2, 2, 1)))
    with patch.object(cla_mod, "cache_fit_shape", lambda t, _l: t), \
         patch.object(cla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock()), \
         patch.object(cla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, create=True), \
         patch.object(cla_mod, "npu_fused_infer_attention_sink_metadata", MagicMock()), \
         patch.object(torch.npu, "get_stream_limit",
                      return_value={"cube_core_num": 24, "vector_core_num": 48}), \
         patch.object(torch.ops.custom, "npu_fused_infer_attention_sink", sink_op, create=True):
        out = host._apply_local_swa(torch.zeros(3, 2, 6), torch.zeros(3, 2, 2), meta, "decode", 2)

    assert out.shape == (3, 2, 6)  # TND, padded back to the captured length
    torch.testing.assert_close(out[2], torch.zeros(2, 6))  # the tail stays zero


@pytest.mark.unit
def test_the_base_global_branch_is_left_to_the_subclasses():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    with pytest.raises(NotImplementedError):
        host._cla_global_attention(None, None, None, "decode", None)


@pytest.mark.unit
def test_forward_delegates_and_passes_the_buffer_through():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.prefix = "layer.7.self_attn"
    buffer = torch.zeros(3, 1, 8)
    op = MagicMock(return_value=torch.ones(3, 4))
    with patch.object(torch.ops.vllm, "npu_cla_forward", op, create=True):
        out, passed = host.forward(torch.zeros(3, 4), "cos", "sin", buffer)
    args = op.call_args.args
    assert args[1] == "cos" and args[2] == "sin" and args[3] == "layer.7.self_attn"
    assert args[4] is buffer
    assert passed is buffer
    torch.testing.assert_close(out, torch.ones(3, 4))


@pytest.mark.unit
@pytest.mark.parametrize("phases", (("prefill",), ("decode",), ("prefill", "decode")))
@pytest.mark.parametrize("with_topk", (False, True))
def test_npu_cla_forward_runs_each_token_slice_with_its_own_slots(phases, with_topk):
    """Each slice gets its own metadata, its own slots and its own output rows."""
    layer = SimpleNamespace(cla_swa_attn_name="swa", prefix="p", layer_idx=7, ena_sp=False, ena_cp=False,
                            o_proj=SimpleNamespace())
    slots = torch.arange(5, dtype=torch.int32)
    slots_2d = torch.stack([slots // 2, slots % 2], dim=-1)
    meta_swa = SimpleNamespace(
        prefill="swa_p", decode="swa_d",
        slot_mapping=slots, get_slot_mapping_2d=lambda: slots_2d,
    )
    metadata = {"swa": meta_swa, "p.attn": SimpleNamespace(prefill="p_p", decode="p_d")}
    ctx = SimpleNamespace(no_compile_layers={"p": layer}, attn_metadata=metadata)

    p_slice, d_slice = slice(0, 2), slice(2, 5)
    desc = (p_slice, d_slice, "prefill" in phases, "decode" in phases)
    calls = []

    def fake_forward_cla(x, cos, sin, attn_meta, swa_meta, caller, local_slots, topk):
        assert (topk is not None) == with_topk
        calls.append((caller, x.size(0), attn_meta, swa_meta,
                      local_slots.slot_mapping.tolist(), local_slots.slot_mapping_2d.size(0)))
        return torch.full((x.size(0), x.size(1)), 1.0 if caller == "prefill" else 2.0)

    layer.forward_cla = fake_forward_cla
    with patch.object(cla_mod, "get_forward_context", return_value=ctx), \
         patch.object(cla_mod, "get_batch_desc", return_value=desc):
        out = npu_cla_forward(torch.zeros(5, 4), torch.zeros(5, 2), torch.zeros(5, 2),
                              "p", torch.zeros(5, 1, 8) if with_topk else None)

    assert [c[0] for c in calls] == list(phases)
    by_caller = {c[0]: c for c in calls}
    if "prefill" in phases:
        caller, rows, attn_meta, swa_meta, sl, sl2 = by_caller["prefill"]
        assert (rows, attn_meta, swa_meta, sl, sl2) == (2, "p_p", "swa_p", [0, 1], 2)
        torch.testing.assert_close(out[p_slice], torch.ones(2, 4))
    else:
        torch.testing.assert_close(out[p_slice], torch.zeros(2, 4))
    if "decode" in phases:
        caller, rows, attn_meta, swa_meta, sl, sl2 = by_caller["decode"]
        assert (rows, attn_meta, swa_meta, sl, sl2) == (3, "p_d", "swa_d", [2, 3, 4], 3)
        torch.testing.assert_close(out[d_slice], torch.full((3, 4), 2.0))
    else:
        torch.testing.assert_close(out[d_slice], torch.zeros(3, 4))


@pytest.mark.unit
def test_npu_cla_forward_fake_keeps_the_input_shape():
    fake = cla_mod.npu_cla_forward_fake(
        torch.zeros(3, 4), torch.zeros(3, 2), torch.zeros(3, 2), "p", torch.zeros(3, 1, 8)
    )
    assert fake.shape == (3, 4) and fake.dtype == torch.float32


@pytest.mark.unit
def test_the_sparse_global_branch_selects_with_the_source_layer_cache():
    host = NPUDeepseekSparseCLAAttention.__new__(NPUDeepseekSparseCLAAttention)
    host.attn = SimpleNamespace(kv_cache="source cache")
    host._q_nope_absorb = lambda q: q * 2
    host._apply_sink_offset = lambda idx: idx + 1
    host._post_attn_absorb = lambda out: out * 10
    seen = {}

    def fake_absorb(q_nope, q_pe, q_cumlens, kv_lens, topk, block_table, kv_cache, attn_metadata):
        seen.update(q=q_nope, topk=topk, kv_cache=kv_cache, dtypes=(q_cumlens.dtype, kv_lens.dtype))
        return torch.ones(2, 1, 3)

    host._apply_attn_absorb = fake_absorb
    meta = SimpleNamespace(query_cumlens=torch.tensor([2], dtype=torch.int64),
                           seq_lens=torch.tensor([8], dtype=torch.int64),
                           block_table="table")
    out = host._cla_global_attention(torch.ones(2, 1, 3), "pe", meta, "decode", torch.zeros(2, 1, 4))

    torch.testing.assert_close(seen["q"], torch.full((2, 1, 3), 2.0))  # its own W_UK_T
    torch.testing.assert_close(seen["topk"], torch.ones(2, 1, 4))  # sink offset applied
    assert seen["kv_cache"] == "source cache"
    assert seen["dtypes"] == (torch.int32, torch.int32)  # the sparse op wants int32
    torch.testing.assert_close(out, torch.full((2, 1, 3), 10.0))


@pytest.mark.unit
def test_the_dense_global_branch_attends_over_the_source_layer_cache():
    host = NPUDeepseekMLACLAAttention.__new__(NPUDeepseekMLACLAAttention)
    host.attn = SimpleNamespace(kv_cache="source cache")
    host.prefix = "layer.7.self_attn"
    host._q_nope_absorb = lambda q: q
    seen = {}

    def fake_attention(q_nope, q_pe, **kwargs):
        seen.update(kwargs)
        return torch.ones(1, 2, 3)  # NTD

    host._apply_attention = fake_attention
    meta = SimpleNamespace(query_cumlens="qc", seq_lens="kv", block_table="table")
    epilog = MagicMock(return_value=torch.ones(2, 3))
    with patch.object(NPUDeepseekMLAAttention, "_post_attn_absorb", epilog):
        out = host._cla_global_attention(torch.zeros(2, 1, 3), "pe", meta, "prefill_absorb", None)

    assert seen["keys"] == "source cache" and seen["num_tokens"] == 2
    assert seen["layer_name"] == "layer.7.self_attn.attn"
    assert seen["metadata_caller"] == "prefill_absorb"
    assert epilog.call_args.args[0].shape == (1, 2, 3)  # NTD goes directly to MLA's epilog.
    assert out.shape == (2, 3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "caller, source_absorb, original_producer",
    (("prefill", True, False), ("prefill", False, False),
     ("decode", True, True), ("decode", False, False)),
)
def test_dense_cla_reads_current_step_metadata_with_either_source_path(caller, source_absorb, original_producer):
    from omni_npu.attention.backends.utils import CrossLayerSharedOp

    window = NPUDeepseekMLACLAAttention._to_metadata_pre_tokens(None)
    build = MagicMock(side_effect=lambda token_count: torch.tensor([token_count], dtype=torch.int32))
    shared = CrossLayerSharedOp(build, (1,), torch.int32,
                                tuple((phase, window) for phase in ("prefill", "prefill_absorb", "decode")),
                                device="cpu")
    for buffer in shared._buffers.values():
        buffer.fill_(-999)  # Detect reading a reserved buffer its producer never wrote.
    host = NPUDeepseekMLACLAAttention.__new__(NPUDeepseekMLACLAAttention)
    host.attn = SimpleNamespace(kv_cache="source cache")
    host.prefix = "layer.7.self_attn"
    host.is_fa_metadata_producer = original_producer
    host.mla_absorb = source_absorb
    with patch.object(CLAReuseLayerMixin, "_init_cla_reuse"):
        host._init_cla_reuse(None, host.prefix, None, None)
    host._q_nope_absorb = lambda q: q

    def attention(q_nope, q_pe, **kwargs):
        tiling = shared({"token_count": q_nope.size(0)}, host.is_fa_metadata_producer,
                       (kwargs["metadata_caller"], window))
        return tiling.to(torch.float32).expand(1, q_nope.size(0), 1)

    host._apply_attention = attention
    producer_phase = "decode" if caller == "decode" else ("prefill_absorb" if source_absorb else "prefill")
    epilog = lambda self, out, w_uv=None: out.transpose(0, 1).flatten(1)
    for token_count in (3, 5):
        # The source refreshes only the tiling for the attention path it actually uses.
        shared({"token_count": token_count}, True, (producer_phase, window))
        meta = SimpleNamespace(query_cumlens=[token_count], seq_lens=[token_count], block_table="table")
        query = torch.zeros(token_count, 1, 1)
        with patch.object(NPUDeepseekMLAAttention, "_post_attn_absorb", epilog):
            output = host._cla_global_attention(query, query, meta, caller, None)
        torch.testing.assert_close(output, torch.full((token_count, 1), float(token_count)))
    assert host.is_fa_metadata_producer is (original_producer or not source_absorb)
    assert build.call_count == (4 if host.is_fa_metadata_producer else 2)


@pytest.mark.unit
@pytest.mark.parametrize("local_weights", (False, True))
def test_dense_cla_v_up_preserves_token_and_head_order(local_weights):
    host = NPUDeepseekMLACLAAttention.__new__(NPUDeepseekMLACLAAttention)
    host.num_local_heads, host.kv_lora_rank, host.v_head_dim = 3, 4, 2
    global_weight = torch.arange(24, dtype=torch.float32).view(3, 4, 2)
    local_weight = global_weight + 100
    host.attn = SimpleNamespace(W_UV=global_weight)
    # FIA returns NTD; CLA hands its epilog a non-contiguous TND view.
    tokens = torch.arange(60, dtype=torch.float32).view(3, 5, 4).transpose(0, 1)

    def batchmatmul(input, weight, perm_y):
        assert input.shape == (3, 5, 4)
        assert perm_y == (1, 0, 2)
        result = torch.bmm(input, weight)
        return result.permute(perm_y)

    with patch("torch_npu.npu_transpose_batchmatmul", side_effect=batchmatmul):
        output = host._post_attn_absorb(tokens, local_weight if local_weights else None)

    expected = torch.einsum("tnl,nlv->tnv", tokens, local_weight if local_weights else global_weight)
    torch.testing.assert_close(output, expected.reshape(5, 6))


def _init_host(layer_idx, ena_cp=False):
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.layer_idx = layer_idx
    host.ena_cp, host.ena_kvsp = ena_cp, False
    host.q_b_proj = SimpleNamespace(tp_size=1)
    host.hidden_size, host.kv_lora_rank, host.qk_rope_head_dim = 1536, 512, 64
    host.qk_nope_head_dim, host.v_head_dim = 128, 128
    host.num_heads, host.num_local_heads = 48, 48
    host.dtype = torch.bfloat16
    host.kv_a_proj_with_mqa = "global down projection"
    host.kv_a_layernorm = "global norm"
    host.kv_b_proj = SimpleNamespace(tp_size=1)
    host.attn = SimpleNamespace()
    host.use_gpt_oss_sink, host.use_gpt_oss_sink_rescale = True, False
    host._inner_mla_attn_kwargs = lambda *_a: {}
    host._cla_swa_page_size_padded = lambda *_a: 1024
    host._init_gpt_oss_swa_sink = MagicMock()
    host._reserve_gpt_oss_fia_metadata = MagicMock()
    return host


@pytest.mark.unit
@pytest.mark.parametrize(
    "layer_idx, nextn, produces, virtual_idx, buffer_exists, ena_cp",
    ((15, 1, False, 48, False, False), (7, 0, True, 39, True, True)),
)
def test_init_cla_reuse_wires_the_local_branch_and_reserves_its_own_keys(
    layer_idx, nextn, produces, virtual_idx, buffer_exists, ena_cp
):
    config = SimpleNamespace(
        num_hidden_layers=32, num_nextn_predict_layers=nextn, rms_norm_eps=1e-5,
        cla_explicit_mapping=[[7, 3], [15, 11]], cla_swa_gate_window=512,
    )
    opt = SimpleNamespace(num_extra_reserved_blocks=2)
    host = _init_host(layer_idx, ena_cp)
    swa_attn = MagicMock()
    existing = MagicMock() if buffer_exists else None

    with patch.object(torch.ops.custom, "_npu_fused_infer_attention_sink_metadata",
                      MagicMock(), create=True), \
         patch.object(cla_mod, "model_extra_config", SimpleNamespace(operator_opt_config=opt)), \
         patch.object(cla_mod, "npu_fused_infer_attention_sink_metadata", existing), \
         patch.object(cla_mod, "ReplicatedFlashCommLinear", MagicMock()), \
         patch.object(cla_mod, "ColumnParallelFlashCommLinear", MagicMock()), \
         patch.object(cla_mod, "RMSNorm", MagicMock()), \
         patch.object(cla_mod, "CLAOutputGate", MagicMock()), \
         patch.object(cla_mod, "NPUMLASWAAttention", swa_attn), \
         patch.object(cla_mod, "CrossLayerSharedOp") as shared_op:
        host._init_cla_reuse(config, f"model.layers.{layer_idx}.self_attn", "cache cfg", None)

    assert host.cla_source_layer_idx == {7: 3, 15: 11}[layer_idx]
    assert host.is_cla_fa_metadata_producer is produces  # only the first reuse layer
    assert not hasattr(host, "kv_a_proj_with_mqa") and not hasattr(host, "kv_a_layernorm")
    assert host.attn.kv_sharing_target_layer_name == (
        f"model.layers.{host.cla_source_layer_idx}.self_attn.attn"
    )
    # Past the real layers and any MTP one, so the cache binder keys it apart.
    assert host.cla_swa_attn_name == f"model.cla_swa_layers.{virtual_idx}.attn"
    swa_kwargs = swa_attn.call_args.kwargs
    assert swa_kwargs["sliding_window"] == 512 and swa_kwargs["page_size_padded"] == 1024
    assert swa_kwargs["num_extra_reserved_blocks"] == 2

    expected = (
        ("cla_decode", 511), ("cla_prefill_absorb", 511), ("cla_prefill", 511),
        ("cla_prefill_absorb_cp", 511),  # reserved here too, or CP recomputes it every layer
    )
    assert shared_op.called is not buffer_exists  # one buffer serves every CLA layer
    if not buffer_exists:
        assert shared_op.call_args.kwargs["callers"] == expected
    host._reserve_gpt_oss_fia_metadata.assert_called_once_with(expected)
    assert host._init_gpt_oss_swa_sink.called


@pytest.mark.unit
def test_the_mixin_builds_its_host_first_then_the_local_branch():
    """The local branch reads attributes the base class sets, so order matters."""
    order = []

    class Base:
        def __init__(self, vllm_config, config, *, cache_config, quant_config, prefix, **kw):
            order.append(("base", vllm_config, config, cache_config, quant_config, prefix, kw))

    class Layer(CLAReuseLayerMixin, Base):
        def _init_cla_reuse(self, config, prefix, cache_config, quant_config):
            order.append(("cla", config, prefix, cache_config, quant_config))

    Layer("vllm cfg", "hf cfg", cache_config="cache", quant_config="quant",
          prefix="p", num_heads=48)

    assert [step[0] for step in order] == ["base", "cla"]
    assert order[0][1:] == ("vllm cfg", "hf cfg", "cache", "quant", "p", {"num_heads": 48})
    assert order[1][1:] == ("hf cfg", "p", "cache", "quant")


@pytest.mark.unit
def test_cla_output_gate_projects_the_global_head_count():
    linear = MagicMock()
    with patch.object(cla_mod, "ColumnParallelFlashCommLinear", linear):
        gate = CLAOutputGate(1536, 48, 6, "quant", "layer.7.swa_gate_local", False)
    assert gate.num_local_heads == 6
    kwargs = linear.call_args.kwargs
    assert linear.call_args.args == (1536, 48)  # every head, narrowed at forward
    assert kwargs["bias"] is False and kwargs["disable_tp"] is False
    assert kwargs["prefix"] == "layer.7.swa_gate_local.w_gate"


@pytest.mark.unit
def test_init_cla_reuse_needs_a_window_size():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.layer_idx = 7
    host.ena_cp = False
    config = SimpleNamespace(num_hidden_layers=32, cla_explicit_mapping=[[7, 3]])
    with pytest.raises(ValueError, match="cla_swa_gate_window is required"):
        host._init_cla_reuse(config, "p", None, None)


@pytest.mark.unit
def test_the_sparse_cla_class_provides_what_its_own_methods_call():
    """Its CP path leans on npu_dsa helpers; stubbing them in a test would hide a rename."""
    module = ast.parse(inspect.getsource(cla_mod))
    sparse = next(n for n in module.body
                  if isinstance(n, ast.ClassDef) and n.name == "NPUDeepseekSparseCLAAttention")
    called = {n.func.attr for n in ast.walk(sparse)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"}
    assert {"_cp_q_lora", "_cp_o_proj", "_cp_attention_metadata", "_cp_rotary_rows"} <= called
    missing = sorted(n for n in called if not hasattr(NPUDeepseekSparseCLAAttention, n))
    assert not missing, f"NPUDeepseekSparseCLAAttention is missing {missing}"


def _sp_step(ena_sp, ena_cp):
    """A 5-token prefill step seen from one rank that holds 3 of them under SP."""
    layer = SimpleNamespace(
        cla_swa_attn_name="swa", prefix="p", layer_idx=7, ena_sp=ena_sp, ena_cp=ena_cp,
        forward_cla=MagicMock(return_value="sp out"),
        _forward_cla_prefill_cp=MagicMock(return_value="cp out"),
    )
    slots = torch.arange(5, dtype=torch.int32)
    meta_swa = SimpleNamespace(prefill="swa_p", decode="swa_d", slot_mapping=slots,
                               get_slot_mapping_2d=lambda: torch.stack([slots // 2, slots % 2], -1))
    metadata = {"swa": meta_swa, "p.attn": SimpleNamespace(prefill="p_p", decode="p_d")}
    ctx = SimpleNamespace(no_compile_layers={"p": layer}, attn_metadata=metadata)
    desc = (slice(0, 5), slice(0, 0), True, False)
    return layer, ctx, desc


@pytest.mark.unit
@pytest.mark.parametrize("ena_cp, with_topk", ((True, True), (False, True), (False, False)))
def test_npu_cla_forward_hands_sp_and_cp_the_whole_step(ena_cp, with_topk):
    """hs is this rank's slice, so it must not be cut by the step's p_slice again."""
    layer, ctx, desc = _sp_step(ena_sp=True, ena_cp=ena_cp)
    hs, cos, sin, topk = torch.zeros(3, 4), torch.zeros(5, 2), torch.zeros(5, 2), torch.zeros(5, 1, 8)
    if not with_topk:
        topk = None
    with patch.object(cla_mod, "get_forward_context", return_value=ctx), \
         patch.object(cla_mod, "get_batch_desc", return_value=desc):
        out = npu_cla_forward(hs, cos, sin, "p", topk)

    taken, skipped = (
        (layer._forward_cla_prefill_cp, layer.forward_cla) if ena_cp
        else (layer.forward_cla, layer._forward_cla_prefill_cp)
    )
    assert out == ("cp out" if ena_cp else "sp out") and not skipped.called
    args = taken.call_args.args
    assert args[0] is hs and args[1] is cos and args[2] is sin
    assert args[3:5] == ("p_p", "swa_p")
    if not ena_cp:
        assert args[5] == "prefill"
    local_slots, buffer = args[-2], args[-1]
    assert local_slots.slot_mapping.tolist() == [0, 1, 2, 3, 4]  # every token's slot
    assert local_slots.slot_mapping_2d.size(0) == 5
    assert buffer is topk


@pytest.mark.unit
@pytest.mark.parametrize("ena_cp, with_topk", ((False, False), (False, True), (True, True)))
def test_npu_cla_forward_runs_a_step_with_decode_without_sp(ena_cp, with_topk):
    """A prefill node still schedules decode-shaped requests (a prompt cached up
    to its last token). As npu_dsa_forward does, that step runs without SP."""
    # A 4-token step on 2 ranks: decode tokens 0-1, prefill tokens 2-3. This is
    # rank 1, holding tokens 2-3; rank 0's rows are all 1, its own all 2.
    class TwoRanks:
        world_size, rank_in_group = 2, 1

        def all_gather(self, x, dim=0):
            return torch.cat([torch.ones_like(x), x], dim=dim)

    calls = []

    def fake_forward_cla(x, cos, sin, attn_meta, swa_meta, caller, local_slots, topk):
        calls.append((caller, layer.ena_sp, x[:, 0].tolist(), topk[:, 0, 0].tolist() if with_topk else None,
                      local_slots.slot_mapping.tolist()))
        return x * 10

    layer = SimpleNamespace(cla_swa_attn_name="swa", prefix="p", layer_idx=7, ena_sp=True, ena_cp=ena_cp,
                            o_proj=SimpleNamespace(), forward_cla=fake_forward_cla,
                            _forward_cla_prefill_cp=MagicMock())
    slots = torch.arange(4, dtype=torch.int32)
    meta_swa = SimpleNamespace(prefill="swa_p", decode="swa_d", slot_mapping=slots,
                               get_slot_mapping_2d=lambda: torch.stack([slots // 2, slots % 2], -1))
    metadata = {"swa": meta_swa, "p.attn": SimpleNamespace(prefill="p_p", decode="p_d")}
    ctx = SimpleNamespace(no_compile_layers={"p": layer}, attn_metadata=metadata)
    desc = (slice(2, 4), slice(0, 2), True, True)
    with patch.object(cla_mod, "get_forward_context", return_value=ctx), \
         patch.object(cla_mod, "get_batch_desc", return_value=desc), \
         patch.object(cla_mod, "get_tp_group", return_value=TwoRanks()), \
         patch("omni_npu.attention.backends.utils.get_tp_group", return_value=TwoRanks()):
        out = npu_cla_forward(torch.full((2, 4), 2.0), torch.zeros(4, 2), torch.zeros(4, 2),
                              "p", torch.full((2, 1, 8), 2, dtype=torch.int32) if with_topk else None)

    assert not layer._forward_cla_prefill_cp.called
    # Both slices see the gathered rows and top-k, with SP off while they run.
    assert calls == [
        ("prefill", False, [2.0, 2.0], [2, 2] if with_topk else None, [2, 3]),
        ("decode", False, [1.0, 1.0], [1, 1] if with_topk else None, [0, 1]),
    ]
    assert layer.ena_sp is True  # restored afterwards
    torch.testing.assert_close(out, torch.full((2, 4), 20.0))  # this rank's rows only


@pytest.mark.unit
def test_a_dense_cla_layer_runs_sp_under_a_cp_config():
    assert NPUDeepseekMLACLAAttention.ena_cp is False


class _RecordingSP:
    """Stands in for SPManager over a 5-token step; this rank holds 3 rows."""

    def __init__(self, calls, tok=5, sp_len=3):
        self.calls, self.tok, self.sp_len = calls, tok, sp_len

    def ag_tokens(self, x):
        assert x.size(0) == self.sp_len
        self.calls.append(("ag_tokens", x.size(0)))
        return x.new_zeros(self.tok, *x.shape[1:])

    def align_tokens(self, x):
        self.calls.append(("align_tokens", x.size(0)))
        return x.new_zeros(6, *x.shape[1:])

    def slice_tokens(self, x):
        self.calls.append(("slice_tokens", x.size(0)))
        return x[: self.sp_len]


def _forward_cla_host(calls, ena_sp, o_proj_tp):
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.ena_sp = ena_sp
    host.o_proj = SimpleNamespace(tp_size=o_proj_tp)
    host.q_a_proj = lambda x: (calls.append(("q_a_proj", x.size(0))) or x,)
    host.q_a_layernorm = lambda q: calls.append(("q_a_layernorm", q.size(0))) or q
    host._q_project = lambda q, _c, _s: (
        calls.append(("q_project", q.size(0))) or (torch.zeros(q.size(0), 2, 3), torch.zeros(q.size(0), 2, 2))
    )
    host.kv_a_proj_with_mqa_swa = lambda x: (calls.append(("kv_a_proj_swa", x.size(0))) or x,)
    host.kv_a_layernorm_swa = "local norm"
    host.attn_cla_swa = SimpleNamespace(kv_cache="local cache")
    host._local_kv_layout_kwargs = lambda: {}
    host._kv_norm_rope_cache = lambda kv, *_a, **_k: calls.append(("cache", kv.size(0)))
    host._cla_global_attention = lambda q, *_a: torch.ones(q.size(0), 6)
    host._apply_local_swa = lambda q, _pe, _m, _c, n: calls.append(("local", n)) or torch.ones(q.size(0), 6)
    host._combine_cla_branches = lambda gx, g, l: calls.append(("gates", gx.size(0))) or g + l
    host._maybe_mome_out = lambda out, _a: calls.append(("mome_out", out.size(0))) or out
    host._apply_o_proj = lambda out: calls.append(("o_proj", out.size(0))) or out[:, :4]
    return host


@pytest.mark.unit
@pytest.mark.parametrize("o_proj_tp", (2, 1))
def test_forward_cla_gathers_each_projection_and_slices_after_o_proj_under_sp(o_proj_tp):
    calls = []
    host = _forward_cla_host(calls, ena_sp=True, o_proj_tp=o_proj_tp)
    sp = _RecordingSP(calls)
    with patch.object(cla_mod.SPManager, "init_sp", return_value=sp) as init_sp:
        out = host.forward_cla(torch.zeros(3, 4), torch.zeros(5, 2), torch.zeros(5, 2),
                               SimpleNamespace(), "swa meta", "prefill", "slots", "topk")

    init_sp.assert_called_once_with(tok=5)  # sized by cos/sin, which span the step
    padded = [("align_tokens", 5), ("o_proj", 6), ("slice_tokens", 6)]
    assert calls == [
        ("q_a_proj", 3), ("ag_tokens", 3), ("q_a_layernorm", 5), ("q_project", 5),
        ("kv_a_proj_swa", 3), ("ag_tokens", 3), ("cache", 5),  # gathered before the cache write
        # The DSA prefill metadata carries no num_tokens; the local branch must
        # still cover all 5 tokens, not this rank's 3.
        ("local", 5),
        ("ag_tokens", 3), ("gates", 5),  # the gates see the same 5 tokens
        ("mome_out", 5),
        *(padded if o_proj_tp > 1 else [("o_proj", 5), ("slice_tokens", 5)]),
    ]
    assert out.shape == (3, 4)  # back to this rank's slice


@pytest.mark.unit
def test_forward_cla_leaves_a_non_sp_step_unsliced():
    calls = []
    host = _forward_cla_host(calls, ena_sp=False, o_proj_tp=2)
    with patch.object(cla_mod.SPManager, "init_sp") as init_sp:
        out = host.forward_cla(torch.zeros(5, 4), torch.zeros(5, 2), torch.zeros(5, 2),
                               SimpleNamespace(), "swa meta", "prefill", "slots", "topk")
    assert not init_sp.called
    assert not any(c[0] in ("ag_tokens", "align_tokens", "slice_tokens") for c in calls)
    assert out.shape == (5, 4)


@pytest.mark.unit
def test_local_kv_under_sp_lands_each_token_in_its_own_slot():
    """Risk 1: rank 1 holds tokens 3-4 plus a pad row; the cache write must see
    every token, in step order, with the pad dropped, matching slot i to token i."""
    tokens = torch.arange(1.0, 6.0).view(5, 1).repeat(1, 4)  # token i carries value i+1
    rank0_rows = tokens[:3] * 10  # what rank 0 projected

    class TwoRanks:
        world_size, rank_in_group, device_group = 2, 1, None

        def all_gather(self, x, dim=0):
            return torch.cat([rank0_rows, x], dim=dim)

    sp = cla_mod.SPManager(TwoRanks())
    sp._scheme_sp_ctrl(5)
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.kv_a_proj_with_mqa_swa = lambda x: (x * 10,)
    host.kv_a_layernorm_swa = "local norm"
    host.attn_cla_swa = SimpleNamespace(kv_cache="local cache")
    host._local_kv_layout_kwargs = lambda: {}
    seen = {}
    host._kv_norm_rope_cache = lambda kv, cos, sin, slots, cache, **kw: seen.update(kv=kv, slots=slots)

    rank1_x = torch.cat([tokens[3:], torch.zeros(1, 4)])  # its slice, padded to sp_len
    slots = SimpleNamespace(slot_mapping=torch.tensor([40, 41, 42, 43, 44]))
    host._write_local_kv(rank1_x, "cos", "sin", slots, sp)

    assert seen["kv"].size(0) == seen["slots"].slot_mapping.size(0) == 5
    torch.testing.assert_close(seen["kv"], tokens * 10)  # row i is token i


def _cp_host(calls):
    host = NPUDeepseekSparseCLAAttention.__new__(NPUDeepseekSparseCLAAttention)
    host._cp_q_lora = lambda x, sp, mome: calls.append(("cp_q_lora", x, sp, mome)) or torch.zeros(4, 3)
    host._q_project = lambda q, cos, sin: (
        calls.append(("q_project", cos, sin)) or (torch.zeros(4, 2, 3), torch.zeros(4, 2, 2))
    )
    host._write_local_kv = lambda *a: calls.append(("write_local_kv", *a))
    host._cla_sparse_attention = lambda *a: calls.append(("global", *a[2:])) or torch.ones(4, 6)
    host._apply_local_swa = lambda q, pe, meta, caller, n: (
        calls.append(("local", meta, caller, n)) or torch.full((4, 6), 2.0)
    )
    host.swa_gate_global = SimpleNamespace(
        forward_cp=lambda x, out, sp: calls.append(("gate_global", x, sp)) or out)
    host.swa_gate_local = SimpleNamespace(
        forward_cp=lambda x, out, sp: calls.append(("gate_local", x, sp)) or out)
    host._cp_o_proj = lambda out, sp: calls.append(("cp_o_proj", out, sp)) or "sp rows"
    return host


@pytest.mark.unit
def test_the_cp_forward_runs_both_branches_on_the_zigzag_layout():
    calls = []
    host = _cp_host(calls)
    q_cumlens, kv_lens = torch.tensor([2, 4]), torch.tensor([3, 8])
    sp = SimpleNamespace(
        cp_attn_meta=lambda: (q_cumlens, kv_lens, "packed table", "cp table"),
        cp_slice=lambda t, cached: f"cp {cached}",
    )
    sp_x, cos, sin = torch.zeros(3, 4), torch.zeros(5, 2), torch.zeros(5, 2)
    local_meta = SimpleNamespace(block_table=torch.tensor([[1, 2], [3, 4]]))
    out = host._forward_cla_prefill_cp(
        sp_x, cos, sin, SimpleNamespace(sp_manager=sp), local_meta, "slots", "cp topk"
    )

    assert out == "sp rows"
    by_name = {c[0]: c[1:] for c in calls}
    assert [c[0] for c in calls] == [
        "cp_q_lora", "q_project", "write_local_kv", "global", "local",
        "gate_global", "gate_local", "cp_o_proj",
    ]
    assert by_name["cp_q_lora"] == (sp_x, sp, None)  # no MoME on a CLA layer
    assert by_name["q_project"] == ("cp cos", "cp sin")  # the query is CP layout
    # The local KV covers every token: full cos/sin, the step's slots, gathered by sp.
    assert by_name["write_local_kv"] == (sp_x, cos, sin, "slots", sp)
    # The global branch reads the source cache with the CP lengths and the CP
    # table, and takes the CP-layout top-k as it is.
    g_q, g_kv, g_table, g_topk, g_meta = by_name["global"]
    assert g_q is q_cumlens and g_kv is kv_lens and g_table == "cp table" and g_topk == "cp topk"
    meta, caller, n = by_name["local"]
    assert meta.query_cumlens is q_cumlens and meta.seq_lens is kv_lens and n == 4
    # Both of a request's fragments read its pages; the spare column covers the zigzag pad.
    assert meta.block_table.tolist() == [[1, 2, 2], [1, 2, 2], [3, 4, 4], [3, 4, 4]]
    assert caller == "prefill_absorb_cp"
    assert by_name["gate_global"] == (sp_x, sp) and by_name["gate_local"] == (sp_x, sp)
    torch.testing.assert_close(by_name["cp_o_proj"][0], torch.full((4, 6), 3.0))


@pytest.mark.unit
def test_the_local_branch_attends_with_the_cp_table_and_key():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.scaling, host.kv_lora_rank = 0.5, 6
    host.cla_swa_gate_window = 512
    host.is_cla_fa_metadata_producer = False
    host.use_gpt_oss_sink = host.use_gpt_oss_sink_rescale = False
    host.attn_cla_swa = SimpleNamespace(
        kv_cache=(torch.zeros(4, 8, 6), torch.zeros(4, 8, 2)), W_UK_T=torch.zeros(1), W_UV=torch.zeros(1),
    )
    host._q_nope_absorb = lambda q, _w: q
    host._post_attn_absorb = lambda out, _w: out
    cp_table = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int32)
    meta = SimpleNamespace(block_table=cp_table, query_cumlens=torch.tensor([2, 4]), seq_lens=torch.tensor([2, 8]))
    sink_op = MagicMock(return_value=(torch.ones(2, 4, 6), None))
    tiling = MagicMock()
    with patch.object(cla_mod, "cache_fit_shape", lambda t, _l: t), \
         patch.object(cla_mod.NPUMLAImpl, "ensure_decode_attn_mask", MagicMock()), \
         patch.object(cla_mod.NPUMLAImpl, "SHARE_MASK_TRIL_SPARSE", None, create=True), \
         patch.object(cla_mod, "npu_fused_infer_attention_sink_metadata", tiling), \
         patch.object(torch.npu, "get_stream_limit",
                      return_value={"cube_core_num": 24, "vector_core_num": 48}), \
         patch.object(torch.ops.custom, "npu_fused_infer_attention_sink", sink_op, create=True):
        host._apply_local_swa(torch.zeros(4, 2, 6), torch.zeros(4, 2, 2), meta, "prefill_absorb_cp", 4)

    assert sink_op.call_args.kwargs["block_table"] is cp_table
    assert tiling.call_args.args[2] == ("cla_prefill_absorb_cp", 511)
    other_keys = {host._cla_metadata_key(c) for c in ("decode", "prefill_absorb", "prefill")}
    assert tiling.call_args.args[2] not in other_keys


@pytest.mark.unit
def test_cla_output_gate_moves_its_logits_to_the_cp_layout():
    gate = CLAOutputGate.__new__(CLAOutputGate)
    torch.nn.Module.__init__(gate)
    gate.num_local_heads = 2
    logits = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    gate.w_gate = lambda _x: logits
    attn_out = torch.arange(12.0).view(3, 4)
    sp = SimpleNamespace(sp_to_cp=MagicMock(side_effect=lambda t: t.flip(0)))

    out = gate.forward_cp(torch.zeros(3, 6), attn_out, sp)

    assert sp.sp_to_cp.call_args.args[0] is logits  # the raw logits are what moves
    torch.testing.assert_close(out, gate._apply_gate(torch.sigmoid(logits.flip(0)), attn_out))
    # The split left the non-CP gate unchanged.
    torch.testing.assert_close(
        gate.forward(torch.zeros(3, 6), attn_out),
        attn_out * torch.sigmoid(logits).repeat_interleave(2, dim=1),
    )


@pytest.mark.unit
def test_forward_cla_rejects_decode_with_sp_still_enabled():
    host = CLAReuseLayerMixin.__new__(CLAReuseLayerMixin)
    host.ena_sp = True
    with pytest.raises(RuntimeError, match="decode requires sequence parallelism to be disabled"):
        host.forward_cla(None, None, None, None, None, "decode", None, None)
