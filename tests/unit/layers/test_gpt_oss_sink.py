# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""GPT-OSS learnable sinks for the high-throughput attention stack."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import omni_npu.v1.layers.attention.gpt_oss_sink as sink_mod
from omni_npu.v1.layers.attention.gpt_oss_sink import GptOssSinkMixin, npu_gpt_oss_fia_fake


def _host(use_sink, rescale_flag, num_heads=8, num_local_heads=4):
    host = GptOssSinkMixin()
    host.num_heads, host.num_local_heads = num_heads, num_local_heads
    opt = SimpleNamespace(use_gpt_oss_sink_rescale=rescale_flag)
    with patch.object(sink_mod, "model_extra_config", SimpleNamespace(operator_opt_config=opt)), \
         patch.object(GptOssSinkMixin, "_create_gpt_oss_sink", lambda self: torch.zeros(num_local_heads)):
        host._init_gpt_oss_sink(SimpleNamespace(use_gpt_oss_sink=use_sink))
    return host


@pytest.mark.unit
@pytest.mark.parametrize(
    "use_sink, flag, expect_sink, expect_rescale",
    ((False, True, False, False), (True, True, True, True), (True, False, True, False)),
)
def test_init_gpt_oss_sink_resolves_both_switches(use_sink, flag, expect_sink, expect_rescale):
    host = _host(use_sink, flag)
    assert host.use_gpt_oss_sink is expect_sink
    assert host.use_gpt_oss_sink_rescale is expect_rescale
    assert (host.gpt_oss_sink is not None) is expect_sink
    assert host.gpt_oss_sink_swa is None  # only a CLA layer adds the second one


@pytest.mark.unit
def test_init_gpt_oss_swa_sink_only_when_the_model_has_sinks():
    for use_sink, expected in ((True, True), (False, False)):
        host = _host(use_sink, True)
        with patch.object(GptOssSinkMixin, "_create_gpt_oss_sink", lambda self: torch.ones(4)):
            host._init_gpt_oss_swa_sink()
        assert (host.gpt_oss_sink_swa is not None) is expected


@pytest.mark.unit
@pytest.mark.parametrize("num_local_heads, expected", ((4, slice(4, 8)), (8, slice(0, 8))))
def test_load_gpt_oss_sink_weight_takes_this_rank_shard(num_local_heads, expected):
    host = GptOssSinkMixin()
    host.num_heads, host.num_local_heads = 8, num_local_heads
    param = torch.nn.Parameter(torch.zeros(num_local_heads), requires_grad=False)
    loaded = torch.arange(8, dtype=torch.float32)
    with patch.object(sink_mod, "get_tp_group", return_value=SimpleNamespace(rank_in_group=1)):
        host._load_gpt_oss_sink_weight(param, loaded)
    assert torch.equal(param.data, loaded[expected])


@pytest.mark.unit
@pytest.mark.parametrize("bad", (torch.zeros(7), torch.zeros(8, 1)))
def test_load_gpt_oss_sink_weight_rejects_a_wrong_shape(bad):
    host = GptOssSinkMixin()
    host.num_heads, host.num_local_heads = 8, 4
    with pytest.raises(ValueError, match="gpt_oss_sink must have shape"):
        host._load_gpt_oss_sink_weight(torch.nn.Parameter(torch.zeros(4)), bad)


@pytest.mark.unit
def test_load_gpt_oss_sink_weight_rejects_a_wrong_local_parameter():
    host = GptOssSinkMixin()
    host.num_heads, host.num_local_heads = 8, 4
    with pytest.raises(ValueError, match="local gpt_oss_sink parameter must have shape"):
        host._load_gpt_oss_sink_weight(
            torch.nn.Parameter(torch.zeros(3)), torch.arange(8, dtype=torch.float32)
        )


@pytest.mark.unit
def test_rescale_gpt_oss_sink_output_scales_by_sigmoid():
    host = GptOssSinkMixin()
    host.num_local_heads = 2
    lse = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])  # [T, N], per head
    sink = torch.tensor([0.5, -0.5])
    scale = torch.sigmoid(lse - sink)  # [T, N]
    tnd = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    scaled = host._rescale_gpt_oss_sink_output(tnd, lse, sink, "TND")
    torch.testing.assert_close(scaled, tnd * scale.unsqueeze(-1))

    ntd = tnd.transpose(0, 1).contiguous()  # [N, T, D]
    scaled_ntd = host._rescale_gpt_oss_sink_output(ntd, lse, sink, "NTD")
    torch.testing.assert_close(scaled_ntd, ntd * scale.transpose(0, 1).unsqueeze(-1))
    # The same numbers, whichever way round the caller holds them.
    torch.testing.assert_close(scaled_ntd, scaled.transpose(0, 1))


@pytest.mark.unit
def test_rescale_gpt_oss_sink_output_rejects_an_unknown_layout():
    host = GptOssSinkMixin()
    host.num_local_heads = 2
    with pytest.raises(ValueError, match="Unsupported GPT-OSS sink output layout"):
        host._rescale_gpt_oss_sink_output(
            torch.ones(3, 2, 4), torch.zeros(3, 2), torch.zeros(2), "BSH"
        )


@pytest.mark.unit
@pytest.mark.parametrize("layout, expected", (("TND", (3, 2, 5)), ("TND_NTD", (2, 3, 5))))
def test_npu_gpt_oss_fia_fake_reports_the_output_shape(layout, expected):
    fake = npu_gpt_oss_fia_fake(
        torch.zeros(3, 2, 6), torch.zeros(1), torch.zeros(1, 5), torch.zeros(1),
        torch.zeros(1), torch.zeros(1), torch.zeros(2), 2, 1, input_layout=layout,
    )
    assert tuple(fake.shape) == expected
    assert "bool batch_invariant=False" in str(torch.ops.vllm.npu_gpt_oss_fia.default._schema)


@pytest.mark.unit
@pytest.mark.parametrize("paged", (True, False))
@pytest.mark.parametrize("batch_invariant", (False, True))
def test_apply_gpt_oss_fia_native_translates_fia_sink_kwargs(paged, batch_invariant):
    """The native op names the sequence lengths and the head count differently."""
    kwargs = {
        "query": torch.zeros(3, 2, 6),
        "key": torch.zeros(4, 8, 6),
        "value": torch.zeros(4, 8, 5),
        "query_rope": torch.zeros(3, 2, 2),
        "key_rope": torch.zeros(4, 8, 2),
        "num_query_heads": 2,
        "num_key_value_heads": 1,
        "input_layout": "TND_NTD" if paged else "TND",
        "atten_mask": None,
        "sparse_mode": 4,
        "pre_tokens": 511,
        "next_tokens": 0,
        "softmax_scale": 0.5,
        "actual_seq_qlen": torch.tensor([3], dtype=torch.int32),
        "actual_seq_kvlen": torch.tensor([8], dtype=torch.int32),
    }
    if paged:
        kwargs["block_table"] = torch.zeros(2, 4, dtype=torch.int32)
        kwargs["block_size"] = 8

    meta = MagicMock(return_value=torch.zeros(1))
    op = MagicMock(return_value=torch.zeros(2, 3, 5))
    with patch.object(sink_mod.model_extra_config.operator_opt_config, "use_batch_invariant_op", batch_invariant), \
         patch.object(sink_mod, "npu_gpt_oss_fia_metadata", meta), \
         patch.object(torch.npu, "current_stream", return_value=None), \
         patch.object(torch.npu, "get_stream_limit",
                      return_value={"cube_core_num": 24, "vector_core_num": 48}), \
         patch.object(torch.ops.vllm, "npu_gpt_oss_fia", op, create=True):
        GptOssSinkMixin._apply_gpt_oss_fia_native(kwargs, torch.zeros(2), ("cla_decode", 511), True)

    meta_args = meta.call_args.args[0]
    assert meta_args["num_heads_q"] == 2 and meta_args["pre_tokens"] == 511
    assert meta_args["input_layout_kv"] == ("BnBsH" if paged else "TND")
    assert meta_args["block_size"] == (8 if paged else 0)
    assert meta_args["batch_size"] == (2 if paged else 1)
    assert meta.call_args.args[1] is True and meta.call_args.args[2] == ("cla_decode", 511)
    # The FIA-sink spellings must not reach the native op.
    passed = op.call_args.kwargs
    assert "actual_seq_qlen" not in passed and "num_query_heads" not in passed
    assert passed["softmax_scale"] == 0.5
    assert passed["batch_invariant"] is meta_args["batch_invariant"] is batch_invariant


@pytest.mark.unit
def test_reserve_gpt_oss_fia_metadata_creates_then_extends():
    # The metadata op is an argument, so it is looked up even with the buffer
    # mocked out, and the operator package does not ship it everywhere.
    with patch.object(torch.ops.custom, "npu_ai_infra_flash_attention_metadata",
                      MagicMock(), create=True), \
         patch.object(sink_mod, "npu_gpt_oss_fia_metadata", None), \
         patch("omni_npu.attention.backends.utils.CrossLayerSharedOp") as ctor:
        GptOssSinkMixin._reserve_gpt_oss_fia_metadata((("decode", 511),))
        ctor.assert_called_once()
        assert ctor.call_args.kwargs["callers"] == (("decode", 511),)

    existing = MagicMock()
    with patch.object(sink_mod, "npu_gpt_oss_fia_metadata", existing):
        GptOssSinkMixin._reserve_gpt_oss_fia_metadata((("cla_decode", 511),))
    existing.reserve.assert_called_once_with((("cla_decode", 511),))


@pytest.mark.unit
def test_create_gpt_oss_sink_wires_the_loader_onto_a_frozen_parameter():
    host = GptOssSinkMixin()
    host.num_local_heads = 4
    with patch.object(sink_mod.torch, "empty",
                      lambda n, device=None, dtype=None: torch.zeros(n, dtype=dtype)):
        sink = host._create_gpt_oss_sink()
    assert sink.shape == (4,) and sink.dtype == torch.float32
    assert sink.requires_grad is False
    assert sink.weight_loader.__func__ is GptOssSinkMixin._load_gpt_oss_sink_weight


@pytest.mark.unit
@pytest.mark.parametrize("batch_invariant", (False, True))
def test_npu_gpt_oss_fia_forwards_to_the_sink_capable_kernel(batch_invariant):
    op = MagicMock(return_value=(torch.zeros(3, 2, 5), torch.zeros(3, 2)))
    with patch.object(torch.ops.custom, "npu_ai_infra_flash_attention", op, create=True):
        out = sink_mod.npu_gpt_oss_fia(
            torch.zeros(3, 2, 6), torch.zeros(4, 8, 6), torch.zeros(4, 8, 5),
            torch.zeros(1024, dtype=torch.int32), torch.tensor([3]), torch.tensor([8]),
            torch.zeros(2), 2, 1, block_size=8, softmax_scale=0.5, batch_invariant=batch_invariant,
        )
    assert out.shape == (3, 2, 5)  # the op returns a tuple; the wrapper unwraps it
    assert op.call_args.kwargs["sinks"].shape == (2,)
    assert op.call_args.kwargs["num_heads"] == 2 and op.call_args.kwargs["block_size"] == 8
    assert op.call_args.kwargs["batch_invariant"] is batch_invariant


@pytest.mark.parametrize("nz", [False, True])
def test_gqa_sink_metadata_and_fake_shape_preserve_per_head_value_dim(nz):
    cache = torch.zeros((2, 2, 1, 128, 16) if nz else (2, 128, 32))
    q, lengths, table = torch.zeros(3, 4, 16), torch.tensor([3]), torch.tensor([[0]])
    args = dict(query=q, key=cache, value=cache, num_query_heads=4,
                num_key_value_heads=2, input_layout="TND", block_table=table,
                block_size=128, sparse_mode=4, pre_tokens=511, next_tokens=0,
                actual_seq_qlen=lengths, actual_seq_kvlen=lengths)
    metadata, kernel = MagicMock(return_value=torch.zeros(1024)), MagicMock(return_value=q)
    with patch.object(sink_mod, "npu_gpt_oss_fia_metadata", metadata), \
         patch.object(torch.npu, "current_stream", return_value=None), \
         patch.object(torch.npu, "get_stream_limit", return_value={"cube_core_num": 24, "vector_core_num": 48}), \
         patch.object(torch.ops.vllm, "npu_gpt_oss_fia", kernel):
        out = GptOssSinkMixin._apply_gpt_oss_fia_native(args, torch.zeros(4), ("gqa", 512), False)
    assert out is q
    assert metadata.call_args.args[0]["head_dim_v"] == 16
    assert metadata.call_args.args[0]["rope_head_dim"] == 0
    assert metadata.call_args.args[1:] == (False, ("gqa", 512))
    fake = npu_gpt_oss_fia_fake(q, cache, cache, torch.zeros(1), lengths, lengths,
                               torch.zeros(4), 4, 2, block_table=table)
    assert fake.shape == q.shape
