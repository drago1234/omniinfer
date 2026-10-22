# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""DFlash-specific contracts of the NPU GQA attention backend."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from omni_npu.attention.backends import attention as backend


def _impl():
    impl = backend.NPUAttentionBackendImpl.__new__(backend.NPUAttentionBackendImpl)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl.head_size_v = 4
    impl.scale = 0.5
    impl.sliding_window = None
    impl.sink_len = None
    impl.use_ai_infra_fa_for_gpt_oss_sink = False
    impl.is_cross_attention = False
    impl.kv_sharing_target_layer_name = None
    impl.attn_sinks = None
    return impl


def _config(kv_nz=False):
    return SimpleNamespace(operator_opt_config=SimpleNamespace(
        kv_nz=kv_nz, use_aicpu_fa_tiling=False,
        enable_kv_rmsnorm_rope_cache=False,
    ))


@pytest.mark.parametrize("causal", [False, True])
def test_metadata_builder_preserves_causal_flag(monkeypatch, causal):
    builder = backend.NPUAttentionMetadataBuilder.__new__(
        backend.NPUAttentionMetadataBuilder
    )
    builder.reorder_batch_threshold = 1
    builder.use_ai_infra_fa_for_gpt_oss_sink = False
    monkeypatch.setattr(backend, "model_extra_config", _config())
    monkeypatch.setattr(backend, "split_decodes_and_prefills", lambda *args, **kwargs: (0, 1, 0, 2))
    common = SimpleNamespace(
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1]),
        max_query_len=2,
        causal=causal,
    )

    metadata = builder.build(0, common)

    assert metadata.causal is causal


@pytest.mark.parametrize("kv_nz", [False, True])
def test_context_kv_update_uses_slots_and_cache_layout(monkeypatch, kv_nz):
    impl = _impl()
    monkeypatch.setattr(backend, "model_extra_config", _config(kv_nz))
    dense_scatter = Mock()
    nz_scatter = Mock()
    monkeypatch.setattr(backend.torch_npu, "npu_scatter_nd_update_", dense_scatter)
    monkeypatch.setattr(backend.torch_npu, "npu_scatter_pa_kv_cache", nz_scatter)
    key = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    value = key + 20
    slots = torch.tensor([4, 1])
    cache = (torch.zeros((2, 4, 4)), torch.zeros((2, 4, 4)))

    impl.do_kv_cache_update(object(), key, value, cache, slots)

    if kv_nz:
        nz_scatter.assert_called_once()
        got_key, got_value, got_key_cache, got_value_cache, got_slots = nz_scatter.call_args.args
        torch.testing.assert_close(got_key, key[:2])
        torch.testing.assert_close(got_value, value[:2])
        assert got_key_cache is cache[0] and got_value_cache is cache[1]
        torch.testing.assert_close(got_slots, slots)
        dense_scatter.assert_not_called()
    else:
        assert dense_scatter.call_count == 2
        torch.testing.assert_close(dense_scatter.call_args_list[0].args[2], key[:2].reshape(2, 4))
        torch.testing.assert_close(dense_scatter.call_args_list[1].args[2], value[:2].reshape(2, 4))
        for call in dense_scatter.call_args_list:
            torch.testing.assert_close(call.args[1], slots.reshape(-1, 1))
        nz_scatter.assert_not_called()


@pytest.mark.parametrize("shared,slots", [
    (True, torch.tensor([0])),
    (False, None),
    (False, torch.tensor([], dtype=torch.long)),
])
def test_context_kv_update_skips_non_writable_slots(monkeypatch, shared, slots):
    impl = _impl()
    impl.kv_sharing_target_layer_name = "shared" if shared else None
    monkeypatch.setattr(backend, "model_extra_config", _config())
    scatter = Mock()
    monkeypatch.setattr(backend.torch_npu, "npu_scatter_nd_update_", scatter)

    impl.do_kv_cache_update(
        object(), torch.ones((1, 1, 4)), torch.ones((1, 1, 4)),
        (torch.zeros((1, 4, 4)), torch.zeros((1, 4, 4))), slots,
    )

    scatter.assert_not_called()


@pytest.mark.parametrize("causal,expected_mode", [(False, 0), (True, 3)])
def test_draft_attention_selects_mask_from_causal_metadata(
    monkeypatch, causal, expected_mode,
):
    impl = _impl()
    causal_mask = torch.ones((2, 2), dtype=torch.bool)
    monkeypatch.setattr(backend.NPUAttentionBackendImpl, "SHARE_MASK_TRIL_SPARSE", causal_mask)
    monkeypatch.setattr(backend, "model_extra_config", _config())
    monkeypatch.setattr(backend, "get_forward_context", lambda: SimpleNamespace(capturing=False))
    monkeypatch.setattr(backend.torch_npu, "npu_scatter_nd_update_", Mock())
    monkeypatch.setattr(
        backend.torch_npu, "_npu_fused_infer_attention_score_v2_infer_output",
        Mock(return_value=(torch.zeros((2, 2, 4)), torch.zeros(2))),
    )
    fused = Mock(return_value=(torch.full((2, 2, 4), 7.),))
    monkeypatch.setattr(backend.torch_npu, "npu_fused_infer_attention_score_v2", fused)
    metadata = backend.NPUMetadata(
        num_actual_tokens=2, block_tables=torch.zeros((1, 1), dtype=torch.int32),
        query_start_loc=[0, 2], seq_lens=[2], max_query_len=2,
        slot_mapping=torch.tensor([0, 1]), causal=causal,
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    cache = (torch.zeros((1, 4, 4)), torch.zeros((1, 4, 4)))
    output = torch.zeros((2, 2, 4))

    result = impl.forward(
        layer, torch.ones((2, 2, 4)), torch.ones((2, 1, 4)),
        torch.ones((2, 1, 4)), cache, metadata, output=output,
    )

    assert result is output
    torch.testing.assert_close(output, torch.full_like(output, 7.))
    assert fused.call_args.kwargs["sparse_mode"] == expected_mode
    assert fused.call_args.kwargs["atten_mask"] is (causal_mask if causal else None)


@pytest.mark.parametrize("causal,sliding_window", [
    (torch.tensor([True, False]), None),
    (False, 16),
])
def test_draft_attention_rejects_unsupported_causal_metadata(
    causal, sliding_window,
):
    impl = _impl()
    impl.sliding_window = sliding_window
    metadata = SimpleNamespace(causal=causal)

    with pytest.raises(NotImplementedError):
        impl.forward(object(), torch.zeros((1, 2, 4)), None, None, (), metadata,
                     output=torch.empty((1, 2, 4)))


def test_draft_attention_rejects_noncausal_sink_path():
    impl = _impl()
    impl.sink_len = 0

    with pytest.raises(NotImplementedError, match="non-causal sink attention"):
        impl.forward(
            object(), torch.zeros((1, 2, 4)), None, None, (),
            SimpleNamespace(causal=False), output=torch.empty((1, 2, 4)),
        )
