# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
from types import SimpleNamespace

from omni_npu.model_config.config_loader.features import (
    apply_eager_mode_config,
    apply_prefill_cudagraph,
    apply_seq_parallel,
)


pytestmark = pytest.mark.unit


def _make_config(graph_mode="eager_mode", moe_multi_stream_tune=False):
    return SimpleNamespace(
        task_config=SimpleNamespace(
            graph_mode=graph_mode,
        ),
        operator_opt_config=SimpleNamespace(
            moe_multi_stream_tune=moe_multi_stream_tune,
            enable_scmoe_multi_stream=True,
            enable_multi_stream=True,
            li_prolog_multi_stream=True,
            enable_prefill_cudagraph=True,
            enable_mhc_multistream=True,
            use_mhc_fusion_op=True,
            use_mega_mhc_fusion_op=True,
            split_q_up_in_multistream=True,
            enable_super_kernel=True,
            enable_sk_scope=True,
            enable_prefetch=True,
            expert_gate_up_prefetch=50,
            expert_down_prefetch=28,
            attn_prefetch=96,
            dense_mlp_prefetch=56,
            lm_head_prefetch=135,
            shared_expert_gate_up_prefetch=28,
            shared_expert_down_prefetch=14,
        ),
    )


def test_eager_mode_does_not_disable_moe_multi_stream_tune():
    """Eager mode should NOT touch moe_multi_stream_tune."""
    cfg = _make_config(graph_mode="eager_mode", moe_multi_stream_tune=True)
    apply_eager_mode_config(cfg)
    assert cfg.operator_opt_config.moe_multi_stream_tune is True


def test_eager_mode_disables_other_optimizations():
    """Eager mode should disable prefetch, super_kernel, etc."""
    cfg = _make_config(graph_mode="eager_mode")
    apply_eager_mode_config(cfg)
    assert cfg.operator_opt_config.enable_prefetch is False
    assert cfg.operator_opt_config.enable_super_kernel is False
    assert cfg.operator_opt_config.enable_sk_scope is False
    assert cfg.operator_opt_config.enable_scmoe_multi_stream is False
    assert cfg.operator_opt_config.expert_gate_up_prefetch == 0
    assert cfg.operator_opt_config.enable_multi_stream is False
    assert cfg.operator_opt_config.li_prolog_multi_stream is False
    assert cfg.operator_opt_config.enable_prefill_cudagraph is False
    assert cfg.operator_opt_config.enable_mhc_multistream is False
    assert cfg.operator_opt_config.use_mhc_fusion_op is False
    assert cfg.operator_opt_config.use_mega_mhc_fusion_op is True
    assert cfg.operator_opt_config.split_q_up_in_multistream is False


def test_non_eager_mode_does_not_touch_config():
    """Non-eager mode should not modify any config."""
    cfg = _make_config(graph_mode="acl_graph", moe_multi_stream_tune=True)
    apply_eager_mode_config(cfg)
    assert cfg.operator_opt_config.moe_multi_stream_tune is True
    assert cfg.operator_opt_config.enable_prefetch is True
    assert cfg.operator_opt_config.enable_multi_stream is True
    assert cfg.operator_opt_config.enable_mhc_multistream is True
    assert cfg.operator_opt_config.use_mhc_fusion_op is True
    assert cfg.operator_opt_config.use_mega_mhc_fusion_op is True
    assert cfg.operator_opt_config.split_q_up_in_multistream is True
    assert cfg.operator_opt_config.li_prolog_multi_stream is True
    assert cfg.operator_opt_config.enable_prefill_cudagraph is True


def _parall_cfg(ena_sp=True, ena_cp=False, ena_attn_sp=False):
    """Build a parall_config namespace for apply_seq_parallel tests."""
    return SimpleNamespace(
        parall_config=SimpleNamespace(
            ena_seq_parallel=ena_sp,
            ena_context_parallel=ena_cp,
            ena_swa_attn_seq_parallel=ena_attn_sp,
        )
    )


def test_apply_seq_parallel_disables_flags_without_model_plugins(monkeypatch):
    """Missing omni model plugins force-disable seq/context/SWA SP flags."""
    monkeypatch.setenv("VLLM_PLUGINS", "other_plugin")
    cfg = _parall_cfg(ena_sp=True, ena_cp=True, ena_attn_sp=True)
    apply_seq_parallel(cfg)
    assert cfg.parall_config.ena_seq_parallel is False
    assert cfg.parall_config.ena_context_parallel is False
    assert cfg.parall_config.ena_swa_attn_seq_parallel is False


def test_apply_seq_parallel_keeps_swa_sp_when_pangu_plugin_present(monkeypatch):
    """omni_pangu_models in VLLM_PLUGINS keeps SWA SP enabled."""
    monkeypatch.setenv("VLLM_PLUGINS", "omni_pangu_models")
    cfg = _parall_cfg(ena_sp=True, ena_attn_sp=True)
    apply_seq_parallel(cfg)
    assert cfg.parall_config.ena_swa_attn_seq_parallel is True


def test_apply_seq_parallel_requires_ena_seq_parallel_for_swa(monkeypatch):
    """ena_swa_attn_seq_parallel cannot be enabled without ena_seq_parallel."""
    monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")
    cfg = _parall_cfg(ena_sp=False, ena_attn_sp=True)
    with pytest.raises(AssertionError, match="ena_swa_attn_seq_parallel"):
        apply_seq_parallel(cfg)


def _prefill_cudagraph_cfg(enabled=True, aicpu_fa_tiling=True):
    return SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            enable_prefill_cudagraph=enabled,
            use_aicpu_fa_tiling=aicpu_fa_tiling,
        )
    )


def test_apply_prefill_cudagraph_needs_aicpu_fa_tiling():
    """Host-side attention lengths would be frozen into the graph."""
    cfg = _prefill_cudagraph_cfg(aicpu_fa_tiling=False)
    apply_prefill_cudagraph(cfg)
    assert cfg.operator_opt_config.enable_prefill_cudagraph is False

    # With the switch off nothing else is inspected, so an environment that
    # cannot capture is not an error.
    off = _prefill_cudagraph_cfg(enabled=False, aicpu_fa_tiling=False)
    apply_prefill_cudagraph(off)
    assert off.operator_opt_config.enable_prefill_cudagraph is False


@pytest.mark.parametrize(
    "task_queue, has_setter, expected",
    [
        # Level 2 without a setter: the capture cannot get to level 1.
        ("2", False, False),
        ("2", True, True),
        ("1", False, True),
        (None, False, True),
        # Parsed as an int, like torch_npu does, not compared as a string.
        ("02", False, False),
        (" 2 ", False, False),
        ("junk", False, True),
    ],
)
def test_apply_prefill_cudagraph_needs_capture_at_level_1(
    monkeypatch, task_queue, has_setter, expected
):
    """Prefill in-graph turns itself off when the capture is stuck at level 2."""
    import torch_npu

    if task_queue is None:
        monkeypatch.delenv("TASK_QUEUE_ENABLE", raising=False)
    else:
        monkeypatch.setenv("TASK_QUEUE_ENABLE", task_queue)
    if has_setter:
        monkeypatch.setattr(torch_npu.npu, "set_task_queue_enable", lambda mode: None,
                            raising=False)
    else:
        monkeypatch.delattr(torch_npu.npu, "set_task_queue_enable", raising=False)

    cfg = _prefill_cudagraph_cfg()
    apply_prefill_cudagraph(cfg)
    assert cfg.operator_opt_config.enable_prefill_cudagraph is expected


def test_apply_prefill_cudagraph_keeps_level_2_under_blocking_launch(monkeypatch):
    """Blocking launch turns the task queue off, so the capture is fine."""
    import torch_npu

    monkeypatch.setenv("TASK_QUEUE_ENABLE", "2")
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    monkeypatch.delattr(torch_npu.npu, "set_task_queue_enable", raising=False)

    cfg = _prefill_cudagraph_cfg()
    apply_prefill_cudagraph(cfg)
    assert cfg.operator_opt_config.enable_prefill_cudagraph is True
