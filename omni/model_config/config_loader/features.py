# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from vllm.logger import logger


def is_mlp_weight_prefetch_on():
    from .loader import model_extra_config
    return (model_extra_config.operator_opt_config.enable_prefetch and 
            model_extra_config.operator_opt_config.expert_gate_up_prefetch and
            model_extra_config.operator_opt_config.expert_down_prefetch)


def apply_eager_mode_config(model_extra_config):
    """Apply eager-mode modifications to the given ModelExtraConfig.

    """
    
    if model_extra_config.task_config.graph_mode == "eager_mode":
        model_extra_config.operator_opt_config.enable_scmoe_multi_stream = False
        model_extra_config.operator_opt_config.enable_multi_stream = False
        model_extra_config.operator_opt_config.enable_mhc_multistream = False
        model_extra_config.operator_opt_config.use_mhc_fusion_op = False
        model_extra_config.operator_opt_config.split_q_up_in_multistream = False
        model_extra_config.operator_opt_config.li_prolog_multi_stream = False
        model_extra_config.operator_opt_config.enable_prefill_cudagraph = False
        model_extra_config.operator_opt_config.enable_super_kernel = False
        model_extra_config.operator_opt_config.enable_sk_scope = False
        model_extra_config.operator_opt_config.enable_prefetch = False
        model_extra_config.operator_opt_config.expert_gate_up_prefetch = 0
        model_extra_config.operator_opt_config.expert_down_prefetch = 0
        model_extra_config.operator_opt_config.attn_prefetch = 0
        model_extra_config.operator_opt_config.dense_mlp_prefetch = 0
        model_extra_config.operator_opt_config.lm_head_prefetch = 0
        model_extra_config.operator_opt_config.shared_expert_gate_up_prefetch = 0
        model_extra_config.operator_opt_config.shared_expert_down_prefetch = 0
        logger.warning(
            f"[WARNING] Eager mode disables all these optimization configurations by default."
        )


def apply_seq_parallel(model_extra_config):
    # validate seq_parallel
    cfg = model_extra_config.parall_config
    ena_sp = cfg.ena_seq_parallel
    ena_cp = cfg.ena_context_parallel
    ena_attn_sp = cfg.ena_swa_attn_seq_parallel

    plugins = os.environ.get("VLLM_PLUGINS", "").split(',')
    if "omni_custom_models" not in plugins and "omni_pangu_models" not in plugins:
        cfg.ena_seq_parallel = False
        cfg.ena_context_parallel = False
        cfg.ena_swa_attn_seq_parallel = False
        logger.warning(
            f"[WARNING] omni_custom_models and omni_pangu_models both not in VLLM_PLUGINS, "
            f"disabled ena_seq_parallel, ena_context_parallel and ena_swa_attn_seq_parallel"
        )

    if ena_cp:
        assert ena_sp, "ena_context_parallel requires ena_seq_parallel"
    if ena_attn_sp:
        assert ena_sp, "ena_swa_attn_seq_parallel requires ena_seq_parallel"


def _task_queue_level():
    """The level torch_npu will actually run at.

    Blocking launch turns the queue off whatever the level variable says, and
    the level itself is an int to torch_npu ("02" is level 2).
    """
    if os.environ.get("ASCEND_LAUNCH_BLOCKING", "").strip() == "1":
        return 0
    raw = os.environ.get("TASK_QUEUE_ENABLE", "").strip()
    if not raw:
        return 1  # torch_npu's default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "[WARNING] unparsable TASK_QUEUE_ENABLE=%r; assuming level 1.", raw
        )
        return 1


def apply_prefill_cudagraph(model_extra_config):
    """Prefill in-graph needs AICPU tiling and a capture at task queue level 1.

    Prefill shapes vary per replay, so every length the kernels run over has to
    come from a device tensor. With AICPU tiling off they are host lists read
    while the graph is built, and a replay would silently use the length of
    whatever batch was captured.

    Level 2 dispatches ops from a side thread, which a capture cannot use.
    ``acl_graph.task_queue_level`` lowers the level around the capture, but only
    where torch_npu exposes the setter; without it the capture would run at
    whatever ``TASK_QUEUE_ENABLE`` asked for, so the feature turns itself off
    instead of capturing graphs that cannot be replayed.
    """
    if not model_extra_config.operator_opt_config.enable_prefill_cudagraph:
        return
    if not model_extra_config.operator_opt_config.use_aicpu_fa_tiling:
        model_extra_config.operator_opt_config.enable_prefill_cudagraph = False
        logger.warning(
            "[WARNING] disabled enable_prefill_cudagraph: it needs "
            "use_aicpu_fa_tiling, or a replay reuses the captured batch's "
            "attention lengths."
        )
        return
    if _task_queue_level() != 2:
        return

    import torch_npu

    if getattr(torch_npu.npu, "set_task_queue_enable", None) is not None:
        return
    model_extra_config.operator_opt_config.enable_prefill_cudagraph = False
    logger.warning(
        "[WARNING] disabled enable_prefill_cudagraph: capture needs task queue "
        "level 1, TASK_QUEUE_ENABLE=2 and this torch_npu has no "
        "set_task_queue_enable to lower it around the capture."
    )


def apply_omni_cache(additional_config):
    pass
