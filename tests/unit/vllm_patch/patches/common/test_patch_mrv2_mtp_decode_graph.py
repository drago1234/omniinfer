# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from omni_npu.vllm_patches.patches.common.patch_mrv2_mtp_decode_graph import (
    build_decode_manager_init, decode_graph_config, draft_compile_config,
    build_mtp_model_loader,
)


def config(sizes=(1, 2, 3), method="mtp"):
    return NS(
        additional_config={"mtp_decode_cudagraph_capture_sizes": list(sizes)},
        speculative_config=NS(method=method, uses_dynamic_speculative_decoding=lambda: False),
        scheduler_config=NS(max_num_seqs=3),
        compilation_config=NS(cudagraph_capture_sizes=[8, 16, 24],
                              max_cudagraph_capture_size=24, compile_sizes=[8, 16, 24]),
    )


def manager(cfg, query_len=1, mode=Mode.FULL_DECODE_ONLY):
    obj = object.__new__(CudaGraphManager)
    obj.vllm_config = cfg
    obj.compilation_config = cfg.compilation_config
    obj.max_num_reqs = 3
    obj.decode_query_len = query_len
    obj.cudagraph_mode = mode
    obj.lora_capture_cases = [0]
    obj._candidates = {}
    obj._capture_descs = {}
    obj._init_candidates()
    obj._graphs_captured = True  # Descriptor lookup only, no graph replay.
    return obj


def test_independent_sizes_and_real_dispatch():
    cfg = config()
    small = decode_graph_config(cfg, 1)
    assert small is not cfg
    assert small.compilation_config is not cfg.compilation_config
    assert cfg.compilation_config.cudagraph_capture_sizes == [8, 16, 24]
    assert cfg.compilation_config.max_cudagraph_capture_size == 24
    assert small.compilation_config.compile_sizes == [8, 16, 24]
    for batch in (1, 2, 3):
        desc = manager(small).dispatch(batch, batch, 1, 0)
        assert (desc.cg_mode, desc.num_tokens, desc.num_reqs) == (Mode.FULL, batch, batch)
        target = manager(cfg, 8, Mode.FULL).dispatch(batch, batch * 8, 8, 0)
        assert target.cg_mode == Mode.FULL and target.num_tokens == batch * 8
        assert manager(cfg).dispatch(batch, batch, 1, 0).cg_mode == Mode.NONE
    assert manager(small, mode=Mode.NONE).dispatch(1, 1, 1, 0).cg_mode == Mode.NONE


def test_sparse_buckets_padding_and_normalization():
    small = decode_graph_config(config((3, 1, 3)), 1)
    assert small.compilation_config.cudagraph_capture_sizes == [1, 3]
    desc = manager(small).dispatch(2, 2, 1, 0)
    assert (desc.num_tokens, desc.num_reqs) == (3, 3)


@pytest.mark.parametrize("sizes", [[], [0], [4], [True], [1.5], "123"])
def test_invalid_sizes(sizes):
    cfg = config()
    cfg.additional_config["mtp_decode_cudagraph_capture_sizes"] = sizes
    with pytest.raises(ValueError):
        decode_graph_config(cfg, 1)


def test_scope_and_default_preserved():
    cfg = config(method="eagle3")
    assert decode_graph_config(cfg, 1) is cfg
    cfg = config()
    cfg.additional_config.clear()
    assert decode_graph_config(cfg, 1) is cfg
    cfg.speculative_config = None
    assert decode_graph_config(cfg, 1) is cfg


def test_reject_incompatible_query_or_dynamic_sd():
    cfg = config()
    with pytest.raises(ValueError):
        decode_graph_config(cfg, 8)
    cfg.speculative_config.uses_dynamic_speculative_decoding = lambda: True
    with pytest.raises(ValueError):
        decode_graph_config(cfg, 1)


def test_init_wrapper_passes_private_config_and_arguments():
    seen = []
    init = build_decode_manager_init(lambda *args, **kwargs: seen.append((args, kwargs)))
    cfg = config()
    obj = object()
    init(obj, cfg, "npu:0", Mode.NONE, decode_query_len=1, lora_capture_cases=[0])
    args, kwargs = seen[0]
    assert args[0] is obj and args[1] is not cfg
    assert args[1].compilation_config.cudagraph_capture_sizes == [1, 2, 3]
    assert args[2:] == ("npu:0", Mode.NONE, 1)
    assert kwargs == {"lora_capture_cases": [0]}


def test_draft_exact_compile_sizes_preserve_target_and_attention_registry():
    cfg = config()
    registry = {"target_layer": object()}
    cfg.compilation_config.static_forward_context = registry
    draft = draft_compile_config(cfg)
    assert draft.compilation_config.compile_sizes == [1, 2, 3, 8, 16, 24]
    assert cfg.compilation_config.compile_sizes == [8, 16, 24]
    assert draft.compilation_config.cudagraph_capture_sizes == [8, 16, 24]
    assert draft.compilation_config.static_forward_context is registry
    draft.compilation_config.static_forward_context["draft_layer"] = object()
    assert "draft_layer" in cfg.compilation_config.static_forward_context
    seen = []
    loader = build_mtp_model_loader(lambda target, config: seen.append((target, config)))
    loader("target", cfg)
    assert seen[0][0] == "target"
    assert seen[0][1].compilation_config.compile_sizes == [1, 2, 3, 8, 16, 24]


def test_unconfigured_loader_preserves_original_config():
    cfg = config()
    cfg.additional_config.clear()
    assert draft_compile_config(cfg) is cfg
