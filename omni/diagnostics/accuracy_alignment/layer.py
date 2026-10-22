# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Decoder-layer boundary targets shared by MHC, attention, and MoE."""

from __future__ import annotations

from typing import Any

import torch

from .config import TrainRefConfig
from .hooks import ModuleTrainRefTarget


def _module(value: Any) -> torch.nn.Module | None:
    if isinstance(value, torch.nn.Module) and type(value).__name__ != "PPMissingLayer":
        return value
    return None


def collect_layer_targets(
    model: Any,
    config: TrainRefConfig,
) -> list[ModuleTrainRefTarget]:
    pangu_model = model.model
    if not (pangu_model.start_layer <= config.layer_id < pangu_model.end_layer):
        return []
    layer = pangu_model.layers[config.layer_id]
    candidates = (
        ("input_layernorm", "input_layernorm"),
        ("self_attn", "self_attention"),
        ("post_attention_layernorm", "post_self_attn_layernorm"),
        ("pre_mlp_layernorm", "pre_mlp_layernorm"),
        ("mlp", "mlp"),
        ("post_mlp_layernorm", "post_mlp_layernorm"),
        ("block_post_layernorm", "block_post_layernorm"),
    )
    targets = []
    for attr_name, training_name in candidates:
        module = _module(getattr(layer, attr_name, None))
        if module is None:
            continue
        targets.append(
            ModuleTrainRefTarget(
                module_name=(f"decoder.layers.{config.layer_id}.{training_name}"),
                training_name=training_name,
                module=module,
                gather_output=False,
            )
        )
    mlp = _module(getattr(layer, "mlp", None))
    if bool(getattr(layer, "use_mhc", False)) and mlp is not None:
        targets.append(
            ModuleTrainRefTarget(
                module_name=f"pangu_transformer_mlp_{config.layer_id + 1}",
                training_name="mlp",
                module=mlp,
                inject_input=False,
                gather_output=False,
            )
        )
    return targets
