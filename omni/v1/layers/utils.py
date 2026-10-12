# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
from contextlib import nullcontext

import torch
import torchair
from transformers import PretrainedConfig


def get_npu_execution_type(stream_label):
    if stream_label is None:
        return nullcontext()
    # Using strings to determine whether to include an item in the image,
    # and later we will use logical differentiation based on parameters.
    elif isinstance(stream_label, str):
        return torchair.scope.npu_stream_switch(stream_label)  # Graph GE/ACL
    elif isinstance(stream_label, torch.npu.Stream):
        return torch.npu.stream(stream_label)  # eager
    return nullcontext()


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def named_stream(name): # not for graph
    if name == "current":
        return torch.npu.current_stream()
    if not hasattr(named_stream, name):
        setattr(named_stream, name, torch.npu.Stream())
    return getattr(named_stream, name)


def get_cla_mapping(config: PretrainedConfig) -> dict[int, int]:
    """Return the zero-based CLA ``reuse_layer -> source_layer`` mapping."""
    raw_mapping = getattr(config, "cla_explicit_mapping", None)
    if not raw_mapping:
        return {}

    items = raw_mapping.items() if isinstance(raw_mapping, dict) else raw_mapping
    mapping: dict[int, int] = {}
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "cla_explicit_mapping entries must be [reuse_layer, source_layer] "
                f"pairs, but got {item!r}."
            )
        reuse_layer, source_layer = int(item[0]), int(item[1])
        if reuse_layer in mapping:
            raise ValueError(f"Duplicate CLA reuse layer {reuse_layer} in cla_explicit_mapping.")
        if source_layer >= reuse_layer:
            raise ValueError(
                f"A CLA source layer must precede its reuse layer, but got "
                f"reuse={reuse_layer}, source={source_layer}."
            )
        if num_hidden_layers is not None and not (0 <= source_layer < reuse_layer < num_hidden_layers):
            raise ValueError(
                "CLA layer indices must be in [0, num_hidden_layers), but got "
                f"reuse={reuse_layer}, source={source_layer}, "
                f"num_hidden_layers={num_hidden_layers}."
            )
        mapping[reuse_layer] = source_layer
    return mapping
