# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared NPU KV cache registration-view normalisation.

NPU attention backends keep inference views as tuple/list strided sub-tensors
over one raw allocation per layer, while vLLM connector registration expects
either a single full-page tensor (Attention) or a list (Mamba/Mome). These
helpers reshape the registration view only; the underlying storage is shared
and unchanged.

Lives in a dependency-free common package because multiple connectors need
it (kv_offload's NPUOffloadingConnector and fusion_pool's NPUMooncakeStoreConnector
both re-export :func:`prepare_kv_caches_for_offloading_registration`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)

KVCacheValue = torch.Tensor | Sequence[torch.Tensor]


def _layer_kv_cache_spec(
    layer_name: str, kv_cache_config: KVCacheConfig
) -> KVCacheSpec:
    for group in kv_cache_config.kv_cache_groups:
        if layer_name not in group.layer_names:
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return spec.kv_cache_specs[layer_name]
        return spec
    raise KeyError(f"Layer {layer_name!r} not found in kv_cache_config")


def _full_page_int8_view(
    first: torch.Tensor,
    num_blocks: int,
    page_size_bytes: int,
) -> torch.Tensor:
    """Build a (num_blocks, page_size_bytes) int8 view over shared raw storage."""
    storage = first.untyped_storage()
    expected_bytes = num_blocks * page_size_bytes
    if storage.nbytes() < expected_bytes:
        raise RuntimeError(
            f"KV storage has {storage.nbytes()} bytes, expected at least "
            f"{expected_bytes} for {num_blocks} blocks x {page_size_bytes} bytes"
        )
    return (
        torch.tensor([], dtype=torch.int8, device=first.device)
        .set_(storage, 0, (num_blocks, page_size_bytes), (page_size_bytes, 1))
    )


def prepare_kv_caches_for_offloading_registration(
    kv_caches: dict[str, KVCacheValue],
    kv_cache_config: KVCacheConfig,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Present NPU KV caches in the shape connector registration expects.

    NPU backends keep inference views as tuple/list strided sub-tensors over
    one raw allocation per layer. Connector registration expects either a
    single full-page tensor (Attention) or a list (Mamba/Mome). We only reshape
    the registration view; underlying storage is unchanged.
    """
    num_blocks = kv_cache_config.num_blocks
    prepared: dict[str, torch.Tensor | list[torch.Tensor]] = {}

    for layer_name, value in kv_caches.items():
        if isinstance(value, torch.Tensor):
            prepared[layer_name] = value
            continue

        if not isinstance(value, (tuple, list)):
            raise TypeError(
                f"Unsupported KV cache type for layer {layer_name!r}: {type(value)!r}"
            )
        if len(value) == 0:
            raise ValueError(f"Empty KV cache sequence for layer {layer_name!r}")

        first = value[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError(
                f"Expected tensor in KV cache sequence for {layer_name!r}, "
                f"got {type(first)!r}"
            )

        layer_spec = _layer_kv_cache_spec(layer_name, kv_cache_config)
        page_size_bytes = layer_spec.page_size_bytes

        if isinstance(layer_spec, MambaSpec):
            if first.storage_offset() != 0:
                logger.warning(
                    "Layer %s first Mamba/Mome state tensor storage_offset=%d "
                    "(expected 0 for page-packed layout)",
                    layer_name,
                    first.storage_offset(),
                )
            prepared[layer_name] = list(value)
            continue

        if isinstance(layer_spec, AttentionSpec):
            if first.storage_offset() != 0:
                logger.warning(
                    "Layer %s first attention KV sub-tensor storage_offset=%d "
                    "(expected 0 for page-packed layout)",
                    layer_name,
                    first.storage_offset(),
                )
            prepared[layer_name] = _full_page_int8_view(
                first, num_blocks, page_size_bytes
            )
            continue

        raise NotImplementedError(
            f"Unsupported KV cache spec {type(layer_spec)!r} for layer {layer_name!r}"
        )

    return prepared
