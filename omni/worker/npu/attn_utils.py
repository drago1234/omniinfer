# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV-cache reshape adaptation for NPU attention backends.

Injected by omni/vllm_patches/patches/common/patch_mrv2_attn_utils.py.
"""

from __future__ import annotations

import torch

from vllm.v1.worker.gpu import attn_utils as up_attn_utils

# Captured at import, which the plugin performs before any patch is applied.
_ORIGINAL = up_attn_utils._reshape_kv_cache


def reshape_kv_cache(
    attn_groups,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config=None,
) -> dict:
    """Reshape each group with either its backend hook or upstream vLLM.

    NPU attention backends expose the shape of one K or V tensor, while the
    raw allocation contains both K and V. Running the upstream reshape first
    therefore attempts to view a 2x allocation as a single tensor and fails
    before the NPU hook can split it. Keep those groups out of the upstream
    path and let their backend own the raw allocation directly.
    """
    backend_groups = []
    upstream_groups = []
    for group in attn_groups:
        hook = getattr(getattr(group, "backend", None), "reshape_kv_cache", None)
        if hook is None:
            upstream_groups.append(group)
        else:
            backend_groups.append((group, hook))

    kv_caches = {}
    if upstream_groups:
        upstream_layer_names = {
            layer_name
            for group in upstream_groups
            for layer_name in group.layer_names
        }
        upstream_shared_layers = {}
        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            if layer_name not in upstream_layer_names:
                continue
            if target_layer_name not in upstream_layer_names:
                continue
            upstream_shared_layers[layer_name] = target_layer_name
        kv_caches.update(
            _ORIGINAL(
                attn_groups=upstream_groups,
                kv_cache_raw_tensors=kv_cache_raw_tensors,
                cache_dtype=cache_dtype,
                kernel_block_sizes=kernel_block_sizes,
                shared_kv_cache_layers=upstream_shared_layers,
                kv_cache_config=kv_cache_config,
            )
        )

    for group, hook in backend_groups:
        spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                continue
            raw = kv_cache_raw_tensors[layer_name]
            raw_size_bytes = raw.numel() * raw.element_size()
            if raw_size_bytes % spec.page_size_bytes:
                raise ValueError(
                    f"{layer_name}: raw size {raw_size_bytes} is not divisible by "
                    f"page size {spec.page_size_bytes}"
                )
            num_blocks = raw_size_bytes // spec.page_size_bytes
            kv_caches[layer_name] = hook(raw, num_blocks, spec)

    # Apply aliases after both reshape paths have populated their target views.
    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        if target_layer_name not in kv_caches:
            raise RuntimeError(
                f"[omni-npu/mrv2] shared KV-cache target {target_layer_name} "
                f"for {layer_name} was not initialized"
            )
        kv_caches[layer_name] = kv_caches[target_layer_name]

    return kv_caches
