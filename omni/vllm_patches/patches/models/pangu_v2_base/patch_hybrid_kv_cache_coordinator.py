# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hybrid APC: Decode FA-skip; other paths run the producer per-group lookup.

PD Decode must reuse local FA prefix cache. vLLM
``find_longest_cache_hit_per_group`` looks up each group independently; the
connector scheduler then takes ``max(per_group_hits)``. On Decode that blocks
FA skip (Mome miss / wrong pull path).

This patch only changes Decode: report FA hit length and clear local
Mome/Mamba computed blocks so PD pulls Prefill's last state. Other paths run
the upstream producer fix below.

Decode is detected by ``OMNI_PD_ROLE=decode`` (``pd_run.sh --role``, set by
ansible ``run_vllm_decode.sh.j2``). Anything else uses
``_find_longest_cache_hit_per_group``.

Producer fix: full-attention blocks keep their true hit so the deep D-side
APC hit is not re-pulled. The scalar is ``min(hit_lengths)`` so SWA/Mome do
not over-skip. A Mamba/Mome snapshot past that minimum is looked up again at
the minimum.
"""

from typing import Callable

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


# Bound at import; find_longest_cache_hit_per_group below replaces the class attr.
_find_longest_cache_hit_per_group: Callable = (
    HybridKVCacheCoordinator.find_longest_cache_hit_per_group
)

_PATH_LOGGED = False


def _is_kv_consumer() -> bool:
    """True when ``OMNI_PD_ROLE=decode`` (ansible Decode / ``pd_run --role``)."""
    import os

    return os.environ.get("OMNI_PD_ROLE", "").strip().lower() == "decode"


def _log_hybrid_apc_path_once(is_consumer: bool) -> None:
    global _PATH_LOGGED
    if _PATH_LOGGED:
        return
    _PATH_LOGGED = True
    try:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "HybridAPC per_group using %s path",
            "consumer-FA-skip"
            if is_consumer
            else "_find_longest_cache_hit_per_group",
        )
    except Exception:  # noqa: BLE001
        pass


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Dispatch: consumer FA-skip, else upstream per-group real-hit lookup."""
    is_consumer = _is_kv_consumer()
    _log_hybrid_apc_path_once(is_consumer)
    if is_consumer:
        return _consumer_fa_skip_per_group(
            self, block_hashes, max_cache_hit_length
        )
    return _producer_per_group(self, block_hashes, max_cache_hit_length)


def _producer_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return per-group real blocks with a uniform min length scalar."""
    # adapt start
    blocks, hit_lengths = (
        HybridAPCConnectorHitPatch._upstream_find_longest_cache_hit_per_group(
            self, block_hashes, max_cache_hit_length
        )
    )
    return blocks, (hit_length,) * len(blocks)


def _consumer_fa_skip_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Per-group lookup; each group reports its own true hit length.

    The scheduler takes ``max(per_group_hits)`` as the FA length, so every
    group must return its real hit length: broadcasting the FA scalar to all
    groups desyncs the scheduler scalar from the per-group block tables
    (skip-padding, external allocation and cache_blocks bookkeeping all key
    off that scalar per group). Mome/Mamba blocks are kept as-is so the
    [null,..,real] layout stays aligned; LLMDataDist still pulls Prefill's
    latest state through the pull list.
    """

    def _hashes(kv_cache_spec: KVCacheSpec):
        if kv_cache_spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, kv_cache_spec.block_size
        )

    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_blocks: list[list] = [[] for _ in range(num_groups)]
    hit_lengths: list[int] = [0] * num_groups

    for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
        blocks = manager_cls.find_longest_cache_hit(
            block_hashes=_hashes(spec),
            max_length=max_cache_hit_length,
            kv_cache_group_ids=group_ids,
            block_pool=self.block_pool,
            kv_cache_spec=spec,
            drop_eagle_block=use_eagle,
            alignment_tokens=self.scheduler_block_size,
        )
        group_hit = len(blocks[0]) * spec.block_size
        if isinstance(spec, FullAttentionSpec):
            _FA_GROUP_IDS.update(group_ids)
            _push_fa_group_ids_to_connector()
        # FIX: report each group's true hit length and keep its blocks.
        # The scheduler takes max(per_group_hits); broadcasting the FA-only
        # scalar to all groups desyncs the scalar from the per-group block
        # tables (add_local_computed_blocks skip-padding, external alloc,
        # cache_blocks bookkeeping all key off the scalar per group), which
        # corrupts the hybrid block layout under int8 page topologies.
        # Mome hits stay: the null-prefix layout [null,..,real] keeps the
        # scalar/block-table pairing, and P's state still reaches D through
        # the pull list built from the request block table.
        for group_id, blks in zip(group_ids, blocks):
            hit_blocks[group_id] = blks
            hit_lengths[group_id] = group_hit

    return tuple(hit_blocks), tuple(hit_lengths)


# KV cache group indices whose spec is full attention. Only these groups
# may skip pulling their prefix-cache hit blocks (connector hash-skip):
# non-FA groups mix null placeholders and window-tail hits in their block
# tables, and skipping those desyncs the P/D positional pairing in
# SchemePull._parse_blocks.
_FA_GROUP_IDS: set[int] = set()


def fa_group_ids() -> frozenset[int]:
    """Full-attention group ids (connector consults this for hash-skip)."""
    return frozenset(_FA_GROUP_IDS)


def _push_fa_group_ids_to_connector():
    """Publish FA group ids to every loaded llmdatadist_connector_v1 module.

    The connector cannot import this module: it may be registered by file
    path (a plain import would resolve to a different module instance with
    an empty registry, and on some deployments this package path does not
    exist at all). Scanning sys.modules by file name is instance-proof and
    needs no importable package path.
    """
    import sys
    for mod in sys.modules.values():
        if mod is None:
            continue
        if getattr(mod, "__file__", None) and mod.__file__.rsplit("/", 1)[-1] == "llmdatadist_connector_v1.py":
            mod.FA_GROUP_IDS_SKIP_PULL = frozenset(_FA_GROUP_IDS)


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("find_longest_cache_hit_per_group")
