# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Save and resize each underlying KV storage once across sleep and wake."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

KVStorageSnapshot = list[tuple["torch.UntypedStorage", int]]


def snapshot_kv_storages(
    kv_caches: Iterable[Iterable[torch.Tensor]],
) -> KVStorageSnapshot:
    """Collect original sizes before releasing any shared storage.

    Keep storage references as well as StorageImpl identities: tensor pointers
    may have offsets, and storage data pointers become zero after release.
    """
    unique: dict[int, tuple[torch.UntypedStorage, int]] = {}
    for row in kv_caches:
        for tensor in row:
            storage = tensor.untyped_storage()
            if storage._cdata not in unique:
                unique[storage._cdata] = (storage, storage.nbytes())
    return list(unique.values())


def resize_kv_storages(snapshot: KVStorageSnapshot, *, release: bool) -> None:
    """Preserve allocations already restored before a later resize failed."""
    for storage, original_nbytes in snapshot:
        target = 0 if release else original_nbytes
        # NPU resize_ may allocate a replacement even when the size is equal.
        if storage.nbytes() != target:
            storage.resize_(target)
