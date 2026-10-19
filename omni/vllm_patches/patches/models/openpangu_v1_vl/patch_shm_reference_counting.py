# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Temporary runtime backport of vLLM PR #59695.

Upstream: https://github.com/vllm-project/vllm/pull/59695
Commit: 443dccd3c9448f586634bfdb7079acd7eab35ead
Baseline: 752a3a504485790a2e8491cacbb35c137339ad34 (current v0.25.1).
Remove this file once the installed vLLM includes the upstream fix.
Wrappers install the upstream counting protocol without editing vendor files.
The old baseline's get() ID check substitutes for the later validation helper.
OpenPanguVLProcessorMixin in omni-models supplies the finally cleanup.
"""

from inspect import signature

from vllm.distributed.device_communicators.shm_object_storage import (
    SingleWriterShmObjectStorage,
)

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("ShmReferenceCountingPatch59695", SingleWriterShmObjectStorage)
class ShmReferenceCountingPatch59695(VLLMPatch):
    _attr_names_to_apply = [
        "__init__", "clear", "get_cached", "release_touches", "touch",
        "default_is_free_check",
    ]

    def __init__(self, *args, **kwargs):
        ShmReferenceCountingPatch59695._upstream__init__(self, *args, **kwargs)
        if self.is_writer:
            self._touched: set[int] = set()

    def clear(self):
        ShmReferenceCountingPatch59695._upstream_clear(self)
        if self.is_writer:
            self._touched.clear()

    def get_cached(self, key):
        address, monotonic_id = ShmReferenceCountingPatch59695._upstream_get_cached(
            self, key,
        )
        self._touched.discard(monotonic_id)
        return address, monotonic_id

    def release_touches(self):
        """Stop protecting touched items that were not looked up."""
        self._touched.clear()

    def touch(self, key, address=0, monotonic_id=0) -> None:
        if self._reader_lock is None:
            if key in self.key_index:
                self._touched.add(self.key_index[key][1])
        else:
            with self.ring_buffer.access_buf(address) as (_, buf_metadata):
                if buf_metadata[0] != monotonic_id:
                    raise ValueError(
                        f"Data for address:id '{address}:{monotonic_id}'"
                        " has been modified or is invalid."
                    )

    def default_is_free_check(self, monotonic_id: int, buf: memoryview) -> bool:
        if monotonic_id in self._touched:
            return False
        reader_count = int.from_bytes(buf[0:4], "little", signed=True)
        writer_count = self.writer_flag[monotonic_id]
        return reader_count >= writer_count * self.n_readers

    def _release_shm_sender_touches(self):
        self._shm_cache.release_touches()

    @classmethod
    def apply(cls):
        # Native #59695 (or this patch already applied): leave it untouched.
        if hasattr(cls._target, "release_touches"):
            return

        from vllm.multimodal.cache import ShmObjectStoreSenderCache
        if "signature" in signature(cls._target.touch).parameters:
            raise RuntimeError("SHM #59695 patch requires the v0.25.1 handle API")
        cls.apply_bypass_conflict("__init__", "clear", "get_cached")
        ShmObjectStoreSenderCache.release_sender_touches = cls._release_shm_sender_touches
