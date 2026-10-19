# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Installation guards and real vLLM 0.25.1 SHM regressions for #59695.

The runtime tests use CPU shared memory without model weights. The installation
checks use isolated interface doubles; each runtime test restores its patch.
"""

from contextlib import contextmanager
from inspect import signature
from multiprocessing import Lock, get_context
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest


def load_patch(patch_env, target):
    patch_env.stub(
        "vllm.distributed.device_communicators.shm_object_storage",
        SingleWriterShmObjectStorage=target,
    )
    sender = type("ShmObjectStoreSenderCache", (), {})
    patch_env.stub("vllm.multimodal.cache", ShmObjectStoreSenderCache=sender)
    module = patch_env.load("patch_shm_reference_counting")
    return module.ShmReferenceCountingPatch59695, sender


def test_native_fix_is_not_overwritten(patch_env):
    release = Mock()
    touch = Mock()
    target = type("NativeStorage", (), {"release_touches": release, "touch": touch})
    patch, sender = load_patch(patch_env, target)
    patch.apply()
    assert target.release_touches is release
    assert target.touch is touch
    assert not hasattr(sender, "release_sender_touches")


def test_signed_handle_api_is_rejected_before_mutation(patch_env):
    def touch(self, key, address=0, monotonic_id=0, signature=None):
        pass

    target = type("SignedStorage", (), {"touch": touch})
    patch, sender = load_patch(patch_env, target)
    with pytest.raises(RuntimeError, match="v0.25.1 handle API"):
        patch.apply()
    assert target.touch is touch
    assert not hasattr(target, "release_touches")
    assert not hasattr(sender, "release_sender_touches")


def test_repeated_apply_keeps_original_wrapper_chain(patch_env):
    class Storage:
        def __init__(self):
            self.is_writer = True

        def clear(self):
            pass

        def get_cached(self, key):
            return 8, 1

        def touch(self, key, address=0, monotonic_id=0):
            pass

    original_init, original_get = Storage.__init__, Storage.get_cached
    patch, sender = load_patch(patch_env, Storage)
    assert patch_env.registered["ShmReferenceCountingPatch59695"] is patch
    patch.apply()
    installed_init = Storage.__init__
    patch.apply()
    assert Storage.__init__ is installed_init
    assert patch._upstream__init__ is original_init
    assert patch._upstream_get_cached is original_get
    storage = Storage()
    storage._touched.add(1)
    assert storage.get_cached("image") == (8, 1)
    assert storage._touched == set()
    storage._touched.add(2)
    cache = sender()
    cache._shm_cache = storage
    cache.release_sender_touches()
    assert storage._touched == set()


@pytest.fixture
def patched_storage():
    pytest.importorskip("vllm")
    from vllm.distributed.device_communicators.shm_object_storage import (
        SingleWriterShmObjectStorage,
    )
    from vllm.multimodal.cache import ShmObjectStoreSenderCache
    from omni_npu.vllm_patches.patches.models.openpangu_v1_vl import (
        patch_shm_reference_counting as module,
    )

    if "signature" in signature(SingleWriterShmObjectStorage.touch).parameters:
        pytest.skip("These handle contracts target the vLLM 0.25.1 backport")
    patch = module.ShmReferenceCountingPatch59695
    missing = object()
    saved = [(SingleWriterShmObjectStorage, name,
              getattr(SingleWriterShmObjectStorage, name, missing))
             for name in patch._attr_names_to_apply]
    saved.append((ShmObjectStoreSenderCache, "release_sender_touches",
                  getattr(ShmObjectStoreSenderCache, "release_sender_touches", missing)))
    try:
        patch.apply()
        yield SingleWriterShmObjectStorage
    finally:
        for target, name, original in saved:
            if original is missing:
                if hasattr(target, name):
                    delattr(target, name)
            else:
                setattr(target, name, original)


@pytest.fixture
def storage_factory(patched_storage):
    from vllm.distributed.device_communicators.shm_object_storage import (
        SingleWriterShmRingBuffer,
    )

    @contextmanager
    def create(n_readers=4, size=102400):
        ring = SingleWriterShmRingBuffer(data_buffer_size=size, create=True)
        writer = patched_storage(max_object_size=1024, n_readers=n_readers,
                                 ring_buffer=ring)
        readers = []
        try:
            handle = writer.handle()
            handle.reader_lock = Lock()
            for _ in range(n_readers):
                readers.append(patched_storage.create_from_handle(handle))
            yield writer, readers
        finally:
            for reader in readers:
                reader.close()
            writer.close()

    return create


def is_evictable(writer, key="image"):
    writer.free_unused()
    return not writer.is_cached(key)


@pytest.mark.parametrize("n_readers", [1, 2, 4])
@pytest.mark.integration
def test_reference_again_before_all_readers_finish(storage_factory, n_readers):
    with storage_factory(n_readers) as (writer, readers):
        first = writer.put("image", "pixels")
        writer.touch("image")
        second = writer.get_cached("image")
        writer.release_touches()
        for reader in readers:
            reader.touch("image", *first)
        for reader in readers:
            assert reader.get(*first) == "pixels"
            reader.touch("image", *second)
            assert reader.get(*second) == "pixels"
        assert is_evictable(writer)


@pytest.mark.parametrize("n_readers", [1, 2, 4])
@pytest.mark.integration
def test_duplicate_new_item_waits_for_last_reader(storage_factory, n_readers):
    with storage_factory(n_readers) as (writer, readers):
        writer.touch("image")
        refs = [writer.put("image", "pixels"), writer.get_cached("image")]
        writer.release_touches()
        for reader in readers[:-1]:
            for ref in refs:
                reader.touch("image", *ref)
            for ref in refs:
                assert reader.get(*ref) == "pixels"
        for ref in refs:
            readers[-1].touch("image", *ref)
        assert not is_evictable(writer)
        assert readers[-1].get(*refs[0]) == "pixels"
        assert not is_evictable(writer)
        assert readers[-1].get(*refs[1]) == "pixels"
        assert is_evictable(writer)


@pytest.mark.integration
def test_touch_protects_until_released_without_adding_reference(storage_factory):
    with storage_factory() as (writer, readers):
        ref = writer.put("image", "pixels")
        for reader in readers:
            reader.get(*ref)
        writer.touch("image")
        writer.touch("image")
        assert not is_evictable(writer)
        writer.release_touches()
        assert is_evictable(writer)


@pytest.mark.integration
def test_get_cached_converts_touch_to_one_reader_reference(storage_factory):
    with storage_factory() as (writer, readers):
        ref = writer.put("image", "pixels")
        for reader in readers:
            reader.get(*ref)
        writer.touch("image")
        ref = writer.get_cached("image")
        assert not is_evictable(writer)
        for reader in readers:
            reader.touch("image", *ref)
            reader.get(*ref)
        # No release_touches: get_cached itself must remove the pin.
        assert is_evictable(writer)


@pytest.mark.integration
def test_clear_removes_old_touches_before_ids_are_reused(storage_factory):
    with storage_factory() as (writer, readers):
        old = writer.put("old", "old pixels")
        writer.touch("old")
        writer.clear()
        new = writer.put("image", "new pixels")
        assert new == old  # The 0.25.1 ring resets its monotonic IDs on clear.
        for reader in readers:
            assert reader.get(*new) == "new pixels"
        assert is_evictable(writer)


@pytest.mark.integration
def test_reader_touch_rejects_stale_id_without_acknowledging(storage_factory):
    with storage_factory() as (writer, readers):
        address, ident = writer.put("image", "pixels")
        with pytest.raises(ValueError, match="modified or is invalid"):
            readers[0].touch("image", address, ident + 1)
        for reader in readers[:-1]:
            reader.get(address, ident)
        assert not is_evictable(writer)
        readers[-1].get(address, ident)
        assert is_evictable(writer)


@pytest.fixture
def sender(patched_storage, monkeypatch):
    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.cache import ShmObjectStoreSenderCache

    monkeypatch.setenv("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", f"omni_ut_{uuid4().hex}")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=4),
        model_config=SimpleNamespace(get_multimodal_config=lambda: MultiModalConfig(
            mm_processor_cache_type="shm",
            mm_processor_cache_gb=32768 / (1024 ** 3),
            mm_shm_cache_max_object_size_mb=1,
        )),
    )
    cache = ShmObjectStoreSenderCache(config)
    try:
        yield cache, config
    finally:
        cache.close()


@pytest.mark.integration
def test_sender_receiver_cache_survives_ring_wraps(sender):
    from vllm.multimodal.cache import ShmObjectStoreReceiverCache
    from vllm.multimodal.inputs import MultiModalKwargsItem

    cache, config = sender
    readers = []
    try:
        lock = Lock()
        for _ in range(4):
            readers.append(ShmObjectStoreReceiverCache(config, lock))
        for index in range(600):
            key = f"image-{index}"
            item = MultiModalKwargsItem.dummy(nbytes=256)
            refs = []
            # Two requests refer to a new image before any worker has read it.
            for _ in range(2):
                cache.touch_sender_cache_item(key)
                refs.append(cache.get_and_update_item((item, []), key)[0])
                cache.release_sender_touches()
            assert all("address" in ref for ref in refs), f"cache stalled at {index}"
            for reader in readers:
                for ref in refs:
                    reader.touch_receiver_cache_item(key, ref)
            for reader in readers:
                for ref in refs:
                    assert reader.get_and_update_item(ref, key) == item
        assert not cache.is_cached_item("image-0")
        assert cache.is_cached_item("image-599")
    finally:
        for reader in readers:
            reader._shm_cache.close()


def _read_in_worker(handle, inbox, done):
    from vllm.distributed.device_communicators.shm_object_storage import (
        SingleWriterShmObjectStorage,
    )

    reader = SingleWriterShmObjectStorage.create_from_handle(handle)
    try:
        while (request := inbox.get()) is not None:
            key, refs = request
            for ref in refs:
                reader.touch(key, *ref)
            for ref in refs:
                assert reader.get(*ref) == key
            done.put(None)
    except BaseException as error:
        done.put(repr(error))
        raise
    finally:
        reader.close()


@pytest.mark.integration
def test_four_processes_balance_references_across_ring_wraps(storage_factory):
    # Fork inherits only this process's temporary patch; vendor files stay intact.
    ctx = get_context("fork")
    with storage_factory(size=8192) as (writer, _):
        handle = writer.handle()
        handle.reader_lock = ctx.Lock()
        done = ctx.Queue()
        inboxes = [ctx.Queue() for _ in range(4)]
        workers = [ctx.Process(target=_read_in_worker, args=(handle, inbox, done))
                   for inbox in inboxes]
        try:
            for worker in workers:
                worker.start()
            for index in range(200):
                key = f"image-{index}-" + "x" * 128
                first = writer.put(key, key)
                writer.touch(key)
                second = writer.get_cached(key)
                writer.release_touches()
                for inbox in inboxes:
                    inbox.put((key, [first, second]))
                for _ in workers:
                    assert done.get(timeout=15) is None
                with writer.ring_buffer.access_buf(first[0]) as (buf, _):
                    assert int.from_bytes(buf[:4], "little", signed=True) == 8
            assert writer.ring_buffer.monotonic_id_start > 0
        finally:
            for inbox in inboxes:
                inbox.put(None)
            for worker in workers:
                if worker.pid is not None:
                    worker.join(timeout=10)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
            for queue in [*inboxes, done]:
                queue.cancel_join_thread()
                queue.close()
        assert all(worker.exitcode == 0 for worker in workers)
