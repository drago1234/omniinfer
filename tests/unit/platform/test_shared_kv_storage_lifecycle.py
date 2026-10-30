# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Exercise actual Worker methods with an allocator that rejects live resize.

Run directly with Python to avoid importing the NPU stack or pytest conftests.
These tests check storage lifecycle semantics, not NPU allocation peaks.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]


class Storage:
    def __init__(self, size, identity):
        self.size = size
        self._cdata = identity
        self.calls = []
        self.fail_once = False

    def nbytes(self):
        return self.size

    def resize_(self, size):
        if size and self.fail_once:
            self.fail_once = False
            raise MemoryError("injected restore failure")
        if size and self.size:
            raise MemoryError("replacement of live NPU storage")
        self.calls.append(size)
        self.size = size


class View:
    def __init__(self, storage):
        self.storage = storage

    def untyped_storage(self):
        return self.storage


class StorageHandle:
    """Separate Python handles that address the same StorageImpl."""

    def __init__(self, storage):
        self.storage = storage
        self._cdata = storage._cdata

    def nbytes(self):
        return self.storage.nbytes()

    def resize_(self, size):
        self.storage.resize_(size)


def make_worker(rows, storages, *, eager=True):
    source = ROOT / "omni/worker/npu_worker.py"
    tree = ast.parse(source.read_text())
    worker_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUWorker"
    )
    worker_class.body = [
        node for node in worker_class.body
        if isinstance(node, ast.FunctionDef) and node.name in ("sleep", "wake_up")
    ]
    worker_class.bases = []
    worker_class.decorator_list = []
    npu = SimpleNamespace(
        empty_cache=Mock(), synchronize=Mock(),
        mem_get_info=lambda: (4096 - sum(s.size for s in storages), 4096),
    )
    namespace = {
        "torch": SimpleNamespace(npu=npu, nn=SimpleNamespace(Module=object)),
        "gc": SimpleNamespace(collect=Mock()),
        "logger": Mock(), "List": List, "Optional": Optional,
        "on_ascend950": lambda: False, "set_aclgraph_recapture": Mock(),
    }
    helper_path = ROOT / "omni/worker/kv_storage.py"
    if helper_path.exists():
        spec = importlib.util.spec_from_file_location("kv_storage_test", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        namespace.update(
            snapshot_kv_storages=helper.snapshot_kv_storages,
            resize_kv_storages=helper.resize_kv_storages,
        )
    module = ast.Module(body=[worker_class], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    worker = namespace["NPUWorker"]()
    worker.is_full_async_rl = False
    worker.model_config = SimpleNamespace(enforce_eager=eager)
    model = SimpleNamespace(to=Mock())
    worker.model_runner = SimpleNamespace(
        kv_caches=rows, drafter=None, get_model=lambda: model,
        unregister_kv_caches=Mock(), reregister_kv_caches=Mock(),
        capture_model=Mock(),
    )
    return worker


class TestSharedKVStorageLifecycle(unittest.TestCase):
    def test_shared_storage_released_and_restored_once_per_round(self):
        storage = Storage(256, 1)
        rows = [[View(storage), View(storage)] for _ in range(3)]
        worker = make_worker(rows, [storage])
        for _ in range(2):
            worker.sleep()
            self.assertEqual(storage.size, 0)
            worker.wake_up(["weights"])
            self.assertEqual(storage.size, 0)
            worker.wake_up(["kv_cache"])
            self.assertEqual(storage.size, 256)
        self.assertEqual(storage.calls, [0, 256, 0, 256])
        self.assertIs(worker.model_runner.kv_caches, rows)

    def test_distinct_storages_keep_original_sizes(self):
        a, b = Storage(128, 1), Storage(512, 2)
        worker = make_worker([[View(a), View(b)], [View(a)]], [a, b])
        worker.sleep()
        worker.wake_up(["kv_cache"])
        self.assertEqual(a.calls, [0, 128])
        self.assertEqual(b.calls, [0, 512])

    def test_storage_identity_is_independent_of_python_handle(self):
        storage = Storage(128, 1)
        rows = [[View(StorageHandle(storage)), View(StorageHandle(storage))]]
        worker = make_worker(rows, [storage])
        worker.sleep()
        worker.wake_up(["kv_cache"])
        self.assertEqual(storage.calls, [0, 128])

    def test_snapshot_failure_does_not_partially_release_storage(self):
        storage = Storage(128, 1)
        bad_view = Mock()
        bad_view.untyped_storage.side_effect = RuntimeError("snapshot failed")
        worker = make_worker([[View(storage), bad_view]], [storage])
        with self.assertRaisesRegex(RuntimeError, "snapshot failed"):
            worker.sleep()
        self.assertEqual(storage.size, 128)
        self.assertEqual(storage.calls, [])

    def test_partial_restore_retry_preserves_successful_storage(self):
        a, b = Storage(128, 1), Storage(512, 2)
        worker = make_worker([[View(a), View(b), View(a)]], [a, b])
        worker.sleep()
        b.fail_once = True
        with self.assertRaisesRegex(MemoryError, "injected"):
            worker.wake_up(["kv_cache"])
        worker.model_runner.reregister_kv_caches.assert_not_called()
        self.assertEqual((a.size, b.size), (128, 0))
        worker.wake_up(["kv_cache"])
        self.assertEqual(a.calls, [0, 128])
        self.assertEqual(b.calls, [0, 512])
        worker.model_runner.reregister_kv_caches.assert_called_once_with()

    def test_repeated_wake_does_not_resize_restored_storage(self):
        storage = Storage(128, 1)
        worker = make_worker([[View(storage)]], [storage])
        worker.sleep()
        worker.wake_up(["kv_cache"])
        worker.wake_up(["kv_cache"])
        self.assertEqual(storage.calls, [0, 128])

    def test_empty_storage_never_resized(self):
        storage = Storage(0, 1)
        worker = make_worker([[View(storage), View(storage)]], [storage])
        worker.sleep()
        worker.wake_up(["kv_cache"])
        self.assertEqual(storage.calls, [])

    def test_empty_cache_collection(self):
        worker = make_worker([], [])
        worker.sleep()
        worker.wake_up(["kv_cache"])
        worker.model_runner.reregister_kv_caches.assert_called_once_with()

    def test_registration_failure_retry_keeps_storage_allocation(self):
        storage = Storage(128, 1)
        worker = make_worker([[View(storage)]], [storage], eager=False)
        worker.sleep()
        worker.model_runner.reregister_kv_caches.side_effect = [
            RuntimeError("register failed"), None,
        ]
        with self.assertRaisesRegex(RuntimeError, "register failed"):
            worker.wake_up(["kv_cache"])
        worker.model_runner.capture_model.assert_not_called()
        worker.wake_up(["kv_cache"])
        self.assertEqual(storage.calls, [0, 128])
        worker.model_runner.capture_model.assert_called_once_with()

    def test_recapture_failure_retry_keeps_storage_allocation(self):
        storage = Storage(128, 1)
        worker = make_worker([[View(storage)]], [storage], eager=False)
        worker.sleep()
        worker.model_runner.capture_model.side_effect = [
            RuntimeError("capture failed"), None,
        ]
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            worker.wake_up(["kv_cache"])
        worker.wake_up(["kv_cache"])
        self.assertEqual(storage.calls, [0, 128])
        self.assertEqual(worker.model_runner.capture_model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
