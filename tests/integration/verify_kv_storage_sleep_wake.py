# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Small real-NPU storage regression using actual Worker lifecycle methods.

The model and transport hooks are stubs; this does not validate a live server,
weight movement, graph capture, or connector registration. Run as a script.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch_npu


MIB = 1 << 20


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean():
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()


def metadata(rows):
    return [
        (tuple(t.shape), tuple(t.stride()), t.storage_offset(), str(t.dtype),
         t.untyped_storage()._cdata)
        for row in rows for t in row
    ]

def make_views(buffers, count):
    rows = []
    dtypes = (torch.bfloat16, torch.float32, torch.int8)
    for index in range(count):
        raw = buffers[index % len(buffers)]
        dtype = dtypes[(index // len(buffers)) % len(dtypes)]
        itemsize = torch.empty((), dtype=dtype).element_size()
        rows.append([raw.view(dtype).as_strided(
            (raw.numel() // 4096, 8, 4),
            (4096 // itemsize, 4, 1),
            storage_offset=256 // itemsize,
        )])
    return rows


def worker(harness, root, rows, buffers, *, eager=True):
    harness.ROOT = root
    result = harness.make_worker(rows, [], eager=eager)
    # Replace the harness's simulated accelerator with the real NPU backend.
    result.sleep.__func__.__globals__.update(torch=torch, gc=gc)
    return result


def measure_wake(instance):
    clean()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    instance.wake_up(["kv_cache"])
    torch.npu.synchronize()
    return {
        "allocated_delta": torch.npu.memory_allocated() - before,
        "peak_delta": torch.npu.max_memory_allocated() - before,
    }


def layout_case(harness, root, *, pools, views, pool_mib, cycles, legacy=False):
    buffers = [torch.empty(pool_mib * MIB, dtype=torch.uint8, device="npu")
               for _ in range(pools)]
    rows = make_views(buffers, views)
    original_metadata = metadata(rows)
    instance = worker(harness, root, rows, buffers)
    records = []
    for cycle in range(cycles):
        instance.sleep()
        assert all(b.untyped_storage().nbytes() == 0 for b in buffers)
        instance.wake_up(["weights"])
        assert all(b.untyped_storage().nbytes() == 0 for b in buffers)
        first = measure_wake(instance)
        assert metadata(rows) == original_metadata
        assert all(b.untyped_storage().nbytes() == pool_mib * MIB for b in buffers)
        expected = pools * pool_mib * MIB
        if legacy:
            # Small safe reproduction of the duplicate peak, not an actual OOM.
            assert first["peak_delta"] > expected * 1.5, first
            records.append({"cycle": cycle + 1, "first_wake": first,
                            "duplicate_peak_detected": True})
            break
        assert first["peak_delta"] <= expected + 2 * MIB, first
        for index, raw in enumerate(buffers):
            raw.fill_(41 + index)
        torch.npu.synchronize()
        pointers = [b.untyped_storage().data_ptr() for b in buffers]
        repeated = measure_wake(instance)
        assert repeated["peak_delta"] == 0, repeated
        assert pointers == [b.untyped_storage().data_ptr() for b in buffers]
        assert metadata(rows) == original_metadata
        assert all(torch.all(raw == 41 + index).item()
                   for index, raw in enumerate(buffers))
        records.append({"cycle": cycle + 1, "first_wake": first,
                        "repeated_wake": repeated,
                        "metadata_preserved": True, "pattern_exact": True,
                        "repeated_wake_pointer_unchanged": True})
    instance.sleep()
    del instance, rows, buffers
    clean()
    return {"pools": pools, "views": views, "pool_mib": pool_mib,
            "cycles": records, "final_allocated": torch.npu.memory_allocated()}


class FailOnceStorage:
    def __init__(self, storage):
        self.storage = storage
        self.failed = False

    def nbytes(self):
        return self.storage.nbytes()

    def resize_(self, size):
        if size and not self.failed:
            self.failed = True
            raise MemoryError("injected second-storage failure")
        self.storage.resize_(size)


def retry_case(harness, root):
    buffers = [torch.empty(8 * MIB, dtype=torch.uint8, device="npu") for _ in range(2)]
    rows = make_views(buffers, 8)
    instance = worker(harness, root, rows, buffers)
    instance.sleep()
    storage, size = instance._kv_storage_snapshot[1]
    instance._kv_storage_snapshot[1] = (FailOnceStorage(storage), size)
    try:
        instance.wake_up(["kv_cache"])
    except MemoryError as error:
        assert "injected" in str(error)
    else:
        raise AssertionError("restore failure was not injected")
    instance.model_runner.reregister_kv_caches.assert_not_called()
    pointer = buffers[0].untyped_storage().data_ptr()
    assert pointer and buffers[1].untyped_storage().nbytes() == 0
    retry = measure_wake(instance)
    assert buffers[0].untyped_storage().data_ptr() == pointer
    assert retry["peak_delta"] <= 8 * MIB + 2 * MIB, retry
    assert buffers[1].untyped_storage().nbytes() == 8 * MIB
    instance.model_runner.reregister_kv_caches.assert_called_once_with()
    instance.sleep()
    del instance, rows, buffers, storage
    clean()
    return {"failure": "injected, not real OOM", "retry": retry,
            "first_storage_pointer_preserved": True,
            "final_allocated": torch.npu.memory_allocated()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    harness = load_module(
        "kv_storage_harness",
        args.root / "tests/unit/platform/test_shared_kv_storage_lifecycle.py",
    )
    source_paths = [args.root / "omni/worker/npu_worker.py",
                    args.root / "omni/worker/kv_storage.py"]
    report = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": args.device, "model_and_transport_hooks": "stubbed",
        "source_sha256": {str(p.relative_to(args.root)):
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    if args.legacy_root:
        report["legacy"] = layout_case(harness, args.legacy_root, pools=1,
                                       views=246, pool_mib=64, cycles=1, legacy=True)
    report["one_storage"] = layout_case(harness, args.root, pools=1,
                                        views=246, pool_mib=64, cycles=3)
    report["five_storages"] = layout_case(harness, args.root, pools=5,
                                          views=92, pool_mib=8, cycles=3)
    report["partial_restore_retry"] = retry_case(harness, args.root)
    assert all(report[k]["final_allocated"] == 0 for k in
               ["one_storage", "five_storages", "partial_restore_retry"])
    report["passed"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
