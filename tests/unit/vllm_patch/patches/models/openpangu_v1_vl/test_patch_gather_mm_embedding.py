# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU tests for gathering multimodal embeddings with placeholder indices.

The upstream runner is an interface double. Tensor operations and patch
registration/application are real.
These tests do not exercise device transfers or asynchronous execution on NPU.
Use --confcutdir=tests/unit/vllm_patch/patches/models to exclude NPU fixtures.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = pytest.mark.unit


class _CpuBufferPair:
    """CPU implementation of the runner's host/device buffer interface."""

    def __init__(self, size, dtype):
        self.cpu = torch.empty(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = torch.empty(size, dtype=dtype)

    def copy_to_gpu(self, count):
        self.gpu[:count].copy_(self.cpu[:count], non_blocking=True)
        return self.gpu[:count]


class _NoDynamicMaskIndices(TorchDispatchMode):
    """Require gather to extract mask positions on CPU through NumPy."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        assert "nonzero" not in name, f"Unexpected dynamic index discovery: {func}"
        return func(*args, **(kwargs or {}))


@pytest.fixture
def gather_patch(patch_env):
    def load(*, native_buffers=True):
        class GPUModelRunner:
            def __init__(self):
                self.max_num_tokens = 16
                self.device = torch.device("cpu")
                self.pin_memory = False
                self.embeddings = []
                self.mask = torch.empty(0, dtype=torch.bool)
                self.upstream_calls = []
                self.buffer_requests = []
                self.upstream_gather = None

            def _gather_mm_embeddings(self, scheduler_output, shift_computed_tokens=0):
                self.upstream_calls.append((scheduler_output, shift_computed_tokens))
                if self.upstream_gather is not None:
                    return self.upstream_gather(self, scheduler_output)
                return self.embeddings, self.mask

        if native_buffers:
            def make_buffer(self, size, dtype):
                self.buffer_requests.append((size, dtype))
                return _CpuBufferPair(size, dtype)

            GPUModelRunner._make_buffer = make_buffer

        patch_env.stub(
            "vllm.v1.worker.gpu_model_runner",
            GPUModelRunner=GPUModelRunner,
            PIN_MEMORY=False,
        )
        module = patch_env.load("patch_gather_mm_embedding")
        patch = patch_env.registered["GPUModelRunnerGatherMMEmbeddings"]
        assert patch is module.GPUModelRunnerGatherMMEmbeddingsPatch
        patch.apply()
        return SimpleNamespace(module=module, runner=GPUModelRunner())

    return load


@pytest.mark.parametrize(
    "has_method,method",
    [(False, None), (True, None), (True, 42)],
    ids=["missing", "none", "not-callable"],
)
def test_incompatible_upstream_warns_and_skips_registration(
    patch_env, caplog, has_method, method,
):
    class GPUModelRunner:
        pass

    if has_method:
        GPUModelRunner._gather_mm_embeddings = method
    original_attributes = dict(vars(GPUModelRunner))
    patch_env.stub(
        "vllm.v1.worker.gpu_model_runner", GPUModelRunner=GPUModelRunner
    )

    with caplog.at_level(logging.WARNING):
        module = patch_env.load("patch_gather_mm_embedding")

    assert "GPUModelRunnerGatherMMEmbeddings" not in patch_env.registered
    assert dict(vars(GPUModelRunner)) == original_attributes
    warnings = [record for record in caplog.records if record.name == module.__name__]
    assert len(warnings) == 1
    assert "Skipping GPUModelRunnerGatherMMEmbeddings patch" in warnings[0].message
    assert "GPUModelRunner._gather_mm_embeddings" in warnings[0].message


@pytest.mark.parametrize("native_buffers", [True, False])
@pytest.mark.parametrize(
    "flags,count,strided,expected",
    [
        ([False, True, False, True, True, False], 6, False, [1, 3, 4]),
        ([False, True, False, True, True, False], 6, True, [1, 3, 4]),
        ([False, False, False, False], 4, False, []),
        ([True, True, True, True], 4, False, [0, 1, 2, 3]),
        ([], 0, False, []),
        ([False, True, False, True, True, True], 4, False, [1, 3]),
    ],
    ids=["sparse", "noncontiguous", "all-false", "all-true", "empty", "dirty-tail"],
)
def test_cpu_mask_yields_ordered_indices_without_touching_mask_buffers(
    gather_patch, native_buffers, flags, count, strided, expected,
):
    runner = gather_patch(native_buffers=native_buffers).runner
    if strided:
        storage = torch.tensor(
            [item for flag in flags for item in (flag, not flag)],
            dtype=torch.bool,
        )
        runner.mask = storage[::2]
        assert not runner.mask.is_contiguous()
    else:
        runner.mask = torch.tensor(flags, dtype=torch.bool)
    features = torch.arange(len(expected) * 3).reshape(-1, 3).float()
    runner.embeddings = [features[:1], features[1:]]
    original_mask = runner.mask.clone()

    # Deliberately conflicting old mask buffers must not influence the CPU path.
    runner.is_mm_embed_idx = 1
    runner.is_mm_embed_buffers = [
        _CpuBufferPair(runner.max_num_tokens, torch.bool) for _ in range(2)
    ]
    for buffer in runner.is_mm_embed_buffers:
        buffer.cpu.fill_(True)
        buffer.gpu.fill_(False)
    old_buffers = runner.is_mm_embed_buffers

    with _NoDynamicMaskIndices():
        embeddings, mask = runner._gather_mm_embeddings(
            SimpleNamespace(total_num_scheduled_tokens=count)
        )

    assert mask is runner.mask
    assert isinstance(embeddings, list)
    assert embeddings is not runner.embeddings
    assert len(embeddings) == len(runner.embeddings)
    for actual, original in zip(embeddings, runner.embeddings):
        assert actual is original
    indices = embeddings.placeholder_indices
    assert indices.dtype == torch.int64
    assert indices.device == runner.device
    assert indices.tolist() == expected
    torch.testing.assert_close(mask, original_mask)
    assert runner.is_mm_embed_buffers is old_buffers
    assert runner.is_mm_embed_idx == 1
    for buffer in old_buffers:
        assert bool(buffer.cpu.all())
        assert not bool(buffer.gpu.any())


@pytest.mark.parametrize("shift", [0, 1, 7])
def test_patch_calls_original_once_and_forwards_scheduler_and_shift(
    gather_patch, shift,
):
    runner = gather_patch().runner
    scheduler = SimpleNamespace(total_num_scheduled_tokens=0)

    if shift == 0:
        embeddings, mask = runner._gather_mm_embeddings(scheduler)
    else:
        embeddings, mask = runner._gather_mm_embeddings(
            scheduler, shift_computed_tokens=shift
        )

    assert runner.upstream_calls == [(scheduler, shift)]
    assert embeddings == []
    assert embeddings.placeholder_indices.shape == (0,)
    assert embeddings.placeholder_indices.dtype == torch.int64
    assert mask is runner.mask


@pytest.mark.parametrize("native_buffers", [True, False])
def test_integer_index_buffers_rotate_and_reuse_independently(
    gather_patch, native_buffers,
):
    runner = gather_patch(native_buffers=native_buffers).runner
    runner.is_mm_embed_idx = 1
    scheduler = SimpleNamespace(total_num_scheduled_tokens=5)
    outputs = []

    for flags, mask_buffer_idx, expected in [
        ([True, False, True, False, True], 1, [0, 2, 4]),
        ([False, True, False, True, False], 1, [1, 3]),
        ([False, False, True, False, False], 0, [2]),
    ]:
        runner.mask = torch.tensor(flags)
        runner.is_mm_embed_idx = mask_buffer_idx
        embeddings, _ = runner._gather_mm_embeddings(scheduler)
        assert embeddings.placeholder_indices.tolist() == expected
        outputs.append(embeddings.placeholder_indices)
        assert runner.is_mm_embed_idx == mask_buffer_idx
        if len(outputs) == 2:
            assert outputs[0].tolist() == [0, 2, 4]

    assert outputs[0].data_ptr() != outputs[1].data_ptr()
    assert outputs[0].data_ptr() == outputs[2].data_ptr()
    assert len(runner.mm_embed_index_buffers) == 2
    assert runner.mm_embed_index_idx == 1
    if native_buffers:
        assert runner.buffer_requests == [
            (runner.max_num_tokens, torch.bool),
            (runner.max_num_tokens, torch.bool),
            (runner.max_num_tokens, torch.int64),
            (runner.max_num_tokens, torch.int64),
        ]


def test_legacy_device_mask_uses_current_cpu_mirror_without_device_readback(
    gather_patch,
):
    runner = gather_patch().runner
    scheduler = SimpleNamespace(total_num_scheduled_tokens=5)
    # A meta tensor supports no host readback: .cpu()/.numpy() would fail.
    runner.mask = torch.empty(7, device="meta", dtype=torch.bool)
    expected_buffers = []

    def gather(current, scheduler_output):
        buffer_idx, flags = expected_buffers.pop(0)
        current.is_mm_embed_idx = buffer_idx
        current.is_mm_embed_buffers[buffer_idx].cpu.fill_(True)
        current.is_mm_embed_buffers[buffer_idx].cpu[:5].copy_(torch.tensor(flags))
        return current.embeddings, current.mask

    runner.upstream_gather = gather
    index_pointers = []
    for mask_buffer_idx, flags, expected in [
        (1, [False, True, False, True, False], [1, 3]),
        (1, [True, False, False, False, True], [0, 4]),
        (0, [False, False, True, False, False], [2]),
    ]:
        expected_buffers.append((mask_buffer_idx, flags))
        with _NoDynamicMaskIndices():
            embeddings, mask = runner._gather_mm_embeddings(scheduler)
        assert mask is runner.mask
        assert mask.device.type == "meta"
        assert runner.is_mm_embed_idx == mask_buffer_idx
        assert embeddings.placeholder_indices.tolist() == expected
        index_pointers.append(embeddings.placeholder_indices.data_ptr())

    assert index_pointers[0] != index_pointers[1]
    assert index_pointers[0] == index_pointers[2]


@pytest.mark.parametrize("native_buffers", [True, False])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64])
def test_buffer_factory_preserves_dtype_numpy_view_and_copy_length(
    gather_patch, native_buffers, dtype,
):
    env = gather_patch(native_buffers=native_buffers)
    buffer = env.module._make_multimodal_buffer(env.runner, dtype)
    assert buffer.cpu.shape == (env.runner.max_num_tokens,)
    assert buffer.gpu.shape == buffer.cpu.shape
    assert buffer.cpu.dtype == buffer.gpu.dtype == dtype
    assert buffer.cpu.device.type == "cpu"
    assert buffer.gpu.device == env.runner.device

    buffer.cpu.zero_()
    buffer.gpu.zero_()
    buffer.np[:3] = [1, 0, 1]
    assert buffer.cpu[:3].tolist() == [1, 0, 1]
    buffer.cpu[3] = 1
    assert buffer.np[3] == 1
    copied = buffer.copy_to_gpu(3)
    assert copied.shape == (3,)
    assert copied.data_ptr() == buffer.gpu.data_ptr()
    assert copied.tolist() == [1, 0, 1]
    assert not bool(buffer.gpu[3:].any())
    assert buffer.copy_to_gpu(0).shape == (0,)
    if native_buffers:
        assert env.runner.buffer_requests == [(env.runner.max_num_tokens, dtype)]
