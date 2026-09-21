# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Execute the original Triton bodies on CPU; real NPU tests remain separate."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class _Pointer:
    """Element-addressed tensor storage, including non-contiguous input views."""

    def __init__(self, tensor, offset=None):
        self.tensor = tensor
        self.offset = tensor.storage_offset() if offset is None else offset

    def __add__(self, offset):
        return _Pointer(self.tensor, self.offset + offset)

    def storage(self):
        size = self.tensor.untyped_storage().nbytes() // self.tensor.element_size()
        return self.tensor.as_strided((size,), (1,), storage_offset=0)


class _Language(ModuleType):
    """Only the tl operations used by these kernels, evaluated with CPU tensors."""

    constexpr = object
    float32 = torch.float32
    arange = staticmethod(torch.arange)
    exp = staticmethod(torch.exp)
    log = staticmethod(torch.log)
    div_rn = staticmethod(torch.div)

    def program_id(self, axis):
        return self.grid_index[axis]

    @staticmethod
    def where(condition, left, right):
        return torch.where(torch.as_tensor(condition), left, right)

    @staticmethod
    def minimum(left, right):
        return torch.minimum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def maximum(left, right):
        return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def max(value, axis):
        return value.max(dim=axis).values

    @staticmethod
    def min(value, axis):
        return value.min(dim=axis).values

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis)

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, mask = torch.broadcast_tensors(
            torch.as_tensor(pointer.offset), torch.as_tensor(mask))
        result = torch.full(offsets.shape, other, dtype=pointer.tensor.dtype)
        addresses = offsets[mask]
        storage = pointer.storage()
        assert bool(((addresses >= 0) & (addresses < storage.numel())).all())
        result[mask] = storage[addresses]
        return result

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, value, mask = torch.broadcast_tensors(
            torch.as_tensor(pointer.offset), torch.as_tensor(value),
            torch.as_tensor(mask))
        addresses = offsets[mask]
        storage = pointer.storage()
        assert bool(((addresses >= 0) & (addresses < storage.numel())).all())
        storage[addresses] = value[mask].to(storage.dtype)


@pytest.fixture
def kernels(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "omni/sample/ops/draft_sampling.py"
    language = _Language("triton.language")
    triton = ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.language = language
    triton_utils = ModuleType("vllm.triton_utils")
    triton_utils.triton = triton
    triton_utils.tl = language
    spec = importlib.util.spec_from_file_location("_draft_sampling_cpu_test", source)
    module = importlib.util.module_from_spec(spec)
    # Isolate imports: never replace or reload the production module, and restore
    # sys.modules before yielding so other CPU/NPU tests keep their real runtime.
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "vllm.triton_utils", triton_utils)
        imports.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
        spec.loader.exec_module(module)

    def launch(fn, rows, tiles, *args):
        args = tuple(_Pointer(a) if isinstance(a, torch.Tensor) else a for a in args)
        for row in range(rows):
            for tile in range(tiles):
                language.grid_index = (row, tile)
                fn(*args)

    return module, launch


@pytest.mark.parametrize("filtering", ["none", "k", "p", "both"])
def test_prepare_kernel_scales_strides_and_expands_filters(kernels, filtering):
    module, launch = kernels
    rows, vocab, slots, block = 4, 9, 2, 4
    x = torch.arange(rows * vocab * 2, dtype=torch.float32).reshape(rows, -1)[:, ::2]
    # Non-unit strides and non-zero offsets in request metadata too.
    temperatures = torch.tensor([-1., 0., -1., 0.5])[1::2]
    k = torch.tensor([-1, 1, -1, 7], dtype=torch.int32)[1::2] if filtering in ("k", "both") else None
    p = torch.tensor([-1., 0.8, -1., 0.9])[1::2] if filtering in ("p", "both") else None
    enabled = filtering != "none"
    out = torch.full((rows, vocab), float("nan"), dtype=torch.bfloat16 if enabled else torch.float32)
    ko = torch.full((rows,), -1, dtype=torch.int32)
    po = torch.full((rows,), float("nan"), dtype=out.dtype)
    before = x.clone()
    launch(module._prepare_draft_logits, rows, 3,
           x, temperatures, k, p, out, ko, po,
           *x.stride(), temperatures.stride(0),
           k.stride(0) if k is not None else 0,
           p.stride(0) if p is not None else 0,
           vocab, slots, k is not None, p is not None, enabled, block)
    scale = torch.tensor([1., 1., 0.5, 0.5])
    torch.testing.assert_close(out, (x / scale[:, None]).to(out.dtype), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    if enabled:
        expected_k = k.repeat_interleave(slots) if k is not None else torch.full_like(ko, vocab)
        expected_p = p.repeat_interleave(slots).to(po.dtype) if p is not None else torch.ones_like(po)
        torch.testing.assert_close(ko, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(po, expected_p, rtol=0, atol=0)
    else:
        assert (ko == -1).all() and po.isnan().all()


def _sample(kernels, x, temperature, noise, scale, all_random):
    module, launch = kernels
    rows, vocab = x.shape
    block = 4
    tiles = (vocab + block - 1) // block
    stats = torch.full((3, rows, tiles), float("nan"))
    indices = torch.full((rows, tiles), -1, dtype=torch.int32)
    q = torch.full_like(x, float("nan"), dtype=torch.float32)
    ids = torch.full((rows,), -1, dtype=torch.int64)
    shared = (*x.stride(), temperature.stride(0), vocab,
              rows // temperature.numel(), tiles, rows * tiles)
    launch(module._draft_stats_and_candidates, rows, tiles,
           x, noise, temperature, stats, indices, *shared,
           all_random, 1e-5, scale, block)
    launch(module._draft_normalize_and_select, rows, tiles,
           x, temperature, stats, indices, q, ids, *shared,
           scale, block, 1 << (tiles - 1).bit_length())
    return ids, q


@pytest.mark.parametrize("scale,all_random", [(False, True), (False, False), (True, True), (True, False)])
def test_reduction_kernels_match_dense_reference(kernels, scale, all_random):
    x = torch.linspace(-3, 3, 72).reshape(4, 18)[:, ::2]
    t = torch.tensor([0., 0.7])
    u = torch.linspace(0.02, 0.98, 36).reshape(4, 9)
    before, noise_before = x.clone(), u.clone()
    ids, q = _sample(kernels, x, t, u, scale, all_random)
    expanded = t.repeat_interleave(2)
    logits = x / torch.where(expanded == 0, 1., expanded)[:, None] if scale else x
    expected_q = logits.softmax(-1)
    expected_ids = (expected_q / -(1 - u).log()).argmax(-1)
    if not all_random:
        expected_ids = torch.where(expanded < 1e-5, logits.argmax(-1), expected_ids)
    torch.testing.assert_close(q, expected_q, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(u, noise_before, rtol=0, atol=0)


def test_reduction_empty_tiles_endpoints_and_cross_tile_tie(kernels):
    x = torch.full((3, 9), -float("inf"))
    x[:, 1] = x[:, 8] = 0
    u = torch.zeros_like(x)
    u[2].fill_(1.)
    ids, q = _sample(kernels, x, torch.tensor([1., 0., 1.]), u, False, False)
    torch.testing.assert_close(q, x.softmax(-1), rtol=0, atol=0)
    assert ids.tolist() == [1, 1, 1]


def test_reduction_large_offset_preserves_noise(kernels):
    x = torch.full((1, 9), 1e20)
    u = torch.full_like(x, 0.5)
    u[0, 8] = 0
    ids, q = _sample(kernels, x, torch.ones(1), u, False, True)
    assert ids.item() == 8
    torch.testing.assert_close(q, x.softmax(-1))


def test_empty_batch_returns_without_launching_kernels(kernels):
    module, _ = kernels
    x = torch.empty(0, 9)
    # Missing other metadata deliberately: empty batches must return before
    # reading filters/RNG or launching any kernels.
    ids, q = module.sample_draft_logits(
        x, SimpleNamespace(temperature=torch.ones(1)), torch.bfloat16, 1e-5)
    assert ids.shape == (0,) and ids.dtype == torch.int64
    assert q.shape == (0, 9) and q.dtype == torch.float32
    assert ids.device.type == q.device.type == "cpu"
