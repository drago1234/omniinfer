# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU allocation/return contracts; real kernel replay is tested on the NPU."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def wrapper(monkeypatch):
    # Load the actual wrapper without importing a PyPTO runtime or compiling DSL.
    pypto = ModuleType("pypto")
    pypto.frontend = SimpleNamespace(jit=lambda **kwargs: lambda function: function)
    pypto.Tensor = lambda *args: None
    pypto.DYNAMIC, pypto.DT_BF16, pypto.DT_FP32 = -1, "bf16", "fp32"
    monkeypatch.setitem(sys.modules, "pypto", pypto)
    path = Path(__file__).resolve().parents[3] / "omni/layers/ops/ai_infra_pypto_engram_proj_gate.py"
    spec = importlib.util.spec_from_file_location("engram_pypto_wrapper_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    return (
        torch.ones(2, 3, 2, 4, dtype=torch.bfloat16),
        torch.ones(2, 3, 8, dtype=torch.bfloat16),
        torch.ones(2, 8, 4, dtype=torch.bfloat16),
        torch.ones(8, 4, dtype=torch.bfloat16),
        torch.ones(2, 4, dtype=torch.bfloat16),
        torch.ones(2, 4, dtype=torch.bfloat16),
    )


def test_inference_does_not_zero_buffers_and_preserves_kernel_output(wrapper, monkeypatch):
    args = inputs()
    expected = torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 2, 4) - 17
    empty = torch.empty

    def poisoned_empty(*shape, **kwargs):
        return empty(*shape, **kwargs).fill_(float("nan"))

    def kernel(*args):
        assert args[-1] is False
        for placeholder in args[6:10]:
            assert placeholder.shape == (1,)
            assert placeholder.isnan().all()
        args[10].copy_(expected.reshape(6, 2, 4))

    compiled = Mock(side_effect=kernel)
    monkeypatch.setattr(wrapper, "ai_infra_engram_proj_gate_kernel", compiled)
    with patch.object(torch, "zeros", side_effect=AssertionError("Inference must not zero buffers")), patch.object(
        torch, "empty", side_effect=poisoned_empty
    ):
        output, *caches = wrapper.ai_infra_engram_proj_gate(*args, return_cache=False)

    compiled.assert_called_once()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert caches == [None] * 4


def test_training_preserves_cache_shapes_dtypes_and_return_order(wrapper, monkeypatch):
    def kernel(*args):
        assert args[-1] is True
        for index, tensor in enumerate(args[6:11], start=1):
            tensor.fill_(index)

    monkeypatch.setattr(wrapper, "ai_infra_engram_proj_gate_kernel", kernel)
    outputs = wrapper.ai_infra_engram_proj_gate(*inputs(), return_cache=True)
    expected = [
        ((2, 3, 2, 4), torch.bfloat16, 5),
        ((2, 3, 2), torch.float32, 3),
        ((2, 3, 2, 4), torch.bfloat16, 2),
        ((2, 3, 4), torch.bfloat16, 1),
        ((2, 3, 2), torch.float32, 4),
    ]
    assert len(outputs) == len(expected)
    for actual, (shape, dtype, value) in zip(outputs, expected):
        assert actual.shape == shape and actual.dtype == dtype
        torch.testing.assert_close(actual, torch.full(shape, value, dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["empty", "strided"])
def test_wrapper_rejects_invalid_inputs_before_launch(wrapper, monkeypatch, invalid):
    args = list(inputs())
    args[0] = args[0][:, :0] if invalid == "empty" else args[0][..., ::2]
    kernel = Mock()
    monkeypatch.setattr(wrapper, "ai_infra_engram_proj_gate_kernel", kernel)
    with pytest.raises(ValueError, match="must not be empty|must be contiguous"):
        wrapper.ai_infra_engram_proj_gate(*args)
    kernel.assert_not_called()


@pytest.mark.parametrize("tokens,width,return_cache", [(129, 32, False), (3, 1600, True)])
def test_kernel_math_and_tile_tails_match_independent_reference(wrapper, tokens, width, return_cache):
    # Execute the original DSL function with CPU tensor primitives. This checks
    # its math and loop tails; it does not validate PyPTO lowering or NPU execution.
    pypto = sys.modules["pypto"]
    pypto.experimental = SimpleNamespace(set_operation_options=lambda **kwargs: None)
    pypto.PrecisionType = SimpleNamespace(INTRINSIC=None)
    for name in ("set_semantic_label", "set_cube_tile_shapes", "set_vec_tile_shapes", "set_pass_options"):
        setattr(pypto, name, lambda *args, **kwargs: None)
    pypto.loop_unroll = lambda start, end, step, unroll_list: (
        (begin, min(unroll_list[0], end - begin)) for begin in range(start, end, unroll_list[0])
    )
    pypto.loop = range
    pypto.cast = lambda value, dtype: value.to({"bf16": torch.bfloat16, "fp32": torch.float32}[dtype])
    pypto.matmul = lambda left, right, dtype: left.float() @ right.float()
    pypto.maximum = torch.clamp_min
    pypto.abs, pypto.sqrt, pypto.sign, pypto.sum = torch.abs, torch.sqrt, torch.sign, torch.sum
    pypto.div = lambda left, right, *args: left / right
    generator = torch.Generator().manual_seed(71)
    shapes = [(1, tokens, 2, width), (1, tokens, 8), (2, 8, width), (8, width), (2, width), (2, width)]
    args = [torch.randn(shape, generator=generator).bfloat16() for shape in shapes]
    hidden, memory, key, value, gamma_key, gamma_query = [x.float() for x in args]
    keys = torch.einsum("bnd,mdh->bnmh", memory, key)
    values = memory @ value
    normalized_keys = torch.nn.functional.rms_norm(keys, (width,), eps=1e-5) * gamma_key
    queries = torch.nn.functional.rms_norm(hidden, (width,), eps=1e-5) * gamma_query
    scores = (normalized_keys * queries).sum(-1) / width ** 0.5
    gates = torch.sigmoid(scores.sign() * scores.abs().clamp_min(1e-6).sqrt())
    expected = (gates[..., None] * values[:, :, None]).bfloat16()
    output, *caches = wrapper.ai_infra_engram_proj_gate(*args, eps=1e-5, return_cache=return_cache)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-4)
    if return_cache:
        for actual, reference in zip(caches, (scores, keys.bfloat16(), values.bfloat16(), gates)):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    else:
        assert caches == [None] * 4
