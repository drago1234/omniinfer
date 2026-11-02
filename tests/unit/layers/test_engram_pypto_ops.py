# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU allocation/return contracts; real kernel replay is tested on the NPU."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

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
    # Keep the package name so --cov=omni_npu traces this dynamically loaded code.
    name = "omni_npu.layers.ops.ai_infra_pypto_engram_proj_gate"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
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


def test_inference_uses_minimal_placeholders_and_preserves_kernel_output(wrapper, monkeypatch):
    args = inputs()
    expected = torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 2, 4) - 17

    def kernel(*args):
        assert args[-1] is False
        for placeholder in args[6:10]:
            assert placeholder.shape == (1,)
        assert args[0].shape == (8, 2, 4)
        torch.testing.assert_close(args[0][:6], inputs()[0].reshape(6, 2, 4))
        torch.testing.assert_close(args[1][:6], inputs()[1].reshape(6, 8))
        assert torch.count_nonzero(args[0][6:]) == 0
        assert torch.count_nonzero(args[1][6:]) == 0
        args[10].fill_(1234)
        args[10][:6].copy_(expected.reshape(6, 2, 4))

    compiled = Mock(side_effect=kernel)
    monkeypatch.setattr(wrapper, "ai_infra_engram_proj_gate_kernel", compiled)
    output, *caches = wrapper.ai_infra_engram_proj_gate(*args, return_cache=False)

    compiled.assert_called_once()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert caches == [None] * 4


@pytest.mark.parametrize("tokens,expected", [
    (1, 1), (3, 4), (1024, 1024),
    (1025, 1536), (1536, 1536), (1537, 2048), (2048, 2048),
    (2049, 2560), (2561, 3072), (3073, 3584),
])
def test_inference_bucket_bounds_padding(wrapper, tokens, expected):
    assert wrapper.inference_token_bucket(tokens) == expected


def test_warmup_covers_bucket_range_and_reuses_plans_for_another_layer(wrapper, monkeypatch):
    calls = []
    monkeypatch.setattr(wrapper, "ai_infra_engram_proj_gate", lambda *args, **kwargs: calls.append(args[0].shape[1]))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    key = torch.zeros(2, 8, 4, dtype=torch.bfloat16)
    value = torch.zeros(8, 4, dtype=torch.bfloat16)
    gamma = torch.ones(2, 4, dtype=torch.bfloat16)
    wrapper.warmup_engram_proj_gate(key, value, gamma, gamma, 3)
    assert calls == [1, 2, 4]
    wrapper.warmup_engram_proj_gate(key, value, gamma, gamma, 2560)
    assert calls == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560]
    assert set(calls) == {wrapper.inference_token_bucket(n) for n in range(1, 2561)}
    wrapper.warmup_engram_proj_gate(key.clone(), value.clone(), gamma, gamma, 2560)
    assert len(calls) == 14
    wrapper.warmup_engram_proj_gate(key, value, gamma, gamma, 4096)
    assert calls[-3:] == [3072, 3584, 4096] and len(calls) == 17


@pytest.mark.parametrize("failure", ["launch", "completion"])
def test_warmup_retries_all_buckets_after_failure(wrapper, monkeypatch, failure):
    calls = []

    def run(query, *args, **kwargs):
        calls.append(query.shape[1])
        if failure == "launch" and len(calls) == 2:
            raise RuntimeError("warmup failed")

    synchronize = Mock(side_effect=[RuntimeError("warmup failed"), None]) if failure == "completion" else Mock()
    monkeypatch.setattr(wrapper, "ai_infra_engram_proj_gate", run)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    _, _, key, value, gamma_key, gamma_query = inputs()
    with pytest.raises(RuntimeError, match="warmup failed"):
        wrapper.warmup_engram_proj_gate(key, value, gamma_key, gamma_query, 3)
    failed_calls = len(calls)

    wrapper.warmup_engram_proj_gate(key, value, gamma_key, gamma_query, 3)

    assert calls[failed_calls:] == [1, 2, 4]


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_warmup_finds_nested_fused_engram_and_uses_local_token_limit(wrapper, monkeypatch, sequence_parallel):
    from unittest.mock import Mock, call
    from omni_npu.layers import engram

    warmup = Mock()
    monkeypatch.setattr(wrapper, "warmup_engram_proj_gate", warmup)

    def fusion(enabled):
        layer = engram.PanGuEngramFusion.__new__(engram.PanGuEngramFusion)
        torch.nn.Module.__init__(layer)
        layer.use_engram_proj_gate_fusion_op = enabled
        layer.use_sequence_parallel = sequence_parallel
        layer.embed_tokens = SimpleNamespace(model_tp_size=8)
        for name in ("proj_weights_key", "proj_weights_value", "k_weight", "q_weight"):
            setattr(layer, name, object())
        layer.clamp_value, layer.eps = 1e-6, 1e-5
        return layer

    first, second, unfused = fusion(True), fusion(True), fusion(False)
    model = torch.nn.Module()
    model.language_model = torch.nn.Sequential(
        torch.nn.Linear(2, 2), first, torch.nn.Sequential(second, unfused)
    )
    # Shared module aliases must not cause duplicate warmup.
    model.shared_alias = first
    wrapper.warmup_engram(model, 20481)

    local_tokens = 2561 if sequence_parallel else 20481
    assert warmup.call_args_list == [
        call(layer.proj_weights_key, layer.proj_weights_value, layer.k_weight, layer.q_weight,
             local_tokens, clamp_value=1e-6, eps=1e-5)
        for layer in (first, second)
    ]


def test_training_preserves_cache_shapes_dtypes_and_return_order(wrapper, monkeypatch):
    def kernel(*args):
        assert args[-1] is True
        assert args[0].shape == (8, 2, 4)
        assert torch.count_nonzero(args[0][6:]) == 0
        assert torch.count_nonzero(args[1][6:]) == 0
        for index, tensor in enumerate(args[6:11], start=1):
            assert tensor.shape[0] == 8
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


@pytest.mark.parametrize("tokens,width,return_cache", [
    (1, 32, False), (129, 32, False), (3, 1600, True),
])
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
    pypto.matmul = lambda left, right, dtype: pypto.cast(left.float() @ right.float(), dtype)
    pypto.maximum = torch.clamp_min
    pypto.abs, pypto.sqrt, pypto.sign, pypto.sum = torch.abs, torch.sqrt, torch.sign, torch.sum
    pypto.div = lambda left, right, *args: left / right
    generator = torch.Generator().manual_seed(71)
    shapes = [(1, tokens, 2, width), (1, tokens, 8), (2, 8, width), (8, width), (2, width), (2, width)]
    args = [torch.randn(shape, generator=generator).bfloat16() for shape in shapes]
    hidden, memory, key, value, gamma_key, gamma_query = [x.float() for x in args]
    # Projection outputs are rounded to BF16 before FP32 normalization/gating.
    keys = torch.einsum("bnd,mdh->bnmh", memory, key).bfloat16().float()
    values = (memory @ value).bfloat16().float()
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
