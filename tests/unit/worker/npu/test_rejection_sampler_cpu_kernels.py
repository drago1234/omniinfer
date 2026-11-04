# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Execute production rejection kernels with Triton's CPU interpreter.

Reuse the NPU cases and independent probability/RNG oracles. This adds Python
line coverage; compilation and device behavior remain covered by NPU tests.
"""
import inspect
import os
from pathlib import Path
from types import FunctionType

import pytest

from tests.integration import test_mrv2_sampling_kernels as cases
from tests.integration.layers.engram_test_utils import run_isolated_pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def cpu_kernels(monkeypatch):
    from triton.runtime.interpreter import InterpretedFunction
    import triton
    import triton.language as tl

    assert os.environ.get("TRITON_INTERPRET") == "1"

    class CpuKernel(InterpretedFunction):
        def __getitem__(self, grid):
            launch = super().__getitem__(grid)

            def call(*args, **kwargs):
                # This Ascend compiler option has no interpreter equivalent.
                kwargs.pop("has_auto_blockify_blacklist_op", None)
                return launch(*args, **kwargs)

            return call

    converted = {}

    def interpret(kernel):
        if id(kernel) in converted:
            return converted[id(kernel)]
        original = kernel.fn
        # Preserve the original code object/filename for coverage, but isolate
        # globals: Ascend's interpreter must not patch torch host-side APIs.
        namespace = {"tl": tl, "triton": triton}
        function = FunctionType(original.__code__, namespace, original.__name__,
                                original.__defaults__)
        function.__annotations__ = inspect.get_annotations(original, eval_str=True)
        result = CpuKernel(function)
        converted[id(kernel)] = result
        for name in original.__code__.co_names:
            value = original.__globals__.get(name)
            if hasattr(value, "fn"):
                namespace[name] = interpret(value)
        return result

    for name, kernel in list(vars(cases.rejection).items()):
        if hasattr(kernel, "fn"):
            monkeypatch.setattr(cases.rejection, name, interpret(kernel))


@pytest.mark.usefixtures("cpu_kernels")
class TestRejectionSamplerCpuKernels:
    # Interpreter imports mutate Triton language objects; keep them in a child.
    __test__ = os.environ.get("OMNI_ENGRAM_TEST_CHILD") == "1"

    @pytest.mark.parametrize("block", [False, True])
    @pytest.mark.parametrize("kind", [
        "bonus", "reject", "no_draft_logits", "stochastic",
        "placeholder", "placeholder_no_draft_logits",
    ])
    def test_rejection_matches_existing_reference(self, block, kind):
        cases.TestMRv2KernelCases().test_rejection_and_bonus_against_cpu_distribution(
            block, kind, device="cpu",
        )

    @pytest.mark.parametrize("block", [False, True])
    def test_greedy_matches_existing_reference(self, block):
        cases.test_rejection_greedy_matches_target_argmax(block, device="cpu")


def test_rejection_cpu_kernels_in_subprocess(tmp_path, monkeypatch):
    # The runner inherits os.environ; pytest restores this after the test.
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    run_isolated_pytest(
        str(Path(__file__).resolve()) + "::TestRejectionSamplerCpuKernels",
        tmp_path, timeout=120, interpreter=True,
    )
