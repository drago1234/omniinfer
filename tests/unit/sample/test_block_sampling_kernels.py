# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import os
import subprocess
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch


@pytest.fixture
def block_kernels(monkeypatch):
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from triton.runtime.interpreter import InterpretedFunction

    utils = ModuleType("vllm.triton_utils")
    utils.triton, utils.tl = triton, tl
    source = Path(__file__).resolve().parents[3] / "omni/sample/ops"
    modules = []
    with monkeypatch.context() as imports:
        imports.setenv("TRITON_INTERPRET", "1")
        imports.setitem(sys.modules, "vllm.triton_utils", utils)
        for filename in ("block_acceptance", "block_selection"):
            spec = importlib.util.spec_from_file_location("_cpu_test_" + filename, source / (filename + ".py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            modules.append(mod)
    assert isinstance(modules[0]._compute_block_prefix_kernel, InterpretedFunction)
    return modules


class TestBlockKernelCases:
    # tl reductions must also be imported with interpretation enabled. Isolate
    # this from vLLM tests that already imported compiled JIT functions.
    __test__ = os.environ.get("OMNI_BLOCK_KERNEL_UT_CHILD") == "1"

    def test_acceptance_and_selection(self, block_kernels):
        acceptance, selection = block_kernels
        # Four requests: later-prefix success, no drafts, zero target mass,
        # and a greedy row whose existing output must remain untouched.
        ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        cu = torch.tensor([2, 2, 3, 5], dtype=torch.int32)
        p = torch.tensor([[.25, .75], [.4, .6], [0., 1.], [.2, .8], [.2, .8]])
        q = torch.tensor([[.5, .5], [.8, .2], [1., 0.], [0., 0.], [0., 0.]])
        greedy = torch.tensor([False, False, False, True])
        tau, accepted, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, q, p, torch.tensor([.9, .2, 0., .5, .5]), 2, greedy,
        )
        # Request 0 has c=(1/2,1/4), h=(1/6,1/4), so tau=2 despite u_1>h_1.
        assert tau.tolist() == [2, 0, 0, 0]
        assert accepted.tolist() == [True, True, False, False, False]
        assert torch.allclose(scale, torch.tensor([1., .5, 1., 1., 1.]), rtol=2e-5, atol=2e-6)
        output = torch.full((4, 3), 99, dtype=torch.int32)
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32),
            torch.tensor([[10], [11], [12], [13]], dtype=torch.int32), ~greedy, 2, -1,
        )
        assert output.tolist() == [[0, 0, 10], [11, -1, -1], [6, -1, -1], [99, 99, 99]]
        assert result.dtype == torch.int64 and torch.equal(result, tau.long())

    def test_partial_residual_mass_covers_tail_tile(self, block_kernels):
        acceptance, _ = block_kernels
        vocab = 4101
        p = torch.full((2, vocab), 0.1 / 4096)
        p[:, 4096:] = 0.18
        q = torch.full_like(p, 1.0 / vocab)
        # A non-unit vocabulary stride must also be honored by the kernel.
        p = torch.stack((p, p), -1)[..., 0]
        partial = acceptance.compute_block_partial_z_triton(
            q, p, torch.tensor([0.5, 0.25]), torch.tensor([2]), 2,
        )
        residual = (0.5 * p[1] - q[1]).clamp_min(0)
        expected = torch.stack((residual[:4096].sum(), residual[4096:].sum()))
        assert partial.shape == (1, 1, 2) and expected[1] > 0
        assert torch.allclose(partial[0, 0], expected, rtol=2e-5, atol=2e-6)

    def test_empty_drafts_preserve_greedy_output(self, block_kernels, monkeypatch):
        acceptance, selection = block_kernels
        monkeypatch.setattr(acceptance, "_finalize_block_acceptance_kernel", object())
        monkeypatch.setattr(selection, "_select_block_tokens_kernel", object())
        ids = torch.empty(0, dtype=torch.int32)
        cu = torch.zeros(3, dtype=torch.int32)
        tau, mask, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, torch.empty(0, 2), torch.empty(0, 2), torch.empty(0), 0,
        )
        assert tau.tolist() == [0, 0, 0] and mask.numel() == scale.numel() == 0
        output = torch.full((3, 3), 99, dtype=torch.int32)
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, ids, torch.tensor([[10], [11], [12]], dtype=torch.int32),
            torch.tensor([True, False, True]), 2, -1,
        )
        assert result.tolist() == [0, 0, 0]
        assert output.tolist() == [[10, -1, -1], [99, 99, 99], [12, -1, -1]]


def test_block_kernels_in_cpu_interpreter(tmp_path):
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", OMNI_BLOCK_KERNEL_UT_CHILD="1",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "--noconftest", "-q", "--tb=short", "--junitxml=" + str(tmp_path / "kernels.xml")],
        env=env, capture_output=True, text=True, timeout=120,
    )
    (tmp_path / "kernels.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr


test_block_kernels_in_cpu_interpreter.__test__ = os.environ.get("OMNI_BLOCK_KERNEL_UT_CHILD") != "1"
