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
    import triton
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

    @pytest.mark.parametrize("uniform, first_tau, first_output", [
        ([.9, .2], 2, [0, 0, 10]),
        ([.1, .9], 1, [0, 5, -1]),
    ])
    def test_acceptance_and_selection(self, block_kernels, uniform, first_tau, first_output):
        acceptance, selection = block_kernels
        # Four requests: later-prefix success, no drafts, zero target mass,
        # and a greedy row whose existing output must remain untouched.
        ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        cu = torch.tensor([2, 2, 3, 5], dtype=torch.int32)
        p = torch.tensor([[.25, .75], [.4, .6], [0., 1.], [.2, .8], [.2, .8]])
        q = torch.tensor([[.5, .5], [.8, .2], [1., 0.], [0., 0.], [0., 0.]])
        greedy = torch.tensor([False, False, False, True])
        tau, accepted, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, q, p, torch.tensor(uniform + [0., .5, .5]), 2, greedy,
        )
        # Request 0 has c=(1/2,1/4), h=(1/6,1/4). A later prefix can
        # pass after an earlier failure; partial acceptance uses recovered[1].
        assert tau.tolist() == [first_tau, 0, 0, 0]
        assert accepted.tolist() == [True, first_tau == 2, False, False, False]
        assert torch.allclose(scale, torch.tensor([1., .5, 1., 1., 1.]), rtol=2e-5, atol=2e-6)
        output = torch.full((4, 3), 99, dtype=torch.int32)
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32),
            torch.tensor([[10], [11], [12], [13]], dtype=torch.int32), ~greedy, 2, -1,
        )
        assert output.tolist() == [first_output, [11, -1, -1], [6, -1, -1], [99, 99, 99]]
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

    @pytest.mark.parametrize("batch", [0, 2])
    def test_single_draft_and_empty_batch(self, block_kernels, batch):
        acceptance, selection = block_kernels
        ids = torch.zeros(batch, dtype=torch.int32)
        cu = torch.arange(1, batch + 1, dtype=torch.int32)
        probs = torch.full((batch, 2), .5)
        tau, accepted, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, probs, probs, torch.full((batch,), .9), 1,
        )
        assert tau.tolist() == [1] * batch
        assert accepted.tolist() == [True] * batch
        assert scale.tolist() == [1.] * batch
        output = torch.full((batch, 2), 99, dtype=torch.int32)
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, torch.full_like(ids, 4),
            torch.full((batch, 1), 10, dtype=torch.int32), None, 1, -1,
        )
        assert output.tolist() == [[0, 10]] * batch
        assert result.tolist() == [1] * batch

    @pytest.mark.parametrize("field, match", [
        ("output_token_ids", "output_token_ids must have shape"),
        ("tau", "tau must have shape"),
        ("bonus_token_ids", "bonus_token_ids must have one token"),
        ("recovered_token_ids", "same number of elements"),
        ("draft_stride", "All token selection tensors must be contiguous"),
        ("fill_this_time", "fill_this_time must have shape"),
        ("fill_stride", "fill_this_time must be contiguous"),
    ])
    def test_selection_rejects_invalid_layout(self, block_kernels, field, match):
        _, selection = block_kernels
        output = torch.full((2, 3), 99, dtype=torch.int32)
        args = dict(
            output_token_ids=output, tau=torch.ones(2, dtype=torch.int32),
            cu_num_draft_tokens=torch.tensor([2, 4], dtype=torch.int32),
            draft_token_ids=torch.arange(4, dtype=torch.int32),
            recovered_token_ids=torch.arange(4, dtype=torch.int32),
            bonus_token_ids=torch.zeros(2, 1, dtype=torch.int32),
            fill_this_time=torch.ones(2, dtype=torch.bool),
            max_spec_len=2, placeholder_token_id=-1,
        )
        if field == "draft_stride":
            args["draft_token_ids"] = torch.arange(8, dtype=torch.int32)[::2]
        elif field == "fill_stride":
            args["fill_this_time"] = torch.ones(4, dtype=torch.bool)[::2]
        else:
            args[field] = args[field][:1]
        with pytest.raises(ValueError, match=match):
            selection.select_block_tokens_triton(**args)
        assert torch.all(output == 99)

    @pytest.mark.parametrize("case, match", [
        ("draft_stride", "draft_token_ids must be contiguous"),
        ("cu_stride", "cu_num_draft_tokens must be contiguous"),
        ("greedy_stride", "is_greedy must be contiguous"),
        ("negative_length", "max_num_draft_tokens must be non-negative"),
        ("uniform_length", "uniform_probs and c must have the same number"),
        ("partial_batch", "partial_z must have one row per request"),
    ])
    def test_acceptance_rejects_invalid_layout(self, block_kernels, case, match):
        acceptance, _ = block_kernels
        ids = torch.zeros(2, dtype=torch.int32)
        cu = torch.tensor([1, 2], dtype=torch.int32)
        probs = torch.full((2, 2), .5)
        with pytest.raises(ValueError, match=match):
            if case == "negative_length":
                acceptance.compute_block_partial_z_triton(probs, probs, torch.ones(2), cu, -1)
            elif case in ("uniform_length", "partial_batch"):
                acceptance.finalize_block_acceptance_triton(
                    torch.ones(2), torch.zeros(1 if case == "partial_batch" else 2, 0, 1),
                    torch.zeros(1 if case == "uniform_length" else 2), cu, draft_token_ids=ids,
                )
            else:
                if case == "draft_stride":
                    ids = torch.zeros(4, dtype=torch.int32)[::2]
                if case == "cu_stride":
                    cu = torch.tensor([1, 0, 2, 0], dtype=torch.int32)[::2]
                greedy = torch.zeros(4, dtype=torch.bool)[::2] if case == "greedy_stride" else None
                acceptance.compute_block_prefix_triton(ids, cu, probs, probs, greedy)


def test_block_kernels_in_cpu_interpreter(tmp_path):
    # Import Triton only in the clean child process: other UTs can leave
    # incompatible Triton modules in the parent, causing a false skip.
    env = dict(os.environ, TRITON_INTERPRET="1", OMNI_BLOCK_KERNEL_UT_CHILD="1",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", "pytest"]
    coverage_module = sys.modules.get("coverage")
    parent_coverage = coverage_module.Coverage.current() if coverage_module else None
    if parent_coverage is not None:
        # The interpreter needs a fresh process, but CI measures the parent.
        # Collect only these real source files and merge their executed lines
        # into the active collector before pytest-cov writes its final report.
        source = Path(__file__).resolve().parents[3] / "omni/sample/ops"
        config = tmp_path / "kernels.coveragerc"
        config.write_text(
            "[run]\nbranch = " + str(parent_coverage.config.branch) + "\ninclude =\n"
            + "\n".join("    " + str(source / name) for name in
                        ("block_acceptance.py", "block_selection.py")) + "\n"
        )
        child_data = tmp_path / ".coverage.kernels"
        command = [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config),
                   "--data-file=" + str(child_data), "-m", "pytest"]
    result = subprocess.run(
        command + [str(Path(__file__).resolve()),
         "--noconftest", "-q", "--tb=short", "--junitxml=" + str(tmp_path / "kernels.xml")],
        env=env, capture_output=True, text=True, timeout=120,
    )
    (tmp_path / "kernels.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if parent_coverage is not None:
        child_coverage = coverage_module.Coverage(data_file=str(child_data))
        child_coverage.load()
        parent_coverage.get_data().update(child_coverage.get_data())


test_block_kernels_in_cpu_interpreter.__test__ = os.environ.get("OMNI_BLOCK_KERNEL_UT_CHILD") != "1"
