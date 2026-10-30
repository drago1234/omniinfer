# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 penalties checked on the NPU and with the Triton CPU interpreter.

The isolated interpreter executes production kernel source for line coverage;
NPU execution also checks compilation and asynchronous launch behavior.

Checks the MRv2 state entry point against a CPU reference with mixed request
parameters, reordered slots, MTP prefixes, packed prompt bits and vocab tails.
"""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")

from omni_npu.vllm_patches.patches.common import patch_mrv2_sampler
from omni_npu.worker.npu.ops import penalties as npu_penalties
import vllm.v1.worker.gpu.sample.penalties as up_penalties

pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture
def patched(monkeypatch):
    device = "cpu" if os.environ.get("OMNI_PENALTIES_INTERPRETER") == "1" else "npu"
    if device == "npu":
        if not torch.npu.is_available():
            pytest.skip("NPU required")
        torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))
    monkeypatch.setattr(up_penalties, "apply_penalties", up_penalties.apply_penalties)
    monkeypatch.setattr(up_penalties, "bincount", up_penalties.bincount)
    ownership = dict(getattr(up_penalties, "_omni_npu_applied_patches", {}))
    ownership.pop("apply_penalties", None)
    ownership.pop("bincount", None)
    monkeypatch.setattr(up_penalties, "_omni_npu_applied_patches", ownership, raising=False)
    patch_mrv2_sampler.MRv2PenaltiesPatch.apply()
    return device


def test_binding_preserves_mrv1(patched):
    from vllm.v1.sample.ops import penalties as v1_penalties

    assert up_penalties.apply_penalties is npu_penalties.apply_penalties
    assert up_penalties.bincount is npu_penalties.bincount
    assert up_penalties.PenaltiesState.apply_staged_writes.__globals__["bincount"] is npu_penalties.bincount
    assert v1_penalties.apply_all_penalties.__module__ == "vllm.v1.sample.ops.penalties"
    assert up_penalties.PenaltiesState.apply_penalties.__globals__["apply_penalties"] is npu_penalties.apply_penalties


@pytest.mark.parametrize("vocab", [257, 8209, 151552])
@pytest.mark.parametrize("width", [1, 8])
def test_penalties_cpu_reference(patched, vocab, width):
    # Per-slot (frequency, presence, repetition); include a neutral request.
    params = [(0., 0., 1.), (.3, 0., 1.), (0., .3, 1.),
              (.3, .3, 1.), (.3, .3, 1.2), (-.3, -.3, .8)]
    slots = [3, 0, 5, 2, 1, 4]
    mapping = torch.tensor([s for s in slots for _ in range(width)], dtype=torch.int32)
    positions = torch.arange(width, dtype=torch.int32).repeat(len(slots))
    ids = torch.tensor(([4, 5, 5, 7, 9, 31, 32, vocab - 1][:width]) * len(slots), dtype=torch.int64)
    rows = len(mapping)
    logits = torch.linspace(-3., 3., vocab).repeat(rows, 1)
    logits[:, 7] = 2.
    logits[:, 9] = 1.
    counts = torch.zeros(len(slots), vocab, dtype=torch.int32)
    prompt = torch.zeros(len(slots), (vocab + 31) // 32, dtype=torch.int32)
    prompt_tokens = [9, 31, 32, vocab - 1]
    for slot in range(len(slots)):
        counts[slot, 5] = slot + 1
        counts[slot, 7] = 1
        counts[slot, vocab - 1] = 2
        for token in prompt_tokens:
            bit = 1 << (token % 32)
            prompt[slot, token // 32] |= bit if bit < 2**31 else bit - 2**32

    expected = logits.clone()
    for row, slot in enumerate(mapping.tolist()):
        freq, pres, rep = params[slot]
        histogram = counts[slot].clone()
        pos = int(positions[row])
        for token in ids[row - pos + 1:row + 1]:
            histogram[token] += 1
        seen = histogram > 0
        repeated = seen.clone()
        repeated[prompt_tokens] = True
        # Independent selective updates, with both positive and negative logits.
        positive = repeated & (expected[row] > 0)
        negative = repeated & ~positive
        expected[row, positive] /= rep
        expected[row, negative] *= rep
        expected[row] -= freq * histogram
        expected[row] -= pres * seen

    # Exercise a padded row stride as well as request remapping.
    storage = torch.empty(rows, vocab + 8, device=patched, dtype=torch.float32)
    actual = storage[:, :vocab]
    actual.copy_(logits)
    state = SimpleNamespace(
        use_penalty=np.array([f != 0 or p != 0 or r != 1 for f, p, r in params]),
        repetition_penalty=SimpleNamespace(gpu=torch.tensor([p[2] for p in params], device=patched)),
        frequency_penalty=SimpleNamespace(gpu=torch.tensor([p[0] for p in params], device=patched)),
        presence_penalty=SimpleNamespace(gpu=torch.tensor([p[1] for p in params], device=patched)),
        prompt_bin_mask=prompt.to(patched),
        output_bin_counts=counts.to(patched),
    )
    before_counts = state.output_bin_counts.cpu().clone()
    before_prompt = state.prompt_bin_mask.cpu().clone()
    up_penalties.PenaltiesState.apply_penalties(
        state, actual, mapping.to(patched), np.array(slots), ids.to(patched), positions.to(patched)
    )
    if patched == "npu":
        torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(state.output_bin_counts.cpu(), before_counts)
    assert torch.equal(state.prompt_bin_mask.cpu(), before_prompt)


@pytest.mark.parametrize("vocab", [257, 8209, 151552])
def test_bincount_cpu_reference(patched, vocab):
    # Reordered slots, repeated tokens, bit 31 and partial final words.
    slots = [3, 0, 2]
    num_slots = 5
    prompt_lens = torch.tensor([0, 0, 1057, 16415, 0], dtype=torch.int32)
    prefill_lens = torch.tensor([37, 0, 1121, 16415, 0], dtype=torch.int32)
    max_len = int(prefill_lens.max())
    ids = torch.zeros(num_slots, max_len + 8, dtype=torch.int64)[:, :max_len]
    pattern = torch.tensor([0, 5, 5, 31, 32, vocab - 1], dtype=torch.int64)
    for slot in slots:
        ids[slot] = pattern.repeat((max_len + len(pattern) - 1) // len(pattern))[:max_len]
    num_words = (vocab + 31) // 32
    expected_prompt = np.full((num_slots, num_words), 0x13579bdf, dtype=np.uint32)
    expected_counts = np.full((num_slots, vocab), 17, dtype=np.int32)
    for slot in slots:
        lp, lf = int(prompt_lens[slot]), int(prefill_lens[slot])
        tokens = ids[slot, :lf].numpy()
        expected_prompt[slot] = 0
        np.bitwise_or.at(
            expected_prompt[slot], tokens[:lp] // 32,
            np.left_shift(np.uint32(1), tokens[:lp] % 32).astype(np.uint32),
        )
        expected_counts[slot] = np.bincount(tokens[lp:], minlength=vocab)
    # Padded rows expose accidental contiguous-layout assumptions.
    token_ids = torch.empty(num_slots, max_len + 8, dtype=torch.int64, device=patched)[:, :max_len]
    token_ids.copy_(ids)
    prompt = torch.full((num_slots, num_words + 4), 0x13579bdf, dtype=torch.int32, device=patched)[:, :num_words]
    counts = torch.full((num_slots, vocab + 8), 17, dtype=torch.int32, device=patched)[:, :vocab]
    args = (
        torch.tensor(slots, dtype=torch.int32, device=patched), token_ids,
        prompt_lens.to(patched), prefill_lens.to(patched), prompt, counts, max_len,
    )
    # Check repeat initialization without synchronizing between launches.
    for _ in range(100 if patched == "npu" else 2):
        up_penalties.bincount(*args)
    if patched == "npu":
        torch.npu.synchronize()
    np.testing.assert_array_equal(prompt.cpu().numpy().view(np.uint32), expected_prompt)
    np.testing.assert_array_equal(counts.cpu().numpy(), expected_counts)


def test_kernel_interpreter(tmp_path):
    # A separate process keeps Triton's interpreter globals out of NPU tests.
    coverage = pytest.importorskip("coverage")
    parent = coverage.Coverage.current()
    source = str(Path(npu_penalties.__file__).resolve())
    data_file = str(tmp_path / ".coverage.penalties")
    bootstrap = tmp_path / "penalties_interpreter.py"
    bootstrap.write_text(
        "import sys, coverage\n"
        "measured = coverage.Coverage(config_file=False, data_file=sys.argv[1],\n"
        "    include=[sys.argv[2]], branch=sys.argv[4] == '1')\n"
        "measured.start()\n"
        "import pytest\n"
        "result = pytest.main([sys.argv[3], '-q', '-o', 'addopts=',\n"
        "    '-k', 'not test_kernel_interpreter', '--tb=short'])\n"
        "measured.stop()\n"
        "measured.save()\n"
        "raise SystemExit(result)\n"
    )
    env = dict(os.environ, TRITON_INTERPRET="1", OMNI_PENALTIES_INTERPRETER="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [
        sys.executable, str(bootstrap), data_file, source, str(Path(__file__).resolve()),
        "1" if parent is not None and parent.config.branch else "0",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = coverage.Coverage(config_file=False, data_file=data_file)
    measured.load()
    _, statements, _, missing, _ = measured.analysis2(source)
    assert statements and (len(statements) - len(missing)) / len(statements) >= 0.8, missing
    if parent is not None:
        # Only merge this module's executed lines; leave global CI exclusions intact.
        parent.get_data().update(measured.get_data())
