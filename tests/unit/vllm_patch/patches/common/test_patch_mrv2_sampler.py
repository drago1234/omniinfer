# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 sampling injection: bindings, MRv1 isolation, missing-target fallback."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import pytest

from omni_npu.vllm_patches.patches.common import patch_mrv2_sampler as patch_mod
from omni_npu.worker.npu import rejection_sampler as npu_rejection
from omni_npu.worker.npu import sampler as npu_sample
from omni_npu.worker.npu.ops import rejection_sampler_utils as npu_rejection_sampler_utils
from unit.vllm_patch.patches.patch_test_utils import applied_patches


def _patch_classes(mod):
    classes = []
    for name in (
        "MRv2GumbelPatch",
        "MRv2SamplerPatch",
        "MRv2SamplerStatesPatch",
        "MRv2SpeculatorGumbelPatch",
        "MRv2DsparkSpeculatorGumbelPatch",
        "MRv2RejectionSamplerPatch",
        "MRv2RejectionSamplerPlaceholderPatch",
        "MRv2RejectionSamplerUtilsPatch",
        "MRv2V1RejectionSamplerTopKTopPPatch",
    ):
        if hasattr(mod, name):
            classes.append(getattr(mod, name))
    return classes


@pytest.fixture
def applied():
    """Apply this file's patches, restoring symbols and ownership on exit."""
    classes = _patch_classes(patch_mod)
    with applied_patches(classes) as applied_classes:
        yield applied_classes


def test_gumbel_binding_covers_the_speculators(applied):
    """Draft sampling is the easy one to miss: each speculator from-imports it."""
    import vllm.v1.worker.gpu.spec_decode.speculator as up_speculator

    assert up_speculator.gumbel_sample is npu_sample.gumbel_sample

    dspark = sys.modules.get("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    if dspark is not None:
        assert dspark.gumbel_sample is npu_sample.gumbel_sample


def test_top_k_top_p_definition_module_is_left_to_mrv1(applied):
    """MRv1 guard: TopKTopPSampler uses the defining module, which stays upstream."""
    from vllm.v1.sample.ops import topk_topp_sampler as up_topk_topp

    assert up_topk_topp.apply_top_k_top_p is not npu_sample.apply_top_k_top_p
    assert (
        up_topk_topp.apply_top_k_top_p.__module__
        == "vllm.v1.sample.ops.topk_topp_sampler"
    )


def test_rejection_sampler_bindings_are_patched(applied):
    """MTP rejection sampling keeps from-imported bindings in several modules."""
    import vllm.v1.sample.rejection_sampler as up_v1_rejection_sampler
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler as up_rejection_sampler
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as up_rejection_sampler_utils

    assert (
        up_rejection_sampler.rejection_sample
        is npu_rejection_sampler_utils.rejection_sample
    )
    assert (
        up_rejection_sampler_utils.rejection_sample
        is npu_rejection_sampler_utils.rejection_sample
    )
    assert up_rejection_sampler.RejectionSampler.__call__ is npu_rejection.rejection_sampler_call
    assert up_v1_rejection_sampler.apply_top_k_top_p is npu_sample.apply_top_k_top_p


def test_missing_target_degrades_to_an_error_log():
    """An unimportable target must log and skip registration, never raise.

    The plugin imports this directory wholesale and exec_module has no
    try/except, so raising here would take MRv1 down with it.

    Runs in a subprocess: sibling tests stub entries in sys.modules, and the
    module is loaded fresh rather than reloaded (reload keeps the old namespace).
    """
    code = (
        "import importlib.util, sys\n"
        "import omni_npu.vllm_patches.patches.common.patch_mrv2_sampler as m\n"
        "sys.modules['vllm.v1.worker.gpu.sample.states'] = None\n"
        "spec = importlib.util.spec_from_file_location('probe', m.__file__)\n"
        "probe = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(probe)  # must not raise\n"
        "leaked = [n for n in ('MRv2GumbelPatch', 'MRv2SamplerPatch',\n"
        "                      'MRv2SamplerStatesPatch', 'MRv2SpeculatorGumbelPatch')\n"
        "          if hasattr(probe, n)]\n"
        "assert not leaked, f'registered despite a missing target: {leaked}'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )

    assert proc.returncode == 0, (
        f"fallback failed:\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )
    assert "not registered" in (proc.stdout + proc.stderr), "expected an error log"


@pytest.mark.parametrize("compute_nans", [False, True])
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_rejection_sampler_call_preserves_sampling_metadata(
    applied, monkeypatch, compute_nans, logprobs_mode,
):
    """The patched entry point must pass placeholder IDs through processing.

    Kernels have independent real-NPU tests; here sentinel outputs distinguish
    raw/processed logits and pre/post-prefill counts at every Python boundary.
    """
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    batch = SimpleNamespace(**{
        name: torch.tensor([i]) for i, name in enumerate((
            "input_ids", "positions", "logits_indices", "expanded_idx_mapping",
            "expanded_local_pos", "idx_mapping_np", "cu_num_logits",
            "idx_mapping", "seq_lens",
        ))
    })
    logits = torch.tensor([[float("nan"), 1.]])
    processed = torch.tensor([[0., 1.]])
    draft_ids = torch.tensor([5, -1])
    positions = torch.tensor([125, 126])
    prefill_len, temperature, seeds = (torch.tensor([v]) for v in (126, 1, 1234))
    draft_logits = torch.tensor([[[1., 0.]]]) if compute_nans else None
    rates = torch.tensor([1.]) if compute_nans else None
    sampled = torch.tensor([[5, -1]])
    before_counts, after_counts, rejected = (torch.tensor([v]) for v in (1, 0, 1))
    nan_counts, logprobs = torch.tensor([1]), object()
    gather = Mock(return_value=(draft_ids, positions))
    process = Mock(return_value=processed)
    verify = Mock(return_value=(sampled, before_counts))
    count = Mock(return_value=(after_counts, rejected))
    nans = Mock(return_value=nan_counts)
    get_logprobs = Mock(return_value=logprobs)
    monkeypatch.setattr(npu_rejection, "gather_draft_sampled", gather)
    monkeypatch.setattr(npu_rejection, "rejection_sample", verify)
    monkeypatch.setattr(npu_rejection, "get_num_sampled_and_rejected", count)
    monkeypatch.setattr(npu_rejection, "get_num_nans", nans)
    # Bypass allocation in __init__, but execute the actually patched class call.
    instance = object.__new__(RejectionSampler)
    instance.sampler = SimpleNamespace(
        compute_nans=compute_nans, logprobs_mode=logprobs_mode,
        use_fp64_gumbel=False, apply_sampling_params=process,
        req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=prefill_len)),
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature),
            seeds=SimpleNamespace(gpu=seeds),
        ),
    )
    instance.num_speculative_steps = 1
    instance.synthetic_conditional_rates = rates
    instance.use_block_verification = not compute_nans
    instance._get_logprobs_tensors = get_logprobs

    events = Mock()
    for name, mock in (("nans", nans), ("gather", gather), ("process", process),
                       ("verify", verify), ("logprobs", get_logprobs), ("count", count)):
        events.attach_mock(mock, name)
    output = instance(logits, batch, draft_logits)
    assert [call[0] for call in events.mock_calls] == (
        (["nans"] if compute_nans else []) +
        ["gather", "process", "verify", "logprobs", "count"]
    )
    gather.assert_called_once_with(
        batch.input_ids, batch.positions, batch.logits_indices,
        batch.expanded_idx_mapping, batch.expanded_local_pos, prefill_len,
    )
    process.assert_called_once_with(
        logits, batch.expanded_idx_mapping, batch.idx_mapping_np,
        positions, draft_ids, batch.expanded_local_pos,
    )
    verify.assert_called_once_with(
        processed, draft_logits, draft_ids, batch.cu_num_logits, positions,
        batch.idx_mapping, batch.expanded_idx_mapping, batch.expanded_local_pos,
        temperature, seeds, 1, rates, use_fp64=False,
        use_block_verification=not compute_nans,
    )
    get_logprobs.assert_called_once_with(
        batch, sampled, before_counts,
        processed if logprobs_mode == "processed_logprobs" else logits,
    )
    count.assert_called_once_with(
        before_counts, batch.seq_lens, batch.cu_num_logits,
        batch.idx_mapping, prefill_len,
    )
    if compute_nans:
        nans.assert_called_once_with(logits)
    else:
        nans.assert_not_called()
    assert output.sampled_token_ids is sampled
    assert output.logprobs_tensors is logprobs
    assert output.num_nans is (nan_counts if compute_nans else None)
    assert output.num_sampled is after_counts
    assert output.num_rejected is rejected
