"""NPU top-k/top-p filtering: FP32 semantics, layouts and graph replay.

Run on an idle device with OMNI_TEST_NPU_DEVICE (default: 0).
"""
import os
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from omni_npu.worker.npu.sampler import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))


def assert_matches(logits, k, p, result):
    expected = apply_top_k_top_p_pytorch(
        logits.cpu().clone(), None if k is None else k.cpu().clone(),
        None if p is None else p.cpu().clone(), allow_cpu_sync=True,
    )
    actual = result.cpu()
    assert actual.dtype == logits.dtype
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [1031, 151552])
@pytest.mark.parametrize("use_k,use_p", [(True, True), (True, False), (False, True), (False, False)])
def test_filter_matches_fp32(vocab, use_k, use_p):
    logits = torch.randn(3, vocab, generator=torch.Generator().manual_seed(17)).npu()
    k = torch.tensor([1, 100, vocab], dtype=torch.int32, device="npu") if use_k else None
    p = torch.tensor([0.8, 0.6, 1.0], device="npu") if use_p else None
    original = logits.clone()
    result = apply_top_k_top_p(logits, k, p)
    assert_matches(original, k, p, result)
    if k is None and p is None:
        assert result is logits


def test_filter_real_batch_and_noncontiguous_inputs():
    # Match a target verification batch: 3 requests * (7 proposals + bonus).
    rows, vocab = 24, 151552
    logits = torch.randn(rows, vocab * 2, generator=torch.Generator().manual_seed(23)).npu()[:, ::2]
    k = torch.full((rows * 2,), 100, dtype=torch.int64, device="npu")[::2]
    p = torch.full((rows * 2,), 0.8, device="npu")[::2]
    original = logits.clone()
    result = apply_top_k_top_p(logits, k, p)
    assert_matches(original, k, p, result)


@pytest.mark.parametrize("case", ["masked", "boundary"])
def test_filter_edge_cases(case):
    vocab = 1031
    x = torch.randn(3, vocab, generator=torch.Generator().manual_seed(19))
    k = torch.tensor([1, 100, vocab], dtype=torch.int32, device="npu")
    p = torch.tensor([0.8, 0.8, 1.0], device="npu")
    if case == "masked":
        x[:, 200:] = -torch.inf
    elif case == "boundary":
        p = torch.tensor([1e-7, 0.9999999, 1.0], device="npu")
    logits = x.npu()
    original = logits.clone()
    result = apply_top_k_top_p(logits, k, p)
    assert_matches(original, k, p, result)


def test_filter_graph_observes_dynamic_inputs():
    source = torch.randn(3, 151552, generator=torch.Generator().manual_seed(37)).npu()
    logits = source.clone()
    k = torch.tensor([1, 100, 151552], dtype=torch.int32, device="npu")
    p = torch.tensor([0.8, 0.6, 1.0], device="npu")
    for _ in range(2):
        logits.copy_(source)
        apply_top_k_top_p(logits, k, p)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = apply_top_k_top_p(logits, k, p)
    for iteration in range(3):
        updated = source.roll(iteration + 1, dims=1) + iteration * 0.125
        logits.copy_(updated)
        k.copy_(torch.tensor([100, 10, 151552], dtype=torch.int32, device="npu"))
        p.copy_(torch.tensor([0.7, 1.0, 0.95], device="npu"))
        graph.replay()
        torch.npu.synchronize()
        assert_matches(updated, k, p, result)


def test_filter_top_k_ties_use_cann_semantics():
    # Unlike the old threshold mask (which kept all ties), CANN retains k items.
    logits = torch.zeros(3, 1031, device="npu")
    k = torch.tensor([1, 100, 1031], dtype=torch.int32, device="npu")
    result = apply_top_k_top_p(logits, k, torch.ones(3, device="npu"))
    assert torch.equal(torch.isfinite(result).sum(-1), k.long())
    assert torch.all(result[torch.isfinite(result)] == 0)


def test_filter_p_one_preserves_underflow_tail():
    # p=1 disables nucleus filtering, including logits whose FP32 softmax is 0.
    logits = torch.tensor([[0., -1000., -2000.]], device="npu")
    original = logits.clone()
    result = apply_top_k_top_p(logits, None, torch.ones(1, device="npu"))
    torch.testing.assert_close(result, original, rtol=0, atol=0)
