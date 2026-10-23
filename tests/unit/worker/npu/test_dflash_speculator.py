# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU DFlash subclass behavior and local speculator selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omni_npu.worker.npu import spec_decode
from omni_npu.worker.npu.spec_decode import dflash_speculator as runtime


def config(method, architectures=(), strategy=None):
    return SimpleNamespace(speculative_config=SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
            architectures=list(architectures),
            dflash_config={"proposal": {"strategy": strategy}})),
    ))


@pytest.mark.parametrize("multimodal", [False, True])
def test_dflash_loader_unwraps_target_and_reuses_upstream_policy(monkeypatch, multimodal):
    language = object()
    target = SimpleNamespace(get_language_model=lambda: language) if multimodal else language
    upstream = Mock(return_value=object())
    monkeypatch.setattr(runtime.DFlashSpeculator, "load_draft_model", upstream)
    speculator = runtime.NPUDFlashSpeculator.__new__(runtime.NPUDFlashSpeculator)
    layers = set()
    assert speculator.load_draft_model(target, layers) is upstream.return_value
    upstream.assert_called_once_with(language, layers)


def test_draft_graph_provider_tracks_current_metadata(monkeypatch):
    import torch

    upstream = Mock(side_effect=[{"layer": object()}, {"layer": object()}])
    monkeypatch.setattr(runtime.DFlashSpeculator, "_build_draft_attn_metadata", upstream)
    monkeypatch.setattr(torch.npu, "Stream", Mock(return_value=object()))
    speculator = runtime.NPUDFlashSpeculator.__new__(runtime.NPUDFlashSpeculator)
    manager = speculator.query_cudagraph_manager = SimpleNamespace()
    desc = SimpleNamespace(num_tokens=8, num_reqs=1)
    first = speculator._build_draft_attn_metadata()
    provider = manager._omni_acl_context_provider
    assert provider(desc).attn_metadata is first
    second = speculator._build_draft_attn_metadata()
    assert provider(desc).attn_metadata is second
    assert provider(desc).batch_descriptor.num_tokens == 8
    assert provider(desc).batch_descriptor.num_reqs == 1
    assert torch.npu.Stream.call_count == 1


def test_metadata_without_graph_manager_preserves_upstream_call(monkeypatch):
    upstream = Mock(return_value={"draft": object()})
    monkeypatch.setattr(runtime.DFlashSpeculator, "_build_draft_attn_metadata", upstream)
    speculator = runtime.NPUDFlashSpeculator.__new__(runtime.NPUDFlashSpeculator)
    speculator.query_cudagraph_manager = None
    positions = object()
    assert speculator._build_draft_attn_metadata(positions, step=2) is upstream.return_value
    upstream.assert_called_once_with(positions, step=2)
    assert not hasattr(speculator, "_omni_runtime_draft_attn_metadata")


@pytest.mark.parametrize("method,architectures,strategy,selected", [
    ("dflash", ["DFlashPanguDraftModel"], "unary", "dflash"),
    ("dflash", ["DFlashPanguDraftModel"], "selector", "dflash2"),
    ("dspark", ["DFlashPanguDraftModel"], "markov", "dspark"),
    ("dflash", ["DFlashQwen3ForCausalLM"], None, "dflash"),
    ("dspark", ["Qwen3DSparkModel"], None, "dspark"),
    ("dflash", ["OtherDraftModel"], "selector", "dflash2"),
    ("dspark", ["OtherDraftModel"], None, "dspark"),
    ("eagle3", [], None, "upstream"),
])
def test_local_factory_selects_npu_speculator(monkeypatch, method, architectures, strategy, selected):
    factories = {name: Mock(return_value=object()) for name in ("dflash", "dflash2", "dspark", "upstream")}
    for name, cls in (("dflash", "NPUDFlashSpeculator"),
                      ("dflash2", "NPUDFlash2Speculator"),
                      ("dspark", "NPUDSparkSpeculator")):
        monkeypatch.setattr(runtime, cls, factories[name])
    monkeypatch.setattr(spec_decode, "vllm_init_speculator", factories["upstream"])
    cfg = config(method, architectures, strategy)
    device = object()
    assert spec_decode.init_speculator(cfg, device) is factories[selected].return_value
    factories[selected].assert_called_once_with(cfg, device)
    for name, factory in factories.items():
        if name != selected:
            factory.assert_not_called()


def test_dspark_loader_unwraps_target_and_reuses_upstream_policy(monkeypatch):
    upstream = Mock(return_value=object())
    monkeypatch.setattr(runtime.DSparkSpeculator, "load_draft_model", upstream)
    speculator = runtime.NPUDSparkSpeculator.__new__(runtime.NPUDSparkSpeculator)
    language = object()
    target = SimpleNamespace(get_language_model=lambda: language)
    layers = set()
    assert speculator.load_draft_model(target, layers) is upstream.return_value
    upstream.assert_called_once_with(language, layers)


@pytest.mark.parametrize("cls", [runtime.NPUDFlash2Speculator, runtime.NPUDSparkSpeculator])
def test_subclasses_inherit_metadata_refresh_without_patching_upstream(cls):
    assert cls._build_draft_attn_metadata is runtime.NPUDFlashSpeculator._build_draft_attn_metadata
    assert runtime.DFlashSpeculator._build_draft_attn_metadata is not cls._build_draft_attn_metadata


def test_dspark_reuses_upstream_generation():
    assert runtime.NPUDSparkSpeculator._generate_draft is runtime.DSparkSpeculator._generate_draft
    assert runtime.NPUDSparkSpeculator._sample_sequential is runtime.DSparkSpeculator._sample_sequential


@pytest.mark.parametrize("bonus_anchor", [False, True])
def test_dspark_preserves_checkpoint_anchor_layout(monkeypatch, bonus_anchor):
    cfg = config("dspark")
    hf_config = cfg.speculative_config.draft_model_config.hf_config
    hf_config.dspark_bonus_anchor = bonus_anchor

    def upstream_init(self, vllm_config, device):
        self.sample_from_anchor = not vllm_config.speculative_config.draft_model_config.hf_config.dspark_bonus_anchor

    monkeypatch.setattr(runtime.DSparkSpeculator, "__init__", upstream_init)
    speculator = runtime.NPUDSparkSpeculator(cfg, "test-device")
    assert speculator.sample_from_anchor is not bonus_anchor
    assert hf_config.dspark_bonus_anchor is bonus_anchor


@pytest.fixture
def selector_factory(monkeypatch):
    import torch

    def init(self, cfg, device):
        self.num_speculative_steps = 2
        self.num_query_per_req = 3
        self.max_num_reqs = 3
        self.vocab_size = 12
        self.draft_logits = (
            torch.zeros((3, 2, 12), dtype=torch.float32, device=device)
            if cfg.speculative_config.draft_sample_method == "probabilistic" else None
        )

    monkeypatch.setattr(runtime.DFlashSpeculator, "__init__", init)

    def make(sample_method="greedy", block_size=3):
        cfg = config("dflash", strategy="selector")
        cfg.speculative_config.draft_sample_method = sample_method
        knobs = cfg.speculative_config.draft_model_config.hf_config.dflash_config
        if block_size is not None:
            knobs["block_size"] = block_size
        return runtime.NPUDFlash2Speculator(cfg, torch.device("cpu"))

    return make


@pytest.mark.parametrize("sample_method", ["greedy", "probabilistic"])
@pytest.mark.parametrize("block_size", [None, 0, 3, 4])
def test_selector_initialization_validates_layout_and_allocates_sampling_buffers(
    selector_factory, sample_method, block_size,
):
    import torch

    if block_size == 4:
        with pytest.raises(ValueError, match=r"block_size=4.*runtime query block size=3"):
            selector_factory(sample_method, block_size)
        return

    spec = selector_factory(sample_method, block_size)
    torch.testing.assert_close(spec._candidate_step_cols, torch.tensor([0, 1], dtype=torch.int32))
    if sample_method == "greedy":
        assert spec._candidate_logits is None
    else:
        assert spec._candidate_logits.shape == (3, 12)
        assert spec._candidate_logits.dtype == spec.draft_logits.dtype
        assert spec._candidate_logits.device.type == "cpu"
        assert torch.isneginf(spec._candidate_logits).all()


@pytest.mark.parametrize("sample_method,missing_buffer", [
    ("greedy", False), ("probabilistic", False), ("probabilistic", True),
])
@pytest.mark.parametrize("capture", [False, True])
def test_selector_samples_from_selected_predecessor_and_preserves_inactive_requests(
    monkeypatch, selector_factory, sample_method, missing_buffer, capture,
):
    import torch
    from vllm.config.compilation import CUDAGraphMode

    spec = selector_factory(sample_method)
    hidden = torch.arange(18, dtype=torch.float32).reshape(9, 2)
    spec._run_model = Mock(return_value=hidden)
    spec.sample_indices = torch.tensor([1, 2, 4, 5, 7, 8])
    spec.input_buffers = SimpleNamespace(input_ids=torch.tensor([1, 0, 0, 2, 0, 0, 3, 0, 0]))
    ids = torch.tensor([[[4, 5], [6, 7]], [[8, 9], [10, 11]]])
    # Step 1 depends on the CANDIDATE INDEX sampled at step 0, not its token id.
    edges = torch.tensor([
        [[[0., 2.], [0., 2.]], [[9., 0.], [0., 9.]]],
        [[[3., 0.], [3., 0.]], [[9., 0.], [0., 9.]]],
    ])
    lattice = Mock(return_value=(ids, edges))
    spec.model = SimpleNamespace(compute_candidate_lattice=lattice)
    spec.sample_idx_mapping = torch.tensor([2, 1, 0, 2, 1, 0])
    spec.sample_pos = torch.tensor([10, 11, 20, 21, 30, 31])
    spec.draft_tokens = torch.full((3, 2), -7, dtype=torch.long)
    spec.temperature = torch.tensor([0.5, 1., 2.])
    spec.seeds = torch.tensor([11, 22, 33])
    spec.use_fp64_gumbel = True
    if spec._candidate_logits is not None:
        spec._candidate_logits.fill_(77.)

    snapshots = []
    # Deliberately choose the non-argmax candidates at step 0. Step 1 must
    # condition on those choices rather than the greedy predecessors.
    selections = [torch.tensor([4, 9]), torch.tensor([6, 11])]

    def sample(logits, mapping, temperature, seeds, positions, **kwargs):
        step = len(snapshots)
        snapshots.append(logits.clone())
        torch.testing.assert_close(mapping, torch.tensor([[2, 1], [0, 2]])[:, step])
        torch.testing.assert_close(positions, torch.tensor([[9, 10], [19, 20]])[:, step])
        assert temperature is spec.temperature and seeds is spec.seeds
        assert kwargs["apply_temperature"] is True
        assert kwargs["output_processed_logits"] is spec.draft_logits
        assert kwargs["output_processed_logits_col"].item() == step
        assert kwargs["use_fp64"] is True
        return selections[step]

    sampler = Mock(side_effect=sample)
    monkeypatch.setattr(runtime, "gumbel_sample", sampler)
    metadata, slots, across_dp = object(), object(), object()
    mode = CUDAGraphMode.FULL if capture else None
    if missing_buffer:
        spec._candidate_logits = None
        with pytest.raises(RuntimeError, match="initialized candidate logits buffer"):
            spec._generate_draft(2, 9, metadata, slots, across_dp, mode)
        sampler.assert_not_called()
        torch.testing.assert_close(spec.draft_tokens, torch.full((3, 2), -7, dtype=torch.long))
        return
    spec._generate_draft(2, 9, metadata, slots, across_dp, mode)

    spec._run_model.assert_called_once_with(
        9, metadata, slots, across_dp, CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE,
    )
    lattice.assert_called_once()
    sampled_hidden, anchors = lattice.call_args.args
    torch.testing.assert_close(sampled_hidden, hidden[torch.tensor([1, 2, 4, 5])].reshape(2, 2, 2))
    torch.testing.assert_close(anchors, torch.tensor([1, 2]))
    torch.testing.assert_close(spec.draft_tokens[2], torch.tensor([-7, -7]))
    if sample_method == "greedy":
        sampler.assert_not_called()
        torch.testing.assert_close(spec.draft_tokens[:2], torch.tensor([[5, 7], [8, 10]]))
    else:
        assert sampler.call_count == 2
        torch.testing.assert_close(spec.draft_tokens[:2], torch.tensor([[4, 6], [9, 11]]))
        expected_first = torch.full((2, 12), float("-inf"))
        expected_first[0, 4:6] = torch.tensor([0., 2.])
        expected_first[1, 8:10] = torch.tensor([3., 0.])
        expected_second = torch.full((2, 12), float("-inf"))
        expected_second[0, 6:8] = torch.tensor([9., 0.])
        expected_second[1, 10:12] = torch.tensor([0., 9.])
        torch.testing.assert_close(snapshots[0], expected_first)
        torch.testing.assert_close(snapshots[1], expected_second)
        torch.testing.assert_close(spec._candidate_logits[2], torch.full((12,), 77.))
