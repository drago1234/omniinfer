# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from omni_npu.diagnostics.accuracy_alignment import compare as compare_module
from omni_npu.diagnostics.accuracy_alignment import pure_real as pure_real_module
from omni_npu.diagnostics.accuracy_alignment.compare import (
    CompareResult,
    _run_parallel,
    compare_one,
    compare_pure_real,
    compare_tensors,
    discover_layers,
    infer_candidates,
    normalize_training_tensor,
    parse_args,
    training_path,
    write_reports,
)
from omni_npu.diagnostics.accuracy_alignment.compare import (
    main as compare_main,
)
from omni_npu.diagnostics.accuracy_alignment.config import (
    AlignmentConfigError,
    PureRealConfig,
    pure_real_requested,
)
from omni_npu.diagnostics.accuracy_alignment.pure_real import (
    PureRealSession,
    pure_real_dump_path,
)
from omni_npu.diagnostics.accuracy_alignment.train_ref import dummy_run_guard
from torch import nn


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Identity()

    def forward(self, value):
        return self.proj(value) + 1


def _pure_config(tmp_path, **overrides):
    values = {
        "infer_dump_dir": tmp_path / "infer",
        "role": "P",
        "run_id": "7",
    }
    values.update(overrides)
    return PureRealConfig(**values)


def test_pure_real_config_accepts_explicit_and_legacy_modes(tmp_path):
    base = {
        "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
        "OMNI_NPU_DEBUG_INFER_DUMP_DIR": str(tmp_path / "infer"),
        "OMNI_NPU_MHC_DEBUG_ROLE": "D",
        "OMNI_NPU_DEBUG_RUN_ID": "9",
    }
    explicit = {**base, "OMNI_NPU_DEBUG_MODE": "pure_real"}
    legacy = {**base, "OMNI_NPU_DEBUG_PURE_REAL": "1"}

    assert pure_real_requested(explicit)
    assert pure_real_requested(legacy)
    assert PureRealConfig.from_env(explicit).run_id == "9"


def test_pure_real_dumps_every_selected_submodule_without_changing_output(tmp_path):
    model = _Block()
    session = PureRealSession(_pure_config(tmp_path), rank=0)
    session.install(model)
    value = torch.tensor([[1.0, 2.0]])

    output = model(value)

    assert torch.equal(output, value + 1)
    files = sorted((tmp_path / "infer" / "prefill").glob("*.pt"))
    assert [item.name for item in files] == ["decorator_proj_run7_call0_rank0.pt"]


def test_pure_real_include_exclude_and_dummy_guard(tmp_path):
    model = _Block()
    config = _pure_config(
        tmp_path,
        include_patterns=("proj",),
        exclude_patterns=("never-match",),
    )
    session = PureRealSession(config, rank=0)
    session.install(model)
    with dummy_run_guard():
        model(torch.ones(1, 2))
    assert not list((tmp_path / "infer").rglob("*.pt"))


def test_pure_real_captures_direct_forward_calls_and_restores_method(tmp_path):
    model = _Block()
    original_forward = model.proj.forward
    config = _pure_config(
        tmp_path,
        run_id="direct",
        include_patterns=(r"^proj$",),
    )
    session = PureRealSession(config, rank=0)
    session.install(model)

    value = torch.ones(2, 3)
    result = model.proj.forward(value)

    assert torch.equal(result, value)
    assert pure_real_dump_path(
        tmp_path / "infer",
        "prefill",
        "proj",
        "direct",
        0,
        0,
    ).is_file()
    session.close()
    assert model.proj.forward == original_forward


def test_comparator_reads_megatron_payload_and_converts_ncl(tmp_path):
    path = tmp_path / "train.pt"
    training = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    torch.save({"tensors": {"input.0": training, "output.0": training}}, path)
    infer = torch.zeros(2, 3)

    result = normalize_training_tensor(
        path,
        key="input.0",
        role="P",
        mock_seq_len=2,
        infer=infer,
    )

    assert torch.equal(result, training.squeeze(0).transpose(0, 1)[:2])


def test_comparator_prefill_uses_configured_length_not_infer_length(tmp_path):
    path = tmp_path / "train.pt"
    training = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.save({"tensors": {"output.0": training}}, path)

    result = normalize_training_tensor(
        path,
        key="output.0",
        role="P",
        mock_seq_len=3,
        infer=torch.zeros(2, 3),
    )

    assert result.shape == (3, 3)


def test_pure_real_compare_pairs_matching_runs(tmp_path):
    root = tmp_path / "infer" / "prefill"
    root.mkdir(parents=True)
    left = root / "decorator_model.layers.2.mlp_run0_call0_rank0.pt"
    right = root / "decorator_model.layers.2.mlp_run1_call0_rank0.pt"
    torch.save(torch.ones(3, 4), left)
    torch.save(torch.ones(3, 4), right)

    results = compare_pure_real(
        tmp_path / "infer",
        role="P",
        rank=0,
        run_a="0",
        run_b="1",
        mock_seq_len=3,
        rtol=1e-5,
        atol=1e-5,
    )

    assert len(results) == 1
    result = results[0]
    assert result.layer_idx == 2
    assert result.module == "mlp"
    assert result.status == "PASS"
    assert result.train_path == str(left)
    assert result.infer_path == str(right)
    assert result.seq_len == 3
    assert result.shapes_match is True


def test_compare_tensors_reports_shape_mismatch():
    passed, metrics = compare_tensors(
        torch.ones(2, 3), torch.ones(3, 2), rtol=1e-5, atol=1e-5
    )
    assert not passed
    assert metrics.get("left_shape") == [2, 3]
    assert metrics.get("right_shape") == [3, 2]
    assert metrics.get("shapes_match") is False


def test_compare_tensors_uses_legacy_relative_error_mask():
    passed, metrics = compare_tensors(
        torch.tensor([0.0, 2.0]),
        torch.tensor([1.0, 1.0]),
        rtol=0.0,
        atol=0.1,
    )

    assert not passed
    assert metrics.get("max_rel_diff_percent") == 50.0
    # The zero reference element is masked, then retained as zero in the mean.
    assert metrics.get("avg_rel_diff_percent") == 25.0


def test_compare_cli_accepts_legacy_report_prefix(tmp_path):
    args = parse_args(
        [
            "--infer-root",
            str(tmp_path),
            "--mock-seq-len",
            "3",
            "--report-prefix",
            "mhc_mock_vs_infer",
        ]
    )

    assert args.report_prefix == "mhc_mock_vs_infer"


def test_pure_real_rejects_invalid_regex_and_non_tensor_outputs(tmp_path):
    with pytest.raises(AlignmentConfigError, match="invalid pure_real"):
        PureRealSession(
            _pure_config(tmp_path, include_patterns=("[",)),
            rank=0,
        )

    session = PureRealSession(_pure_config(tmp_path), rank=0)
    session._dump("scalar", torch.tensor(1.0))
    session._dump("object", object())
    assert not list((tmp_path / "infer").rglob("*.pt"))


def test_pure_real_prefill_only_writes_world_rank_zero(tmp_path):
    session = PureRealSession(_pure_config(tmp_path), rank=1)
    session._dump("proj", torch.ones(1, 2))
    assert not list((tmp_path / "infer").rglob("*.pt"))


def test_install_pure_real_validates_model_and_replaces_previous_session(
    monkeypatch, tmp_path
):
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Identity()

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    monkeypatch.setattr(pure_real_module, "_ultra_model_classes", lambda: (Base, Inner))
    monkeypatch.setattr(pure_real_module, "_rank", lambda: 3)
    runner = SimpleNamespace(
        get_model=lambda: Base(),
        model_config=SimpleNamespace(enforce_eager=False),
    )
    config = _pure_config(tmp_path, role="D")
    with pytest.raises(AlignmentConfigError, match="enforce_eager"):
        pure_real_module.install_pure_real(runner, config)

    runner.model_config.enforce_eager = True
    first = pure_real_module.install_pure_real(runner, config)
    assert first is not None
    assert first.rank == 3
    second = pure_real_module.install_pure_real(runner, config)
    assert second is not first
    assert first.installed_hook_count == 0

    runner.get_model = lambda: nn.Identity()
    assert pure_real_module.install_pure_real(runner, config) is None
    assert second.installed_hook_count == 0


def test_normalize_training_tensor_loads_shards_and_validates_length(tmp_path):
    first = tmp_path / "tensor_rank0_0.pt"
    second = tmp_path / "tensor_rank1_0.pt"
    torch.save(torch.ones(2, 2), first)
    torch.save(torch.full((2, 2), 2.0), second)
    infer = torch.zeros(2, 4)

    merged = normalize_training_tensor(
        first,
        key="output.0",
        role="P",
        mock_seq_len=2,
        infer=infer,
    )
    assert merged.shape == (2, 4)
    assert torch.equal(merged[:, 2:], torch.full((2, 2), 2.0))

    with pytest.raises(ValueError, match="decode index"):
        normalize_training_tensor(
            first,
            key="output.0",
            role="D",
            mock_seq_len=3,
            infer=infer,
        )
    with pytest.raises(ValueError, match="prefill length"):
        normalize_training_tensor(
            first,
            key="output.0",
            role="P",
            mock_seq_len=3,
            infer=infer,
        )


def test_compare_tensors_handles_empty_and_correlated_values():
    passed, metrics = compare_tensors(
        torch.empty(0), torch.empty(0), rtol=0.0, atol=0.0
    )
    assert passed
    assert metrics.get("all_close") is True

    passed, metrics = compare_tensors(
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert passed
    assert metrics.get("corr_coef") == pytest.approx(1.0)


def test_candidate_and_training_path_conventions(tmp_path):
    candidates = infer_candidates(
        tmp_path,
        layer=2,
        module="dsa_attention_topk_indices",
        mode="real",
        role="D",
        rank=1,
    )
    assert len(candidates) == 4
    assert candidates[0].name.endswith("decode_real_rank1.pt")

    common = {
        "train_root": tmp_path / "train",
        "custom_root": tmp_path / "custom",
        "layer_prefix": "layers",
        "model_prefix": "model",
        "lm_head_name": "head",
    }
    cases = (
        (2, "pangu_transformer_o_conv", "input.0"),
        (2, "gmm_experts_output", "output.0"),
        (2, "dsa_attention_topk_indices", "output.0"),
        (None, "embedding.word_embeddings", "input.0"),
        (None, "rotary_pos_emb.sin", "output.1"),
        (None, "lm_head", "output.0"),
        (None, "other", "output.0"),
        (2, "self_attention.qa_conv", "input.0"),
    )
    for layer, module, expected_key in cases:
        path, key = training_path(layer=layer, module=module, **common)
        assert path.suffix == ".pt"
        assert key == expected_key


def _compare_kwargs(tmp_path, *, layer=0, module="mlp", mode="train_ref", rank=0):
    return {
        "layer": layer,
        "module": module,
        "train_root": tmp_path / "train",
        "infer_root": tmp_path / "infer",
        "custom_root": tmp_path / "custom",
        "mode": mode,
        "role": "P",
        "mock_seq_len": 2,
        "rank": rank,
        "rtol": 1e-5,
        "atol": 1e-5,
        "layer_prefix": "layers",
        "model_prefix": "model",
        "lm_head_name": "head",
    }


def _save_compare_inputs(kwargs, train, infer):
    infer_path = infer_candidates(
        kwargs.get("infer_root"),
        layer=kwargs.get("layer"),
        module=kwargs.get("module"),
        mode=kwargs.get("mode"),
        role=kwargs.get("role"),
        rank=kwargs.get("rank"),
    )[0]
    train_path, key = training_path(
        kwargs.get("train_root"),
        kwargs.get("custom_root"),
        layer=kwargs.get("layer"),
        module=kwargs.get("module"),
        layer_prefix=kwargs.get("layer_prefix"),
        model_prefix=kwargs.get("model_prefix"),
        lm_head_name=kwargs.get("lm_head_name"),
    )
    infer_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(infer, infer_path)
    torch.save({"tensors": {key: train}}, train_path)
    return train_path, infer_path


def test_compare_one_reports_expected_and_missing_inputs(tmp_path):
    skipped = compare_one(
        **_compare_kwargs(tmp_path, module="self_attention.core_attention")
    )
    assert skipped.status == "SKIP"

    kwargs = _compare_kwargs(tmp_path)
    missing_infer = compare_one(**kwargs)
    assert missing_infer.message == "inference dump missing"
    infer_path = infer_candidates(
        kwargs.get("infer_root"),
        layer=0,
        module="mlp",
        mode="train_ref",
        role="P",
        rank=0,
    )[0]
    infer_path.parent.mkdir(parents=True)
    torch.save(torch.ones(2, 3), infer_path)
    missing_train = compare_one(**kwargs)
    assert missing_train.message == "training/reference dump missing"


@pytest.mark.parametrize(
    ("module", "layer", "train", "infer", "rank"),
    (
        ("mlp", 0, torch.ones(2, 3), torch.ones(2, 3), 0),
        ("lm_head", None, torch.ones(2, 1, 3), torch.ones(1, 3), 0),
        ("moe_gate_output", 0, torch.ones(4, 3), torch.ones(2, 3), 1),
        ("gmm_experts_output", 0, torch.ones(2, 3), torch.ones(2, 3), 0),
        ("pangu_transformer_o_conv", 0, torch.ones(2, 4), torch.ones(2, 3), 0),
    ),
)
def test_compare_one_normalizes_special_modules(
    tmp_path, module, layer, train, infer, rank
):
    kwargs = _compare_kwargs(tmp_path, module=module, layer=layer, rank=rank)
    _save_compare_inputs(kwargs, train, infer)
    assert compare_one(**kwargs).status == "PASS"


def test_compare_one_real_router_uses_train_ref_dump_and_reports_bad_payload(tmp_path):
    kwargs = _compare_kwargs(tmp_path, module="moe_router_scores", mode="real")
    _save_compare_inputs(kwargs, torch.zeros(2, 3), torch.ones(2, 3))
    train_ref_path = infer_candidates(
        kwargs.get("infer_root"),
        layer=0,
        module="moe_router_scores",
        mode="train_ref",
        role="P",
        rank=0,
    )[0]
    torch.save(torch.ones(2, 3), train_ref_path)
    assert compare_one(**kwargs).status == "PASS"

    torch.save({"bad": "payload"}, train_ref_path)
    result = compare_one(**kwargs)
    assert result.status == "ERROR"
    assert "does not contain tensor" in result.message


def test_pure_real_compare_reports_missing_and_invalid_pair(tmp_path):
    root = tmp_path / "infer" / "prefill"
    root.mkdir(parents=True)
    left = root / "decorator_layers.1.proj_run0_call1_rank0.pt"
    right = root / "decorator_layers.1.proj_run1_call1_rank0.pt"
    torch.save(torch.ones(2, 3), left)
    results = compare_pure_real(
        tmp_path / "infer",
        role="P",
        rank=0,
        run_a="0",
        run_b="1",
        mock_seq_len=2,
        rtol=1e-5,
        atol=1e-5,
    )
    assert results[0].status == "SKIP"

    torch.save({"bad": "payload"}, right)
    assert (
        compare_pure_real(
            tmp_path / "infer",
            role="P",
            rank=0,
            run_a="0",
            run_b="1",
            mock_seq_len=2,
            rtol=1e-5,
            atol=1e-5,
        )[0].status
        == "ERROR"
    )


def test_discovery_reports_and_parallel_helpers(tmp_path):
    stage = tmp_path / "infer" / "prefill"
    stage.mkdir(parents=True)
    (stage / "decoder.layers.3.mlp.pt").touch()
    (stage / "decoder.layers.1.mlp.pt").touch()
    assert discover_layers(tmp_path / "infer", "P") == [1, 3]

    results = [
        CompareResult(None, "head", "PASS", max_abs_diff=0.0),
        CompareResult(1, "mlp", "FAIL", message="different", max_abs_diff=1.0),
        CompareResult(2, "skip", "SKIP", message="missing"),
    ]
    report = tmp_path / "report"
    write_reports(results, report, "unit")
    assert (report / "summary_unit.json").is_file()
    assert "different" in (report / "summary_unit.md").read_text(encoding="utf-8")
    assert (report / "mlp" / "all_layers_unit.json").is_file()

    assert _run_parallel([], lambda value: value, 2) == []
    assert _run_parallel([1], lambda value: value + 1, 1) == [2]
    assert _run_parallel([1, 2], lambda value: value + 1, 2) == [2, 3]


def test_compare_main_covers_cli_validation_and_both_modes(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="--train-root"):
        compare_main(["--infer-root", str(tmp_path), "--mock-seq-len", "1"])
    with pytest.raises(SystemExit, match="must be used together"):
        compare_main(
            [
                "--infer-root",
                str(tmp_path),
                "--train-root",
                str(tmp_path),
                "--mock-seq-len",
                "1",
                "--layer-start",
                "0",
            ]
        )

    monkeypatch.setattr(
        compare_module,
        "compare_pure_real",
        lambda *_args, **_kwargs: [CompareResult(None, "proj", "PASS")],
    )
    assert (
        compare_main(
            [
                "--infer-root",
                str(tmp_path),
                "--mock-seq-len",
                "1",
                "--compare-mode",
                "pure_real_decorator",
                "--report-dir",
                str(tmp_path / "pure-report"),
            ]
        )
        == 0
    )

    monkeypatch.setattr(
        compare_module,
        "compare_one",
        lambda **_kwargs: CompareResult(0, "mlp", "FAIL"),
    )
    assert (
        compare_main(
            [
                "--infer-root",
                str(tmp_path),
                "--train-root",
                str(tmp_path),
                "--mock-seq-len",
                "1",
                "--layer-start",
                "0",
                "--layer-end",
                "1",
                "--modules",
                "mlp",
                "--workers",
                "1",
                "--report-dir",
                str(tmp_path / "train-report"),
            ]
        )
        == 1
    )
