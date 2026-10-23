# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from omni_npu.diagnostics.accuracy_alignment import installer as installer_module
from omni_npu.diagnostics.accuracy_alignment import train_ref as train_ref_module
from omni_npu.diagnostics.accuracy_alignment.config import (
    AlignmentConfigError,
    TrainRefConfig,
    real_requested,
    requested_mode,
    train_ref_requested,
)
from omni_npu.diagnostics.accuracy_alignment.dump_io import TrainRefDataError
from omni_npu.diagnostics.accuracy_alignment.installer import (
    _AttributeOverride,
    collect_mhc_train_ref_targets,
    collect_train_ref_targets,
    install_train_ref,
)
from omni_npu.diagnostics.accuracy_alignment.mhc import (
    MhcTrainRefHooks,
    MhcTrainRefTarget,
)
from omni_npu.diagnostics.accuracy_alignment.train_ref import (
    TrainRefSession,
    TrainRefTarget,
    dummy_run_guard,
)
from torch import nn


def _config(tmp_path, *, role="P", mock_shape=2, layer_id=0):
    train_dir = tmp_path / "train"
    infer_dir = tmp_path / "infer"
    train_dir.mkdir()
    infer_dir.mkdir()
    return TrainRefConfig(
        train_dump_dir=train_dir,
        infer_dump_dir=infer_dir,
        layer_id=layer_id,
        role=role,
        mock_shape=mock_shape,
    )


def _save_input(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": {"input.0": tensor, "output.0": tensor}}, path)


def test_attribute_override_restores_mhc_fusion_setting():
    model = SimpleNamespace(use_mhc_fusion_op=True)
    override = _AttributeOverride(model, "use_mhc_fusion_op", False)

    assert model.use_mhc_fusion_op is False
    override.close()
    assert model.use_mhc_fusion_op is True


def test_installer_disables_fused_mla_prolog_only_for_session(
    tmp_path,
    monkeypatch,
):
    cfg = _config(tmp_path)

    class Base:
        pass

    class Pangu:
        pass

    class NoopHooks:
        installed_hook_count = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def install(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    attention = SimpleNamespace(use_mlaprolog=True)
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    pangu = Pangu()
    pangu.layers = [layer]
    pangu.start_layer = 0
    pangu.end_layer = 1
    pangu.use_mhc_fusion_op = False
    model = Base()
    model.model = pangu
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        get_model=lambda: model,
    )

    monkeypatch.setattr(
        installer_module,
        "_ultra_model_classes",
        lambda: (Base, Pangu),
    )
    monkeypatch.setattr(installer_module, "ModelLevelTrainRefHooks", NoopHooks)
    monkeypatch.setattr(installer_module, "AttentionTrainRefHooks", NoopHooks)
    monkeypatch.setattr(installer_module, "MoeTrainRefHooks", NoopHooks)
    monkeypatch.setattr(
        installer_module,
        "collect_train_ref_targets",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        installer_module,
        "collect_layer_targets",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        installer_module,
        "collect_mhc_train_ref_targets",
        lambda *_args: [],
    )

    session = install_train_ref(runner, cfg)

    assert session is not None
    assert attention.use_mlaprolog is False
    session.close()
    assert attention.use_mlaprolog is True


class _Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, hidden, marker, *, scale=1):
        self.seen = (hidden.clone(), marker, scale)
        return hidden * scale, marker


def test_train_ref_requested_accepts_explicit_and_legacy_selection():
    assert not train_ref_requested({})
    assert train_ref_requested(
        {"OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True"}
    )
    assert train_ref_requested(
        {
            "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
            "OMNI_NPU_DEBUG_MODE": "train_ref",
        }
    )


def test_legacy_mode_resolution_distinguishes_unset_and_empty_whitelist():
    base = {"OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True"}
    assert requested_mode(base) == "train_ref"
    assert requested_mode({**base, "OMNI_NPU_DEBUG_MOCK_MODULES": ""}) == "real"
    assert real_requested({**base, "OMNI_NPU_DEBUG_MOCK_MODULES": "mlp.*"})


def test_real_module_whitelist_supports_exact_and_prefix_matches(tmp_path):
    cfg = _config(tmp_path)
    cfg = TrainRefConfig(
        **{
            **vars(cfg),
            "mode": "real",
            "mock_modules": ("self_attention.*", "lm_head"),
        }
    )
    assert cfg.should_mock("self_attention.linear_qkv")
    assert not cfg.should_mock("self_attention")
    assert cfg.should_mock("lm_head")
    assert not cfg.should_mock("mlp.linear_fc1")


def test_train_ref_explicit_whitelist_limits_injected_boundaries(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    cfg = TrainRefConfig.from_env(
        {
            "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
            "OMNI_NPU_DEBUG_MODE": "train_ref",
            "OMNI_NPU_DEBUG_TRAIN_DUMP_DIR": str(train_dir),
            "OMNI_NPU_DEBUG_INFER_DUMP_DIR": str(tmp_path / "infer"),
            "OMNI_NPU_DEBUG_MOCK_SHAPE": "2",
            "OMNI_NPU_MHC_DEBUG_ROLE": "P",
            "OMNI_NPU_DEBUG_MOCK_MODULES": (
                "self_attention.q_layernorm,lm_head"
            ),
        }
    )

    assert cfg.mock_modules == ("self_attention.q_layernorm", "lm_head")
    assert cfg.should_mock("self_attention.q_layernorm")
    assert cfg.should_mock("lm_head")
    assert not cfg.should_mock("embedding")


def test_config_accepts_legacy_dump_environment_names(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    cfg = TrainRefConfig.from_env(
        {
            "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
            "OMNI_NPU_DEBUG_MODE": "train_ref",
            "OMNI_NPU_MHC_DEBUG_TRAIN_DUMP_DIR": str(train_dir),
            "OMNI_NPU_MHC_DEBUG_INFER_DUMP_DIR": str(tmp_path / "infer"),
            "OMNI_NPU_MHC_DEBUG_MOCK_SHAPE": "7",
            "OMNI_NPU_MHC_DEBUG_ROLE": "D",
        }
    )
    assert cfg.role == "D"
    assert cfg.mock_shape == 7
    assert cfg.stage == "decode"


def test_config_accepts_optional_mhc_custom_dump_dir(tmp_path):
    train_dir = tmp_path / "train"
    custom_dir = tmp_path / "custom"
    train_dir.mkdir()
    custom_dir.mkdir()
    cfg = TrainRefConfig.from_env(
        {
            "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
            "OMNI_NPU_DEBUG_MODE": "train_ref",
            "OMNI_NPU_DEBUG_TRAIN_DUMP_DIR": str(train_dir),
            "OMNI_NPU_DEBUG_INFER_DUMP_DIR": str(tmp_path / "infer"),
            "OMNI_NPU_DEBUG_MOCK_SHAPE": "2",
            "OMNI_NPU_MHC_DEBUG_ROLE": "P",
            "OMNI_NPU_MHC_CUSTOM_DUMP_DIR": str(custom_dir),
        }
    )
    assert cfg.mhc_custom_dump_dir == custom_dir
    assert cfg.mhc_training_dump_path("attn", "pre_in") == (
        custom_dir / "pangu_transformer_hc_pre_in_1_rank0.pt"
    )


def test_config_parses_layer_list_ranges_and_all(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    base = {
        "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
        "OMNI_NPU_DEBUG_MODE": "real",
        "OMNI_NPU_DEBUG_TRAIN_DUMP_DIR": str(train_dir),
        "OMNI_NPU_DEBUG_INFER_DUMP_DIR": str(tmp_path / "infer"),
        "OMNI_NPU_DEBUG_MOCK_SHAPE": "2",
        "OMNI_NPU_MHC_DEBUG_ROLE": "P",
    }
    selected = TrainRefConfig.from_env(
        {**base, "OMNI_NPU_DEBUG_LAYER_IDS": "0,3-5,3"}
    )
    all_layers = TrainRefConfig.from_env({**base, "OMNI_NPU_DEBUG_LAYER_IDS": "all"})

    assert selected.requested_layer_ids(8) == (0, 3, 4, 5)
    assert all_layers.requested_layer_ids(4) == (0, 1, 2, 3)


def test_config_rejects_incomplete_enabled_mode(tmp_path):
    with pytest.raises(AlignmentConfigError, match="incomplete"):
        TrainRefConfig.from_env(
            {
                "OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE": "True",
                "OMNI_NPU_DEBUG_MODE": "train_ref",
                "OMNI_NPU_DEBUG_TRAIN_DUMP_DIR": str(tmp_path),
            }
        )


def test_session_replaces_only_first_tensor_and_dumps_tuple_output(tmp_path):
    cfg = _config(tmp_path)
    module = _Probe()
    train_path = cfg.training_dump_path("decoder_layer")
    golden = torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
    _save_input(train_path, golden)
    target = TrainRefTarget("decoder_layer_0", "decoder_layer", module, train_path)
    session = TrainRefSession(cfg, rank=3)
    session.install([target])

    marker = object()
    original = torch.zeros(2, 2)
    result = module(original, marker, scale=2)

    assert torch.equal(module.seen[0], golden[:2])
    assert module.seen[1] is marker
    assert module.seen[2] == 2
    assert torch.equal(result[0], golden[:2] * 2)
    output_path = (
        cfg.infer_dump_dir / "prefill" / "decoder_layer_0_prefill_train_ref_rank3.pt"
    )
    assert torch.equal(torch.load(output_path, weights_only=False), result[0])


def test_real_session_dumps_without_loading_or_replacing_unselected_input(tmp_path):
    cfg = _config(tmp_path)
    cfg = TrainRefConfig(**{**vars(cfg), "mode": "real", "mock_modules": ()})
    module = _Probe()
    missing = cfg.training_dump_path("decoder_layer")
    session = TrainRefSession(cfg, rank=1)
    session.install(
        [TrainRefTarget("decoder_layer_0", "decoder_layer", module, missing)]
    )
    runtime = torch.zeros(2, 2)

    output = module(runtime, object())

    assert torch.equal(module.seen[0], runtime)
    assert torch.equal(output[0], runtime)
    assert (
        cfg.infer_dump_dir / "prefill" / "decoder_layer_0_prefill_real_rank1.pt"
    ).is_file()


def test_real_session_ignores_nonmatching_request_shape(tmp_path):
    cfg = _config(tmp_path, mock_shape=3)
    cfg = TrainRefConfig(**{**vars(cfg), "mode": "real", "mock_modules": ()})
    module = _Probe()
    session = TrainRefSession(cfg, rank=0)
    session.install(
        [
            TrainRefTarget(
                "decoder_layer_0",
                "decoder_layer",
                module,
                cfg.training_dump_path("decoder_layer"),
            )
        ]
    )

    module(torch.zeros(2, 2), object())

    assert not list(cfg.infer_dump_dir.rglob("*.pt"))


def test_shape_mismatch_fails_instead_of_using_runtime_input(tmp_path):
    cfg = _config(tmp_path, mock_shape=3)
    module = _Probe()
    train_path = cfg.training_dump_path("decoder_layer")
    _save_input(train_path, torch.ones(3, 2))
    session = TrainRefSession(cfg, rank=0)
    session.install(
        [TrainRefTarget("decoder_layer_0", "decoder_layer", module, train_path)]
    )
    with pytest.raises(TrainRefDataError, match="runtime token count"):
        module(torch.zeros(2, 2), object())


def test_prefill_lm_head_accepts_single_selected_token(tmp_path):
    cfg = _config(tmp_path, mock_shape=3)
    module = _Probe()
    train_path = cfg.training_dump_path("lm_head")
    golden = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8)
    _save_input(train_path, golden)
    session = TrainRefSession(cfg, rank=0)
    session.install([TrainRefTarget("lm_head", "lm_head", module, train_path)])

    module(torch.zeros(1, 8), object())

    assert torch.equal(module.seen[0], golden[2, 0].unsqueeze(0))


def test_missing_training_file_fails_during_install(tmp_path):
    cfg = _config(tmp_path)
    session = TrainRefSession(cfg, rank=0)
    with pytest.raises(TrainRefDataError, match="does not exist"):
        session.install(
            [
                TrainRefTarget(
                    "embedding",
                    "embedding",
                    _Probe(),
                    cfg.training_dump_path("embedding"),
                )
            ]
        )


def test_dummy_run_does_not_replace_or_dump(tmp_path):
    cfg = _config(tmp_path)
    module = _Probe()
    train_path = cfg.training_dump_path("decoder_layer")
    _save_input(train_path, torch.ones(2, 2))
    session = TrainRefSession(cfg, rank=0)
    session.install(
        [TrainRefTarget("decoder_layer_0", "decoder_layer", module, train_path)]
    )
    runtime = torch.zeros(2, 2)
    with dummy_run_guard():
        result = module(runtime, object())
    assert torch.equal(module.seen[0], runtime)
    assert torch.equal(result[0], runtime)
    assert not list(cfg.infer_dump_dir.rglob("*.pt"))


def test_capture_does_not_replace_or_dump(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    module = _Probe()
    train_path = cfg.training_dump_path("decoder_layer")
    _save_input(train_path, torch.ones(2, 2))
    session = TrainRefSession(cfg, rank=0)
    session.install(
        [TrainRefTarget("decoder_layer_0", "decoder_layer", module, train_path)]
    )
    monkeypatch.setattr(train_ref_module, "_capturing", lambda: True)
    runtime = torch.zeros(2, 2)

    result = module(runtime, object())

    assert torch.equal(module.seen[0], runtime)
    assert torch.equal(result[0], runtime)
    assert not list(cfg.infer_dump_dir.rglob("*.pt"))


def test_unset_forward_context_keeps_direct_eager_hook_active(monkeypatch):
    from vllm import forward_context

    def raise_unset_context():
        raise AssertionError("Forward context is not set")

    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        raise_unset_context,
    )

    assert train_ref_module.should_skip_runtime_hook() is False


def test_reinstall_removes_old_hooks(tmp_path):
    cfg = _config(tmp_path)
    module = _Probe()
    train_path = cfg.training_dump_path("decoder_layer")
    _save_input(train_path, torch.ones(2, 2))
    target = TrainRefTarget("decoder_layer_0", "decoder_layer", module, train_path)
    session = TrainRefSession(cfg, rank=0)
    session.install([target])
    session.install([target])
    assert session.installed_hook_count == 2


def test_collect_targets_obeys_local_pp_ownership(tmp_path):
    cfg = _config(tmp_path, layer_id=1)

    class PPMissingLayer(nn.Module):
        pass

    layer0 = PPMissingLayer()
    layer1 = nn.Identity()
    pangu = SimpleNamespace(
        embed_tokens=PPMissingLayer(),
        layers=[layer0, layer1],
        start_layer=1,
        end_layer=2,
    )
    model = SimpleNamespace(model=pangu, lm_head=nn.Identity())
    targets = collect_train_ref_targets(model, cfg)
    assert [target.module_name for target in targets] == [
        "decoder_layer_1",
    ]


class _MhcProbe:
    def __init__(self):
        self.seen = {}

    def mhc_pre(self, hidden_states):
        self.seen["pre"] = hidden_states.clone()
        return hidden_states + 1, hidden_states + 2, hidden_states + 3

    def resolve_sinkhorn(self, h_res, _task_key):
        self.seen["sinkhorn"] = h_res.clone()
        return h_res + 4

    def resolve_fused_split_sinkhorn(self, h_post, h_res, _task_key):
        self.seen["fused_sinkhorn"] = (h_post.clone(), h_res.clone())
        return h_post + 5, h_res + 6

    def mhc_post(self, hidden_states, h_post, residual, h_res):
        self.seen["post"] = tuple(
            value.clone() for value in (hidden_states, h_post, residual, h_res)
        )
        return hidden_states + h_post + residual + h_res

    def mhc_sandwich_norm_post_preonly(
        self,
        hidden_states,
        residual,
        h_post,
        h_res,
        *_args,
        **_kwargs,
    ):
        self.seen["sandwich"] = tuple(
            value.clone() for value in (hidden_states, residual, h_post, h_res)
        )
        return hidden_states + h_post + h_res, residual

    def launch_fused_split_sinkhorn(self, residual, _task_key):
        self.seen["launch"] = residual.clone()
        return residual + 7, residual + 8


class _DecoderProbe:
    def forward_mhc_fused(self, hidden_states):
        return hidden_states + 9, None


def _save_mhc_inputs(cfg, branch, values):
    for boundary, value in values.items():
        path = cfg.mhc_training_dump_path(branch, boundary)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)


def test_mhc_hooks_inject_normal_boundaries_and_restore_methods(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    cfg = _config(tmp_path)
    cfg = TrainRefConfig(**{**vars(cfg), "mhc_custom_dump_dir": custom_dir})
    references = {
        "pre_in": torch.full((3, 2), 10.0),
        "post_in": torch.full((3, 2), 20.0),
        "post_h_post": torch.full((3, 2), 30.0),
        "post_residual": torch.full((3, 2), 40.0),
        "post_h_res": torch.full((3, 2), 50.0),
    }
    _save_mhc_inputs(cfg, "attn", references)
    module = _MhcProbe()
    decoder = _DecoderProbe()
    hooks = MhcTrainRefHooks(cfg, rank=2)
    hooks.install([MhcTrainRefTarget(0, "attn", module, decoder)])

    pre_output = module.mhc_pre(torch.zeros(2, 2))
    pre_seen = module.seen.get("pre")
    pre_reference = references.get("pre_in")
    assert pre_seen is not None and pre_reference is not None
    assert torch.equal(pre_seen, pre_reference[:2])
    assert torch.equal(pre_output[0], pre_reference[:2] + 1)

    module.resolve_sinkhorn(torch.zeros(2, 2), "task")
    post_output = module.mhc_post(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
    )
    post_seen = module.seen.get("post")
    assert post_seen is not None
    assert torch.equal(post_seen[0], references.get("post_in")[:2])
    assert torch.equal(post_seen[1], references.get("post_h_post")[:2])
    assert torch.equal(post_seen[2], references.get("post_residual")[:2])
    assert torch.equal(post_seen[3], references.get("post_h_res")[:2])
    assert torch.equal(
        post_output,
        sum(value[:2] for key, value in references.items() if key != "pre_in"),
    )
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "layer_0_attn_mhc_pre_prefill_train_ref_rank2.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "pangu_transformer_hc_pre_out_1_prefill_train_ref_rank2.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "layer_0_attn_mhc_sinkhorn_prefill_train_ref_rank2.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "layer_0_attn_mhc_post_prefill_train_ref_rank2.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "pangu_transformer_hc_post_out_1_prefill_train_ref_rank2.pt"
    ).is_file()

    hooks.close()
    module.mhc_pre(torch.zeros(2, 2))
    assert torch.equal(module.seen.get("pre"), torch.zeros(2, 2))


def test_mhc_hooks_cover_fused_method_boundaries(tmp_path):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    cfg = _config(tmp_path)
    cfg = TrainRefConfig(**{**vars(cfg), "mhc_custom_dump_dir": custom_dir})
    references = {
        "pre_in": torch.full((3, 2), 1.0),
        "post_in": torch.full((3, 2), 2.0),
        "post_h_post": torch.full((3, 2), 3.0),
        "post_residual": torch.full((3, 2), 4.0),
        "post_h_res": torch.full((3, 2), 5.0),
    }
    _save_mhc_inputs(cfg, "mlp", references)
    module = _MhcProbe()
    decoder = _DecoderProbe()
    hooks = MhcTrainRefHooks(cfg, rank=0)
    hooks.install([MhcTrainRefTarget(0, "mlp", module, decoder)])

    module.launch_fused_split_sinkhorn(torch.zeros(2, 2), "task")
    assert torch.equal(module.seen.get("launch"), references.get("pre_in")[:2])
    module.resolve_fused_split_sinkhorn(torch.zeros(2, 2), torch.zeros(2, 2), "task")
    module.mhc_sandwich_norm_post_preonly(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
    )
    sandwich_seen = module.seen.get("sandwich")
    assert sandwich_seen is not None
    assert torch.equal(sandwich_seen[0], references.get("post_in")[:2])
    assert torch.equal(sandwich_seen[1], references.get("post_residual")[:2])
    decoder.forward_mhc_fused(torch.zeros(2, 2))
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "layer_0_mlp_mhc_post_fused_prefill_train_ref_rank0.pt"
    ).is_file()
    assert (
        cfg.infer_dump_dir / "prefill" / "decoder_layer_0_prefill_train_ref_rank0.pt"
    ).is_file()


def test_mhc_hooks_dump_without_optional_custom_references(tmp_path):
    cfg = _config(tmp_path)
    module = _MhcProbe()
    decoder = _DecoderProbe()
    hooks = MhcTrainRefHooks(cfg, rank=0)
    hooks.install([MhcTrainRefTarget(0, "attn", module, decoder)])
    runtime = torch.zeros(2, 2)

    output = module.mhc_pre(runtime)

    assert torch.equal(module.seen.get("pre"), runtime)
    assert torch.equal(output[0], runtime + 1)
    assert (
        cfg.infer_dump_dir
        / "prefill"
        / "pangu_transformer_hc_pre_out_1_prefill_train_ref_rank0.pt"
    ).is_file()


def test_collect_mhc_targets_supports_dump_only_without_custom_dir(tmp_path):
    cfg = _config(tmp_path)
    layer = SimpleNamespace(
        use_mhc=True,
        attn_mhc_module=object(),
        mlp_mhc_module=object(),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(start_layer=0, end_layer=1, layers=[layer])
    )
    targets = collect_mhc_train_ref_targets(model, cfg)
    assert [target.branch for target in targets] == ["attn", "mlp"]

    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    cfg = TrainRefConfig(**{**vars(cfg), "mhc_custom_dump_dir": custom_dir})
    branch_values = {
        boundary: torch.ones(2, 2)
        for boundary in (
            "pre_in",
            "post_in",
            "post_h_post",
            "post_residual",
            "post_h_res",
        )
    }
    _save_mhc_inputs(cfg, "attn", branch_values)
    _save_mhc_inputs(cfg, "mlp", branch_values)
    targets = collect_mhc_train_ref_targets(model, cfg)
    assert [target.branch for target in targets] == ["attn", "mlp"]


def test_installer_rejects_graph_mode_before_model_hooks(tmp_path, monkeypatch):
    cfg = _config(tmp_path)

    class Base:
        pass

    class Pangu:
        pass

    model = Base()
    model.model = Pangu()
    monkeypatch.setattr(
        installer_module,
        "_ultra_model_classes",
        lambda: (Base, Pangu),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        get_model=lambda: model,
    )
    with pytest.raises(AlignmentConfigError, match="enforce_eager"):
        install_train_ref(runner, cfg)


def test_installer_skips_unsupported_model_before_mode_checks(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        installer_module,
        "_ultra_model_classes",
        lambda: (type("Base", (), {}), type("Pangu", (), {})),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        get_model=lambda: object(),
    )

    assert install_train_ref(runner, cfg) is None
