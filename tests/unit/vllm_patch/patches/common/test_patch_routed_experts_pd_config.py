# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# pylint: skip-file
"""Tests for routed-experts compatibility with PD configuration."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load directly so the tests do not require an NPU runtime.
_PATCH_PATH = (
    Path(__file__).resolve().parents[5]
    / "omni"
    / "vllm_patches"
    / "patches"
    / "common"
    / "patch_routed_experts.py"
)

KV_GUARD_MSG = (
    "--enable-return-routed-experts is incompatible with KV "
    "connectors (PD disaggregation, KV cache offload)."
)


@pytest.fixture(scope="module")
def patch_mod():
    spec = importlib.util.spec_from_file_location(
        "_t_patch_routed_experts_pd_config", _PATCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(
    *,
    enabled=True,
    kv_role="kv_producer",
    pipeline_parallel_size=1,
    use_v2_model_runner=False,
):
    kv = SimpleNamespace(kv_role=kv_role) if kv_role is not None else None
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=enabled),
        kv_transfer_config=kv,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            worker_cls="auto",
        ),
        use_v2_model_runner=use_v2_model_runner,
        v2_validated=0,
    )

    def _validate_v2_model_runner():
        cfg.v2_validated += 1

    cfg._validate_v2_model_runner = _validate_v2_model_runner
    return cfg


def _stub_post_init(patch_mod, monkeypatch, *, first_pass_error=None):
    """Stub the KV guard and later configuration side effects."""
    observed = []

    def fake(cfg):
        flag = cfg.model_config.enable_return_routed_experts
        observed.append(flag)
        if flag and cfg.kv_transfer_config is not None:
            if first_pass_error is not None:
                raise first_pass_error
            raise ValueError(KV_GUARD_MSG)
        cfg.parallel_config.worker_cls = "stub_platform.worker.NPUWorker"
        cfg.compilation_mode = "PIECEWISE"
        if cfg.use_v2_model_runner:
            cfg._validate_v2_model_runner()

    monkeypatch.setattr(patch_mod, "_ORIGINAL_VLLM_CONFIG_POST_INIT", fake)
    return observed


def _call_patch(patch_mod, cfg):
    """Invoke the patched ``__post_init__`` the way VllmConfig.__init__ does."""
    return patch_mod.VllmConfigPDCompatPatch.__dict__["__post_init__"](cfg)


def test_chat_final_forwards_optional_kwargs(patch_mod, monkeypatch):
    """The relay forwards optional parser and multimodal arguments."""
    observed = {}
    response = SimpleNamespace(choices=[SimpleNamespace()])

    async def upstream(
        serving,
        request,
        result_generator,
        *args,
        **kwargs,
    ):
        observed["args"] = args
        observed["kwargs"] = kwargs
        observed["results"] = [item async for item in result_generator]
        return response

    monkeypatch.setattr(patch_mod, "_ORIGINAL_CHAT_FULL", upstream)
    monkeypatch.setattr(patch_mod.envs, "OMNI_REUSE_PREFILLED_TOKENS", False)
    monkeypatch.setattr(patch_mod.envs, "OMNI_SKIP_DECODE_TOKENIZE", False)

    final_res = SimpleNamespace(
        outputs=[SimpleNamespace(routed_experts=None)],
        kv_transfer_params=None,
    )
    request = SimpleNamespace(kv_transfer_params=None)

    async def results():
        yield final_res

    result = asyncio.run(
        patch_mod.ExpertIdServingChatFinal.chat_completion_full_generator(
            SimpleNamespace(),
            request,
            results(),
            "req-1",
            "model",
            [],
            None,
            None,
            parser="parser-sentinel",
            mm_token_counts={"image": 2},
        )
    )

    assert result is response
    assert observed["results"] == [final_res]
    assert observed["kwargs"] == {
        "parser": "parser-sentinel",
        "mm_token_counts": {"image": 2},
    }


def test_pd_decode_final_only_uses_request_saved_before_upstream_free(
    patch_mod, monkeypatch
):
    """Snapshot terminal requests before native cleanup removes them."""
    request = SimpleNamespace(request_id="req-final")
    existing = SimpleNamespace(
        request_id="req-final",
        finish_reason="length",
        new_token_ids=[303],
        routed_experts="upstream-last-step",
    )
    scheduler = SimpleNamespace(
        routed_experts_mgr=SimpleNamespace(
            is_pd_disaggregation=True,
            is_pd_prefill=False,
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enable_return_routed_experts=True)
        ),
        requests={"req-final": request},
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-final": 1})
    saved_slot_mapping = object()
    recomputed = object()
    observed = {}

    def get_slot_mapping(current_scheduler, current_request):
        assert current_scheduler is scheduler
        assert current_request is request
        return saved_slot_mapping

    def original(current_scheduler, *_args):
        current_scheduler.requests.pop("req-final")
        return {0: SimpleNamespace(outputs=[existing])}

    def recompute(mgr, current_request, slot_mapping, **kwargs):
        observed.update(
            mgr=mgr,
            request=current_request,
            slot_mapping=slot_mapping,
            kwargs=kwargs,
        )
        return recomputed

    monkeypatch.setattr(patch_mod, "_get_request_slot_mapping", get_slot_mapping)
    monkeypatch.setattr(
        patch_mod.SchedulerRoutedExpertsPatch, "_upstream_update_from_output", original
    )
    monkeypatch.setattr(
        patch_mod, "_get_routed_experts_from_slot_mapping", recompute
    )

    result = patch_mod.SchedulerRoutedExpertsPatch.update_from_output(
        scheduler,
        scheduler_output,
        SimpleNamespace(),
    )

    assert result[0].outputs[0].routed_experts is recomputed
    assert observed == {
        "mgr": scheduler.routed_experts_mgr,
        "request": request,
        "slot_mapping": saved_slot_mapping,
        "kwargs": {"stopped": True, "num_new_tokens": 1},
    }


def test_guard_matches_upstream_pd_rejection_message(patch_mod):
    cfg = _cfg(enabled=True, kv_role="kv_producer")
    exc = ValueError(KV_GUARD_MSG)
    assert patch_mod._is_pd_routed_experts_guard(cfg, exc) is True


def test_guard_ignores_unrelated_value_error(patch_mod):
    cfg = _cfg(enabled=True, kv_role="kv_producer")
    exc = ValueError("some other config problem: pipeline parallelism > 1")
    assert patch_mod._is_pd_routed_experts_guard(cfg, exc) is False


def test_guard_ignores_when_feature_disabled(patch_mod):
    cfg = _cfg(enabled=False, kv_role="kv_producer")
    exc = ValueError(KV_GUARD_MSG)
    assert patch_mod._is_pd_routed_experts_guard(cfg, exc) is False


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_bypass_runs_full_post_init_and_keeps_side_effects(
    patch_mod, monkeypatch, role
):
    """The retry preserves initializer side effects and restores the flag."""
    cfg = _cfg(enabled=True, kv_role=role)
    observed = _stub_post_init(patch_mod, monkeypatch)

    _call_patch(patch_mod, cfg)

    assert observed == [True, False]
    assert cfg.model_config.enable_return_routed_experts is True
    assert cfg.parallel_config.worker_cls == "stub_platform.worker.NPUWorker"
    assert cfg.compilation_mode == "PIECEWISE"


def test_bypass_restores_flag_when_second_pass_also_fails(
    patch_mod, monkeypatch
):
    """A genuine error in the re-run must propagate - and still not leak the
    hidden flag state."""
    cfg = _cfg(enabled=True, kv_role="kv_producer")

    def fake(cfg):
        if cfg.model_config.enable_return_routed_experts:
            raise ValueError(KV_GUARD_MSG)
        raise ValueError("some other config problem")

    monkeypatch.setattr(patch_mod, "_ORIGINAL_VLLM_CONFIG_POST_INIT", fake)

    with pytest.raises(ValueError, match="some other config problem"):
        _call_patch(patch_mod, cfg)
    assert cfg.model_config.enable_return_routed_experts is True


def test_kv_both_is_not_bypassed(patch_mod, monkeypatch):
    """kv_both / KV offload have no relay path; upstream's rejection stands."""
    cfg = _cfg(enabled=True, kv_role="kv_both")
    _stub_post_init(patch_mod, monkeypatch)

    with pytest.raises(ValueError, match="incompatible with KV"):
        _call_patch(patch_mod, cfg)
    assert cfg.model_config.enable_return_routed_experts is True
    assert cfg.parallel_config.worker_cls == "auto"


def test_unrelated_value_error_propagates_untouched(patch_mod, monkeypatch):
    cfg = _cfg(enabled=True, kv_role="kv_producer")
    _stub_post_init(
        patch_mod,
        monkeypatch,
        first_pass_error=ValueError(
            "--enable-return-routed-experts is incompatible with "
            "pipeline parallelism (PP > 1)."
        ),
    )

    with pytest.raises(ValueError, match="pipeline parallelism"):
        _call_patch(patch_mod, cfg)
    assert cfg.model_config.enable_return_routed_experts is True


def test_bypass_reasserts_pp_rejection(patch_mod, monkeypatch):
    """Even if the first-pass error was the KV guard, PP > 1 must not slip
    through the bypass (defensive re-assert; upstream checks PP first)."""
    cfg = _cfg(enabled=True, kv_role="kv_producer", pipeline_parallel_size=2)
    _stub_post_init(patch_mod, monkeypatch)

    with pytest.raises(ValueError, match="pipeline parallelism"):
        _call_patch(patch_mod, cfg)
    assert cfg.model_config.enable_return_routed_experts is True


def test_v2_model_runner_revalidated_after_flag_restore(
    patch_mod, monkeypatch
):
    """With the flag hidden, the re-run's V2 validation does not see routed
    experts capture; the patch must re-validate once the flag is back so an
    explicit VLLM_USE_V2_MODEL_RUNNER=1 still gets upstream's refusal."""
    cfg = _cfg(enabled=True, kv_role="kv_consumer", use_v2_model_runner=True)
    _stub_post_init(patch_mod, monkeypatch)

    _call_patch(patch_mod, cfg)

    assert cfg.v2_validated == 2


def test_v2_revalidation_skipped_when_v1(patch_mod, monkeypatch):
    cfg = _cfg(enabled=True, kv_role="kv_consumer", use_v2_model_runner=False)
    _stub_post_init(patch_mod, monkeypatch)

    _call_patch(patch_mod, cfg)

    assert cfg.v2_validated == 0


def test_kv_both_is_not_an_allowed_pd_role():
    """Only producer and consumer roles have a prompt-routing relay."""
    src = _PATCH_PATH.read_text()
    assert 'role not in ("kv_producer", "kv_consumer")' in src, (
        "PD bypass must remain restricted to kv_producer / kv_consumer; "
        "widening it lets kv_both / KV offload through, where slot semantics "
        "genuinely break."
    )
