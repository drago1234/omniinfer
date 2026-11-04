# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EPLB metadata update contract of OpenPanguV2ForCausalLM.

A rebalance may move experts between devices but must never change how many
physical experts a single device holds: the expert weight buffers were sized
for that count at build time. The check has to survive `python -O`, so it is a
raise rather than an assert.
"""

from types import SimpleNamespace

import pytest

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


def _bare_model(num_local_physical_experts, num_logical_experts=8):
    model = model_mod.OpenPanguV2ForCausalLM.__new__(
        model_mod.OpenPanguV2ForCausalLM
    )
    model.num_local_physical_experts = num_local_physical_experts
    model.num_logical_experts = num_logical_experts
    model.model = SimpleNamespace(layers=[])
    return model


def test_update_metadata_rejects_a_changed_per_device_count():
    """A different per-device count would desync the expert weight buffers."""
    model = _bare_model(num_local_physical_experts=4)

    with pytest.raises(ValueError, match="per-device expert count"):
        model.update_physical_experts_metadata(16, 8)


def test_update_metadata_recomputes_redundant_experts():
    """Matching counts update the physical/redundant totals in place."""
    model = _bare_model(num_local_physical_experts=4, num_logical_experts=8)

    model.update_physical_experts_metadata(12, 4)

    assert model.num_physical_experts == 12
    assert model.num_local_physical_experts == 4
    assert model.num_redundant_experts == 4


@pytest.fixture
def build_moe(monkeypatch):
    """Run the real constructor with CPU weights and fake distributed groups."""
    import torch
    from unittest.mock import Mock

    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: "test_ep")
    group = SimpleNamespace(
        rank=lambda: 1, size=lambda: 2, _get_backend=lambda device: backend
    )
    monkeypatch.setattr(model_mod, "get_ep_group", lambda: SimpleNamespace(
        device_group=group, rank_in_group=1, world_size=2
    ))
    monkeypatch.setattr(model_mod, "get_tp_group", lambda: SimpleNamespace(
        rank_in_group=0, world_size=1
    ))
    monkeypatch.setattr(model_mod, "current_platform", SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(model_mod, "on_ascend910b", lambda: False)
    monkeypatch.setattr(model_mod, "ReplicatedLinear", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(model_mod, "OpenPanguV2MLP", lambda **kwargs: torch.nn.Identity())
    runner = Mock(return_value=SimpleNamespace(
        routed_experts=SimpleNamespace(w13_weight=torch.zeros(1))
    ))
    monkeypatch.setattr(model_mod, "FusedMoE", runner)
    monkeypatch.setattr(model_mod, "model_extra_config", SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            moe_comm_strategy="allgather", gmm_fr_token_threshold=0,
            moe_dispatch_combine_max_batch_size=16, moe_seq_split_length=16,
        )
    ))
    config = SimpleNamespace(
        num_hidden_layers=4, n_routed_experts=8, n_shared_experts=1,
        routed_scaling_factor=1.0, hidden_size=4, hidden_act="silu",
        moe_intermediate_size=8, num_experts_per_tok=2, norm_topk_prob=True,
    )

    def build(*, layer_idx=3, enabled=True, redundant=4):
        parallel = SimpleNamespace(
            use_sequence_parallel_moe=False, enable_eplb=enabled,
            eplb_config=SimpleNamespace(num_redundant_experts=redundant),
        )
        moe = model_mod.OpenPanguV2MOE(
            config, parallel, prefix=f"model.layers.{layer_idx}.mlp"
        )
        return moe, runner.call_args.kwargs

    return build


@pytest.mark.parametrize(
    "layer_idx,enabled,redundant,expected_enabled,expected_redundant",
    [
        (3, True, 4, True, 4),
        (4, True, 4, False, 0),
        (5, True, 4, False, 0),
        (3, True, 0, True, 0),
        (4, True, 0, True, 0),
        (3, False, 4, False, 0),
        (4, False, 4, False, 0),
    ],
)
def test_moe_init_eplb_slots_and_mtp_isolation(
    build_moe, layer_idx, enabled, redundant, expected_enabled, expected_redundant
):
    moe, runner_args = build_moe(
        layer_idx=layer_idx, enabled=enabled, redundant=redundant
    )
    assert moe.is_mtp_layer == (layer_idx >= 4)
    assert moe.enable_eplb is expected_enabled
    assert moe.n_redundant_experts == expected_redundant
    assert runner_args["enable_eplb"] is expected_enabled
    assert runner_args["num_redundant_experts"] == expected_redundant
    assert moe.n_logical_experts == 8
    assert moe.n_physical_experts == 8 + expected_redundant
    assert moe.n_local_physical_experts == (8 + expected_redundant) // 2
    assert moe.physical_expert_start == moe.n_local_physical_experts
    assert moe.physical_expert_end == moe.n_physical_experts


def test_moe_init_rejects_uneven_physical_expert_partition(build_moe):
    with pytest.raises(ValueError, match=r"logical=8, redundant=1, ep_size=2"):
        build_moe(redundant=1)
