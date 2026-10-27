# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Branch coverage for ``OpenPanguV2ForCausalLM.load_weights``.

Commit 5d5fb3eee3c5917a850a114818c930d248f44a46 refactored ``load_weights`` in
``pangu_v2_moe`` to deduplicate the layout with ``pangu_ultra_moe``. The change
only touched two spots:

* ``_skip_weight`` now merges the ``rotary_emb.inv_freq`` and the tied
  ``lm_head.weight`` checks into a single ``or`` expression.
* The main loop renamed the results of ``_try_load_stacked`` /
  ``_try_load_expert`` and collapsed the expert "skip / record" branches into
  one ``if expert_name == "" or expert_name is not None:`` block.

These tests pin every branch of the new code paths so the incremental
coverage of the commit stays at 100%.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from omni_npu.v1.models.pangu import pangu_v2_moe as pangu_moe

pytestmark = pytest.mark.unit


def _make_owner(params_dict, config=None, num_redundant_experts=0):
    """Build a minimal ``OpenPanguV2ForCausalLM`` shell for ``load_weights``.

    Only the attributes read by ``load_weights`` are wired up; the heavy vLLM
    model construction is bypassed via ``__new__`` and a stubbed
    ``named_parameters`` so each branch can be exercised in isolation.
    """
    owner = pangu_moe.OpenPanguV2ForCausalLM.__new__(pangu_moe.OpenPanguV2ForCausalLM)
    nn.Module.__init__(owner)
    owner.config = config or SimpleNamespace(
        tie_word_embeddings=False,
        n_routed_experts=0,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    owner.num_redundant_experts = num_redundant_experts
    # ``fused_moe_make_expert_params_mapping`` is patched per-test, so the model
    # payload itself is never inspected.
    owner.model = object()
    owner.named_parameters = lambda: list(params_dict.items())
    return owner


def _copying_default_loader(param, loaded_weight):
    """A ``default_weight_loader`` stand-in that actually copies the data.

    Using this as the ``side_effect`` of a MagicMock lets a test assert both the
    call count and that the weight really landed in the parameter, without
    pulling in vLLM's real loader (which assumes a specific runtime).
    """
    with torch.no_grad():
        param.data.copy_(loaded_weight)


@contextmanager
def _patched_deps(**overrides):
    """Patch every ``load_weights`` external dependency in one place."""
    defaults = {
        "fused_moe_make_expert_params_mapping": MagicMock(return_value=[]),
        "is_pp_missing_parameter": MagicMock(return_value=False),
        "high_throughout": MagicMock(return_value=False),
        "maybe_remap_kv_scale_name": MagicMock(side_effect=lambda name, _: name),
        "get_spec_layer_idx_from_weight_name": MagicMock(return_value=None),
        "default_weight_loader": MagicMock(side_effect=_copying_default_loader),
    }
    defaults.update(overrides)
    patchers = {name: patch.object(pangu_moe, name, value) for name, value in defaults.items()}
    mocks = {name: patcher.start() for name, patcher in patchers.items()}
    try:
        yield mocks
    finally:
        for patcher in patchers.values():
            patcher.stop()


# ---------------------------------------------------------------------------
# _skip_weight: the merged ``or`` / ``and`` condition
# ---------------------------------------------------------------------------


def test_skip_weight_skips_rotary_emb_inv_freq():
    """The ``or`` left operand (``rotary_emb.inv_freq``) short-circuits to True."""
    owner = _make_owner({})
    with _patched_deps() as mocks:
        loaded = owner.load_weights(
            [("model.layers.0.self_attn.rotary_emb.inv_freq", torch.zeros(4))]
        )
    assert loaded == set()
    # Skipped before any loader runs.
    mocks["default_weight_loader"].assert_not_called()


def test_skip_weight_skips_tied_lm_head_weight():
    """The ``or`` right operand (tied ``lm_head.weight``) is True when tied."""
    config = SimpleNamespace(
        tie_word_embeddings=True,
        n_routed_experts=0,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    owner = _make_owner({"lm_head.weight": nn.Parameter(torch.zeros(2, 2))}, config=config)
    with _patched_deps() as mocks:
        loaded = owner.load_weights([("lm_head.weight", torch.zeros(2, 2))])
    assert loaded == set()
    mocks["default_weight_loader"].assert_not_called()


def test_skip_weight_does_not_skip_lm_head_when_untied():
    """``and`` right operand is False when ``tie_word_embeddings`` is False."""
    config = SimpleNamespace(
        tie_word_embeddings=False,
        n_routed_experts=0,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    param = nn.Parameter(torch.zeros(2, 2))
    owner = _make_owner({"lm_head.weight": param}, config=config)
    with _patched_deps() as mocks:
        loaded = owner.load_weights([("lm_head.weight", torch.ones(2, 2))])
    assert loaded == {"lm_head.weight"}
    mocks["default_weight_loader"].assert_called_once()
    torch.testing.assert_close(param, torch.ones(2, 2))


def test_skip_weight_skips_speculative_layer_weight():
    """The trailing ``get_spec_layer_idx_from_weight_name`` check still skips."""
    spec_name = "model.layers.1.input_layernorm.weight"
    owner = _make_owner({spec_name: nn.Parameter(torch.zeros(2, 2))})
    with _patched_deps(
        get_spec_layer_idx_from_weight_name=MagicMock(return_value=1)
    ) as mocks:
        loaded = owner.load_weights([(spec_name, torch.ones(2, 2))])
    assert loaded == set()
    mocks["default_weight_loader"].assert_not_called()


# ---------------------------------------------------------------------------
# _try_load_stacked: ``stacked_name`` path
# ---------------------------------------------------------------------------


def test_stacked_weight_loaded_and_added():
    """A stacked weight is loaded via the param's ``weight_loader`` and recorded."""
    param = nn.Parameter(torch.zeros(2, 2))
    param.weight_loader = MagicMock()
    owner = _make_owner({"model.layers.0.mlp.gate_up_proj.weight": param})
    with _patched_deps() as mocks:
        loaded = owner.load_weights(
            [("model.layers.0.mlp.gate_proj.weight", torch.ones(2, 2))]
        )
    assert loaded == {"model.layers.0.mlp.gate_up_proj.weight"}
    param.weight_loader.assert_called_once()
    # ``continue`` skips the expert / default paths.
    mocks["default_weight_loader"].assert_not_called()


# ---------------------------------------------------------------------------
# _try_load_expert: the collapsed ``if expert_name == "" or expert_name is not None:``
# ---------------------------------------------------------------------------


def test_expert_seen_but_not_loaded_returns_empty_string():
    """``_try_load_expert`` returns ``""`` -> continue without recording."""
    expert_mapping = [("model.expert_down", "routed_expert_down", 0, 0)]
    # The mapped name is intentionally absent from params_dict so the expert
    # weight_loader is never invoked and ``saw_expert_key`` stays True.
    owner = _make_owner({})
    with _patched_deps(
        fused_moe_make_expert_params_mapping=MagicMock(return_value=expert_mapping)
    ) as mocks:
        loaded = owner.load_weights(
            [("model.layers.0.mlp.experts.0.routed_expert_down.weight", torch.ones(2, 2))]
        )
    assert loaded == set()
    mocks["default_weight_loader"].assert_not_called()


def test_expert_loaded_successfully_is_recorded():
    """``_try_load_expert`` returns a name -> recorded and we ``continue``."""
    expert_mapping = [("model.expert_down", "routed_expert_down", 0, 0)]
    param = nn.Parameter(torch.zeros(2, 2))
    param.weight_loader = MagicMock(return_value=True)
    mapped_name = "model.layers.0.mlp.experts.0.model.expert_down.weight"
    owner = _make_owner({mapped_name: param})
    with _patched_deps(
        fused_moe_make_expert_params_mapping=MagicMock(return_value=expert_mapping)
    ) as mocks:
        loaded = owner.load_weights(
            [("model.layers.0.mlp.experts.0.routed_expert_down.weight", torch.ones(2, 2))]
        )
    assert loaded == {mapped_name}
    param.weight_loader.assert_called_once()
    mocks["default_weight_loader"].assert_not_called()


def test_expert_none_falls_through_to_default_loader():
    """``_try_load_expert`` returns ``None`` -> fall through to the default path."""
    param = nn.Parameter(torch.zeros(2, 2))
    owner = _make_owner({"model.norm.weight": param})
    with _patched_deps() as mocks:
        loaded = owner.load_weights([("model.norm.weight", torch.ones(2, 2))])
    assert loaded == {"model.norm.weight"}
    mocks["default_weight_loader"].assert_called_once()
    torch.testing.assert_close(param, torch.ones(2, 2))


# ---------------------------------------------------------------------------
# Fall-through edge cases that the new ``continue`` semantics must preserve
# ---------------------------------------------------------------------------


def test_bias_not_in_params_dict_is_skipped():
    """A ``.bias`` name absent from params_dict is skipped before kv-scale remap."""
    owner = _make_owner({})
    with _patched_deps() as mocks:
        loaded = owner.load_weights([("model.layers.0.mlp.fc1.bias", torch.zeros(2))])
    assert loaded == set()
    mocks["maybe_remap_kv_scale_name"].assert_not_called()
    mocks["default_weight_loader"].assert_not_called()


def test_kv_scale_remap_returns_none_is_skipped():
    """``maybe_remap_kv_scale_name`` returning ``None`` skips the weight."""
    owner = _make_owner({})
    with _patched_deps(
        maybe_remap_kv_scale_name=MagicMock(return_value=None)
    ) as mocks:
        loaded = owner.load_weights([("model.layers.0.kv_scale", torch.zeros(1))])
    assert loaded == set()
    mocks["default_weight_loader"].assert_not_called()


def test_remapped_name_missing_from_params_warns_and_skips():
    """A remapped name not present in params_dict warns and skips loading."""
    owner = _make_owner({})
    with _patched_deps() as mocks, patch.object(pangu_moe.logger, "warning") as warn:
        loaded = owner.load_weights([("model.layers.0.unknown.weight", torch.zeros(1))])
    assert loaded == set()
    mocks["default_weight_loader"].assert_not_called()
    warn.assert_called_once()


def test_q_b_proj_split_marks_synthetic_children_loaded():
    """Loading ``q_b_proj`` also marks the synthetic ``q_b_nope/q_b_pe`` children."""
    parent = nn.Parameter(torch.zeros(2, 2))
    nope = nn.Parameter(torch.zeros(2, 2))
    pe = nn.Parameter(torch.zeros(2, 2))
    parent_name = "model.layers.0.self_attn.q_b_proj.weight"
    nope_name = "model.layers.0.self_attn.q_b_nope_proj.weight"
    pe_name = "model.layers.0.self_attn.q_b_pe_proj.weight"
    owner = _make_owner({parent_name: parent, nope_name: nope, pe_name: pe})
    with _patched_deps() as mocks:
        loaded = owner.load_weights([(parent_name, torch.ones(2, 2))])
    assert loaded == {parent_name, nope_name, pe_name}
    mocks["default_weight_loader"].assert_called_once()
