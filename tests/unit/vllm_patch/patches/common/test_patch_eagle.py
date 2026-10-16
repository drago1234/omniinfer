# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from omni_npu.attention.backends.mome import NPUMomeAttentionMetadataBuilder
from omni_npu.vllm_patches.patches.common import patch_eagle as eagle_mod
from omni_npu.vllm_patches.patches.common.patch_eagle import (
    DraftAttnGroup,
    EagleProposerPatch,
)


def _set_greedy_draft_sampling_state(proposer):
    """Supply sampling state when a test bypasses the real initializer."""
    proposer.speculative_config = SimpleNamespace(
        draft_sample_method="greedy",
        rejection_sample_method="standard",
    )
    proposer._enable_probabilistic_draft_probs = False
    proposer._temperature_only = False
    proposer._draft_top_k = None


def _fake_proposer(attn_metadata):
    fake = SimpleNamespace()
    _set_greedy_draft_sampling_state(fake)
    fake.runner = SimpleNamespace(
        _omni_spec_decode_common_attn_metadata=attn_metadata,
        batch_execution_and_padding_state=(
            CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=2),
            None,
        ),
        dp_parallel_lmhead=False,
        local_parallel_lmhead=False,
    )
    fake.attn_layer_names = ["layer0"]
    fake.num_speculative_tokens = 1
    fake.n_predict = 1
    fake.prefill_single_step = False
    fake.supports_mm_inputs = False
    fake.method = "eagle"
    fake.vllm_config = MagicMock()
    fake.input_ids = torch.arange(4)
    fake.inputs_embeds = torch.zeros(4, 2)
    fake.hidden_states = torch.zeros(4, 2)
    fake.model = MagicMock(return_value=torch.zeros(2, 2))
    fake.mtp_exec_count = 0

    def _arange_positions(num_tokens):
        return torch.arange(num_tokens)

    fake._get_positions = _arange_positions
    fake.build_per_group_and_layer_attn_metadata = MagicMock(
        return_value=(None, {"layer0": "built"})
    )
    return fake


@pytest.mark.unit
def test_eagle_dummy_run_pads_stashed_common_metadata(monkeypatch):
    orig_slots = torch.tensor([7, 8, 9], dtype=torch.int32)
    orig_blocks = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    meta = SimpleNamespace(
        slot_mapping=orig_slots.clone(),
        block_table_tensor=orig_blocks.clone(),
    )
    fake = _fake_proposer(meta)
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    EagleProposerPatch.dummy_run(fake, num_tokens=2, attn_metadata=None)

    assert torch.equal(orig_slots, torch.tensor([7, 8, 9], dtype=torch.int32))
    assert torch.equal(orig_blocks, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
    assert torch.equal(
        meta.slot_mapping,
        torch.full_like(orig_slots, PADDING_SLOT_ID),
    )
    assert torch.equal(
        meta.block_table_tensor,
        torch.full_like(orig_blocks, NULL_BLOCK_ID),
    )
    fake.build_per_group_and_layer_attn_metadata.assert_called_once()
    fake.model.assert_called_once()
    assert fake.runner.batch_execution_and_padding_state is None


@pytest.mark.unit
def test_eagle_dummy_run_without_metadata_skips_padding(monkeypatch):
    fake = _fake_proposer(None)
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    EagleProposerPatch.dummy_run(fake, num_tokens=2, attn_metadata=None)

    fake.build_per_group_and_layer_attn_metadata.assert_not_called()
    fake.model.assert_called_once()


@pytest.mark.unit
@pytest.mark.parametrize("method, expected", [("mtp", 3), ("eagle", 0)])
def test_draft_dummy_run_counts_each_forward(monkeypatch, method, expected):
    fake = _fake_proposer(None)
    fake.method = method
    fake.n_predict = 3
    fake.num_speculative_tokens = 3
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    EagleProposerPatch.dummy_run(fake, num_tokens=2, is_profile=True)

    assert fake.model.call_count == 3
    assert fake.mtp_exec_count == expected


@pytest.mark.unit
def test_draft_graph_capture_counts_each_forward(monkeypatch):
    fake = _fake_proposer(None)
    fake.method = "mtp"
    fake.n_predict = 2
    fake.num_speculative_tokens = 3
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    EagleProposerPatch.dummy_run(fake, num_tokens=2, is_graph_capturing=True)

    assert fake.model.call_count == 2
    assert fake.mtp_exec_count == 2


@pytest.mark.unit
def test_draft_forward_counts_before_model_call():
    fake = SimpleNamespace(
        method="mtp",
        mtp_exec_count=0,
        model=MagicMock(side_effect=RuntimeError("forward failed")),
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        eagle_mod._forward_draft_model(fake, input_ids=None)
    assert fake.mtp_exec_count == 1


@pytest.mark.unit
def test_draft_counter_starts_at_zero(monkeypatch):
    def original_init(proposer, **_kwargs):
        _set_greedy_draft_sampling_state(proposer)
        proposer.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(n_predict=1)
        )
        proposer.num_speculative_tokens = 3

    monkeypatch.setattr(eagle_mod, "EagleProposer_original_init", original_init)
    fake = SimpleNamespace(method="mtp")
    config = SimpleNamespace(kv_transfer_config=None)
    EagleProposerPatch.__init__(fake, config, torch.device("cpu"))
    assert fake.mtp_exec_count == 0


@pytest.mark.unit
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_mtp_proposal_counts_each_draft_forward(monkeypatch, num_speculative_tokens):
    fake = _fake_proposer(None)
    fake.method = "mtp"
    fake.input_ids = torch.arange(4, dtype=torch.int32)
    fake.n_predict = num_speculative_tokens
    fake.allowed_attn_types = None
    fake.parallel_drafting = False
    fake.model_returns_tuple = lambda: False
    fake.set_inputs_first_pass = MagicMock(
        return_value=(2, torch.tensor([0]), SimpleNamespace(slot_mapping=torch.zeros(2)))
    )
    fake._save_and_change_target_input = MagicMock()
    fake.build_model_inputs_first_pass = MagicMock(return_value=(
        {"input_ids": None, "inputs_embeds": torch.zeros(2, 2)},
        2,
    ))
    fake._get_slot_mapping = MagicMock(return_value=torch.zeros(2))
    fake._sample_draft_tokens = MagicMock(return_value=(torch.tensor([9]), None))
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        eagle_mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )

    output = EagleProposerPatch.propose(
        fake,
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=torch.zeros(2, dtype=torch.int64),
        target_positions=torch.zeros(2, dtype=torch.int64),
        target_hidden_states=torch.zeros(2, 2),
        next_token_ids=torch.zeros(1, dtype=torch.int64),
        token_indices_to_sample=torch.tensor([0]),
        common_attn_metadata=SimpleNamespace(slot_mapping=torch.zeros(2)),
        sampling_metadata=SimpleNamespace(),
    )

    assert output.shape == (1, num_speculative_tokens)
    assert fake.model.call_count == num_speculative_tokens
    assert fake.mtp_exec_count == num_speculative_tokens

# --------------------------------------------------------------------------
# draft attention groups: which one is the base, and what step0 passes down
# --------------------------------------------------------------------------


def _fake_attn_group(layer_names, block_size=16):
    """One entry of ``runner.attn_groups`` with a builder of a given block size."""
    builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size))
    return SimpleNamespace(
        layer_names=list(layer_names),
        get_metadata_builder=lambda: builder,
    )


def _fake_backend_proposer(attn_layer_names, attn_groups):
    fake = SimpleNamespace()
    fake.attn_layer_names = list(attn_layer_names)
    fake.runner = SimpleNamespace(attn_groups=attn_groups)
    return fake


class TestInitializeAttnBackendBaseGroup:
    """``initialize_attn_backend`` has to visit the base group first.

    ``_rebuild_per_group_metadata_for_step`` updates the base group's
    CommonAttentionMetadata in place and lets every other group shallow-copy
    from it, so the base group must be both first in the list *and* the one
    ``kv_cache_gid`` names -- not whichever group happens to be enumerated
    first.  Hybrid MLA + MoME puts the draft layers in different kv cache
    groups, which is where "first group with a draft layer wins" got it wrong.
    """

    def test_base_group_is_hoisted_and_drives_kv_cache_gid(self):
        # The base layer (attn_layer_names[0]) sits in the *second* kv cache
        # group -- the ordering the old rule got wrong.
        proposer = _fake_backend_proposer(
            ["draft.0", "draft.1"],
            [
                [_fake_attn_group(["draft.1"], block_size=32)],
                [_fake_attn_group(["draft.0"], block_size=16)],
            ],
        )

        EagleProposerPatch.initialize_attn_backend(proposer, kv_cache_config=None)

        first, second = proposer.draft_attn_groups
        assert first.is_base
        assert first.kv_cache_group_id == 1
        assert proposer.kv_cache_gid == 1
        assert not second.is_base
        assert second.kv_cache_group_id == 0
        # block_size is taken off the first group's spec, i.e. the base one
        assert proposer.block_size == 16

    def test_non_base_groups_keep_their_enumeration_order(self):
        # Every draft layer is one this drafter owns, but none of the groups
        # holds attn_layer_names[0], so the recording order is untouched.
        proposer = _fake_backend_proposer(
            ["draft.0", "draft.1", "draft.2"],
            [[_fake_attn_group(["draft.1"])], [_fake_attn_group(["draft.2"])]],
        )

        EagleProposerPatch.initialize_attn_backend(proposer, kv_cache_config=None)

        assert [g.kv_cache_group_id for g in proposer.draft_attn_groups] == [0, 1]

    def test_group_without_the_base_layer_leaves_kv_cache_gid_unset(self):
        """A draft layer whose group does not hold ``attn_layer_names[0]``.

        Nothing is flagged base, so pin that the fallback is an explicit
        ``None`` -- not group 0, which would silently update the wrong CM in
        place.
        """
        proposer = _fake_backend_proposer(
            ["draft.0", "draft.1"],
            [[_fake_attn_group(["draft.1"], block_size=8)]],
        )

        EagleProposerPatch.initialize_attn_backend(proposer, kv_cache_config=None)

        assert not proposer.draft_attn_groups[0].is_base
        assert proposer.kv_cache_gid is None
        assert proposer.block_size == 8

    def test_only_draft_layers_are_kept_and_sorted(self):
        group = _fake_attn_group(["target.0", "draft.9", "draft.2"])
        proposer = _fake_backend_proposer(["draft.2", "draft.9"], [[group]])

        EagleProposerPatch.initialize_attn_backend(proposer, kv_cache_config=None)

        assert proposer.draft_attn_groups[0].layer_names == ["draft.2", "draft.9"]
        assert proposer.draft_attn_groups[0].is_base

    def test_is_base_defaults_to_false(self):
        group = DraftAttnGroup(kv_cache_group_id=0, layer_names=["draft.0"])
        assert group.is_base is False


def _fake_mome_builder():
    """A real (uninitialised) MoME builder, so the patch's isinstance holds."""
    builder = NPUMomeAttentionMetadataBuilder.__new__(NPUMomeAttentionMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(block_size=16)
    builder.build_for_drafting = MagicMock(return_value="mome-metadata")
    return builder


def _plain_builder(block_size=4):
    """A non-MoME (MLA-like) builder: gets no extra metadata kwargs."""
    return SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=block_size),
        build_for_drafting=MagicMock(return_value="mla-metadata"),
    )


class TestBuildPerGroupStep0Offset:
    """``use_mome_step0_offset`` withholds the accepted count on step0.

    Handing step0 a count would take ``build``'s ``num_accepted_tokens is not
    None`` branch and index the committed tail with the wrong offset; step0 is
    supposed to derive its own -- see
    tests/unit/attention/backends/test_mome_num_prompt_tokens.py.
    """

    @staticmethod
    def _proposer(builder, layer_names=("draft.0",)):
        fake = SimpleNamespace()
        fake.kv_cache_gid = 0
        fake.draft_attn_groups = [
            DraftAttnGroup(
                kv_cache_group_id=0, layer_names=list(layer_names), builder=builder
            )
        ]
        fake.runner = SimpleNamespace(
            num_accepted_tokens=SimpleNamespace(
                gpu=torch.tensor([1, 2, 3], dtype=torch.int32)
            ),
            num_prompt_tokens=SimpleNamespace(
                gpu=torch.tensor([8, 9, 10], dtype=torch.int32)
            ),
        )
        # The real helper returns base_cm unchanged when the group *is* the
        # base group, which is the case here -- bind it rather than stubbing.
        fake._build_common_attn_metadata_for_group = partial(
            EagleProposerPatch._build_common_attn_metadata_for_group, fake
        )
        return fake

    def _call(self, builder, **kwargs):
        EagleProposerPatch.build_per_group_and_layer_attn_metadata(
            self._proposer(builder),
            common_attn_metadata=SimpleNamespace(num_reqs=3),
            draft_index=kwargs.pop("draft_index", 0),
            **kwargs,
        )
        return builder.build_for_drafting.call_args.kwargs

    def test_step0_omits_the_accepted_count(self):
        kwargs = self._call(_fake_mome_builder(), use_mome_step0_offset=True)

        assert "num_accepted_tokens" not in kwargs
        assert kwargs["num_prompt_tokens"].tolist() == [8, 9, 10]
        assert kwargs["draft_index"] == 0

    def test_default_still_passes_the_runner_count(self):
        kwargs = self._call(_fake_mome_builder())

        assert kwargs["num_accepted_tokens"].tolist() == [1, 2, 3]
        assert kwargs["num_prompt_tokens"].tolist() == [8, 9, 10]

    def test_non_mome_builder_gets_neither_argument(self):
        kwargs = self._call(_plain_builder(), use_mome_step0_offset=True)

        assert "num_accepted_tokens" not in kwargs
        assert "num_prompt_tokens" not in kwargs

    def test_all_layers_of_a_group_share_one_metadata_object(self):
        builder = _fake_mome_builder()
        per_group, per_layer = EagleProposerPatch.build_per_group_and_layer_attn_metadata(
            self._proposer(builder, layer_names=("draft.0", "draft.1")),
            common_attn_metadata=SimpleNamespace(num_reqs=3),
            draft_index=0,
        )

        assert per_group == ["mome-metadata"]
        assert per_layer == {"draft.0": "mome-metadata", "draft.1": "mome-metadata"}


def _fake_block_table(tensor, slot_buf_len=8):
    # Real BlockTable exposes a persistent per-group slot_mapping.gpu buffer;
    # non-base rebuilds write their draft slots through a slice of it.
    return SimpleNamespace(
        get_device_tensor=lambda num_reqs: tensor,
        slot_mapping=SimpleNamespace(
            gpu=torch.zeros(slot_buf_len, dtype=torch.int64)
        ),
    )


def _rebuild_proposer(groups, block_tables, *, kv_cache_gid):
    fake = SimpleNamespace()
    fake.uses_mrope = False
    fake.kv_cache_gid = kv_cache_gid
    fake.draft_attn_groups = list(groups)
    fake.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=[_fake_block_table(t) for t in block_tables]
        ),
        num_accepted_tokens=SimpleNamespace(
            gpu=torch.tensor([1, 2], dtype=torch.int32)
        ),
        num_prompt_tokens=SimpleNamespace(gpu=torch.tensor([8, 9], dtype=torch.int32)),
    )
    return fake


class TestRebuildPerGroupMetadataForStep:
    """``_rebuild_per_group_metadata_for_step`` writes slots into the live buffer.

    Each group's view covers this step's execution tokens, backed by its
    persistent block-table buffer. Only padding within that view is consumed
    by this step; storage beyond the view is not part of its metadata.
    """

    BUFFER_LEN = 8

    def _run(self, proposer, cm, positions, token_index=0):
        EagleProposerPatch._rebuild_per_group_metadata_for_step(
            proposer,
            common_attn_metadata=cm,
            clamped_positions=torch.tensor(positions, dtype=torch.int64),
            exceeds_max_model_len=torch.zeros(len(positions), dtype=torch.bool),
            token_index=token_index,
            per_layer_attn_metadata={},
        )

    def test_tail_of_the_live_buffer_is_padded(self):
        buf = torch.arange(self.BUFFER_LEN, dtype=torch.int64)
        cm = SimpleNamespace(num_reqs=2, num_actual_tokens=self.BUFFER_LEN, slot_mapping=buf)
        group = DraftAttnGroup(
            kv_cache_group_id=0, layer_names=["draft.0"], builder=_plain_builder(4)
        )
        # One row of block ids per request: positions 3 and 5 land in blocks 0
        # and 1, gathered out of req 0's and req 1's row respectively.
        block_table = torch.tensor([[10, 11], [12, 13]], dtype=torch.int64)
        proposer = _rebuild_proposer([group], [block_table], kv_cache_gid=0)
        proposer.runner.input_batch.block_table[0].slot_mapping.gpu = buf

        self._run(proposer, cm, [3, 5])

        # 10*4 + 3 = 43 and 13*4 + 1 = 53; the rest of the live buffer is padded
        assert cm.slot_mapping.data_ptr() == buf.data_ptr()
        assert cm.slot_mapping.tolist() == [43, 53, -1, -1, -1, -1, -1, -1]

    def test_accepted_count_only_rides_along_on_step1(self):
        """step1 locates the committed tail inside the full step0 window."""
        for token_index, expected in ((0, [1, 2]), (1, None)):
            builder = _fake_mome_builder()
            cm = SimpleNamespace(
                num_reqs=2, num_actual_tokens=8, slot_mapping=torch.arange(8, dtype=torch.int64)
            )
            group = DraftAttnGroup(
                kv_cache_group_id=0, layer_names=["draft.0"], builder=builder
            )
            proposer = _rebuild_proposer(
                [group],
                [torch.tensor([[10, 20], [30, 40]], dtype=torch.int64)],
                kv_cache_gid=0,
            )

            self._run(proposer, cm, [1, 2], token_index=token_index)

            kwargs = builder.build_for_drafting.call_args.kwargs
            assert kwargs["draft_index"] == token_index + 1
            assert kwargs["num_prompt_tokens"].tolist() == [8, 9]
            if expected is None:
                assert "num_accepted_tokens" not in kwargs
            else:
                assert kwargs["num_accepted_tokens"].tolist() == expected

    def test_non_base_group_gets_a_distinct_metadata_copy(self):
        slot_buf = torch.full((self.BUFFER_LEN,), -7, dtype=torch.int64)
        cm = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=self.BUFFER_LEN,
            slot_mapping=torch.arange(self.BUFFER_LEN, dtype=torch.int64),
        )
        base = DraftAttnGroup(
            kv_cache_group_id=0,
            layer_names=["draft.0"],
            builder=_plain_builder(4),
            is_base=True,
        )
        other = DraftAttnGroup(
            kv_cache_group_id=1, layer_names=["draft.1"], builder=_plain_builder(4)
        )
        block_tables = [
            torch.tensor([[10, 20], [30, 40]], dtype=torch.int64),
            torch.tensor([[50, 60], [70, 80]], dtype=torch.int64),
        ]
        proposer = _rebuild_proposer([base, other], block_tables, kv_cache_gid=0)
        # The non-base group's slot buffer lives on its own block table.
        proposer.runner.input_batch.block_table[1].slot_mapping.gpu = slot_buf

        self._run(proposer, cm, [3, 5])

        base_cm = base.builder.build_for_drafting.call_args.kwargs["common_attn_metadata"]
        other_cm = other.builder.build_for_drafting.call_args.kwargs["common_attn_metadata"]
        assert base_cm is cm
        assert other_cm is not cm
        assert other_cm.block_table_tensor is block_tables[1]
        assert cm.block_table_tensor is block_tables[0]
        # Non-base slots are computed from its own block table (50*4+3, 80*4+1)
        # and written through its own buffer; the base buffer is untouched
        # apart from the base group's own in-place update.
        assert other_cm.slot_mapping is not cm.slot_mapping
        # views the non-base group's own buffer (same storage, full length)
        assert other_cm.slot_mapping.data_ptr() == slot_buf.data_ptr()
        assert other_cm.slot_mapping.tolist() == [203, 321, -1, -1, -1, -1, -1, -1]
        assert slot_buf[:2].tolist() == [203, 321]


def _propose_step0_offset_flag(n_predict, num_speculative_tokens):
    """Run ``propose`` far enough to read ``use_mome_step0_offset``.

    The flag is what makes single-head multi-step MTP fall back to the
    drafter's own step0 count, so it is worth pinning the exact predicate.  The
    run is stopped by propose's own "runner has not determined padding yet"
    guard, right after the metadata build it is asked for.
    """
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return [], {}

    fake = SimpleNamespace()
    fake.method = "mtp"
    fake.n_predict = n_predict
    _set_greedy_draft_sampling_state(fake)
    # Non-None runner, but no padding state: propose raises right after the build.
    fake.runner = SimpleNamespace(batch_execution_and_padding_state=None)
    fake.set_inputs_first_pass = MagicMock(return_value=(4, None, object()))
    fake.build_per_group_and_layer_attn_metadata = MagicMock(side_effect=_capture)

    with pytest.raises(ValueError, match="Propose of drafter"):
        EagleProposerPatch.propose(
            fake,
            num_speculative_tokens=num_speculative_tokens,
            target_token_ids=torch.zeros(4, dtype=torch.int64),
            target_positions=torch.zeros(4, dtype=torch.int64),
            target_hidden_states=torch.zeros(4, 2),
            next_token_ids=torch.zeros(2, dtype=torch.int64),
            token_indices_to_sample=None,
            common_attn_metadata=SimpleNamespace(),
            sampling_metadata=SimpleNamespace(),
        )
    assert seen["draft_index"] == 0
    return seen["use_mome_step0_offset"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_predict, num_speculative_tokens, expected",
    [
        (1, 3, True),  # single head, multi step: the case the fix is for
        (1, 1, False),  # plain single step has no step1 to locate a tail for
        (2, 3, False),  # multi-head keeps the runner count
    ],
)
def test_propose_step0_offset_predicate(n_predict, num_speculative_tokens, expected):
    assert (
        _propose_step0_offset_flag(n_predict, num_speculative_tokens) is expected
    )


@pytest.mark.parametrize("role,method,n_predict,expected", [
    pytest.param("kv_producer", "mtp", 1, True, id="prefill-mtp"),
    pytest.param("kv_consumer", "mtp", 1, False, id="decode-mtp"),
    pytest.param(None, "mtp", 1, False, id="no-kv-transfer"),
])
def test_prefill_single_step_scope(monkeypatch, role, method, n_predict, expected):
    """Only a single-head MTP producer enables the initialization-time flag."""
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role) if role else None,
    )
    fake = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(n_predict=n_predict)),
        num_speculative_tokens=7,
    )
    _set_greedy_draft_sampling_state(fake)
    monkeypatch.setattr(eagle_mod, "EagleProposer_original_init", lambda *args, **kwargs: None)
    EagleProposerPatch.__init__(fake, config, torch.device("cpu"))
    assert fake.prefill_single_step is expected


@pytest.mark.parametrize("prefill_single_step,expected_forwards", [(True, 1), (False, 7)])
def test_dummy_run_respects_prefill_single_step(monkeypatch, prefill_single_step, expected_forwards):
    """Warmup runs one step on P and preserves seven steps on D."""
    fake = _fake_proposer(None)
    fake.method = "mtp"
    fake.num_speculative_tokens = 7
    fake.prefill_single_step = prefill_single_step
    fake._determine_batch_execution_and_padding = MagicMock(return_value=(None, 2, None))
    monkeypatch.setattr(eagle_mod, "set_forward_context", lambda **kw: nullcontext())
    monkeypatch.setattr(eagle_mod, "get_forward_context", lambda: SimpleNamespace())
    EagleProposerPatch.dummy_run(fake, num_tokens=2)
    assert fake.model.call_count == expected_forwards
    assert fake.num_speculative_tokens == 7


class TestDraftSlotExecutionSize:
    @pytest.mark.parametrize('initial,execution,valid', [
        (2048, 3, 3), (24, 3, 3), (8, 1, 1),
        (2048, 8, 3), (24, 24, 3), (3, 3, 3),
    ])
    @pytest.mark.parametrize('uses_mrope', [False, True])
    def test_each_group_uses_current_execution_view(self, initial, execution, valid, uses_mrope):
        # Base group is deliberately group 1, with a different block size.
        capacity = max(initial, execution) + 4
        tables = [
            torch.tensor([[10 + 10*i, 11 + 10*i, 12 + 10*i] for i in range(valid)]),
            torch.tensor([[100 + 10*i, 101 + 10*i, 102 + 10*i] for i in range(valid)]),
        ]
        groups = [
            DraftAttnGroup(kv_cache_group_id=1, layer_names=['base'], builder=_plain_builder(16), is_base=True),
            DraftAttnGroup(kv_cache_group_id=0, layer_names=['other'], builder=_plain_builder(4)),
        ]
        proposer = _rebuild_proposer(groups, tables, kv_cache_gid=1)
        proposer.uses_mrope = uses_mrope
        buffers = [torch.full((capacity,), -7, dtype=torch.int64) for _ in tables]
        for table, buffer in zip(proposer.runner.input_batch.block_table, buffers):
            table.slot_mapping.gpu = buffer
        cm = SimpleNamespace(num_reqs=valid, num_actual_tokens=execution, slot_mapping=buffers[1][:initial])
        positions = torch.arange(valid, dtype=torch.int64) + 3
        supplied_positions = torch.stack([positions]*3) if uses_mrope else positions
        exceeds = torch.zeros(valid, dtype=torch.bool)
        if valid > 1:
            exceeds[-1] = True
        per_layer = {}
        # Repeated draft steps must keep storage addresses and group isolation.
        for step in (0, 1):
            EagleProposerPatch._rebuild_per_group_metadata_for_step(
                proposer, cm, supplied_positions, exceeds, step, per_layer,
            )
            for group in groups:
                gid = group.kv_cache_group_id
                group_cm = group.builder.build_for_drafting.call_args.kwargs['common_attn_metadata']
                size = group.builder.kv_cache_spec.block_size
                ids = tables[gid].gather(1, (positions // size).view(-1, 1)).view(-1)
                expected = ids * size + positions % size
                expected.masked_fill_(exceeds, PADDING_SLOT_ID)
                assert group_cm.slot_mapping.numel() == execution
                assert group_cm.slot_mapping.data_ptr() == buffers[gid].data_ptr()
                assert torch.equal(group_cm.slot_mapping[:valid], expected)
                assert (group_cm.slot_mapping[valid:] == PADDING_SLOT_ID).all()
                assert (buffers[gid][execution:] == -7).all()
                assert group_cm.block_table_tensor is tables[gid]
                assert (group_cm is cm) == (gid == proposer.kv_cache_gid)
            assert set(per_layer) == {'base', 'other'}


@pytest.mark.unit
@pytest.mark.parametrize("method,rejection,draft,upstream_enabled,expected_enabled", [
    ("mtp", "block", "probabilistic", False, True),
    ("mtp", "standard", "probabilistic", True, True),
    ("mtp", "block", "greedy", False, False),
    ("mtp", "block", "temperature_only", False, False),
    ("mtp", "standard", "temperature_only", False, True),
    ("eagle", "block", "probabilistic", False, False),
    ("eagle3", "block", "probabilistic", False, False),
    ("dflash", "block", "probabilistic", False, False),
])
def test_mtp_block_draft_probs_initialized_before_top_k(
    monkeypatch, method, rejection, draft, upstream_enabled, expected_enabled,
):
    def original_init(proposer, **_kwargs):
        proposer.method = method
        proposer.speculative_config = SimpleNamespace(
            rejection_sample_method=rejection, draft_sample_method=draft,
        )
        proposer._enable_probabilistic_draft_probs = upstream_enabled
        proposer.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(n_predict=1))
        proposer.num_speculative_tokens = 7

    monkeypatch.setattr(eagle_mod, "EagleProposer_original_init", original_init)
    monkeypatch.setattr(eagle_mod.omni_envs, "OMNI_DRAFT_TOP_K", 5)
    proposer = SimpleNamespace()
    EagleProposerPatch.__init__(proposer, SimpleNamespace(kv_transfer_config=None), torch.device("cpu"))
    assert proposer._enable_probabilistic_draft_probs is expected_enabled
    top_k_enabled = upstream_enabled or (method == "mtp" and rejection == "block" and draft == "probabilistic")
    assert proposer._draft_top_k == (5 if top_k_enabled else None)
