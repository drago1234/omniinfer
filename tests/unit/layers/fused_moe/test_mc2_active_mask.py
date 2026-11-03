"""MC2 metadata masks must describe unpadded input rows and survive reuse."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from omni_npu.attention.backends.utils import generate_mc2_active_mask
from omni_npu.layers.fused_moe import prepare_permute_unpermute_finalize as module


def test_mask_updates_in_place_from_device_query_tail():
    builder = NS(mc2_mask=torch.zeros(640, dtype=torch.bool))
    pointer = builder.mc2_mask.data_ptr()
    for active in [512, 256, 160, 152, 20, 1, 0, 160]:
        query = torch.tensor([0, active, active], dtype=torch.int32)
        mask = generate_mc2_active_mask(builder, query[-1], torch.arange(640))
        assert mask.data_ptr() == pointer
        assert torch.equal(mask, torch.arange(640) < active)


def _get(monkeypatch, metadata, width):
    monkeypatch.setattr(module, 'get_forward_context', lambda: NS(attn_metadata=metadata))
    return module.DispatchCombinePrepPmtAndUnpmtFinal._get_mc2_mask(NS(), width)


def test_mome_without_mask_falls_back_to_mla(monkeypatch):
    mask = torch.tensor([True, True, True, False])
    result = _get(monkeypatch, {'mome': NS(decode=NS(), num_decode_tokens=4, num_actual_tokens=4),
                                       'mla': NS(mc2_mask=mask)}, 4)
    assert result.data_ptr() == mask.data_ptr()


def test_single_metadata_uses_batch_mask(monkeypatch):
    mask = torch.tensor([True, False, False, False])
    result = _get(monkeypatch, NS(mc2_mask=mask), 4)
    assert result.data_ptr() == mask.data_ptr()


@pytest.mark.parametrize('mask', [torch.ones(3, dtype=torch.bool),
                                torch.ones(4), torch.ones(4, 1, dtype=torch.bool)])
def test_invalid_mask_warns_and_falls_back(monkeypatch, mask):
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning_once", warning)
    assert _get(monkeypatch, {'mla': NS(mc2_mask=mask)}, 4) is None
    warning.assert_called_once()
    assert "falling back to unmasked dispatch" in warning.call_args.args[0]


def test_missing_batch_mask_is_optional(monkeypatch):
    assert _get(monkeypatch, {'mome': NS(), 'mla': NS(decode=None)}, 4) is None


def test_prefill_batch_uses_outer_mask(monkeypatch):
    mask = torch.tensor([True, True, False, False])
    result = _get(monkeypatch, {'mla': NS(decode=None, mc2_mask=mask)}, 4)
    assert result.data_ptr() == mask.data_ptr()


def test_accessor_uses_only_first_metadata(monkeypatch):
    first = torch.tensor([True, False, False, False])
    later = torch.ones(4, dtype=torch.bool)
    result = _get(monkeypatch, {
        'first': NS(mc2_mask=first),
        'later': NS(mc2_mask=later),
    }, 4)
    assert result.data_ptr() == first.data_ptr()


def test_missing_metadata_returns_none(monkeypatch):
    assert _get(monkeypatch, None, 4) is None


def test_empty_metadata_dict_returns_none(monkeypatch):
    assert _get(monkeypatch, {}, 4) is None


def test_decode_only_mask_is_ignored(monkeypatch):
    metadata = NS(decode=NS(mc2_mask=torch.ones(4, dtype=torch.bool)),
                  num_decode_tokens=4, num_actual_tokens=4)
    assert _get(monkeypatch, {'mla': metadata}, 4) is None


def test_batch_mask_preserves_graph_padding(monkeypatch):
    mask = torch.arange(512) < 256
    metadata = NS(mc2_mask=mask, decode=NS(),
                  num_decode_tokens=8, num_actual_tokens=512)
    result = _get(monkeypatch, {'mome': NS(decode=None), 'mla': metadata}, 512)
    assert result.data_ptr() == mask.data_ptr()
    assert result[:256].all()
    assert not result[256:].any()


def test_batch_mask_takes_priority_over_decode_mask(monkeypatch):
    mask = torch.tensor([True, True, False, False])
    metadata = NS(mc2_mask=mask, decode=NS(mc2_mask=torch.ones(4, dtype=torch.bool)),
                  num_decode_tokens=1, num_actual_tokens=4)
    assert _get(monkeypatch, {'mla': metadata}, 4).data_ptr() == mask.data_ptr()
