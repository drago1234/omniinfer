# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MTP's sequence-parallel shards when a FULL graph serves the step."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from omni_npu.v1.models.pangu import pangu_v2_moe_mtp as mtp_mod


def _bare(cls):
    """__new__ skips the module tree; tests set only what the path reads."""
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    return obj


class TestMtpPrefillGraph(unittest.TestCase):
    def test_graph_step_reads_its_shards_from_the_persistent_buffers(self):
        p = _bare(mtp_mod.OpenPanguV2MultiTokenPredictor)
        p.num_mtp_layers, p.mtp_start_layer_idx, p.wrapped_layers = 1, 3, None
        p.layers = {"3": MagicMock(return_value=torch.zeros(1))}
        p.inputs_embeds_shard = torch.zeros(8, 2)
        p.previous_hidden_states_shard = torch.zeros(8, 2)
        sp = MagicMock(slice_tokens=lambda t: t[2:6])
        tp = MagicMock(all_gather=lambda t, dim=0: t)
        meta = {"layer0": SimpleNamespace(prefill_graph_width=16)}
        with patch.object(mtp_mod.model_extra_config.parall_config, "ena_seq_parallel", True), \
             patch.object(mtp_mod.SPManager, "init_sp", return_value=sp), \
             patch.object(mtp_mod, "get_tp_group", return_value=tp), \
             patch.object(mtp_mod, "get_forward_context",
                          return_value=SimpleNamespace(attn_metadata=meta)):
            p.forward(input_ids=torch.zeros(16, dtype=torch.long),
                      positions=torch.arange(16),
                      previous_hidden_states=torch.ones(16, 2),
                      inputs_embeds=torch.ones(16, 2))
        # A per-forward shard is a fresh view every step; the graph froze the
        # buffer's address.
        _, _, hidden, embeds, _ = p.layers["3"].call_args.args
        self.assertEqual(embeds.data_ptr(), p.inputs_embeds_shard.data_ptr())
        self.assertEqual(hidden.data_ptr(), p.previous_hidden_states_shard.data_ptr())
        self.assertEqual(embeds.size(0), 4)

    def test_compile_reslices_the_shard_to_this_capture_width(self):
        layer = _bare(mtp_mod.OpenPanguV2MultiTokenPredictorLayer)
        layer.need_tp_padding = True
        layer.enorm = MagicMock(side_effect=lambda x: x)
        layer.hnorm = MagicMock(side_effect=lambda x: x)
        layer.eh_proj = MagicMock(side_effect=lambda x: (x, None))
        block = MagicMock(return_value=("hidden", None, None, None, None))
        block.self_attn.rotary_emb.get_cos_sin.return_value = (None, None)
        block.mhc_head.return_value = (None,) * 5
        layer.mtp_block = block
        # The shards keep the first capture's length (4); this capture is 9
        # wide over TP 4, so cdiv gives 3 rows where floor would drop one.
        with patch.object(mtp_mod.model_extra_config.operator_opt_config,
                          "enable_prefill_cudagraph", True), \
             patch.object(torch.compiler, "is_compiling", return_value=True), \
             patch.object(mtp_mod, "get_tp_group",
                          return_value=SimpleNamespace(world_size=4)):
            layer.forward(input_ids=torch.zeros(9, dtype=torch.long),
                          positions=torch.arange(9),
                          previous_hidden_states=torch.zeros(4, 2),
                          inputs_embeds=torch.zeros(4, 2))
        self.assertEqual(layer.enorm.call_args.args[0].shape[0], 3)
        self.assertEqual(layer.hnorm.call_args.args[0].shape[0], 3)
