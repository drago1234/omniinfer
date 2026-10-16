# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from omni_npu.attention.backends import mome as mome_mod
from omni_npu.v1.layers.attention import npu_pangu as pangu_mod


class TestMomeSPThreshold(unittest.TestCase):
    def build_metadata(self, tokens, threshold, *, enabled=True, decodes=0, prefills=1):
        builder = mome_mod.NPUMomeAttentionMetadataBuilder.__new__(
            mome_mod.NPUMomeAttentionMetadataBuilder
        )
        builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False))
        builder.compilation_config = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        )
        builder.mome_block_size = 128
        builder.reorder_batch_threshold = 1
        builder.fake_num_spec = 0
        builder.is_decode_node = False
        builder.decode_cudagraph_max_bs = 16
        builder.enable_flashcomm2 = False
        # One token per decode row, with the remaining tokens in the prefill row.
        lengths = [1] * decodes + ([tokens - decodes] if prefills else [])
        query_start = torch.tensor([0] + list(range(1, decodes + 1)) + ([tokens] if prefills else []), dtype=torch.int32)
        computed = torch.zeros(len(lengths), dtype=torch.int32)
        common = SimpleNamespace(
            num_reqs=len(lengths), query_start_loc=query_start, query_start_loc_cpu=query_start,
            seq_lens=torch.tensor(lengths, dtype=torch.int32),
            block_table_tensor=torch.zeros((len(lengths), 1), dtype=torch.int32),
            max_query_len=max(lengths, default=0), num_computed_tokens_cpu=computed,
            # The decision must use the unpadded CPU query total instead.
            num_actual_tokens=tokens + 1024,
            compute_num_computed_tokens=lambda: computed,
        )
        sp_metadata = (object(), object(), object())
        config = SimpleNamespace(enable_mome_sp=enabled, mome_sp_token_threshold=threshold)
        with patch.object(mome_mod, "model_extra_config", SimpleNamespace(operator_opt_config=config)), patch.object(
            mome_mod, "split_decodes_and_prefills",
            return_value=(decodes, prefills, decodes, tokens - decodes),
        ), patch.object(builder, "_build_for_sp", return_value=sp_metadata) as build_sp:
            result = builder.build(0, common)
        return result, build_sp, sp_metadata

    def test_threshold_boundaries_and_unpadded_total(self):
        for threshold in (8192, 10240, 16384, 20480):
            for tokens, expected in ((threshold - 1, False), (threshold, True), (threshold + 1, True)):
                with self.subTest(threshold=threshold, tokens=tokens):
                    metadata, build_sp, sp_metadata = self.build_metadata(tokens, threshold)
                    self.assertEqual(metadata.use_mome_sp, expected)
                    self.assertEqual(build_sp.call_count, int(expected))
                    self.assertIs(metadata.conv_sp_meta, sp_metadata if expected else None)

    def test_disabled_master_switch(self):
        metadata, build_sp, _ = self.build_metadata(24576, 8192, enabled=False)
        self.assertFalse(metadata.use_mome_sp)
        self.assertIsNone(metadata.conv_sp_meta)
        build_sp.assert_not_called()

    def test_zero_threshold_builds_sp_for_small_prefill(self):
        metadata, build_sp, _ = self.build_metadata(2, 0)
        self.assertTrue(metadata.use_mome_sp)
        build_sp.assert_called_once()

    def test_mixed_and_decode_metadata_keep_sp_disabled(self):
        for threshold in (0, 8192):
            for tokens, decodes, prefills in ((24576, 1, 1), (2, 2, 0)):
                with self.subTest(threshold=threshold, decodes=decodes, prefills=prefills):
                    metadata, build_sp, _ = self.build_metadata(
                        tokens, threshold, decodes=decodes, prefills=prefills
                    )
                    for part in (metadata, metadata.prefill, metadata.decode):
                        if part is not None:
                            self.assertFalse(part.use_mome_sp)
                            self.assertIsNone(part.conv_sp_meta)
                    build_sp.assert_not_called()

    def test_successive_steps_do_not_mutate_layer_or_previous_metadata(self):
        prefix = "model.layers.7.self_attn"
        for route in ("cp", "sp", "fc2"):
            layer = SimpleNamespace(
                prefix=prefix, is_cla_reuse_layer=False, is_dsa_layer=False, use_mome=True,
                is_cp_layer=route == "cp", is_attn_sp_layer=route == "sp",
                is_flashcomm2_layer=route == "fc2", tp_size=1,
                moe_comm_strategy="allgather_reducescatter",
            )
            seen = []
            consume = Mock(side_effect=lambda hidden, cos, sin, attn, mome: seen.append(mome) or hidden)
            setattr(layer, {"cp": "_forward_prefill_cp", "sp": "_forward_prefill_sp", "fc2": "_forward_prefill_FC2"}[route], consume)
            original_state = vars(layer).copy()
            for tokens in (4096, 24576, 4096, 8192):
                metadata, _, _ = self.build_metadata(tokens, 8192)
                attn = SimpleNamespace(num_actual_tokens=tokens, num_decode_tokens=0,
                                       num_decodes=0, num_prefills=1, prefill=SimpleNamespace())
                attn_name = f"{prefix}.attn"
                context = SimpleNamespace(no_compile_layers={"layer": layer},
                                          attn_metadata={attn_name: attn, f"{attn_name}.mome": metadata})
                with patch.object(pangu_mod, "get_forward_context", return_value=context):
                    pangu_mod.npu_pangu_forward(
                        torch.empty(0), torch.empty(0), torch.empty(0), "layer", attn_name
                    )
                self.assertIs(seen[-1], metadata)
                self.assertEqual(vars(layer), original_state)
            self.assertEqual([m.use_mome_sp for m in seen], [False, True, False, True])
            # Metadata copies used by block-table updates keep the decision.
            copied = copy.copy(seen[-1])
            self.assertTrue(copied.use_mome_sp)
            self.assertIs(copied.conv_sp_meta, seen[-1].conv_sp_meta)


if __name__ == "__main__":
    unittest.main()
