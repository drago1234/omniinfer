# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for ``omni_npu.v1.config.reasoning.ReasoningConfig``."""

from __future__ import annotations

import unittest
import pytest
from unittest.mock import MagicMock, patch

from omni_npu.v1.config import ReasoningConfig
from vllm.config.model import ModelConfig


class TestReasoningConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = MagicMock(spec=ModelConfig)

    def test_defaults_not_enabled(self) -> None:
        cfg = ReasoningConfig()
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.reasoning_start_token_ids)
        self.assertIsNone(cfg.reasoning_end_token_ids)

    def test_ban_tool_start_in_thinking_default_false(self) -> None:
        """The opt-in switch for the worker-side tool-call ban defaults to off."""
        cfg = ReasoningConfig()
        self.assertFalse(cfg.ban_tool_start_in_thinking)

    def test_ban_tool_end_in_thinking_default_false(self) -> None:
        """The opt-in switch for the tool-call-END ban defaults to off."""
        cfg = ReasoningConfig()
        self.assertFalse(cfg.ban_tool_end_in_thinking)

    @pytest.mark.skip(reason="as_argparse_dict was removed in vllm 0.25.1")
    def test_as_argparse_dict(self) -> None:
        arg_dict = ReasoningConfig.as_argparse_dict()
        self.assertEqual(arg_dict["type"], str)
        self.assertIsNone(arg_dict["default"])
        self.assertIn("JSON string", arg_dict["help"])
        self.assertIn("reasoning", arg_dict["help"].lower())

    @pytest.mark.skip(reason="as_argparse_dict was removed in vllm 0.25.1")
    def test_as_argparse_dict_fallback_doc(self) -> None:
        with patch.object(ReasoningConfig, "__doc__", None):
            d = ReasoningConfig.as_argparse_dict()
        self.assertIn("Reasoning configuration.", d["help"])

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_short_circuit_already_set(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        cfg = ReasoningConfig()
        cfg._reasoning_start_token_ids = [1]
        cfg._reasoning_end_token_ids = [2]
        cfg.initialize_token_ids(self.model_config)
        mock_omni_tok.assert_not_called()
        mock_vllm_tok.assert_not_called()
        self.assertTrue(cfg.enabled)
        # Direct ID injection without strings: the forced sequence doubles
        # as the natural end marker (no tokenizer round-trip).
        self.assertEqual(cfg.natural_reasoning_end_token_ids, [2])

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_missing_start_raises(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        cfg = ReasoningConfig(reasoning_end_str="</t>")
        with self.assertRaises(KeyError):
            cfg.initialize_token_ids(self.model_config)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_missing_end_raises(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        cfg = ReasoningConfig(reasoning_start_str="<t>")
        with self.assertRaises(KeyError):
            cfg.initialize_token_ids(self.model_config)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_uses_think_alias_strings(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [10] if s == "<a>" else [20]
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(think_start_str="<a>", think_end_str="<b>")
        cfg.initialize_token_ids(self.model_config)

        self.assertEqual(cfg.reasoning_start_token_ids, [10])
        self.assertEqual(cfg.reasoning_end_token_ids, [20])
        self.assertTrue(cfg.enabled)
        mock_vllm_tok.assert_called_once_with(model_config=self.model_config)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_reasoning_str_over_alias(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = (
            lambda s, add_special_tokens=False: [1] if s == "RS" else [2] if s == "RE" else [99]
        )
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="RS",
            reasoning_end_str="RE",
            think_start_str="XX",
            think_end_str="YY",
        )
        cfg.initialize_token_ids(self.model_config)
        mock_enc.encode.assert_any_call("RS", add_special_tokens=False)
        mock_enc.encode.assert_any_call("RE", add_special_tokens=False)
        self.assertEqual(cfg.reasoning_start_token_ids, [1])
        self.assertEqual(cfg.reasoning_end_token_ids, [2])

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_initialize_token_ids_raises_when_encode_empty(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.return_value = []
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<x>", reasoning_end_str="<y>")
        with self.assertRaises(ValueError) as ctx:
            cfg.initialize_token_ids(self.model_config)
        self.assertIn("failed to tokenize", str(ctx.exception))

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_not_resolved_when_ban_off(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """Ban off → resolution skipped, field stays None even if vocab has the key."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|tool_call_start|>": 999}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<s>", reasoning_end_str="<e>")
        cfg.initialize_token_ids(self.model_config)
        self.assertIsNone(cfg.tool_call_start_token_id)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_resolved_when_ban_on(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|tool_call_start|>": 999}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_start_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_start_token_id, 999)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_falls_back_to_unused11(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        # No <|tool_call_start|>, but [unused11] is present.
        mock_enc.get_vocab.return_value = {"[unused11]": 11}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_start_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_start_token_id, 11)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_start_token_id_none_when_no_key(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"foo": 1, "bar": 2}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_start_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertIsNone(cfg.tool_call_start_token_id)

    # --- ban_tool_end_in_thinking resolution --------------------------

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_end_token_id_not_resolved_when_ban_off(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """End-ban off → end id stays None even when vocab has the key."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|tool_call_end|>": 888}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<s>", reasoning_end_str="<e>")
        cfg.initialize_token_ids(self.model_config)
        self.assertIsNone(cfg.tool_call_end_token_id)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_end_token_id_resolved_when_ban_on(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|tool_call_end|>": 888}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_end_token_id, 888)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_end_token_id_falls_back_to_unused12(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        # No <|tool_call_end|>, but [unused12] is present.
        mock_enc.get_vocab.return_value = {"[unused12]": 12}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_end_token_id, 12)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_tool_call_end_token_id_none_when_no_key(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"foo": 1, "bar": 2}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertIsNone(cfg.tool_call_end_token_id)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_start_and_end_bans_resolve_independently(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """Only the start ban is on → only the start id resolves (end stays None)."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {
            "<|tool_call_start|>": 999,
            "<|tool_call_end|>": 888,
        }
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tool_start_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.tool_call_start_token_id, 999)
        # End ban off → end id never probed.
        self.assertIsNone(cfg.tool_call_end_token_id)

    # --- natural_reasoning_end_token_ids (vLLM 0.30.0 backport) -------

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_natural_end_falls_back_to_reasoning_end_str(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """No parser configured → the forced end sequence doubles as natural."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = (
            lambda s, add_special_tokens=False: [1]
            if s == "<s>"
            else [2]
            if s == "<e>"
            else [99]
        )
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<s>", reasoning_end_str="<e>")
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.reasoning_end_token_ids, [2])
        self.assertEqual(cfg.natural_reasoning_end_token_ids, [2])

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.ReasoningParserManager")
    def test_natural_end_uses_parser_marker(
        self,
        mock_parser_mgr: MagicMock,
        mock_omni_tok: MagicMock,
        mock_vllm_tok: MagicMock,
    ) -> None:
        """Parser configured → its end marker is the natural one, even when
        the forced ``reasoning_end_str`` carries an extra transition phrase."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = (
            lambda s, add_special_tokens=False: [1]
            if s == "<s>"
            else [2]
            if s == "Enough.</n>"
            else [7]
            if s == "</n>"
            else [99]
        )
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc
        mock_parser = MagicMock()
        mock_parser.reasoning_end_str = "</n>"
        mock_parser_mgr.get_reasoning_parser.return_value = (
            lambda tokenizer: mock_parser
        )

        cfg = ReasoningConfig(
            reasoning_parser="deepseek_r1",
            reasoning_start_str="<s>",
            reasoning_end_str="Enough.</n>",
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.reasoning_end_token_ids, [2])
        self.assertEqual(cfg.natural_reasoning_end_token_ids, [7])

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.ReasoningParserManager")
    def test_natural_end_empty_encode_raises(
        self,
        mock_parser_mgr: MagicMock,
        mock_omni_tok: MagicMock,
        mock_vllm_tok: MagicMock,
    ) -> None:
        """A parser marker that fails to tokenize raises instead of silently
        disabling the V2 thinking budget."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = (
            lambda s, add_special_tokens=False: [1]
            if s == "<s>"
            else [2]
            if s == "<e>"
            else []
        )
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc
        mock_parser = MagicMock()
        mock_parser.reasoning_end_str = "</n>"
        mock_parser_mgr.get_reasoning_parser.return_value = (
            lambda tokenizer: mock_parser
        )

        cfg = ReasoningConfig(
            reasoning_parser="deepseek_r1",
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
        )
        with self.assertRaises(ValueError) as ctx:
            cfg.initialize_token_ids(self.model_config)
        self.assertIn("natural reasoning end", str(ctx.exception))

    # --- ban_sequence_end_in_thinking ----------------------------------

    def test_ban_sequence_end_default_false(self) -> None:
        """The sequence-end ban is opt-in and defaults to off."""
        cfg = ReasoningConfig()
        self.assertFalse(cfg.ban_sequence_end_in_thinking)
        self.assertEqual(cfg.sequence_end_token_ids, ())

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_sequence_end_not_resolved_when_ban_off(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """Ban off → ids stay empty even when the vocab has both markers."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {
            "<|message_end|>": 148902,
            "<|pangu_text_end|>": 148900,
        }
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(reasoning_start_str="<s>", reasoning_end_str="<e>")
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.sequence_end_token_ids, ())

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_sequence_end_resolved_when_ban_on(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """Both markers resolve, in declaration order (turn then document)."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {
            "<|message_end|>": 148902,
            "<|pangu_text_end|>": 148900,
        }
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_sequence_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.sequence_end_token_ids, (148902, 148900))

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_sequence_end_partial_vocab(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """A vocab with only the turn boundary still resolves that one side."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"<|message_end|>": 148902}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_sequence_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.sequence_end_token_ids, (148902,))

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_sequence_end_none_when_no_keys(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """Unknown vocab → empty tuple, no exception (holder warns instead)."""
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {"foo": 1, "bar": 2}
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_sequence_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.sequence_end_token_ids, ())

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_sequence_end_gate_includes_new_flag(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        """The resolve gate must accept the sequence-end flag on its own.

        With every tool ban off, only ``ban_sequence_end_in_thinking`` is set;
        resolution has to happen rather than early-returning.
        """
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        mock_enc.get_vocab.return_value = {
            "<|message_end|>": 148902,
            "<|pangu_text_end|>": 148900,
        }
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc

        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_sequence_end_in_thinking=True,
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.sequence_end_token_ids, (148902, 148900))
        # Tool bans stay off, so their ids are never probed.
        self.assertIsNone(cfg.tool_call_start_token_id)
        self.assertIsNone(cfg.tool_call_end_token_id)


if __name__ == "__main__":
    unittest.main()
