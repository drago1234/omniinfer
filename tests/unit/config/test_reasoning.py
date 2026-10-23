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

    def test_ban_tokens_in_thinking_default_empty(self) -> None:
        """The ban list is opt-in and defaults to empty."""
        cfg = ReasoningConfig()
        self.assertEqual(cfg.ban_tokens_in_thinking, [])
        self.assertEqual(cfg.banned_token_ids, ())

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
        self.assertEqual(cfg.natural_reasoning_end_token_ids, [2])
        self.assertIsNot(cfg.natural_reasoning_end_token_ids, cfg.reasoning_end_token_ids)

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

    def _install_tokenizer(
        self,
        mock_omni_tok: MagicMock,
        mock_vllm_tok: MagicMock,
        vocab: dict[str, int],
    ) -> MagicMock:
        mock_enc = MagicMock()
        mock_enc.encode.side_effect = (
            lambda s, add_special_tokens=False: [1] if s == "<s>" else [2]
        )
        mock_enc.get_vocab.return_value = vocab
        mock_omni_tok.return_value = mock_enc
        mock_vllm_tok.return_value = mock_enc
        return mock_enc

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_empty_ban_list_skips_vocab_lookup(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = self._install_tokenizer(
            mock_omni_tok,
            mock_vllm_tok,
            {"<|tool_call_start|>": 999, "<|message_end|>": 148902},
        )
        cfg = ReasoningConfig(reasoning_start_str="<s>", reasoning_end_str="<e>")
        cfg.initialize_token_ids(self.model_config)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.banned_token_ids, ())
        mock_enc.get_vocab.assert_not_called()

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_ban_list_skipped_when_reasoning_disabled(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        mock_enc = self._install_tokenizer(
            mock_omni_tok, mock_vllm_tok, {"<|message_end|>": 148902}
        )
        cfg = ReasoningConfig(ban_tokens_in_thinking=["<|message_end|>"])
        # An empty parser name is still looked up upstream when the start and
        # end strings are missing. Clear it so this case stays on the
        # disabled path instead of raising KeyError.
        cfg.reasoning_parser = None
        cfg.initialize_token_ids(self.model_config)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.banned_token_ids, ())
        mock_enc.get_vocab.assert_not_called()

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_ban_list_resolves_literal_strings_in_order(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        self._install_tokenizer(
            mock_omni_tok,
            mock_vllm_tok,
            {
                "<|tool_call_start|>": 999,
                "<|message_end|>": 148902,
                "<|pangu_text_end|>": 148900,
            },
        )
        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tokens_in_thinking=["<|tool_call_start|>", "<|message_end|>"],
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.banned_token_ids, (999, 148902))

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_ban_list_skips_missing_vocab_entry(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        self._install_tokenizer(
            mock_omni_tok, mock_vllm_tok, {"<|message_end|>": 148902}
        )
        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tokens_in_thinking=["<|message_end|>", "<|pangu_text_end|>"],
        )
        with self.assertLogs(
            "omni_npu.v1.config.reasoning", level="WARNING"
        ) as logs:
            cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.banned_token_ids, (148902,))
        self.assertIn("<|pangu_text_end|>", logs.output[0])

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_ban_list_collapses_duplicate_ids(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        self._install_tokenizer(
            mock_omni_tok,
            mock_vllm_tok,
            {"<|tool_call_start|>": 7, "<|message_end|>": 7, "<|tool_call_end|>": 8},
        )
        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tokens_in_thinking=[
                "<|tool_call_start|>",
                "<|message_end|>",
                "<|tool_call_end|>",
                "<|tool_call_start|>",
            ],
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.banned_token_ids, (7, 8))

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    def test_unlisted_vocab_token_is_not_resolved(
        self, mock_omni_tok: MagicMock, mock_vllm_tok: MagicMock
    ) -> None:
        self._install_tokenizer(
            mock_omni_tok,
            mock_vllm_tok,
            {"<|message_end|>": 148902, "<|pangu_text_end|>": 148900},
        )
        cfg = ReasoningConfig(
            reasoning_start_str="<s>",
            reasoning_end_str="<e>",
            ban_tokens_in_thinking=["<|message_end|>"],
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.banned_token_ids, (148902,))


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
        mock_omni_tok.assert_not_called()
        self.assertEqual(mock_enc.encode.call_count, 2)

    @patch("vllm.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.cached_tokenizer_from_config")
    @patch("omni_npu.v1.config.reasoning.ReasoningParserManager")
    def test_natural_end_without_parser_marker_reuses_end_ids(
        self, mock_parser_mgr, mock_omni_tok, mock_vllm_tok
    ) -> None:
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda s, **kw: [1] if s == "<s>" else [2, 3]
        mock_omni_tok.return_value = mock_vllm_tok.return_value = tokenizer
        mock_parser_mgr.get_reasoning_parser.return_value.return_value.reasoning_end_str = ""

        cfg = ReasoningConfig(
            reasoning_parser="deepseek_r1",
            reasoning_start_str="<s>", reasoning_end_str="<e>",
        )
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(cfg.natural_reasoning_end_token_ids, [2, 3])
        self.assertEqual(tokenizer.encode.call_count, 2)
        cfg.initialize_token_ids(self.model_config)
        self.assertEqual(tokenizer.encode.call_count, 2)
        mock_parser_mgr.get_reasoning_parser.assert_called_once_with("deepseek_r1")

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


if __name__ == "__main__":
    unittest.main()
