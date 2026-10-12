# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.reasoning import ReasoningConfig as VllmReasoningConfig
from vllm.config.utils import config
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig(VllmReasoningConfig):
    """vLLM reasoning configuration extended with Omni thinking bans.

    The upstream fields and token initialisation remain authoritative. Omni
    adds only compatibility aliases, the Pangu tool-marker ban switches, and a
    server-side ``thinking_token_budget`` default. Per-request budgets still
    use vLLM's native ``SamplingParams.thinking_token_budget`` field and native
    ``ThinkingBudgetStateHolder``; ``thinking_token_budget`` here is the
    launch-time default applied when a request omits its own (see
    ``patch_thinking_limit.py``).
    """

    think_start_str: str = ""
    """Deprecated alias for `reasoning_start_str`."""
    think_end_str: str = ""
    """Deprecated alias for `reasoning_end_str`."""

    thinking_token_budget: int | None = None
    """Server-side default thinking-token budget. Applied per-request when the
    request omits its own ``thinking_token_budget`` (see
    ``patch_thinking_limit.py``). The request's own budget always wins and is
    carried by vLLM's native ``SamplingParams.thinking_token_budget`` field."""

    ban_tool_start_in_thinking: bool = False
    """Ban the tool-call start marker while reasoning is still open."""
    ban_tool_end_in_thinking: bool = False
    """Ban the tool-call end marker while reasoning is still open."""

    ban_sequence_end_in_thinking: bool = False
    """Ban the sequence/turn end tokens while reasoning is still open.

    Pangu tends to emit ``<|message_end|>`` (turn boundary) -- and right after
    it ``<|pangu_text_end|>`` (document boundary) -- at the position where it
    should emit ``</think>``. Both terminate decoding: the first through
    ``stop_token_ids``, the second through ``eos_token_id`` (see
    ``vllm.v1.core.sched.utils.check_stop``). Sampling either one while
    reasoning is open truncates the stream before a single content token is
    generated, which reaches the client as reasoning-only output. Banning both
    pushes the model back onto ``</think>`` so the answer can actually be
    written.
    """

    _tool_call_start_token_id: int | None = field(
        default=None, init=False, repr=False
    )
    _tool_call_end_token_id: int | None = field(
        default=None, init=False, repr=False
    )
    _sequence_end_token_ids: tuple[int, ...] = field(
        default=(), init=False, repr=False
    )

    @property
    def tool_call_start_token_id(self) -> int | None:
        return self._tool_call_start_token_id

    @property
    def tool_call_end_token_id(self) -> int | None:
        return self._tool_call_end_token_id

    @property
    def sequence_end_token_ids(self) -> tuple[int, ...]:
        """Token ids banned while reasoning is open."""
        return self._sequence_end_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Run native reasoning initialisation, then resolve ban marker IDs."""
        if not self.reasoning_start_str and self.think_start_str:
            self.reasoning_start_str = self.think_start_str
        if not self.reasoning_end_str and self.think_end_str:
            self.reasoning_end_str = self.think_end_str

        # This preserves vLLM 0.25.1's parser-derived marker support.
        super().initialize_token_ids(model_config)
        if not self.enabled or not (
            self.ban_tool_start_in_thinking
            or self.ban_tool_end_in_thinking
            or self.ban_sequence_end_in_thinking
        ):
            return

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        vocab = tokenizer.get_vocab()

        # Pangu's legacy markers are kept as fallbacks for old checkpoints.
        if self.ban_tool_start_in_thinking:
            for marker in ("<|tool_call_start|>", "[unused11]"):
                if marker in vocab:
                    self._tool_call_start_token_id = int(vocab[marker])
                    break
        if self.ban_tool_end_in_thinking:
            for marker in ("<|tool_call_end|>", "[unused12]"):
                if marker in vocab:
                    self._tool_call_end_token_id = int(vocab[marker])
                    break
        if self.ban_sequence_end_in_thinking:
            # The turn boundary is the one that fires in practice; the document
            # boundary is the tokenizer's declared eos and shows up right after
            # it. Both must be held back or the stream can end with no content.
            # A missing marker leaves the side disabled; the ban holder warns.
            self._sequence_end_token_ids = tuple(
                int(vocab[marker])
                for marker in ("<|message_end|>", "<|pangu_text_end|>")
                if marker in vocab
            )
