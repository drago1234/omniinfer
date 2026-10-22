# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.reasoning import ReasoningConfig as VllmReasoningConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

logger = init_logger(__name__)


@config
class ReasoningConfig(VllmReasoningConfig):
    """vLLM reasoning configuration extended with Omni thinking bans.

    The upstream fields and token initialisation remain authoritative. Omni
    adds only compatibility aliases, a list of tokenizer strings banned while
    thinking is open, and a server-side ``thinking_token_budget`` default.
    Per-request budgets still use vLLM's native
    ``SamplingParams.thinking_token_budget`` field and native
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

    ban_tokens_in_thinking: list[str] = field(default_factory=list)
    """Tokenizer strings banned while reasoning is still open.

    Each entry is looked up as-is in the tokenizer vocab, for example
    ``<|tool_call_start|>`` or ``<|message_end|>``. A string absent from the
    vocab is skipped. Resolved ids are suppressed on every logits row while
    the request is still inside ``<think>``.
    """

    _banned_token_ids: tuple[int, ...] = field(
        default=(), init=False, repr=False
    )

    @property
    def banned_token_ids(self) -> tuple[int, ...]:
        """Token ids banned while reasoning is open."""
        return self._banned_token_ids

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Run native reasoning initialisation, then resolve ban marker IDs."""
        if not self.reasoning_start_str and self.think_start_str:
            self.reasoning_start_str = self.think_start_str
        if not self.reasoning_end_str and self.think_end_str:
            self.reasoning_end_str = self.think_end_str

        # This preserves vLLM 0.25.1's parser-derived marker support.
        super().initialize_token_ids(model_config)
        if not self.enabled or not self.ban_tokens_in_thinking:
            return

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        vocab = tokenizer.get_vocab()
        resolved: list[int] = []
        missing: list[str] = []
        seen: set[int] = set()
        for marker in self.ban_tokens_in_thinking:
            if marker not in vocab:
                missing.append(marker)
                continue
            tid = int(vocab[marker])
            if tid in seen:
                continue
            seen.add(tid)
            resolved.append(tid)
        if missing:
            logger.warning(
                "ReasoningConfig: ban_tokens_in_thinking entries not in the "
                "tokenizer vocab, skipped: %s",
                missing,
            )
        self._banned_token_ids = tuple(resolved)
