# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Admit generation prompts independently of the requested output budget.

vLLM 0.25.1 validates prompt + requested output during rendering, before
serving.get_max_tokens can clamp output to the remaining context. In PD this
can reject D after P has already completed with max_tokens=1.

Reserve one output token during rendering; keep the original request intact
so serving still applies its output/default/platform limits. Explicit prompt
truncation remains opt-in. This does not handle cleanup of other D-side errors.
"""

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_chat_build_tok_params = ChatCompletionRequest.build_tok_params
_completion_build_tok_params = CompletionRequest.build_tok_params


@register_patch("ChatOutputTokenBudget", ChatCompletionRequest)
class ChatOutputTokenBudgetPatch(VLLMPatch):
    _attr_names_to_apply = ["build_tok_params"]

    def build_tok_params(self, model_config):
        # Change only a local copy used to construct rendering parameters.
        # Both fields must be overridden: max_completion_tokens takes priority.
        rendering_request = self.model_copy(update={
            "max_tokens": 1, "max_completion_tokens": 1,
        })
        return _chat_build_tok_params(rendering_request, model_config)


@register_patch("CompletionOutputTokenBudget", CompletionRequest)
class CompletionOutputTokenBudgetPatch(VLLMPatch):
    _attr_names_to_apply = ["build_tok_params"]

    def build_tok_params(self, model_config):
        rendering_request = self.model_copy(update={"max_tokens": 1})
        return _completion_build_tok_params(rendering_request, model_config)
