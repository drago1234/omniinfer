# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect the native thinking-budget state machine to MRv2 and MTP rows."""

from __future__ import annotations

from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder


class MRv2ThinkingBudget:
    def __init__(self, reasoning_config, device):
        self.reasoning_config = reasoning_config
        self.device = device
        # Persistent request slots are independent of each step's batch rows.
        self.requests = {}

    def add_request(self, req_idx, new_req_data):
        self.requests.pop(req_idx, None)
        params = new_req_data.sampling_params
        if params is None or params.thinking_token_budget is None:
            return
        prompt = new_req_data.prompt_token_ids
        outputs = list(new_req_data.prefill_token_ids[len(prompt):])
        holder = ThinkingBudgetStateHolder(
            self.reasoning_config, 1, 0, self.device, is_pin_memory=False
        )
        holder.sync_batch(BatchUpdate(1, [], [(0, params, prompt, outputs)], []))
        self.requests[req_idx] = (holder, outputs, len(prompt))

    def remove_request(self, req_idx):
        self.requests.pop(req_idx, None)

    def apply(self, logits, req_states, idx_mapping_np,
              expanded_idx_mapping=None, input_ids=None):
        active = [int(idx) for idx in idx_mapping_np if int(idx) in self.requests]
        if not active:
            return logits
        # Read only committed history: chunked-prefill samples are discarded,
        # and resumption can include output tokens in prefill_token_ids.
        total_lens = req_states.total_len.gpu[active].cpu().tolist()
        mapping = (expanded_idx_mapping.cpu().tolist()
                   if expanded_idx_mapping is not None else idx_mapping_np.tolist())
        draft_inputs = input_ids.cpu().tolist() if input_ids is not None else None
        for idx, total_len in zip(active, total_lens):
            holder, outputs, prompt_len = self.requests[idx]
            start = prompt_len + len(outputs)
            if total_len > start:
                outputs.extend(req_states.all_token_ids.gpu[idx, start:total_len]
                               .cpu().tolist())
            holder.update_state([outputs], None)
            if not holder.has_tracked_requests():
                continue
            rows = [row for row, slot in enumerate(mapping) if slot == idx]
            # Row j predicts after draft[0:j] has been accepted. Its input is
            # draft[j-1]; row 0's input is already in committed history.
            # Keep this hypothetical prefix separate from persistent state:
            # verification may reject any of these tokens.
            committed_state = holder._state[0].copy()
            virtual_outputs = list(outputs)
            try:
                for step, row in enumerate(rows):
                    if step:
                        virtual_outputs.append(draft_inputs[row])
                        holder.update_state([virtual_outputs], None)
                    state = holder._state[0]
                    if state["in_end"] and state["force_index"]:
                        # A delta target distribution makes standard rejection
                        # reject any other draft and resample the end token.
                        token = holder.think_end_token_ids[state["end_count"]]
                        logits[row].fill_(float("-inf"))
                        logits[row, token] = 0
            finally:
                holder._state[0] = committed_state
        return logits
