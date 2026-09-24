# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine


class PanguStreamingParserEngine(StreamingParserEngine):
    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:
        if self.skip_tool_parsing and terminal in self._tool_terminals:
            transition = self.config.transitions.get((self.state, terminal))
            # implicit_tool_end path: the marker would end reasoning and move
            # to TOOL_ARGS.  Mirror the base branch: close reasoning and force
            # state -> CONTENT so the following JSON body reaches the content
            # channel (BFCL reads it from there), but emit NO marker text.
            if transition is not None and EventType.REASONING_END in transition.events:
                self.state = ParserState.CONTENT
                return [
                    SemanticEvent(
                        EventType.REASONING_END,
                        value="",  # drop the marker text
                        tool_index=self.tool_index,
                    )
                ]
            # Any other tool terminal while skip_tool_parsing is active (incl.
            # the no-transition case, e.g. tool_end arriving in CONTENT, or
            # tool_start arriving in REASONING when implicit_tool_end is off):
            # drop the marker entirely and leave the state untouched so
            # subsequent prose keeps flowing in the current channel.
            return []
        return super()._on_terminal(terminal, value)
