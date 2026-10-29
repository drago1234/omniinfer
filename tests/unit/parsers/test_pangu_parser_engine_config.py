# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pangu ParserEngine 配置组装的分支覆盖测试。

覆盖 _build_terminal_map / _build_transitions / pangu_parser_engine_config
在 thinking_enabled 与 implicit_tool_end 开关组合下的全部分支。
"""

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import ParserState

from omni_npu.v1.parsers._pangu_parser_engine_config import (
    PanguTerminals,
    _build_terminal_map,
    _build_transitions,
    pangu_parser_engine_config,
)


TERMINALS = PanguTerminals(
    think_start="<|think_start|>",
    think_end="<|think_end|>",
    tool_start="<|tool_call_start|>",
    tool_end="<|tool_call_end|>",
)


class TestBuildTerminalMap:
    """终结符映射:思考模式关闭时仅工具符,开启时附加 THINK 起止符。"""

    def test_thinking_disabled_only_tool_terminals(self):
        terminal_map = _build_terminal_map(TERMINALS, thinking_enabled=False)

        assert terminal_map == {
            "TOOL_START": TERMINALS.tool_start,
            "TOOL_END": TERMINALS.tool_end,
        }

    def test_thinking_enabled_adds_think_terminals(self):
        terminal_map = _build_terminal_map(TERMINALS, thinking_enabled=True)

        assert terminal_map == {
            "TOOL_START": TERMINALS.tool_start,
            "TOOL_END": TERMINALS.tool_end,
            "THINK_START": TERMINALS.think_start,
            "THINK_END": TERMINALS.think_end,
        }


class TestBuildTransitions:
    """状态转移表:基础工具转移 + 思考模式分支(含 implicit_tool_end)。"""

    def test_base_tool_transitions_without_thinking(self):
        transitions = _build_transitions(
            thinking_enabled=False, implicit_tool_end=False
        )

        assert set(transitions) == {
            (ParserState.CONTENT, "TOOL_START"),
            (ParserState.TOOL_ARGS, "TOOL_END"),
        }
        # 思考模式关闭时不应出现任何 REASONING 相关转移
        assert not any(state == ParserState.REASONING for state, _ in transitions)

    @pytest.mark.parametrize("implicit_tool_end", [True, False])
    def test_thinking_enabled_transitions(self, implicit_tool_end):
        transitions = _build_transitions(
            thinking_enabled=True, implicit_tool_end=implicit_tool_end
        )

        expected = {
            (ParserState.CONTENT, "TOOL_START"),
            (ParserState.TOOL_ARGS, "TOOL_END"),
            (ParserState.REASONING, "THINK_START"),
            (ParserState.REASONING, "THINK_END"),
            (ParserState.CONTENT, "THINK_START"),
            (ParserState.CONTENT, "THINK_END"),
        }
        if implicit_tool_end:
            expected.add((ParserState.REASONING, "TOOL_START"))

        assert set(transitions) == expected

    def test_thinking_enabled_reasoning_end_emits_event(self):
        transitions = _build_transitions(
            thinking_enabled=True, implicit_tool_end=False
        )

        reasoning_end = transitions[(ParserState.REASONING, "THINK_END")]
        assert reasoning_end.events == (EventType.REASONING_END,)

    def test_implicit_tool_end_chain_emits_reasoning_end_then_tool_start(self):
        transitions = _build_transitions(
            thinking_enabled=True, implicit_tool_end=True
        )

        transition = transitions[(ParserState.REASONING, "TOOL_START")]
        assert transition.events == (
            EventType.REASONING_END,
            EventType.TOOL_CALL_START,
        )
        assert transition.next_state == ParserState.TOOL_ARGS

    def test_thinking_disabled_ignores_implicit_tool_end(self):
        """implicit_tool_end 仅在思考模式内生效;关闭思考模式时参数无效。"""
        with_flag = _build_transitions(
            thinking_enabled=False, implicit_tool_end=True
        )
        without_flag = _build_transitions(
            thinking_enabled=False, implicit_tool_end=False
        )

        assert set(with_flag) == set(without_flag)


class TestPanguParserEngineConfig:
    """组装入口:初始状态、token 终结符、保留 token 与各开关联动。"""

    @pytest.mark.parametrize(
        ("thinking_enabled", "expected_initial"),
        [(True, ParserState.REASONING), (False, ParserState.CONTENT)],
    )
    def test_initial_state_follows_thinking_mode(
        self, thinking_enabled, expected_initial
    ):
        config = pangu_parser_engine_config(
            TERMINALS,
            thinking_enabled=thinking_enabled,
            implicit_tool_end=False,
        )

        assert config.initial_state == expected_initial

    def test_token_terminals_mirror_string_terminals(self):
        config = pangu_parser_engine_config(
            TERMINALS,
            thinking_enabled=True,
            implicit_tool_end=False,
        )

        assert config.token_id_terminals == dict(config.terminals)
        assert config.terminals["THINK_START"] == TERMINALS.think_start

    def test_thinking_disabled_preserves_think_tokens(self):
        """思考模式关闭时,THINK 起止 token 需保留在输出流中。"""
        config = pangu_parser_engine_config(
            TERMINALS,
            thinking_enabled=False,
            implicit_tool_end=False,
        )

        assert config.preserve_tokens == frozenset(
            {TERMINALS.think_start, TERMINALS.think_end}
        )

    def test_thinking_enabled_preserves_nothing(self):
        """思考模式开启时,THINK 起止符由状态机消费,不再透传。"""
        config = pangu_parser_engine_config(
            TERMINALS,
            thinking_enabled=True,
            implicit_tool_end=False,
        )

        assert config.preserve_tokens == frozenset()

    def test_content_events_cover_all_states(self):
        config = pangu_parser_engine_config(
            TERMINALS,
            thinking_enabled=True,
            implicit_tool_end=True,
        )

        assert config.content_events == {
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        }
        assert config.tool_args_json is False
        assert config.name == "pangu"
