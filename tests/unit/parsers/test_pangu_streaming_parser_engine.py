# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Branch coverage for PanguStreamingParserEngine._on_terminal."""

import pytest

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import ParserState

from omni_npu.v1.parsers._pangu_parser_engine_config import (
    PanguTerminals,
    pangu_parser_engine_config,
)
from omni_npu.v1.parsers._pangu_streaming_parser_engine import (
    PanguStreamingParserEngine,
)
from omni_npu.v1.parsers.pangu_parser_engine import PanguParserEngine

TERMINALS = PanguTerminals(
    think_start="<|think_start|>",
    think_end="<|think_end|>",
    tool_start="<|tool_call_start|>",
    tool_end="<|tool_call_end|>",
)


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    @property
    def all_special_tokens(self) -> list[str]:
        return []

    @property
    def all_special_ids(self) -> list[int]:
        return []

    def decode(self, _token_ids: list[int]) -> str:
        return ""


def _build_engine(*, thinking_enabled: bool = True, implicit_tool_end: bool = True):
    vocab = {
        TERMINALS.think_start: 100,
        TERMINALS.think_end: 101,
        TERMINALS.tool_start: 102,
        TERMINALS.tool_end: 103,
    }
    config = pangu_parser_engine_config(
        TERMINALS,
        thinking_enabled=thinking_enabled,
        implicit_tool_end=implicit_tool_end,
    )
    return PanguStreamingParserEngine(config, _FakeTokenizer(vocab), vocab=vocab)


@pytest.mark.parametrize("tool_index", [-1, 3])
def test_skip_tool_parsing_implicit_tool_end_closes_reasoning(tool_index):
    engine = _build_engine(thinking_enabled=True, implicit_tool_end=True)
    engine.state = ParserState.REASONING
    engine.skip_tool_parsing = True
    engine.tool_index = tool_index

    events = engine._on_terminal("TOOL_START", TERMINALS.tool_start)

    assert engine.state == ParserState.CONTENT
    assert [e.type for e in events] == [EventType.REASONING_END]
    assert events[0].value == ""
    assert events[0].tool_index == tool_index


def test_skip_tool_parsing_tool_end_in_content_returns_empty():
    engine = _build_engine()
    engine.state = ParserState.CONTENT
    engine.skip_tool_parsing = True

    events = engine._on_terminal("TOOL_END", TERMINALS.tool_end)

    assert events == []
    assert engine.state == ParserState.CONTENT


def test_skip_tool_parsing_tool_end_in_tool_args_returns_empty():
    engine = _build_engine()
    engine.state = ParserState.TOOL_ARGS
    engine.skip_tool_parsing = True
    engine.tool_index = 0

    events = engine._on_terminal("TOOL_END", TERMINALS.tool_end)

    assert events == []
    assert engine.state == ParserState.TOOL_ARGS


def test_skip_tool_parsing_tool_start_in_content_returns_empty():
    engine = _build_engine(thinking_enabled=True, implicit_tool_end=False)
    engine.state = ParserState.CONTENT
    engine.skip_tool_parsing = True

    events = engine._on_terminal("TOOL_START", TERMINALS.tool_start)

    assert events == []
    assert engine.state == ParserState.CONTENT


def test_non_tool_terminal_falls_through_to_base():
    engine = _build_engine(thinking_enabled=True, implicit_tool_end=True)
    engine.state = ParserState.REASONING
    engine.skip_tool_parsing = True

    events = engine._on_terminal("THINK_END", TERMINALS.think_end)

    assert engine.state == ParserState.CONTENT
    assert [e.type for e in events] == [EventType.REASONING_END]
    assert events[0].value == TERMINALS.think_end


def test_skip_tool_parsing_off_delegates_to_base_for_tool_start():
    engine = _build_engine()
    engine.state = ParserState.CONTENT
    engine.skip_tool_parsing = False
    engine.tool_index = -1

    events = engine._on_terminal("TOOL_START", TERMINALS.tool_start)

    assert engine.state == ParserState.TOOL_ARGS
    assert [e.type for e in events] == [EventType.TOOL_CALL_START]
    assert events[0].value == TERMINALS.tool_start
    assert engine.tool_index == 0
    assert events[0].tool_index == 0


def test_pangu_parser_engine_uses_streaming_subclass():
    engine = PanguParserEngine(_FakeTokenizer({
        TERMINALS.think_start: 100,
        TERMINALS.think_end: 101,
        TERMINALS.tool_start: 102,
        TERMINALS.tool_end: 103,
    }))
    assert isinstance(engine._engine, PanguStreamingParserEngine)
