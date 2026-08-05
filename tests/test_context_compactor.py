from unittest.mock import AsyncMock

import pytest

from core.managers.chat_message import ChatMessage
from core.managers.context_compactor import ContextCompactor


class FakeAI:
    def __init__(self, summary="摘要", usage=None, error=None):
        self.chat_completion = AsyncMock()
        if error:
            self.chat_completion.side_effect = error
        else:
            self.chat_completion.return_value = (summary, usage)


def message(role, content, timestamp, **kwargs):
    return ChatMessage(role=role, content=content, timestamp=timestamp, **kwargs)


def make_compactor(ai, threshold=1, recent=1, max_summary=123):
    return ContextCompactor(ai, threshold, recent, max_summary)


@pytest.mark.asyncio
async def test_below_threshold_does_not_call_ai():
    ai = FakeAI()
    compactor = make_compactor(ai, threshold=10_000)
    messages = [message("user", "hello", 1)]

    result = await compactor.compact(messages)

    assert result.compacted is False
    assert result.messages == messages
    ai.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_above_threshold_compacts_without_force():
    ai = FakeAI()
    compactor = make_compactor(ai, threshold=1, recent=1)
    messages = [message("user", "old", 1), message("user", "recent", 2)]

    result = await compactor.compact(messages)

    assert result.compacted is True
    ai.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_without_old_messages_does_not_call_ai():
    ai = FakeAI()
    compactor = make_compactor(ai, recent=100)
    messages = [message("user", "only", 1)]

    result = await compactor.compact(messages, force=True)

    assert result.compacted is False
    assert result.messages == messages
    ai.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_input_sequence_is_not_modified():
    ai = FakeAI()
    compactor = make_compactor(ai, recent=1)
    messages = [message("user", "old", 1), message("user", "recent", 2)]
    original = list(messages)

    await compactor.compact(messages, force=True)

    assert messages == original


@pytest.mark.asyncio
async def test_force_compacts_and_builds_summary_request():
    ai = FakeAI("  summary  ", {"total_tokens": 7})
    compactor = make_compactor(ai, recent=1, max_summary=321)
    messages = [message("user", "old", 42), message("user", "recent", 43)]

    result = await compactor.compact(messages, force=True)

    assert result.compacted is True
    assert result.usage == {"total_tokens": 7}
    assert result.messages[0].content == "【历史对话摘要】\nsummary"
    assert result.messages[0].role == "assistant"
    assert result.messages[0].name == "系统"
    assert result.messages[0].timestamp == 42
    assert result.messages[1] is messages[1]
    request = ai.chat_completion.await_args.kwargs
    assert request["max_tokens"] == 321
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_empty_none_whitespace_and_error_leave_input_unchanged():
    messages = [message("user", "old", 1), message("user", "recent", 2)]
    cases = [
        (FakeAI(""), None),
        (FakeAI(None, {"total_tokens": 4}), {"total_tokens": 4}),
        (FakeAI("   \n  ", {"total_tokens": 2}), {"total_tokens": 2}),
        (FakeAI(error=RuntimeError("offline")), None),
    ]

    for ai, expected_usage in cases:
        result = await make_compactor(ai).compact(messages, force=True)
        assert result.compacted is False
        assert result.messages == messages
        assert result.usage == expected_usage


@pytest.mark.asyncio
async def test_tool_call_and_result_stay_together():
    tool_calls = [{"id": "call-1", "function": {"name": "search", "arguments": "{}"}}]
    messages = [
        message("user", "old", 1),
        message("assistant", "", 2, tool_calls=tool_calls),
        message("tool", "result", 3, tool_call_id="call-1"),
    ]
    ai = FakeAI()
    result = await make_compactor(ai, recent=1).compact(messages, force=True)

    assert result.compacted is True
    remaining = result.messages[1:]
    assert [item.role for item in remaining] == ["assistant", "tool"]
    assert remaining[0].tool_calls == tool_calls
