from types import SimpleNamespace

import pytest

from core.ai.protocol import AssistantMessage, ContextOverflowError
from core.engine.agent_engine import AgentEngine
from core.engine.conversation_event_log import (
    ConversationEventLog,
    EventKind,
    TurnKind,
)
from core.tools._types import ToolResult
from core.tools.tool_loop import ToolLoop


@pytest.mark.asyncio
async def test_tool_loop_rebuilds_prompt_once_after_context_overflow():
    class FakeAI:
        model = "test"

        def __init__(self):
            self.requests = []
            self.attempts = 0

        async def chat_completion_with_tools(self, *, messages, tools):
            self.requests.append(list(messages))
            self.attempts += 1
            if self.attempts == 1:
                raise ContextOverflowError("context length exceeded")
            return AssistantMessage(content="done"), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ai = FakeAI()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=ai,
            max_tool_rounds=2,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    loop = ToolLoop(ctx)
    rebuilt = []

    async def rebuild(service, elapsed_ms):
        rebuilt.append(service)
        return ([{"role": "system", "content": "compressed"}], [])

    replies = []

    async def reply_callback(*args, **kwargs):
        replies.append(kwargs.get("content", args[1] if len(args) > 1 else ""))

    await loop.run(
        messages=[{"role": "system", "content": "original"}],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="reply",
        reply_callback=reply_callback,
        model_context_overflow_callback=rebuild,
    )

    assert len(rebuilt) == 1
    assert ai.requests == [
        [{"role": "system", "content": "original"}],
        [{"role": "system", "content": "compressed"}],
    ]
    assert replies == ["done"]


@pytest.mark.asyncio
async def test_tool_loop_propagates_turn_kind_to_protocol_events(monkeypatch, tmp_path):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "inspect",
                                "arguments": "{}",
                            }
                        ]
                    ),
                    AssistantMessage(content="done"),
                ]
            )

        async def chat_completion_with_tools(self, *, messages, tools):
            response = next(self.responses)
            if response.tool_calls and isinstance(response.tool_calls[0], dict):
                from core.ai.protocol import AssistantToolCall

                response.tool_calls = [AssistantToolCall(**response.tool_calls[0])]
            return response, None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    async def fake_execute(name, args, tool_context, permission_manager):
        return ToolResult(content="inspection result")

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=FakeAI(),
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )

    async def reply_callback(**kwargs):
        return None

    await ToolLoop(ctx).run(
        messages=[{"role": "user", "content": "run"}],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="turn-1",
        reply_callback=reply_callback,
        event_log=event_log,
        turn_id="turn-1",
        turn_kind=TurnKind.SYSTEM,
    )

    events = await event_log.snapshot_events("chat", include_internal=True)
    assert [event.kind for event in events.events] == [
        EventKind.ASSISTANT_TOOL_CALL,
        EventKind.TOOL_RESULT,
        EventKind.ASSISTANT_TOOL_CALL,
    ]
    turn = (await event_log.snapshot_turns("chat", include_internal=True)).turns[0]
    assert turn.turn_kind is TurnKind.SYSTEM
    await event_log.close()
