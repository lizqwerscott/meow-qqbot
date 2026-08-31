import pytest

from core.engine.conversation_timeline import TimelineEvent
from core.engine.model_context_transcript import ModelContextScope, ModelContextSnapshot
from core.engine.prompt_builder import PromptBuilder
from core.message import InputMessage


@pytest.mark.asyncio
async def test_prompt_build_requires_authoritative_timeline_snapshot():
    builder = PromptBuilder.__new__(PromptBuilder)

    with pytest.raises(ValueError, match="timeline_snapshot is required"):
        await builder.build(
            chat_id="chat",
            is_group=False,
            user_nickname="alice",
            sender_id="alice",
            input_message=InputMessage(
                id="message",
                sender_id="alice",
                chat_id="chat",
                content="hello",
                is_group=False,
            ),
        )


def test_timeline_history_is_authoritative_and_keeps_only_visible_events():
    snapshot = (
        TimelineEvent(
            chat_id="chat",
            seq=1,
            event_id="user:m1",
            role="user",
            content="question",
            message_id="m1",
            sender_id="alice",
            timestamp=1_700_000_000,
        ),
        TimelineEvent(
            chat_id="chat",
            seq=2,
            event_id="delivery:d1",
            role="assistant",
            content="answer",
            event_kind="delivery",
            delivery_kind="response",
            timestamp=1_700_000_001,
        ),
    )

    projected = PromptBuilder._timeline_history(snapshot)

    assert [message["role"] for message in projected] == ["user", "assistant"]
    assert projected[0] == {
        "role": "user",
        "content": "[alice 在 2023-11-15 06:13:20]: question",
    }
    assert projected[1]["content"] == "answer"
    assert all(set(message) == {"role", "content"} for message in projected)
    assert PromptBuilder._timeline_history(()) == []


def test_empty_model_context_snapshot_preserves_existing_timeline_assistant_messages():
    timeline = (
        TimelineEvent(
            chat_id="chat",
            seq=1,
            event_id="user:m1",
            role="user",
            content="question",
            message_id="m1",
            sender_id="alice",
            timestamp=1_700_000_000,
        ),
        TimelineEvent(
            chat_id="chat",
            seq=2,
            event_id="delivery:d1",
            role="assistant",
            content="answer",
            event_kind="delivery",
            delivery_kind="response",
            timestamp=1_700_000_001,
        ),
    )
    snapshot = ModelContextSnapshot(
        scope=ModelContextScope(chat_id="chat", principal_id="alice"),
        events=(),
    )

    history = PromptBuilder._model_context_history(snapshot, timeline)

    assert [message["role"] for message in history] == ["user", "assistant"]


def test_protocol_snapshot_is_limited_to_the_requested_turn():
    from core.engine.turn_protocol_history import ProtocolEvent, TurnProtocolHistory

    events = (
        ProtocolEvent(
            turn_id="active",
            seq=1,
            event_id="assistant:0",
            role="assistant",
            content="internal plan",
            tool_calls=({"id": "call-1", "type": "function"},),
        ),
        ProtocolEvent(
            turn_id="active",
            seq=2,
            event_id="tool:call-1",
            role="tool",
            content="tool result",
            tool_call_id="call-1",
        ),
    )

    wire = PromptBuilder._timeline_history(())
    wire.extend(TurnProtocolHistory.to_wire_messages(events))

    assert wire == [
        {
            "role": "assistant",
            "content": "internal plan",
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
    ]
