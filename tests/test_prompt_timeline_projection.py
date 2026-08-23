from core.engine.conversation_timeline import TimelineEvent
from core.engine.model_context_transcript import ModelContextScope, ModelContextSnapshot
from core.engine.prompt_builder import PromptBuilder


def test_timeline_snapshot_replaces_matching_user_content_without_reordering_protocol():
    history = [
        {
            "role": "user",
            "content": "[old-name 在 2020-01-01 00:00:00]: stale",
            "raw_content": "stale",
            "message_id": "m1",
            "sender_id": "user",
            "name": "old-name",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    snapshot = (
        TimelineEvent(
            chat_id="chat",
            seq=1,
            event_id="user:m1",
            role="user",
            content="authoritative",
            message_id="m1",
            sender_id="user",
            timestamp=1_700_000_000,
        ),
    )

    projected = PromptBuilder._apply_timeline_snapshot(history, snapshot)

    assert [message["role"] for message in projected] == ["user", "assistant", "tool"]
    assert projected[0]["raw_content"] == "authoritative"
    assert projected[0]["content"].endswith(": authoritative")
    assert projected[1] == history[1]
    assert projected[2] == history[2]


def test_timeline_snapshot_does_not_inject_events_missing_from_legacy_history():
    history = [{"role": "assistant", "content": "existing"}]
    snapshot = (
        TimelineEvent(
            chat_id="chat",
            seq=1,
            event_id="user:m1",
            role="user",
            content="not inserted",
            message_id="m1",
            sender_id="user",
        ),
    )

    assert PromptBuilder._apply_timeline_snapshot(history, snapshot) == history


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
