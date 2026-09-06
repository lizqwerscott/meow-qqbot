import sqlite3

import pytest

from core.engine.turn_protocol_history import (
    ProtocolInvariantError,
    TurnProtocolHistory,
)


@pytest.mark.asyncio
async def test_protocol_history_keeps_assistant_tool_pairing_and_is_idempotent(
    tmp_path,
):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    assistant = await history.append_assistant(
        turn_id="turn-1",
        event_id="assistant:0",
        content="I will inspect it",
        tool_calls=({"id": "call-1", "name": "read_file"},),
        reasoning_content="inspect",
    )
    tool = await history.append_tool_result(
        turn_id="turn-1",
        event_id="tool:call-1",
        tool_call_id="call-1",
        tool_name="read_file",
        content="file contents",
    )
    duplicate = await history.append_tool_result(
        turn_id="turn-1",
        event_id="tool:call-1",
        tool_call_id="call-1",
        tool_name="read_file",
        content="changed",
    )

    assert assistant.seq == 1
    assert tool.seq == 2
    assert duplicate == tool
    assert [event.role for event in await history.snapshot("turn-1")] == [
        "assistant",
        "tool",
    ]
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_rejects_orphan_and_duplicate_tool_results(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))

    with pytest.raises(ProtocolInvariantError, match="no assistant call"):
        await history.append_tool_result(
            turn_id="turn-1",
            event_id="tool:orphan",
            tool_call_id="orphan",
            tool_name="read_file",
            content="bad",
        )

    await history.append_assistant(
        turn_id="turn-1",
        event_id="assistant:0",
        content="",
        tool_calls=({"id": "call-1", "name": "read_file"},),
    )
    await history.append_tool_result(
        turn_id="turn-1",
        event_id="tool:call-1",
        tool_call_id="call-1",
        tool_name="read_file",
        content="ok",
    )
    with pytest.raises(ProtocolInvariantError, match="duplicate tool result"):
        await history.append_tool_result(
            turn_id="turn-1",
            event_id="tool:call-2",
            tool_call_id="call-1",
            tool_name="read_file",
            content="again",
        )
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_isolated_by_turn_id(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await history.append_assistant(
        turn_id="turn-a",
        event_id="assistant:0",
        content="a",
    )
    await history.append_assistant(
        turn_id="turn-b",
        event_id="assistant:0",
        content="b",
    )

    assert [event.content for event in await history.snapshot("turn-a")] == ["a"]
    assert [event.content for event in await history.snapshot("turn-b")] == ["b"]
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_deletes_one_chat_and_caches_full_event_tokens(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await history.append_assistant(
        chat_id="chat-a",
        turn_id="turn-a",
        event_id="assistant:0",
        content="answer",
        tool_calls=[
            {"id": "call-a", "function": {"name": "read_file", "arguments": "{}"}}
        ],
        reasoning_content="reasoning",
    )
    await history.append_assistant(
        chat_id="chat-b",
        turn_id="turn-b",
        event_id="assistant:0",
        content="keep",
    )

    summary = await history.session_summary("chat-a")
    assert summary["estimated_tokens"] > len("answer") // 4

    await history.delete_chat("chat-a")

    assert await history.history("chat-a") == []
    assert len(await history.history("chat-b")) == 1
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_bounded_read_returns_latest_events(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    for index in range(4):
        await history.append_assistant(
            chat_id="chat-a",
            turn_id=f"turn-{index}",
            event_id=f"assistant:{index}",
            content=str(index),
            timestamp=float(index),
        )

    bounded = await history.history("chat-a", max_events=2)

    assert [message["content"] for message in bounded] == ["2", "3"]
    assert await history.history("chat-a", max_events=0) == []
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_isolates_same_turn_id_across_chats(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await history.append_assistant(
        chat_id="chat-a",
        turn_id="shared-turn",
        event_id="assistant:a",
        content="a",
        tool_calls=({"id": "call-a", "name": "read_file"},),
    )
    await history.append_tool_result(
        chat_id="chat-a",
        turn_id="shared-turn",
        event_id="tool:a",
        tool_call_id="call-a",
        tool_name="read_file",
        content="a-result",
    )
    await history.append_assistant(
        chat_id="chat-b",
        turn_id="shared-turn",
        event_id="assistant:b",
        content="b",
        tool_calls=({"id": "call-b", "name": "read_file"},),
    )

    assert [event.content for event in await history.snapshot(
        "shared-turn", chat_id="chat-a"
    )] == ["a", "a-result"]
    assert [event.content for event in await history.snapshot(
        "shared-turn", chat_id="chat-b"
    )] == ["b"]
    with pytest.raises(ProtocolInvariantError, match="ambiguous turn"):
        await history.snapshot("shared-turn")
    with pytest.raises(ProtocolInvariantError, match="no assistant call"):
        await history.append_tool_result(
            chat_id="chat-b",
            turn_id="shared-turn",
            event_id="tool:wrong-chat",
            tool_call_id="call-a",
            tool_name="read_file",
            content="must not cross chats",
        )
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_upgrades_legacy_turn_scoped_primary_key(tmp_path):
    path = tmp_path / "protocol.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE turn_protocol_history (
            turn_id TEXT NOT NULL,
            chat_id TEXT NOT NULL DEFAULT '',
            seq INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_call_id TEXT NOT NULL DEFAULT '',
            tool_name TEXT NOT NULL DEFAULT '',
            tool_calls TEXT NOT NULL DEFAULT '[]',
            reasoning_content TEXT NOT NULL DEFAULT '',
            timestamp REAL NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (turn_id, seq),
            UNIQUE (turn_id, event_id)
        )
        """)
    conn.execute(
        "INSERT INTO turn_protocol_history "
        "(turn_id, chat_id, seq, event_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("legacy-turn", "chat-a", 1, "assistant:1", "assistant", "old", 1.0),
    )
    conn.commit()
    conn.close()

    history = TurnProtocolHistory(str(path))
    await history.append_assistant(
        chat_id="chat-b",
        turn_id="legacy-turn",
        event_id="assistant:2",
        content="new",
    )

    assert [event["content"] for event in await history.history("chat-a")] == [
        "old"
    ]
    assert [event["content"] for event in await history.history("chat-b")] == [
        "new"
    ]
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_claims_legacy_turn_by_message_id(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await history.append_assistant(
        turn_id="message-1",
        event_id="assistant:0",
        content="legacy",
    )

    assert await history.claim_orphan_turns("chat-a", ["message-1"]) == 1
    assert len(await history.history("chat-a")) == 1
    await history.close()


def test_protocol_event_wire_projection_hides_storage_metadata():
    from core.engine.turn_protocol_history import ProtocolEvent

    events = (
        ProtocolEvent(
            turn_id="turn-1",
            seq=1,
            event_id="assistant:0",
            role="assistant",
            content="working",
            tool_calls=(
                {"id": "call-1", "type": "function", "function": {"name": "read_file"}},
            ),
            reasoning_content="internal",
            timestamp=123,
        ),
        ProtocolEvent(
            turn_id="turn-1",
            seq=2,
            event_id="tool:call-1",
            role="tool",
            content="result",
            tool_call_id="call-1",
            tool_name="read_file",
            timestamp=124,
        ),
    )

    assert TurnProtocolHistory.to_wire_messages(events) == [
        {
            "role": "assistant",
            "content": "working",
            "reasoning_content": "internal",
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "read_file"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
