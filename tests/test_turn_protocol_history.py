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
