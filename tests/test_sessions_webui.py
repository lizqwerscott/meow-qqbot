import httpx
import pytest

from core.engine.turn_protocol_history import TurnProtocolHistory
from core.webui.app import create_app


class _ContextManager:
    async def get_all_disk_chat_ids_async(self):
        return ["group-1", "private-1"]

    async def get_archived_sessions_summary_async(self):
        return {}

    async def get_session_summary_async(self, chat_id):
        return {"message_count": 0, "last_activity": 0, "estimated_tokens": 0}

    async def get_chat_history_async(self, chat_id, max_messages=None):
        if chat_id == "legacy-1":
            return [
                {"role": "assistant", "content": "旧回复", "timestamp": 1},
                {
                    "role": "tool",
                    "tool_name": "read_file",
                    "tool_call_id": "call-legacy",
                    "content": "旧工具结果",
                    "timestamp": 2,
                },
            ]
        return []

    async def clear_chat_history_async(self, chat_id):
        return None

    async def get_archived_files_async(self, chat_id):
        return []

    def get_chat_type(self, chat_id):
        return chat_id.startswith("group")


@pytest.mark.asyncio
async def test_sessions_protocol_view_and_kind_filter(tmp_path):
    protocol = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await protocol.append_assistant(
        chat_id="heartbeat:events",
        turn_id="turn-1",
        event_id="assistant:0",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "function": {"name": "heartbeat_respond", "arguments": "{}"},
            }
        ],
    )
    await protocol.append_tool_result(
        chat_id="heartbeat:events",
        turn_id="turn-1",
        event_id="tool:call-1",
        tool_call_id="call-1",
        tool_name="heartbeat_respond",
        content="done",
    )
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "protocol_history": protocol,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        protocol_page = await client.get("/sessions/heartbeat:events/protocol")
        heartbeat_page = await client.get("/sessions?kind=heartbeat")
        group_page = await client.get("/sessions?kind=group")

    assert protocol_page.status_code == 200
    assert "heartbeat_respond" in protocol_page.text
    assert "done" in protocol_page.text
    assert heartbeat_page.status_code == 200
    assert "heartbeat:events" in heartbeat_page.text
    assert "group-1" not in heartbeat_page.text
    assert group_page.status_code == 200
    assert "group-1" in group_page.text
    assert "private-1" not in group_page.text
    await protocol.close()


@pytest.mark.asyncio
async def test_protocol_view_falls_back_to_legacy_context_history(tmp_path):
    protocol = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "protocol_history": protocol,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions/legacy-1/protocol")

    assert response.status_code == 200
    assert "旧回复" in response.text
    assert "旧工具结果" in response.text
    await protocol.close()


@pytest.mark.asyncio
async def test_session_clear_removes_protocol_history(tmp_path):
    protocol = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))
    await protocol.append_assistant(
        chat_id="clear-me",
        turn_id="turn-clear",
        event_id="assistant:0",
        content="will be deleted",
    )
    app = create_app(
        {
            "context_manager": _ContextManager(),
            "protocol_history": protocol,
        },
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/sessions/clear-me/clear")

    assert response.status_code == 303
    assert await protocol.history("clear-me") == []
    await protocol.close()
