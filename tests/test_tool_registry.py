import json
from unittest.mock import AsyncMock

import pytest

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_internal_control_rejects_mark_important_at_execution():
    handler = AsyncMock()
    registry = ToolRegistry()
    registry.register(
        ToolEntry(
            name="mark_important",
            description="remember",
            parameters={},
            handler=handler,
        )
    )
    context = ToolContext(
        chat_id="task:1",
        is_group=False,
        reply_to="message",
        sender_id="system",
        reply_callback=AsyncMock(),
        internal_control=True,
    )

    result = await registry.execute("mark_important", {}, context)

    assert json.loads(result.content)["error"] == "内部任务不允许写入长期记忆"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_turn_can_execute_mark_important():
    handler = AsyncMock(return_value=ToolResult(content="ok"))
    registry = ToolRegistry()
    registry.register(
        ToolEntry(
            name="mark_important",
            description="remember",
            parameters={},
            handler=handler,
        )
    )
    context = ToolContext(
        chat_id="chat",
        is_group=False,
        reply_to="message",
        sender_id="user",
        reply_callback=AsyncMock(),
    )

    await registry.execute("mark_important", {}, context)

    handler.assert_awaited_once_with({}, context)
