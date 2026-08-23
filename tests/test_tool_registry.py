import json
from unittest.mock import AsyncMock

import pytest

from core.engine.turn_capabilities import TurnCapabilities
from core.managers.session_manager import InboundIntent
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


@pytest.mark.asyncio
async def test_registry_rejects_unapproved_ambient_media_uri():
    handler = AsyncMock(return_value=ToolResult(content="should not run"))
    registry = ToolRegistry()
    registry.register(
        ToolEntry(name="image", description="inspect", parameters={}, handler=handler)
    )
    context = ToolContext(
        chat_id="group",
        is_group=True,
        reply_to="message",
        sender_id="user",
        reply_callback=AsyncMock(),
        capabilities=TurnCapabilities.for_intent(
            InboundIntent.GROUP_AMBIENT,
            chat_id="group",
            sender_id="user",
            reply_to="message",
            allowed_media_uris=frozenset({"media://inbound/allowed"}),
        ),
    )

    result = await registry.execute(
        "image", {"media_uri": "media://inbound/other", "question": "what?"}, context
    )


@pytest.mark.asyncio
async def test_registry_rejects_disallowed_delivery_kind_before_handler():
    handler = AsyncMock(return_value=ToolResult(content="should not run"))
    registry = ToolRegistry()
    registry.register(
        ToolEntry(
            name="send_message",
            description="send",
            parameters={},
            handler=handler,
            delivery_kind="message",
        )
    )
    context = ToolContext(
        chat_id="chat",
        is_group=False,
        reply_to="message",
        sender_id="user",
        reply_callback=AsyncMock(),
        capabilities=TurnCapabilities(
            intent=InboundIntent.DIRECT_TASK,
            allowed_delivery_kinds=frozenset({"emoji"}),
        ),
    )

    result = await registry.execute("send_message", {}, context)

    assert json.loads(result.content)["error"] == "投递类型不在当前 turn capability 内"
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_rejects_tool_after_turn_is_cancelled():
    handler = AsyncMock(return_value=ToolResult(content="should not run"))
    registry = ToolRegistry()
    registry.register(
        ToolEntry(name="write_file", description="write", parameters={}, handler=handler)
    )
    context = ToolContext(
        chat_id="chat",
        is_group=False,
        reply_to="message",
        sender_id="user",
        reply_callback=AsyncMock(),
        turn_active_callback=AsyncMock(return_value=False),
    )

    result = await registry.execute("write_file", {}, context)

    assert json.loads(result.content)["error"].startswith("TURN_NOT_ACTIVE")
    handler.assert_not_awaited()
