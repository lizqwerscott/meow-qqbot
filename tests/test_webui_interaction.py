from __future__ import annotations

import asyncio

import httpx
import pytest

from core.approval.approval_manager import ApprovalManager
from core.engine.conversation_delivery import ChannelDeliveryRouter, DeliveryRequest
from core.engine.conversation_event_log import ConversationEventLog, TurnStatus
from core.session_identity import (
    ApprovalPrompt,
    ChannelRegistry,
    DeliveryTarget,
    QQAdapter,
)
from core.webui.app import create_app
from core.webui.interaction import WebUiConversationGateway


@pytest.mark.asyncio
async def test_turn_cursor_keeps_cutoff_and_loads_older_turns(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    for index in range(1, 4):
        await event_log.append_user_message(
            chat_id="chat",
            turn_id=f"turn-{index}",
            message_id=f"message-{index}",
            content=f"问题 {index}",
        )
        await event_log.append_accepted_delivery(
            chat_id="chat",
            turn_id=f"turn-{index}",
            delivery_id=f"delivery-{index}",
            content=f"答案 {index}",
        )
        await event_log.append_turn_terminal(
            chat_id="chat", turn_id=f"turn-{index}", status=TurnStatus.COMPLETED
        )

    first, has_more, next_before = await event_log.snapshot_turn_cursor("chat", limit=2)
    assert [turn.turn_id for turn in first.turns] == ["turn-3", "turn-2"]
    assert has_more is True
    assert next_before == 2

    await event_log.append_user_message(
        chat_id="chat",
        turn_id="turn-4",
        message_id="message-4",
        content="新消息",
    )
    older, older_has_more, _ = await event_log.snapshot_turn_cursor(
        "chat", limit=2, before_turn_sequence=next_before, cutoff_seq=first.cutoff_seq
    )
    assert [turn.turn_id for turn in older.turns] == ["turn-1"]
    assert older_has_more is False
    await event_log.close()


@pytest.mark.asyncio
async def test_webui_gateway_is_idempotent_and_never_uses_qq(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    calls = []
    channel_registry = ChannelRegistry()

    async def route(*, input_message, reply_callback, get_user_nickname):
        calls.append(input_message)
        await event_log.append_user_message(
            chat_id=input_message.chat_id,
            turn_id=input_message.id,
            message_id=input_message.id,
            content=input_message.content,
            sender_id=input_message.sender_id,
            session_kind="private",
        )
        await reply_callback(
            content="WebUI 回复",
            chat_id=input_message.chat_id,
            message_id=input_message.id,
            is_group=False,
        )
        await event_log.append_accepted_delivery(
            chat_id=input_message.chat_id,
            turn_id=input_message.id,
            delivery_id=f"webui:{input_message.id}",
            content="WebUI 回复",
            message_id=input_message.id,
        )
        await event_log.append_turn_terminal(
            chat_id=input_message.chat_id,
            turn_id=input_message.id,
            status=TurnStatus.COMPLETED,
        )

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        event_log=event_log,
        store_path=str(tmp_path / "webui.sqlite3"),
        channel_registry=channel_registry,
    )
    session = await gateway.create_session()
    receipt = await gateway.submit(
        session.session_id, content="你好", request_id="same-request"
    )
    duplicate = await gateway.submit(
        session.session_id, content="不应重复", request_id="same-request"
    )
    await asyncio.sleep(0.05)

    assert receipt.turn_id == duplicate.turn_id
    assert duplicate.duplicate is True
    assert len(calls) == 1
    assert calls[0].delivery_target.channel == "webui"
    assert calls[0].delivery_target.channel != "qq"
    assert ("webui", "admin") in channel_registry.list_accounts()
    events = list(gateway.hub._events[session.session_id])
    assert [event.event_type for event in events] == [
        "session.ready",
        "turn.accepted",
        "message.created",
        "turn.completed",
    ]
    await event_log.close()


@pytest.mark.asyncio
async def test_channel_delivery_router_unifies_message_and_approval_delivery():
    sent = []
    approvals = []
    manager = None

    async def send(target, content, **options):
        sent.append((target, content, options))
        return {"id": "qq-message-1"}

    async def send_approval(target, prompt, **_options):
        approvals.append((target, prompt))
        for future in manager._pending.values():
            future.set_result("allow-once")
        return True

    registry = ChannelRegistry()
    registry.register(
        QQAdapter(
            send_callback=send,
            approval_callback=send_approval,
        )
    )
    router = ChannelDeliveryRouter(registry)
    target = DeliveryTarget("qq", "default", "direct", "admin")

    result = await router.deliver(
        DeliveryRequest(
            target=target, content="普通通知", options={"delivery_id": "d1"}
        )
    )
    manager = ApprovalManager(
        api_client=None,
        admin_ids=["admin"],
        delivery_router=router,
    )
    decision = await manager.request_approval(
        "chat",
        "exec",
        "需要审批",
        details="echo test",
        timeout=1,
    )

    assert result == {"id": "qq-message-1"}
    assert sent[0][1] == "普通通知"
    assert decision == "allow-once"
    assert approvals[0][0] == target
    assert isinstance(approvals[0][1], ApprovalPrompt)


@pytest.mark.asyncio
async def test_webui_approval_uses_the_originating_channel_target():
    approvals = []
    manager = None

    async def send_approval(target, prompt, **_options):
        approvals.append((target, prompt))
        for future in manager._pending.values():
            future.set_result("allow-once")
        return True

    registry = ChannelRegistry()
    from core.webui.interaction.delivery import WebUiDeliveryAdapter
    from core.webui.interaction.events import EventHub

    registry.register(WebUiDeliveryAdapter(account_id="admin", hub=EventHub()))
    webui_adapter = registry.get("webui", "admin")
    webui_adapter.send_approval = send_approval
    router = ChannelDeliveryRouter(registry)
    manager = ApprovalManager(
        api_client=None,
        admin_ids=["admin"],
        delivery_router=router,
    )

    decision = await manager.request_approval(
        "agent:main:webui:admin:direct:s-session",
        "exec",
        "需要审批",
        details="echo test",
        timeout=1,
        delivery_target=DeliveryTarget("webui", "admin", "direct", "s-session"),
    )

    assert decision == "allow-once"
    assert approvals[0][0] == DeliveryTarget("webui", "admin", "direct", "s-session")


@pytest.mark.asyncio
async def test_webui_approval_fallback_uses_the_originating_channel_target():
    sent = []

    async def send(target, content, **_options):
        sent.append((target, content))
        return True

    registry = ChannelRegistry()
    from core.webui.interaction.delivery import WebUiDeliveryAdapter
    from core.webui.interaction.events import EventHub

    webui_adapter = WebUiDeliveryAdapter(account_id="admin", hub=EventHub())
    registry.register(webui_adapter)
    router = ChannelDeliveryRouter(registry)
    webui_adapter.send_message = send
    manager = ApprovalManager(
        api_client=None,
        admin_ids=["admin"],
        delivery_router=router,
    )

    manager._spawn_admin_notice(
        "审批兜底",
        delivery_id="fallback-1",
        delivery_target=DeliveryTarget("webui", "admin", "direct", "s-session"),
    )
    await asyncio.sleep(0.05)

    assert sent == [
        (DeliveryTarget("webui", "admin", "direct", "s-session"), "审批兜底")
    ]


@pytest.mark.asyncio
async def test_webui_agent_session_cannot_switch_back_to_chat(tmp_path):
    async def route(**_kwargs):
        return None

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session(mode="agent")

    with pytest.raises(ValueError, match="cannot switch back"):
        await gateway.submit(session.session_id, content="hello", mode="chat")


@pytest.mark.asyncio
async def test_webui_chat_session_persists_agent_upgrade(tmp_path):
    async def route(*, input_message, **_kwargs):
        input_message.session_mode = "agent"

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session(mode="chat")
    await gateway.submit(session.session_id, content="请执行任务")
    await asyncio.sleep(0.05)

    assert gateway.get_session(session.session_id).mode == "agent"


@pytest.mark.asyncio
async def test_webui_resource_references_are_authorized_to_session(tmp_path):
    class Store:
        async def authorize(self, chat_id, media_uri):
            if (
                chat_id != "agent:main:webui:admin:direct:s-resource"
                or media_uri.endswith("not-owned")
            ):
                return None
            return type(
                "Record",
                (),
                {
                    "media_id": "media-1",
                    "media_uri": media_uri,
                    "resource_type": "image",
                    "sha256": "hash-1",
                    "mime_type": "image/png",
                    "size": 12,
                    "filename": "image.png",
                },
            )()

    async def route(**_kwargs):
        return None

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        media_service=type("Media", (), {"store": Store()})(),
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    with pytest.raises(ValueError, match="authorized"):
        await gateway._resources(
            [
                {
                    "resource_type": "image",
                    "media_uri": "media://inbound/not-owned",
                }
            ],
            session_key="agent:main:webui:admin:direct:s-resource",
        )

    resources = await gateway._resources(
        [
            {
                "resource_type": "image",
                "media_uri": "media://inbound/media-1",
            }
        ],
        session_key="agent:main:webui:admin:direct:s-resource",
    )
    assert resources[0].media_id == "media-1"


@pytest.mark.asyncio
async def test_webui_delivery_starts_a_new_message_each_turn():
    from core.webui.interaction.delivery import WebUiDeliveryAdapter
    from core.webui.interaction.events import EventHub

    hub = EventHub()
    adapter = WebUiDeliveryAdapter(session_id="s1", hub=hub)
    await adapter.deliver(turn_id="t1", content="first")
    await adapter.deliver(turn_id="t2", content="second")

    assert [event.event_type for event in hub._events["s1"]] == [
        "message.created",
        "message.created",
    ]


def test_enabled_webui_requires_an_administrator_token():
    with pytest.raises(RuntimeError, match="administrator token"):
        create_app({}, {"enabled": True})


@pytest.mark.asyncio
async def test_webui_interaction_api_exposes_structured_turns(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    async def route(*, input_message, reply_callback, get_user_nickname):
        await event_log.append_user_message(
            chat_id=input_message.chat_id,
            turn_id=input_message.id,
            message_id=input_message.id,
            content=input_message.content,
        )
        await reply_callback(
            content="答案",
            chat_id=input_message.chat_id,
            message_id=input_message.id,
            is_group=False,
        )
        await event_log.append_accepted_delivery(
            chat_id=input_message.chat_id,
            turn_id=input_message.id,
            delivery_id=f"delivery:{input_message.id}",
            content="答案",
        )
        await event_log.append_turn_terminal(
            chat_id=input_message.chat_id,
            turn_id=input_message.id,
            status=TurnStatus.COMPLETED,
        )

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        event_log=event_log,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    app = create_app(
        {"webui_gateway": gateway, "conversation_event_log": event_log}, {}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/chat/sessions", json={"title": "测试"})
        session_id = created.json()["session"]["session_id"]
        submitted = await client.post(
            f"/api/chat/sessions/{session_id}/turns",
            json={"content": "问题", "request_id": "api-request"},
        )
        assert submitted.status_code == 200
        await asyncio.sleep(0.05)
        history = await client.get(f"/api/sessions/{session_id}/turns?limit=10")

    assert created.status_code == 200
    assert history.status_code == 200
    body = history.json()
    assert body["unstable_cursor"] is False
    assert body["items"][0]["blocks"]
    assert {
        block["text"] for block in body["items"][0]["blocks"] if "text" in block
    } == {
        "问题",
        "答案",
    }
    await event_log.close()


@pytest.mark.asyncio
async def test_webui_interaction_api_keeps_resource_only_messages_visible(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="resource-api-chat",
        turn_id="turn-1",
        message_id="message-1",
        content="",
        resources=(
            {
                "resource_type": "image",
                "media_id": "image-api-1",
                "media_uri": "media://inbound/image-api-1",
                "mime_type": "image/png",
                "filename": "猫.png",
            },
        ),
    )
    await event_log.append_turn_terminal(chat_id="resource-api-chat", turn_id="turn-1")
    app = create_app(
        {"conversation_event_log": event_log},
        {},
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/sessions/resource-api-chat/turns?limit=10")

    assert response.status_code == 200
    blocks = response.json()["items"][0]["blocks"]
    resource_blocks = [block for block in blocks if block["type"] == "image"]
    assert len(resource_blocks) == 1
    assert resource_blocks[0]["resource"]["preview_url"] == "/media/image-api-1/content"
    await event_log.close()


@pytest.mark.asyncio
async def test_webui_history_rejects_sessions_outside_webui_store(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        event_log=event_log,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    app = create_app(
        {"webui_gateway": gateway, "conversation_event_log": event_log}, {}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/sessions/not-a-webui-session/turns")

    assert response.status_code == 404
    await event_log.close()
