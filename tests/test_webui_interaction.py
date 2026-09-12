from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from core.approval.approval_manager import ApprovalManager
from core.engine.conversation_delivery import ChannelDeliveryRouter, DeliveryRequest
from core.engine.conversation_event_log import (
    ConversationEvent,
    ConversationEventLog,
    EventKind,
    TurnKind,
    TurnStatus,
)
from core.engine.tool_events import ToolLifecycleEvent
from core.media.store import MediaStore
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
    accepted = next(event for event in events if event.event_type == "turn.accepted")
    assert accepted.payload["mode"] == "agent"
    assert accepted.payload["model_group"] == "auto"
    await event_log.close()


@pytest.mark.asyncio
async def test_webui_can_browse_external_channel_sessions_read_only(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    await event_log.append_user_message(
        chat_id="qq-private-42",
        turn_id="qq-turn-1",
        message_id="qq-message-1",
        content="来自 QQ",
        session_kind="private",
    )
    await event_log.append_accepted_delivery(
        chat_id="qq-private-42",
        turn_id="qq-turn-1",
        delivery_id="qq-delivery-1",
        content="QQ 回复",
        session_kind="private",
    )
    await event_log.append_turn_terminal(
        chat_id="qq-private-42", turn_id="qq-turn-1", status=TurnStatus.COMPLETED
    )
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        event_log=event_log,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    page = await gateway.list_external_sessions()

    assert len(page.items) == 1
    assert page.items[0].session_id == "qq-private-42"
    assert page.items[0].read_only is True
    with pytest.raises(KeyError):
        gateway.get_session("qq-private-42")
    history_session = await gateway.resolve_history_session("qq-private-42")
    assert history_session.session_key == "qq-private-42"
    await event_log.close()


@pytest.mark.asyncio
async def test_webui_gateway_projects_tool_lifecycle_events_without_tool_details(
    tmp_path,
):
    async def route(*, input_message, tool_event_callback, **_kwargs):
        await tool_event_callback(
            ToolLifecycleEvent(
                event_type="tool.started",
                session_id=input_message.chat_id,
                turn_id=input_message.id,
                tool_call_id="call-1",
                tool_name="execute_command",
                status="started",
                metadata={"path": "/secret/file", "reason": "started"},
            )
        )
        await tool_event_callback(
            ToolLifecycleEvent(
                event_type="tool.finished",
                session_id=input_message.chat_id,
                turn_id=input_message.id,
                tool_call_id="call-1",
                tool_name="execute_command",
                status="completed",
                elapsed_ms=12,
            )
        )

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()
    await gateway.submit(session.session_id, content="run", request_id="tool-events")
    await asyncio.sleep(0.05)

    events = list(gateway.hub._events[session.session_id])
    lifecycle = [event for event in events if event.event_type.startswith("tool.")]
    assert [(event.event_type, event.payload["status"]) for event in lifecycle] == [
        ("tool.started", "started"),
        ("tool.finished", "completed"),
    ]
    assert lifecycle[0].payload["metadata"] == {"reason": "started"}
    assert all(event.turn_id == events[1].turn_id for event in lifecycle)


@pytest.mark.asyncio
async def test_webui_gateway_projects_tool_arguments_and_result_for_live_cards(
    tmp_path,
):
    async def route(*, input_message, tool_event_callback, **_kwargs):
        await tool_event_callback(
            ToolLifecycleEvent(
                event_type="tool.updated",
                session_id=input_message.chat_id,
                turn_id=input_message.id,
                tool_call_id="call-emoji",
                tool_name="send_emoji",
                status="running",
                metadata={"arguments": {"emoji_hash": "abc123", "reason": "开心"}},
            )
        )
        await tool_event_callback(
            ToolLifecycleEvent(
                event_type="tool.finished",
                session_id=input_message.chat_id,
                turn_id=input_message.id,
                tool_call_id="call-emoji",
                tool_name="send_emoji",
                status="completed",
                metadata={"result": '{"success": true}'},
            )
        )

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()
    await gateway.submit(session.session_id, content="send", request_id="tool-details")
    await asyncio.sleep(0.05)

    events = list(gateway.hub._events[session.session_id])
    updated = next(event for event in events if event.event_type == "tool.updated")
    finished = next(event for event in events if event.event_type == "tool.finished")
    assert updated.payload["arguments"] == {"emoji_hash": "abc123", "reason": "开心"}
    assert finished.payload["result"] == '{"success": true}'


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
async def test_channel_delivery_router_keeps_webui_and_qq_targets_isolated():
    qq_messages = []

    async def send_qq(target, content, **_options):
        qq_messages.append((target, content))
        return {"id": "qq-message"}

    from core.webui.interaction.delivery import WebUiDeliveryAdapter
    from core.webui.interaction.events import EventHub

    hub = EventHub()
    registry = ChannelRegistry()
    registry.register(QQAdapter(send_callback=send_qq))
    registry.register(WebUiDeliveryAdapter(account_id="admin", hub=hub))
    router = ChannelDeliveryRouter(registry)

    webui_target = DeliveryTarget("webui", "admin", "direct", "session-1")
    qq_target = DeliveryTarget("qq", "default", "direct", "qq-admin")
    await router.deliver(
        DeliveryRequest(
            target=webui_target,
            content="只给 WebUI",
            options={"turn_id": "turn-webui"},
        )
    )
    await router.deliver(DeliveryRequest(target=qq_target, content="只给 QQ"))
    await router.deliver_approval(
        webui_target,
        ApprovalPrompt(
            session_key="approval-webui",
            title="WebUI 审批",
            description="请确认",
        ),
    )

    assert qq_messages == [(qq_target, "只给 QQ")]
    assert [event.event_type for event in hub._events["session-1"]] == [
        "message.created",
        "approval.requested",
    ]
    assert hub._events["session-1"][0].payload["content"] == "只给 WebUI"


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
async def test_webui_pending_approval_endpoint_recovers_evicted_event(tmp_path):
    manager = ApprovalManager(
        api_client=None,
        admin_ids=["admin"],
        webui_admin_ids=["admin"],
    )
    session_key = "approval:webui:exec:recover"
    future = asyncio.get_running_loop().create_future()
    manager._pending[session_key] = future
    manager._pending_info[session_key] = {
        "tool_name": "exec",
        "reason": "命令需要审批",
        "details": "echo safe",
        "created_at": time.time(),
        "expires_at": time.time() + 60,
        "delivery_target": DeliveryTarget("webui", "admin", "direct", "s-1"),
        "cwd": "/tmp",
        "timeout_sec": 60,
    }
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        approval_pending_callback=manager.list_webui_pending,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = gateway._store.create(
        session_id="s-1",
        session_key="agent:main:webui:admin:direct:s-1",
        operator_id="admin",
        title="审批会话",
        mode="agent",
    )
    app = create_app({"webui_gateway": gateway}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        recovered = await client.get(
            f"/api/chat/sessions/{session.session_id}/approvals"
        )
        empty = await client.get("/api/chat/sessions/other/approvals")

    assert recovered.status_code == 200
    recovered_item = recovered.json()["items"][0]
    assert recovered_item["session_key"] == session_key
    assert recovered_item["title"] == "🔐 exec 审批请求"
    assert recovered_item["description"] == "原因: 命令需要审批"
    assert recovered_item["command_preview"] == "echo safe"
    assert recovered_item["cwd"] == "/tmp"
    assert recovered_item["severity"] == "info"
    assert recovered_item["timeout_sec"] == 60
    assert 0 < recovered_item["remaining_secs"] <= 60
    assert empty.status_code == 404
    manager.resolve_webui(session_key, "deny", "admin")


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
async def test_webui_model_group_is_a_validated_single_turn_override(tmp_path):
    captured = []

    async def route(*, input_message, **_kwargs):
        captured.append(input_message)

    class Registry:
        def list_group_options(self):
            return [{"id": "fast", "label": "fast", "model_count": 1}]

        def has_group(self, group_name):
            return group_name == "fast"

        def get_group(self, group_name):
            return ["provider/fast"] if group_name == "fast" else []

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        model_registry=Registry(),
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()

    await gateway.submit(session.session_id, content="use fast", model_group="fast")
    await asyncio.sleep(0.05)

    assert captured[0].model_chain == ["provider/fast"]
    assert gateway.model_options() == [
        {"id": "auto", "label": "自动路由", "model_count": 0},
        {"id": "fast", "label": "fast", "model_count": 1},
    ]
    with pytest.raises(ValueError, match="unknown model group"):
        await gateway.submit(
            session.session_id,
            content="bad model",
            request_id="bad-model",
            model_group="not-registered",
        )


@pytest.mark.asyncio
async def test_webui_model_options_endpoint_is_read_only(tmp_path):
    class Registry:
        def list_group_options(self):
            return [{"id": "primary", "label": "primary", "model_count": 2}]

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        model_registry=Registry(),
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    app = create_app({"webui_gateway": gateway}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/chat/options")

    assert response.status_code == 200
    assert response.json() == {
        "model_groups": [
            {"id": "auto", "label": "自动路由", "model_count": 0},
            {"id": "primary", "label": "primary", "model_count": 2},
        ],
        "controls": {
            "thinking_effort": {
                "enabled": False,
                "default": "provider",
                "reason": "当前模型服务按全局配置固定思考强度",
            },
            "quick_mode": {
                "enabled": False,
                "default": "off",
                "reason": "快速模式尚未提供按 Turn 的安全覆写",
            },
            "context_compaction": {
                "enabled": False,
                "default": "manual",
                "reason": "当前运行时未启用上下文压缩",
            },
        },
    }


@pytest.mark.asyncio
async def test_webui_reasoning_effort_is_a_validated_single_turn_override(tmp_path):
    captured = []

    async def route(*, input_message, **_kwargs):
        captured.append(input_message)

    class Registry:
        def list_reasoning_effort_options(self, model_chain=None):
            return ["low", "medium", "high"]

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        model_registry=Registry(),
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()

    await gateway.submit(
        session.session_id,
        content="think carefully",
        reasoning_effort="high",
    )
    await asyncio.sleep(0.05)

    assert captured[0].reasoning_effort == "high"
    await gateway.submit(
        session.session_id,
        content="inherit provider settings",
        request_id="provider-effort",
        reasoning_effort="provider",
    )
    await asyncio.sleep(0.05)
    assert captured[1].reasoning_effort is None
    assert gateway.chat_options()["controls"]["thinking_effort"] == {
        "enabled": True,
        "default": "provider",
        "options": ["low", "medium", "high"],
        "reason": "可在当前 Turn 覆写，未选择时跟随模型配置",
    }
    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        await gateway.submit(
            session.session_id,
            content="invalid",
            request_id="invalid-effort",
            reasoning_effort="extreme",
        )


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
async def test_webui_upload_resource_is_bound_to_its_session(tmp_path):
    store = MediaStore(str(tmp_path / "media"))
    media_service = type(
        "Media",
        (),
        {
            "enabled": True,
            "store": store,
            "max_image_bytes": 10,
            "max_file_bytes": 20,
        },
    )()

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        media_service=media_service,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()
    other_session = await gateway.create_session()

    resource = await gateway.upload_resource(
        session.session_id,
        filename="note.txt",
        mime_type="text/plain",
        data=b"hello",
    )
    assert resource["resource_type"] == "file"
    assert resource["media_uri"].startswith("media://inbound/")
    assert await store.authorize(session.session_key, resource["media_uri"])
    assert (
        await store.authorize(other_session.session_key, resource["media_uri"]) is None
    )

    with pytest.raises(ValueError, match="empty"):
        await gateway.upload_resource(
            session.session_id,
            filename="empty.txt",
            mime_type="text/plain",
            data=b"",
        )
    with pytest.raises(ValueError, match="SVG"):
        await gateway.upload_resource(
            session.session_id,
            filename="icon.svg",
            mime_type="image/svg+xml",
            data=b"<svg />",
        )
    with pytest.raises(ValueError, match="upload limit"):
        await gateway.upload_resource(
            session.session_id,
            filename="large.txt",
            mime_type="text/plain",
            data=b"x" * 21,
        )
    await store.close()


@pytest.mark.asyncio
async def test_webui_draft_upload_is_claimed_or_released(tmp_path):
    store = MediaStore(str(tmp_path / "media"))
    media_service = type("Media", (), {"enabled": True, "store": store})()

    async def route(**_kwargs):
        return None

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=route,
        get_user_nickname=lambda _user_id: "admin",
        media_service=media_service,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()
    submitted = await gateway.upload_resource(
        session.session_id,
        filename="submitted.txt",
        mime_type="text/plain",
        data=b"submitted",
    )
    receipt = await gateway.submit(
        session.session_id,
        content="",
        resources=[submitted],
        request_id="claim-upload",
    )
    claimed = await store.authorize(session.session_key, submitted["media_uri"])

    assert claimed is not None
    assert claimed.message_id == receipt.turn_id
    with pytest.raises(ValueError, match="no longer available"):
        await gateway.discard_upload(
            session.session_id,
            media_uri=submitted["media_uri"],
            upload_id=submitted["extra"]["webui_upload_id"],
        )

    discarded = await gateway.upload_resource(
        session.session_id,
        filename="discarded.txt",
        mime_type="text/plain",
        data=b"discarded",
    )
    await gateway.discard_upload(
        session.session_id,
        media_uri=discarded["media_uri"],
        upload_id=discarded["extra"]["webui_upload_id"],
    )

    assert await store.authorize(session.session_key, discarded["media_uri"]) is None
    actions = [item["action"] for item in gateway._store.list_audit(session.session_id)]
    assert {"attachment.uploaded", "attachment.discarded", "turn.submitted"} <= set(
        actions
    )
    await store.close()


@pytest.mark.asyncio
async def test_webui_stale_draft_uploads_are_cleaned_on_session_access(tmp_path):
    store = MediaStore(str(tmp_path / "media"))
    media_service = type("Media", (), {"enabled": True, "store": store})()
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        media_service=media_service,
        store_path=str(tmp_path / "webui.sqlite3"),
        draft_upload_ttl_seconds=0,
    )
    session = await gateway.create_session()
    resource = await gateway.upload_resource(
        session.session_id,
        filename="stale.txt",
        mime_type="text/plain",
        data=b"stale",
    )

    await gateway.list_sessions()

    assert await store.authorize(session.session_key, resource["media_uri"]) is None
    await store.close()


@pytest.mark.asyncio
async def test_webui_session_list_uses_stable_cursor_pages(tmp_path):
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    for index in range(5):
        await gateway.create_session(title=f"会话 {index}")

    first = await gateway.list_sessions(limit=2)
    second = await gateway.list_sessions(limit=2, cursor=first.next_cursor or "")
    third = await gateway.list_sessions(limit=2, cursor=second.next_cursor or "")

    assert len(first.items) == 2
    assert len(second.items) == 2
    assert len(third.items) == 1
    assert first.has_more is True
    assert second.has_more is True
    assert third.has_more is False
    all_sessions = await gateway.list_sessions(limit=100)
    assert {
        session.session_id
        for page in (first, second, third)
        for session in page.items
    } == {session.session_id for session in all_sessions.items}

    with pytest.raises(ValueError, match="invalid session cursor"):
        await gateway.list_sessions(cursor="invalid")
    with pytest.raises(ValueError, match="invalid session cursor"):
        await gateway.list_sessions(cursor="////")
    with pytest.raises(ValueError, match="invalid session cursor"):
        await gateway.list_sessions(
            cursor="eyJ1cGRhdGVkX2F0IjpOYU4sInNlc3Npb25faWQiOiJzLTEifQ"
        )


@pytest.mark.asyncio
async def test_webui_upload_endpoint_accepts_multipart_resource(tmp_path):
    store = MediaStore(str(tmp_path / "media"))
    media_service = type(
        "Media",
        (),
        {"enabled": True, "store": store, "max_file_bytes": 20},
    )()
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        media_service=media_service,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()
    app = create_app({"webui_gateway": gateway}, {"token": "secret"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        csrf_response = await client.get(
            "/api/csrf", headers={"Authorization": "Bearer secret"}
        )
        csrf_token = csrf_response.json()["token"]
        response = await client.post(
            f"/api/chat/sessions/{session.session_id}/uploads",
            files={"file": ("说明.txt", b"hello", "text/plain")},
            headers={
                "Authorization": "Bearer secret",
                "X-CSRF-Token": csrf_token,
            },
        )
        resource = response.json()["resource"]
        discarded = await client.request(
            "DELETE",
            f"/api/chat/sessions/{session.session_id}/uploads",
            json={
                "media_uri": resource["media_uri"],
                "upload_id": resource["extra"]["webui_upload_id"],
            },
            headers={
                "Authorization": "Bearer secret",
                "X-CSRF-Token": csrf_token,
            },
        )

    assert response.status_code == 200
    assert response.json()["resource"]["filename"] == "说明.txt"
    assert discarded.status_code == 200
    await store.close()


@pytest.mark.asyncio
async def test_webui_audit_endpoint_limits_details_to_safe_metadata(tmp_path):
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session(mode="chat")
    gateway._store.record_audit(
        session.session_id,
        "turn.submitted",
        mode="chat",
        resource_count=2,
        content="must not be exposed",
        turn_id="must not be exposed",
    )
    app = create_app({"webui_gateway": gateway}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/api/chat/sessions/{session.session_id}/audit?limit=10"
        )
        missing = await client.get("/api/chat/sessions/not-a-webui-session/audit")

    assert response.status_code == 200
    submitted = next(
        item for item in response.json()["items"] if item["action"] == "turn.submitted"
    )
    assert submitted["details"] == {"mode": "chat", "resource_count": 2}
    assert missing.status_code == 404


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


@pytest.mark.asyncio
async def test_webui_delivery_exposes_emoji_preview_in_live_resource_event():
    from core.webui.interaction.delivery import WebUiDeliveryAdapter
    from core.webui.interaction.events import EventHub

    hub = EventHub()
    adapter = WebUiDeliveryAdapter(session_id="s1", hub=hub)
    await adapter.deliver(
        turn_id="t1",
        content="",
        resources=[
            {
                "resource_type": "emoji",
                "resource_id": "emoji-hash",
                "filename": "emoji-hash.png",
                "extra": {"preview_url": "/static/emojis/emoji-hash.png"},
            }
        ],
    )

    resource = hub._events["s1"][0].payload["resources"][0]
    assert resource["resource_id"] == "emoji-hash"
    assert resource["preview_url"] == "/static/emojis/emoji-hash.png"
    assert resource["is_image"] is True


@pytest.mark.asyncio
async def test_webui_backpressure_is_a_retryable_turn_failure():
    from core.webui.interaction.delivery import WebUiDeliveryAdapter
    from core.webui.interaction.events import EventHub

    hub = EventHub()
    adapter = WebUiDeliveryAdapter(session_id="s1", hub=hub)
    receipt = await adapter.deliver(
        turn_id="t1",
        content="当前会话任务较多，请稍后重试。",
        delivery_event="delivery.backpressure",
    )

    assert receipt.status == "accepted"
    assert [event.event_type for event in hub._events["s1"]] == [
        "delivery.backpressure",
        "turn.failed",
    ]
    assert hub._events["s1"][-1].payload["retryable"] is True


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
        direct_session = await client.get(f"/api/chat/sessions/{session_id}")

    assert created.status_code == 200
    assert history.status_code == 200
    assert direct_session.status_code == 200
    assert direct_session.json()["session"]["session_id"] == session_id
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
async def test_webui_turn_projection_preserves_sender_and_reasoning_blocks(tmp_path):
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        event_log=event_log,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session(mode="agent")
    await event_log.append_user_message(
        chat_id=session.session_key,
        turn_id="turn-projection",
        message_id="message-projection",
        content="来自外部用户",
        sender_id="user-42",
    )
    await event_log.append_event(
        ConversationEvent(
            chat_id=session.session_key,
            turn_id="turn-projection",
            event_id="assistant:projection",
            role="assistant",
            kind=EventKind.ASSISTANT_TOOL_CALL,
            content="",
            reasoning_content="先检查现有状态",
            tool_calls=(
                {
                    "id": "call-projection",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{}"},
                },
            ),
        ),
        turn_kind=TurnKind.AI,
    )
    await event_log.append_event(
        ConversationEvent(
            chat_id=session.session_key,
            turn_id="turn-projection",
            event_id="tool:projection",
            role="tool",
            kind=EventKind.TOOL_RESULT,
            content="检查完成",
            tool_call_id="call-projection",
            tool_name="inspect",
        ),
        turn_kind=TurnKind.AI,
    )
    await event_log.append_turn_terminal(
        chat_id=session.session_key,
        turn_id="turn-projection",
        status=TurnStatus.COMPLETED,
    )
    app = create_app(
        {"webui_gateway": gateway, "conversation_event_log": event_log}, {}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/sessions/{session.session_id}/turns")

    assert response.status_code == 200
    blocks = response.json()["items"][0]["blocks"]
    assert {block["sender_id"] for block in blocks if block["role"] == "user"} == {
        "user-42"
    }
    assert any(
        block["type"] == "reasoning" and block["text"] == "先检查现有状态"
        for block in blocks
    )
    assert any(block["type"] == "tool" for block in blocks)
    await event_log.close()


@pytest.mark.asyncio
async def test_webui_compaction_is_session_bound_and_audited(tmp_path):
    calls = []

    async def compact(**kwargs):
        calls.append(kwargs)
        return {
            "changed": True,
            "tier": 3,
            "operation": "compact_replace",
            "before_tokens": 1000,
            "after_tokens": 400,
            "saved_tokens": 600,
            "reason": "tier3 summary",
            "scope_count": 1,
        }

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        context_compact_callback=compact,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()

    result = await gateway.compact_session(session.session_id)

    assert result["changed"] is True
    assert calls[0]["chat_id"] == session.session_key
    assert calls[0]["principal_id"] == "admin"
    audit = gateway.list_audit(session.session_id)
    compact_audit = next(
        item for item in audit if item["action"] == "context.compacted"
    )
    assert compact_audit == {
        "action": "context.compacted",
        "details": {
            "after_tokens": 400,
            "before_tokens": 1000,
            "changed": True,
            "operation": "compact_replace",
            "reason": "summary",
            "saved_tokens": 600,
            "tier": 3,
        },
        "created_at": compact_audit["created_at"],
    }
    with pytest.raises(KeyError):
        await gateway.compact_session("not-webui-session")


@pytest.mark.asyncio
async def test_webui_compaction_endpoint_requires_csrf_and_returns_safe_reason(
    tmp_path,
):
    async def compact(**_kwargs):
        return {
            "changed": False,
            "tier": 0,
            "operation": "none",
            "before_tokens": 0,
            "after_tokens": 0,
            "saved_tokens": 0,
            "reason": "database details should not leak",
            "scope_count": 0,
        }

    gateway = WebUiConversationGateway(
        operator_id="admin",
        route_callback=lambda **_kwargs: None,
        get_user_nickname=lambda _user_id: "admin",
        context_compact_callback=compact,
        store_path=str(tmp_path / "webui.sqlite3"),
    )
    session = await gateway.create_session()
    app = create_app({"webui_gateway": gateway}, {"token": "secret"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        csrf_response = await client.get(
            "/api/csrf", headers={"Authorization": "Bearer secret"}
        )
        response = await client.post(
            f"/api/chat/sessions/{session.session_id}/compact",
            json={},
            headers={
                "Authorization": "Bearer secret",
                "X-CSRF-Token": csrf_response.json()["token"],
            },
        )

    assert response.status_code == 200
    assert response.json()["result"]["reason"] == "not_changed"


@pytest.mark.asyncio
async def test_webui_csrf_endpoint_issues_token_for_browser_api():
    app = create_app({}, {"token": "secret"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/csrf", headers={"Authorization": "Bearer secret"}
        )

        assert response.status_code == 200
        assert response.json()["token"]
        assert client.cookies.get("webui_csrf")


@pytest.mark.asyncio
async def test_webui_event_tail_cursor_skips_old_events():
    from core.webui.interaction.events import EventHub

    hub = EventHub()
    await hub.publish("session", "old")
    stream = hub.stream("session", after_event_id="__tail__")
    next_event = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    await hub.publish("session", "new")

    event = await asyncio.wait_for(next_event, timeout=1)
    assert event is not None
    assert event.event_type == "new"
    await stream.aclose()


@pytest.mark.asyncio
async def test_webui_event_resync_includes_latest_available_event_id():
    from core.webui.interaction.events import EventHub

    hub = EventHub(max_events_per_session=1)
    await hub.publish("session", "old")
    latest = await hub.publish("session", "latest")
    stream = hub.stream("session", after_event_id="evicted")
    event = await asyncio.wait_for(anext(stream), timeout=1)

    assert event is not None
    assert event.event_type == "resync_required"
    assert event.payload["latest_event_id"] == latest.event_id
    await stream.aclose()


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
