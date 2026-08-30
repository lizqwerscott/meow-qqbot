import httpx
import pytest

from core.engine.mode_router import ModeRouteInput, ModeRouter
from core.engine.routing_audit import RoutingAuditStore
from core.message import InputMessage
from core.webui.app import create_app


@pytest.mark.asyncio
async def test_routing_audit_persists_content_free_records(tmp_path):
    store = RoutingAuditStore(str(tmp_path / "orchestration.sqlite"), max_rows=2)
    record = await store.append(
        chat_id="private-123456",
        message_id="message-123456",
        source="user",
        intent="private_conversation",
        mode="chat",
        reason_code="default_chat",
        reason="no deterministic work request was found",
        capability_profile="private_chat",
        policy_version="mode-router/v1",
        scheduler_revision=2,
        work_plan_hint=None,
        trace=("source:user", "selected_rule:default_chat"),
    )

    records = await store.list_records(mode="chat", chat_prefix="private-")

    assert records == [record]
    assert records[0].trace == ("source:user", "selected_rule:default_chat")
    assert await store.get(record.id) == record
    await store.close()


def test_mode_router_trace_contains_only_fixed_rule_evidence():
    message = InputMessage(
        id="message-1",
        sender_id="user-1",
        chat_id="private-1",
        content="帮我修 core/client.py",
        is_group=False,
    )

    decision = ModeRouter().route(ModeRouteInput(message))

    assert decision.trace == (
        "source:user",
        "intent:private_conversation",
        "selected_rule:explicit_work",
    )
    assert message.content not in " ".join(decision.trace)


@pytest.mark.asyncio
async def test_routing_webui_lists_and_shows_audit_detail(tmp_path):
    store = RoutingAuditStore(str(tmp_path / "orchestration.sqlite"))
    record = await store.append(
        chat_id="chat-1",
        message_id="message-1",
        source="user",
        intent="private_conversation",
        mode="agent",
        reason_code="explicit_work",
        reason="an explicit work action and target require Agent capabilities",
        capability_profile="agent_full",
        policy_version="mode-router/v1",
        scheduler_revision=1,
        work_plan_hint=None,
        trace=("source:user", "selected_rule:explicit_work"),
    )
    app = create_app(
        {"agent_engine": type("Engine", (), {"routing_audit_store": store})()}, {}
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listing = await client.get("/routing?mode=agent")
        detail = await client.get(f"/routing/{record.id}")

    assert listing.status_code == 200
    assert "explicit_work" in listing.text
    assert "agent_full" in listing.text
    assert detail.status_code == 200
    assert "selected_rule:explicit_work" in detail.text
    await store.close()
