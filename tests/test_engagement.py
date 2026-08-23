import logging

import pytest

from core.engine.ambient_delivery import decide_ambient_delivery
from core.engine.engagement_config import (
    EngagementConfig,
    normalize_engagement_config,
)
from core.engine.group_engagement import (
    EngagementPhase,
    GroupEngagementManager,
)
from core.managers.session_manager import AdmissionOrigin, InboundIntent, PendingInbound
from core.message import InputMessage


def test_normalize_engagement_config_rejects_invalid_values(caplog):
    caplog.set_level(logging.WARNING)
    config = normalize_engagement_config(
        {
            "conversation_collect_idle_ms": -1,
            "conversation_collect_max_messages": 1.5,
            "group_ambient_mode": "invalid",
            "group_ambient_allow_single_question": "yes",
        }
    )

    assert config.conversation_collect_idle_ms == 700
    assert config.conversation_collect_max_messages == 8
    assert config.group_ambient_mode == "off"
    assert config.group_ambient_allow_single_question is True
    assert len(caplog.records) == 4


def test_normalize_active_chat_allowlist():
    config = normalize_engagement_config(
        {"group_ambient_active_chats": [" chat-1 ", "chat-1", "chat-2"]}
    )
    assert config.group_ambient_active_chats == ("chat-1", "chat-2")

    media_config = normalize_engagement_config(
        {
            "media_batch_max_resources": 3,
            "media_batch_max_chars": 4096,
            "media_batch_max_download_bytes": 2 * 1024 * 1024,
            "media_batch_capability_timeout_seconds": 45,
        }
    )
    assert media_config.media_batch_max_resources == 3
    assert media_config.media_batch_max_chars == 4096
    assert media_config.media_batch_max_download_bytes == 2 * 1024 * 1024
    assert media_config.media_batch_capability_timeout_seconds == 45.0

    tool = decide_ambient_delivery(
        "answer", delivery_mode="automatic", tool_delivered=True, reply_anchor_id="m1"
    )
    assert tool.should_deliver is False
    assert tool.reason == "already_delivered"

    silent = decide_ambient_delivery("NO_REPLY", delivery_mode="automatic")
    assert silent.should_deliver is False
    assert silent.reason == "silent_final_reply"

    tool_only = decide_ambient_delivery(
        "answer", delivery_mode="message_tool_only", reply_anchor_id="m2"
    )
    assert tool_only.should_deliver is False
    assert tool_only.reason == "automatic_delivery_disabled"
    assert tool_only.reply_anchor_id == "m2"

    automatic = decide_ambient_delivery(
        "  answer  ", delivery_mode="automatic", reply_anchor_id="m3"
    )
    assert automatic.should_deliver is True
    assert automatic.content == "answer"


def pending(message_id: str, content: str, now: float) -> PendingInbound:
    return PendingInbound(
        InputMessage(message_id, "user", "chat", content, True),
        content,
        InboundIntent.GROUP_AMBIENT,
        AdmissionOrigin.USER_MESSAGE,
        enqueued_at=now,
    )


def active_config(**overrides) -> EngagementConfig:
    values = {
        "group_ambient_mode": "active",
        "group_ambient_active_chats": ("chat",),
        "group_ambient_cooldown_seconds": 30.0,
        "group_ambient_quiet_cooldown_seconds": 10.0,
        "group_ambient_window_seconds": 300.0,
        "group_ambient_max_age_seconds": 600.0,
        "group_ambient_min_messages": 2,
    }
    values.update(overrides)
    return EngagementConfig(**values)


@pytest.mark.asyncio
async def test_active_engagement_reserves_and_applies_cooldown():
    now = [100.0]
    manager = GroupEngagementManager(active_config(), clock=lambda: now[0])
    batch = [pending("one", "hello", now[0]), pending("two", "?", now[0])]

    decision = await manager.evaluate("chat", batch=batch)
    assert decision.allowed is True
    assert decision.reply_anchor_id == "two"
    assert await manager.start(decision) is True
    assert manager.phase("chat") is EngagementPhase.THINKING
    assert await manager.complete(decision, delivered=True, silent=False) is True
    assert manager.phase("chat") is EngagementPhase.COOLDOWN

    blocked = await manager.evaluate("chat", batch=batch)
    assert blocked.reason == "cooldown"
    now[0] += 31
    allowed_again = await manager.evaluate("chat", batch=batch)
    assert allowed_again.allowed is True


@pytest.mark.asyncio
async def test_single_question_and_budget_rules():
    now = [100.0]
    manager = GroupEngagementManager(
        active_config(
            group_ambient_min_messages=2,
            group_ambient_max_turns_per_window=1,
            group_ambient_cooldown_seconds=0,
            group_ambient_quiet_cooldown_seconds=0,
        ),
        clock=lambda: now[0],
    )
    question = await manager.evaluate(
        "chat", batch=[pending("question", "吃什么？", now[0])]
    )
    assert question.allowed is True
    await manager.start(question)
    await manager.complete(question, delivered=False, silent=True)
    second = await manager.evaluate(
        "chat", batch=[pending("question-2", "去哪？", now[0])]
    )
    assert second.reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_shadow_and_generation_isolation():
    now = [100.0]
    shadow = GroupEngagementManager(
        EngagementConfig(group_ambient_mode="shadow"), clock=lambda: now[0]
    )
    decision = await shadow.evaluate(
        "chat", batch=[pending("one", "a", now[0]), pending("two", "b", now[0])]
    )
    assert decision.allowed is False
    assert decision.shadow is True

    shadow.observe(decision)
    assert shadow.snapshot_metrics() == {
        "reason:batch_threshold": 1,
        "shadow_candidates": 1,
    }

    manager = GroupEngagementManager(active_config(), clock=lambda: now[0])
    first = await manager.evaluate(
        "chat", batch=[pending("one", "a", now[0]), pending("two", "b", now[0])]
    )
    stale = await manager.evaluate(
        "chat", batch=[pending("three", "c", now[0]), pending("four", "d", now[0])]
    )
    assert stale.reason == "already_reserved"
    assert await manager.complete(first, delivered=True, silent=False) is True
