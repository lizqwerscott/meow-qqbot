import pytest
from unittest.mock import AsyncMock

from core.engine.mode_router import (
    ActiveWorkPlanHint,
    ModeReasonCode,
    ModeRouteInput,
    ModeRouter,
    ModeRouteSource,
)
from core.engine.prompt_snapshot import PromptMode
from core.managers.session_manager import InboundIntent
from core.message import InputMessage


def message(
    content: str,
    *,
    is_group: bool = False,
    sender_id: str = "user-1",
    chat_id: str = "chat-1",
    is_at_mention: bool = False,
) -> InputMessage:
    return InputMessage(
        id="message-1",
        sender_id=sender_id,
        chat_id=chat_id,
        content=content,
        is_group=is_group,
        is_at_mention=is_at_mention,
    )


@pytest.mark.parametrize(
    "content",
    ["你好", "解释这个报错是什么意思", "你会写 Python 吗？", "服务器为什么会重启？"],
)
def test_general_conversation_and_nouns_without_actions_stay_chat(content):
    decision = ModeRouter().route(ModeRouteInput(message(content)))

    assert decision.mode is PromptMode.CHAT
    assert decision.reason_code is ModeReasonCode.DEFAULT_CHAT
    assert decision.capability_profile == "private_chat"


@pytest.mark.parametrize(
    "content",
    [
        "帮我修 core/foo.py 的报错",
        "运行刚才的测试",
        "登录服务器重启 api 服务",
        "写一个 Python 脚本解析日志",
        "创建一个定时任务",
    ],
)
def test_explicit_private_work_requests_use_agent(content):
    decision = ModeRouter().route(ModeRouteInput(message(content)))

    assert decision.mode is PromptMode.AGENT
    assert decision.reason_code is ModeReasonCode.EXPLICIT_WORK
    assert decision.capability_profile == "agent_full"


def test_discussion_only_constraint_overrides_work_verbs():
    decision = ModeRouter().route(
        ModeRouteInput(message("先给方案，不要修改 core/foo.py"))
    )

    assert decision.mode is PromptMode.CHAT
    assert decision.reason_code is ModeReasonCode.DISCUSSION_ONLY


def test_group_work_requires_an_explicit_wake():
    unwoken = ModeRouter().route(
        ModeRouteInput(message("帮我修 core/foo.py 的报错", is_group=True))
    )
    woken = ModeRouter().route(
        ModeRouteInput(
            message("猫猫帮我修 core/foo.py 的报错", is_group=True),
            intent=InboundIntent.DIRECT_TASK,
        )
    )

    assert unwoken.mode is PromptMode.CHAT
    assert unwoken.capability_profile == "group_reply"
    assert woken.mode is PromptMode.AGENT
    assert woken.capability_profile == "agent_full"


@pytest.mark.parametrize(
    ("source", "expected_reason", "expected_profile"),
    [
        (ModeRouteSource.AMBIENT, ModeReasonCode.AMBIENT_CHAT, "group_ambient"),
        (ModeRouteSource.PROACTIVE, ModeReasonCode.PROACTIVE_CHAT, "group_proactive"),
    ],
)
def test_ambient_and_proactive_always_stay_chat(
    source, expected_reason, expected_profile
):
    decision = ModeRouter().route(
        ModeRouteInput(
            message("先给方案，不要修改 core/foo.py", is_group=True, is_at_mention=True),
            source=source,
            intent=InboundIntent.GROUP_AMBIENT,
        )
    )

    assert decision.mode is PromptMode.CHAT
    assert decision.reason_code is expected_reason
    assert decision.capability_profile == expected_profile


def test_validated_work_plan_follow_up_has_the_highest_priority():
    inbound = message("只解释刚才的结果", sender_id="owner", chat_id="chat-plan")
    decision = ModeRouter().route(
        ModeRouteInput(
            inbound,
            scheduler_revision=7,
            active_work_plan=ActiveWorkPlanHint(
                work_plan_id="plan-123",
                chat_id="chat-plan",
                owner_id="owner",
                revision=7,
            ),
        )
    )

    assert decision.mode is PromptMode.AGENT
    assert decision.reason_code is ModeReasonCode.WORK_PLAN_FOLLOW_UP
    assert decision.work_plan_hint == "plan-123"


@pytest.mark.parametrize(
    "hint",
    [
        ActiveWorkPlanHint("plan-123", "other-chat", "owner", 7),
        ActiveWorkPlanHint("plan-123", "chat-plan", "other-owner", 7),
        ActiveWorkPlanHint("plan-123", "chat-plan", "owner", 6),
        ActiveWorkPlanHint("plan-123", "chat-plan", "owner", 7, is_eligible=False),
    ],
)
def test_invalid_work_plan_candidates_cannot_take_over_a_message(hint):
    decision = ModeRouter().route(
        ModeRouteInput(
            message("你好", sender_id="owner", chat_id="chat-plan"),
            scheduler_revision=7,
            active_work_plan=hint,
        )
    )

    assert decision.mode is PromptMode.CHAT
    assert decision.work_plan_hint is None


def test_mode_decision_projects_immutable_pending_inbound_metadata():
    decision = ModeRouter(policy_version="test-policy/v2").route(
        ModeRouteInput(message("帮我修 core/foo.py 的报错"), scheduler_revision=12)
    )

    metadata = decision.to_metadata()

    assert metadata.mode == "agent"
    assert metadata.capability_profile == "agent_full"
    assert metadata.reason_code == "explicit_work"
    assert metadata.policy_version == "test-policy/v2"


@pytest.mark.asyncio
async def test_dispatch_proactive_uses_group_ambient_intent_and_proactive_source():
    from core.engine.agent_engine import AgentEngine

    engine = object.__new__(AgentEngine)
    engine.dispatch = AsyncMock()
    reply_callback = AsyncMock()
    nickname = lambda _: "system"

    await AgentEngine.dispatch_proactive(
        engine, "group-1", "分享一个今天的冷知识", reply_callback, nickname
    )

    engine.dispatch.assert_awaited_once()
    kwargs = engine.dispatch.await_args.kwargs
    assert kwargs["_source"] is ModeRouteSource.PROACTIVE
    assert kwargs["_intent"] is InboundIntent.GROUP_AMBIENT
    proactive_message = engine.dispatch.await_args.args[0]
    assert proactive_message.is_group is True
    assert proactive_message.sender_id == "system"
