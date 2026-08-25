import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.ai.protocol import AssistantMessage, AssistantToolCall
from core.engine.agent_engine import (
    AgentEngine,
    BackgroundTaskResult,
    _TurnRequest,
    _TurnResult,
)
from core.engine.conversation_scheduler import ConversationScheduler
from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
)
from core.engine.engagement_config import EngagementConfig
from core.engine.group_engagement import GroupEngagementManager
from core.engine.prompt_builder import PromptBuildResult
from core.engine.turn_capabilities import TurnCapabilities
from core.engine.turn_state import TurnPhase
from core.managers.session_manager import (
    AdmissionOrigin,
    InboundIntent,
    PendingInbound,
    SessionTaskManager,
)
from core.message import InputMessage
from core.tools._types import ToolResult
from core.tools.tool_loop import ToolLoop


@pytest.mark.asyncio
async def test_admission_duplicate_history_does_not_produce_prompt_fragment():
    class DuplicateContextManager(FakeContextManager):
        async def add_user_message_async(self, *args, **kwargs):
            return False

    engine = make_engine(FakeToolLoop())
    engine.context_manager = DuplicateContextManager()
    pending = PendingInbound(
        InputMessage("duplicate", "user", "chat", "hello", False),
        "hello",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )

    admitted = await engine._admit_pending_message(
        pending,
        source="initial",
        get_user_nickname=lambda sender_id: sender_id,
    )

    assert admitted is None
    assert engine._admitted_ids == {}

    result = BackgroundTaskResult(result="done", error=None)

    value, error = result

    assert value == "done"
    assert error is None


@pytest.mark.asyncio
async def test_user_admission_writes_conversation_timeline_projection():
    engine = make_engine(FakeToolLoop())

    class Timeline:
        def __init__(self):
            self.calls = []

        async def append_user_message(self, **kwargs):
            self.calls.append(kwargs)

    timeline = Timeline()
    engine.timeline = timeline
    pending = PendingInbound(
        InputMessage("timeline-message", "user", "chat", "hello", False),
        "hello",
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )

    admitted = await engine._admit_pending_message(
        pending,
        source="initial",
        get_user_nickname=lambda sender_id: sender_id,
    )

    assert admitted is not None
    assert timeline.calls == [
        {
            "chat_id": "chat",
            "message_id": "timeline-message",
            "content": "hello",
            "sender_id": "user",
            "timestamp": pending.message.timestamp,
            "session_kind": "private",
        }
    ]

    class ArchiveTracker:
        def __init__(self, context_manager):
            self.context_manager = context_manager
            self.calls = []

        async def archive_if_stale(self, chat_id, is_group):
            self.calls.append(
                (chat_id, is_group, list(self.context_manager.user_messages))
            )

    engine = make_engine(FakeToolLoop())
    tracker = ArchiveTracker(engine.context_manager)
    engine._archive_manager = tracker
    pending = PendingInbound(
        InputMessage("late", "user", "chat", "hello", False),
        "hello",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )

    admitted = await engine._admit_pending_message(
        pending,
        source="initial",
        get_user_nickname=lambda sender_id: sender_id,
    )

    assert admitted is not None
    assert tracker.calls == [
        (
            "chat",
            False,
            [
                (
                    "chat",
                    "hello",
                    "late",
                    {
                        "sender_id": "user",
                        "name": "user",
                        "timestamp": pending.message.timestamp,
                    },
                )
            ],
        )
    ]


@pytest.mark.asyncio
async def test_admission_duplicate_check_prefers_timeline_projection():
    engine = make_engine(FakeToolLoop())
    engine.timeline = SimpleNamespace(
        history=AsyncMock(return_value=[{"role": "user", "message_id": "message-1"}])
    )

    class LegacyContextManager:
        async def get_chat_history_async(self, chat_id):
            raise AssertionError("legacy history should not be read")

    engine.context_manager = LegacyContextManager()

    assert await engine._is_message_admitted("chat", "message-1") is True


@pytest.mark.asyncio
async def test_history_migration_summary_scans_legacy_and_timeline_session_union(
    tmp_path,
):
    from core.engine.conversation_timeline import ConversationTimeline

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="shared",
        message_id="m1",
        content="hello",
        sender_id="user",
        timestamp=1,
    )
    await timeline.append_user_message(
        chat_id="timeline-only",
        message_id="m2",
        content="new",
        sender_id="user",
        timestamp=2,
    )

    class ContextManager:
        async def get_all_chat_ids_async(self):
            return ["shared", "legacy-only"]

        async def get_all_disk_chat_ids_async(self):
            return ["disk-only"]

        async def get_chat_history_async(self, chat_id):
            if chat_id == "shared":
                return [{"role": "user", "raw_content": "hello"}]
            if chat_id == "legacy-only":
                return [
                    {"role": "user", "raw_content": "old"},
                    {"role": "tool", "content": "result", "tool_call_id": "call"},
                ]
            return []

    engine = make_engine(FakeToolLoop())
    engine.context_manager = ContextManager()
    engine.timeline = timeline

    summary = await engine.get_history_migration_summary()

    assert summary == {
        "session_count": 4,
        "sessions_with_missing_legacy_visible": 1,
        "sessions_with_legacy_protocol": 1,
        "sessions_ready_for_legacy_read_removal": 3,
        "sessions_with_scan_errors": 0,
        "ready_for_legacy_read_removal": False,
    }
    assert "hello" not in str(summary)
    await timeline.close()


@pytest.mark.asyncio
async def test_internal_control_admission_skips_durable_side_effects(caplog):
    engine = make_engine(FakeToolLoop())
    calls = []

    async def record(effect, payload):
        calls.append((effect, payload))
        return True

    engine._run_hindsight_side_effect = lambda payload: record("hindsight", payload)
    engine._run_learner_side_effect = lambda payload: record("learner", payload)
    caplog.set_level(logging.INFO)
    pending = PendingInbound(
        InputMessage("task", "system", "task:1", "do work", False),
        "do work",
        "agent",
        AdmissionOrigin.INTERNAL_CONTROL,
    )

    await engine._admit_pending_message(
        pending,
        source="initial",
        get_user_nickname=lambda _: "system",
    )

    assert calls == []
    assert [message[1] for message in engine.context_manager.user_messages] == [
        "do work"
    ]
    assert any("跳过内部控制准入副作用" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_duplicate_admission_keeps_pending_outbox_effects():
    class DuplicateContextManager(FakeContextManager):
        async def add_user_message_async(self, *args, **kwargs):
            return False

        async def get_chat_history_async(self, *args, **kwargs):
            return [{"role": "user", "message_id": "duplicate"}]

    class Outbox:
        def __init__(self):
            self.prepared = False
            self.cancelled = False

        async def prepare(self, *args, **kwargs):
            self.prepared = True
            return False

        async def cancel(self, *args, **kwargs):
            self.cancelled = True

    class ProcessTracker:
        def __init__(self):
            self.calls = 0

        async def __call__(self):
            self.calls += 1

    engine = make_engine(FakeToolLoop())
    engine.context_manager = DuplicateContextManager()
    engine._admission_outbox = Outbox()
    process_tracker = ProcessTracker()
    engine._process_admission_outbox = process_tracker
    pending = PendingInbound(
        InputMessage("duplicate", "user", "chat", "hello", False),
        "hello",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )

    admitted = await engine._admit_pending_message(
        pending,
        source="initial",
        get_user_nickname=lambda sender_id: sender_id,
    )

    assert admitted is None
    assert engine._admission_outbox.prepared is False
    assert engine._admission_outbox.cancelled is False
    assert process_tracker.calls == 1


def test_legacy_tuple_result_shape_remains_supported():
    execution = ("done", None)
    assert getattr(execution, "result", None) is None
    assert not hasattr(execution, "tool_delivered")
    result, error = execution
    assert result == "done"
    assert error is None


class FakeContextManager:
    def __init__(self):
        self.rollbacks = []
        self.user_messages = []
        self.recorded_chat_types = []

    async def remove_last_user_message_if_async(self, chat_id, message_id):
        self.rollbacks.append((chat_id, message_id))

    async def add_user_message_async(self, chat_id, content, message_id, **kwargs):
        self.user_messages.append((chat_id, content, message_id, kwargs))

    async def record_chat_type(self, chat_id, is_group):
        self.recorded_chat_types.append((chat_id, is_group))

    def get_chat_type(self, chat_id):
        return False


class FakeToolLoop:
    def __init__(
        self,
        *,
        error=None,
        pause=None,
        replies=None,
        on_run=None,
        sends_tool_text=False,
        final_reply_silent=False,
    ):
        self.calls = []
        self.error = error
        self.pause = pause
        self.replies = ["reply"] if replies is None else replies
        self.on_run = on_run
        self.sends_tool_text = sends_tool_text
        self.final_reply_silent = final_reply_silent

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_run:
            await self.on_run(kwargs)
        if self.pause:
            await self.pause.wait()
        if self.error:
            raise self.error
        for reply in self.replies:
            await kwargs["reply_callback"](
                chat_id=kwargs["chat_id"],
                content=reply,
                message_id=kwargs["reply_to"],
                is_group=kwargs["is_group"],
            )
        if self.sends_tool_text:
            await kwargs["tool_reply_callback"](
                chat_id=kwargs["chat_id"],
                content="工具消息",
                message_id="",
                is_group=kwargs["is_group"],
            )
            await kwargs["delivery_state_callback"]()
        if "reply_state_callback" in kwargs and self.final_reply_silent:
            await kwargs["reply_state_callback"](True)
        return True, True


def make_engine(tool_loop, *, rule_router=None, model_registry=None):
    engine = AgentEngine.__new__(AgentEngine)
    engine._bot_id = "bot"
    engine.context_manager = FakeContextManager()
    engine.session_manager = SessionTaskManager()
    engine.tool_loop = tool_loop
    engine.rule_router = rule_router
    engine.model_registry = model_registry
    engine._session_binding = object()
    engine.cost_tracker = object()
    engine._system_events = None
    engine._admin_id = []
    engine._archive_manager = None
    engine.hindsight = None
    engine.learners = None
    engine._nm = None
    engine._admitted_ids = __import__("collections").OrderedDict()
    engine._admitted_side_effect_ids = set()
    engine._processed_ids = __import__("collections").OrderedDict()
    engine._max_processed_ids = 1000
    engine._dedup_lock = asyncio.Lock()
    engine._admission_in_progress = set()
    return engine


@pytest.mark.asyncio
async def test_start_repairs_enabled_model_context_before_workers():
    engine = make_engine(FakeToolLoop())
    events = []

    class Transcript:
        async def repair(self):
            events.append("repair")
            return {
                "abandoned_compaction_count": 1,
                "orphan_event_count": 0,
                "invalid_event_count": 0,
                "invalid_pair_count": 0,
                "fallback_count": 0,
            }

    engine.model_context_enabled = True
    engine.model_context = Transcript()
    engine._process_admission_outbox = AsyncMock(
        side_effect=lambda: events.append("outbox")
    )
    engine._ensure_delivery_recovery_worker = AsyncMock(
        side_effect=lambda: events.append("delivery")
    )
    engine._resume_preserved_consumers = AsyncMock(
        side_effect=lambda: events.append("consumers")
    )

    await engine.start()

    assert events == ["repair", "outbox", "delivery", "consumers"]


@pytest.mark.asyncio
async def test_start_disables_model_context_when_repair_fails():
    engine = make_engine(FakeToolLoop())

    class Transcript:
        async def repair(self):
            raise RuntimeError("broken transcript")

    engine.model_context_enabled = True
    engine.model_context_read_enabled = True
    engine.model_context_write_enabled = True
    engine.model_context_shadow = True
    engine.model_context = Transcript()
    engine._process_admission_outbox = AsyncMock()
    engine._ensure_delivery_recovery_worker = AsyncMock()
    engine._resume_preserved_consumers = AsyncMock()

    await engine.start()

    assert engine.model_context_enabled is False
    assert engine.model_context_read_enabled is False
    assert engine.model_context_write_enabled is False
    assert engine.model_context_shadow is False
    engine._process_admission_outbox.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_loop_isolates_send_message_callback_from_other_tools(monkeypatch):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(
                        content=None,
                        tool_calls=[
                            AssistantToolCall(
                                "send", "send_message", '{"text":"sent"}'
                            ),
                            AssistantToolCall("other", "other_tool", "{}"),
                        ],
                    ),
                    AssistantMessage(content="final"),
                ]
            )

        async def chat_completion_with_tools(self, **kwargs):
            return next(self.responses), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ai = FakeAI()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=ai,
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    loop = ToolLoop(ctx)
    callback_calls = []

    async def capture_callback(**kwargs):
        callback_calls.append(("normal", kwargs["content"]))

    async def tool_callback(**kwargs):
        callback_calls.append(("special", kwargs["content"]))
        return DeliveryReceipt(status="accepted", logical_delivery_id="tool:task:1")

    async def fake_execute(name, args, tool_ctx, permission_manager):
        if name == "send_message":
            assert tool_ctx.reply_callback is tool_callback
            await tool_ctx.reply_callback(
                chat_id=tool_ctx.chat_id,
                content="sent",
                message_id="",
                is_group=tool_ctx.is_group,
            )
            return ToolResult(
                content="ok",
                delivery_receipt=DeliveryReceipt(
                    status="accepted", logical_delivery_id="tool:task:1"
                ),
                delivery_kind="message",
            )
        assert tool_ctx.reply_callback is capture_callback
        await tool_ctx.reply_callback(
            chat_id=tool_ctx.chat_id,
            content="other output",
            message_id=tool_ctx.reply_to,
            is_group=tool_ctx.is_group,
        )
        return ToolResult(content="ok")

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    await loop.run(
        messages=[],
        tools=[],
        chat_id="task:1",
        is_group=True,
        reply_to="reply",
        reply_callback=capture_callback,
        tool_reply_callback=tool_callback,
        tool_reply_names={"send_message"},
    )

    assert callback_calls == [("special", "sent"), ("normal", "other output")]


@pytest.mark.asyncio
async def test_tool_loop_prepares_media_delivery_before_execution(monkeypatch):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(
                        content=None,
                        tool_calls=[
                            AssistantToolCall(
                                "emoji-call",
                                "send_emoji",
                                '{"emoji_hash":"abc123"}',
                            )
                        ],
                    ),
                    AssistantMessage(content="done"),
                ]
            )

        async def chat_completion_with_tools(self, **kwargs):
            return next(self.responses), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ai = FakeAI()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=ai,
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    loop = ToolLoop(ctx)
    controller = DeliveryController(DeliveryLedger(":memory:"))
    observed = []

    async def fake_execute(name, args, tool_ctx, permission_manager):
        record = await controller.ledger.get("tool:chat:turn-1:send_emoji:emoji-call")
        observed.append(record.status if record else None)
        return ToolResult(
            content="emoji sent",
            sent_emoji=True,
            delivery_receipt=DeliveryReceipt(
                status="accepted", logical_delivery_id="emoji:abc123"
            ),
            delivery_kind="emoji",
        )

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    await loop.run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=True,
        reply_to="reply",
        reply_callback=AsyncMock(),
        delivery_controller=controller,
        turn_id="turn-1",
    )

    assert observed == ["prepared"]
    record = await controller.ledger.get("tool:chat:turn-1:send_emoji:emoji-call")
    assert record is not None
    assert record.status == "sent"
    await controller.ledger.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "delivery_kind", "arguments"),
    [
        ("send_emoji", "emoji", '{"emoji_hash":"abc123"}'),
        ("synthesize_speech", "voice", '{"text":"晚安，主人～"}'),
    ],
)
async def test_tool_loop_delivers_final_text_after_sending_media(
    monkeypatch,
    tool_name,
    delivery_kind,
    arguments,
):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(
                        content=None,
                        tool_calls=[
                            AssistantToolCall(
                                "media-call",
                                tool_name,
                                arguments,
                            )
                        ],
                    ),
                    AssistantMessage(content="晚安，主人～"),
                ]
            )

        async def chat_completion_with_tools(self, **kwargs):
            return next(self.responses), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=FakeAI(),
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    loop = ToolLoop(ctx)
    controller = DeliveryController(DeliveryLedger(":memory:"))
    delivered = []

    async def reply_callback(**kwargs):
        delivered.append(kwargs["content"])

    async def fake_execute(name, args, tool_ctx, permission_manager):
        return ToolResult(
            content="media sent",
            delivery_receipt=DeliveryReceipt(
                status="accepted", logical_delivery_id="media:abc123"
            ),
            delivery_kind=delivery_kind,
        )

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    await loop.run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=True,
        reply_to="reply",
        reply_callback=reply_callback,
        delivery_controller=controller,
        turn_id="turn-1",
    )

    assert delivered == ["晚安，主人～"]
    await controller.ledger.close()


@pytest.mark.asyncio
async def test_tool_loop_suppresses_assistant_text_when_response_has_tool_calls(
    monkeypatch,
):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(
                        content="I will inspect it.",
                        tool_calls=[AssistantToolCall("call-1", "read_file", "{}")],
                    ),
                    AssistantMessage(content="The inspection is complete."),
                ]
            )

        async def chat_completion_with_tools(self, **kwargs):
            return next(self.responses), None

    class FakeContext:
        def __init__(self):
            self.legacy_assistant = []
            self.legacy_tools = []

        async def add_assistant_message_async(self, *args, **kwargs):
            self.legacy_assistant.append((args, kwargs))

        async def add_tool_result_async(self, *args, **kwargs):
            self.legacy_tools.append((args, kwargs))

    context = FakeContext()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=FakeAI(),
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=context,
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    delivered = []

    async def reply_callback(**kwargs):
        delivered.append(kwargs["content"])

    async def fake_execute(*args, **kwargs):
        return ToolResult(content="inspection data")

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    await ToolLoop(ctx).run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="message",
        reply_callback=reply_callback,
        capabilities=TurnCapabilities.for_intent(InboundIntent.PRIVATE_CONVERSATION),
    )

    assert context.legacy_assistant == []
    assert context.legacy_tools == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_call", "capabilities"),
    [
        (
            AssistantToolCall("malformed", "read_file", "not-json"),
            TurnCapabilities.for_intent(InboundIntent.PRIVATE_CONVERSATION),
        ),
        (
            AssistantToolCall(
                "unauthorized-media",
                "image",
                '{"media_uri":"media://inbound/other"}',
            ),
            TurnCapabilities.for_intent(
                InboundIntent.GROUP_AMBIENT,
                allowed_media_uris=frozenset({"media://inbound/allowed"}),
            ),
        ),
    ],
)
async def test_capability_tool_rejection_does_not_write_legacy_protocol(
    monkeypatch, tool_call, capabilities
):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(content=None, tool_calls=[tool_call]),
                    AssistantMessage(content="done"),
                ]
            )

        async def chat_completion_with_tools(self, **kwargs):
            return next(self.responses), None

    class FakeContext:
        def __init__(self):
            self.legacy_assistant = []
            self.legacy_tools = []

        async def add_assistant_message_async(self, *args, **kwargs):
            self.legacy_assistant.append((args, kwargs))

        async def add_tool_result_async(self, *args, **kwargs):
            self.legacy_tools.append((args, kwargs))

    context = FakeContext()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=FakeAI(),
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=context,
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )

    async def reply_callback(**kwargs):
        return None

    await ToolLoop(ctx).run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=capabilities.intent is InboundIntent.GROUP_AMBIENT,
        reply_to="message",
        reply_callback=reply_callback,
        capabilities=capabilities,
    )

    assert context.legacy_assistant == []
    assert context.legacy_tools == []


@pytest.mark.asyncio
async def test_tool_loop_stops_at_turn_gate_after_completed_tool(monkeypatch):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.calls = 0

        async def chat_completion_with_tools(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                pytest.fail("terminated turn must not start another provider request")
            return (
                AssistantMessage(
                    content="working",
                    tool_calls=[AssistantToolCall("call-1", "read_file", "{}")],
                ),
                None,
            )

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            raise AssertionError("terminated turn must not commit a tool result")

    ai = FakeAI()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=ai,
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    active = True
    executions = []
    delivered = []

    async def turn_is_active():
        return active

    async def reply_callback(**kwargs):
        delivered.append(kwargs["content"])

    async def fake_execute(name, args, tool_ctx, permission_manager):
        nonlocal active
        executions.append(name)
        active = False
        return ToolResult(content="done")

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    result = await ToolLoop(ctx).run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="message",
        reply_callback=reply_callback,
        turn_active_callback=turn_is_active,
    )

    assert result == (False, False)
    assert ai.calls == 1
    assert executions == ["read_file"]


@pytest.mark.asyncio
async def test_tool_loop_tracks_revision_after_approval_transition(monkeypatch):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.calls = 0

        async def chat_completion_with_tools(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return (
                    AssistantMessage(
                        tool_calls=[AssistantToolCall("call-1", "first_tool", "{}")]
                    ),
                    None,
                )
            if self.calls == 2:
                return (
                    AssistantMessage(
                        tool_calls=[AssistantToolCall("call-2", "second_tool", "{}")]
                    ),
                    None,
                )
            return AssistantMessage(content="done"), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=FakeAI(),
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    seen_revisions = []

    async def transition(**kwargs):
        seen_revisions.append(kwargs["expected_revision"])
        return SimpleNamespace(revision=kwargs["expected_revision"] + 1)

    async def fake_execute(name, args, tool_ctx, permission_manager):
        seen_revisions.append((name, tool_ctx.turn_revision))
        if name == "first_tool":
            awaiting = await tool_ctx.transition_turn(
                expected_revision=tool_ctx.turn_revision,
                phase=TurnPhase.AWAITING_APPROVAL,
                approval_plan_id="approval-1",
            )
            await tool_ctx.transition_turn(
                expected_revision=awaiting.revision,
                phase=TurnPhase.ACTIVE,
            )
        return ToolResult(content="ok")

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    await ToolLoop(ctx).run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="message",
        reply_callback=AsyncMock(),
        turn_id="turn",
        transition_turn=transition,
        turn_revision=7,
    )

    assert seen_revisions == [
        ("first_tool", 7),
        7,
        8,
        ("second_tool", 9),
        9,
    ]


@pytest.mark.asyncio
async def test_explicit_delivery_transport_is_blocked_after_turn_terminates(
    monkeypatch,
):
    class FakeAI:
        model = "test"

        async def chat_completion_with_tools(self, **kwargs):
            return (
                AssistantMessage(
                    tool_calls=[AssistantToolCall("send-1", "send_message", "{}")]
                ),
                None,
            )

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=FakeAI(),
            max_tool_rounds=1,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    active = True
    transport_calls = []
    controller = DeliveryController(DeliveryLedger(":memory:"))

    async def turn_is_active():
        return active

    async def raw_transport(**kwargs):
        transport_calls.append(kwargs)

    async def fake_execute(name, args, tool_ctx, permission_manager):
        nonlocal active
        active = False
        receipt = await tool_ctx.reply_callback(
            chat_id=tool_ctx.chat_id,
            content="must not send",
            message_id=tool_ctx.reply_to,
            is_group=tool_ctx.is_group,
        )
        return ToolResult(content="delivery attempted", delivery_receipt=receipt)

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", fake_execute)
    await ToolLoop(ctx).run(
        messages=[],
        tools=[],
        chat_id="chat",
        is_group=True,
        reply_to="message",
        reply_callback=AsyncMock(),
        tool_reply_callback=raw_transport,
        tool_reply_names={"send_message"},
        delivery_controller=controller,
        turn_id="turn",
        turn_active_callback=turn_is_active,
    )

    assert transport_calls == []
    assert await controller.ledger.status_counts() == {}
    await controller.ledger.close()
    engine = make_engine(FakeToolLoop())

    from core.message import InputMessage

    message = InputMessage(
        "id",
        "user",
        "chat",
        "",
        True,
        replied_author_id="bot",
    )

    assert engine._should_dispatch_to_ai(message) is True


def test_classifies_inbound_intent_once_at_ingress():
    engine = make_engine(FakeToolLoop())

    private = InputMessage("private", "user", "chat", "🙂", False)
    direct = InputMessage("direct", "user", "chat", "猫猫执行", True)
    ambient = InputMessage("ambient", "user", "chat", "闲聊", True)

    assert engine._classify_inbound_intent(private).value == "private_conversation"
    assert engine._classify_inbound_intent(direct).value == "direct_task"
    assert engine._classify_inbound_intent(ambient).value == "group_ambient"


def test_should_not_dispatch_to_ai_for_reply_to_other_user():
    engine = make_engine(FakeToolLoop())

    from core.message import InputMessage

    message = InputMessage(
        "id",
        "user",
        "chat",
        "",
        True,
        replied_author_id="other-user",
    )

    assert engine._should_dispatch_to_ai(message) is False

    engine = make_engine(FakeToolLoop())
    from core.message import InputMessage, ResourceMeta

    message = InputMessage(
        "id",
        "user",
        "chat",
        "",
        False,
        resources=[ResourceMeta(resource_type="image", media_uri="media://inbound/1")],
    )

    assert engine._should_dispatch_to_ai(message) is True


def test_should_dispatch_to_ai_for_waking_group_image_message():
    engine = make_engine(FakeToolLoop())
    from core.message import InputMessage, ResourceMeta

    message = InputMessage(
        "id",
        "user",
        "chat",
        "猫猫看看",
        True,
        resources=[ResourceMeta(resource_type="image", media_uri="media://inbound/1")],
    )

    assert engine._should_dispatch_to_ai(message) is True


def test_should_not_dispatch_to_ai_for_unwoken_group_image_message():
    engine = make_engine(FakeToolLoop())
    from core.message import InputMessage, ResourceMeta

    message = InputMessage(
        "id",
        "user",
        "chat",
        "",
        True,
        resources=[ResourceMeta(resource_type="image", media_uri="media://inbound/1")],
    )

    assert engine._should_dispatch_to_ai(message) is False


@pytest.mark.asyncio
async def test_ingress_keeps_media_references_without_preparing_them():
    engine = make_engine(FakeToolLoop())

    class MediaService:
        def __init__(self):
            self.calls = 0

        async def prepare_for_ai(self, message):
            self.calls += 1
            raise AssertionError("ingress must not prepare media")

    from core.message import ResourceMeta

    engine.media_service = MediaService()
    message = InputMessage(
        "media",
        "user",
        "chat",
        "请看附件",
        True,
        resources=[ResourceMeta(resource_type="image", media_uri="media://inbound/a")],
    )

    pending = await engine._prepare_pending_inbound(
        message, intent=InboundIntent.GROUP_AMBIENT
    )

    assert pending.resource_refs == ("media://inbound/a",)
    assert "[媒体引用: media://inbound/a]" in pending.prepared_content
    assert engine.media_service.calls == 0


@pytest.mark.asyncio
async def test_background_task_media_is_explicitly_authorized_and_lazy():
    engine = make_engine(FakeToolLoop())
    prompt_contexts = []

    class Store:
        async def authorize(self, chat_id, media_uri):
            assert chat_id == "source-chat"
            assert media_uri == "media://inbound/a"
            return SimpleNamespace(
                resource_type="image",
                media_uri=media_uri,
                media_id="a",
                sha256="digest",
                mime_type="image/png",
                size=12,
                filename="a.png",
            )

    class MediaService:
        def __init__(self):
            self.store = Store()
            self.prepare_calls = 0

        async def prepare_for_ai(self, message):
            self.prepare_calls += 1
            assert message.chat_id == "source-chat"
            assert [resource.media_uri for resource in message.resources] == [
                "media://inbound/a"
            ]
            from core.media.models import MediaTurnContext

            return MediaTurnContext(current_blocks=("understood",))

    engine.media_service = MediaService()

    async def build_task_messages(**kwargs):
        prompt_contexts.append(kwargs["media_context"])
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    await engine.execute_background_task(
        "task-chat",
        "inspect the attachment",
        "system",
        delivery_channel="source-chat",
        auto_media_understanding=True,
        media_refs=("media://inbound/a",),
        media_source_chat_id="source-chat",
    )

    assert engine.media_service.prepare_calls == 1
    assert prompt_contexts[0].as_text() == "understood"


@pytest.mark.asyncio
async def test_background_task_media_understanding_is_off_by_default():
    engine = make_engine(FakeToolLoop())

    class MediaService:
        def __init__(self):
            self.prepare_calls = 0

        async def prepare_for_ai(self, message):
            self.prepare_calls += 1
            raise AssertionError("disabled task must not prepare media")

    engine.media_service = MediaService()
    engine.prompt_builder = SimpleNamespace(
        build_task_messages=lambda **kwargs: pytest.fail("sync callback")
    )

    async def build_task_messages(**kwargs):
        assert kwargs["media_context"] is None
        return [], []

    engine.prompt_builder.build_task_messages = build_task_messages
    await engine.execute_background_task(
        "task-chat",
        "do work",
        "system",
        delivery_channel="source-chat",
        media_refs=("media://inbound/a",),
    )

    assert engine.media_service.prepare_calls == 0


@pytest.mark.asyncio
async def test_run_turn_builds_routes_forwards_and_collects_replies():
    prompt_calls = 0
    forwarded = []
    router = SimpleNamespace(classify=lambda text: "medium")
    registry = SimpleNamespace(get_chain=lambda tier: ["primary"])
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop, rule_router=router, model_registry=registry)

    async def build_prompt():
        nonlocal prompt_calls
        prompt_calls += 1
        return [{"role": "user", "content": "hello"}], [{"name": "tool"}]

    async def reply_callback(**kwargs):
        forwarded.append(kwargs)

    request = _TurnRequest(
        chat_id="chat",
        sender_id="user",
        is_group=True,
        reply_to="message",
        route_text="hello",
        prompt_factory=build_prompt,
        reply_callback=reply_callback,
        delivery_channel="delivery",
        reply_to_message_id="original",
    )

    result = await engine._run_turn(request)

    assert prompt_calls == 1
    assert result.replies == ("reply",)
    assert result.sent_emoji and result.text_committed
    assert forwarded[0]["content"] == "reply"
    assert tool_loop.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert tool_loop.calls[0]["tools"] == [{"name": "tool"}]
    assert tool_loop.calls[0]["model_chain"] == ["primary"]
    assert tool_loop.calls[0]["tier"] == "medium"
    assert tool_loop.calls[0]["delivery_channel"] == "delivery"
    assert tool_loop.calls[0]["reply_to_message_id"] == "original"
    assert tool_loop.calls[0]["binding_manager"] is engine._session_binding


@pytest.mark.asyncio
async def test_run_turn_uses_supplied_route_without_reclassifying():
    tool_loop = FakeToolLoop()
    router = SimpleNamespace(classify=lambda text: pytest.fail("must not classify"))
    engine = make_engine(tool_loop, rule_router=router, model_registry=object())

    async def build_prompt():
        return [], []

    async def reply_callback(**kwargs):
        pass

    await engine._run_turn(
        _TurnRequest(
            chat_id="chat",
            sender_id="user",
            is_group=False,
            reply_to="message",
            route_text="ignored",
            prompt_factory=build_prompt,
            reply_callback=reply_callback,
            model_chain=["chosen"],
            tier="simple",
        )
    )

    assert tool_loop.calls[0]["model_chain"] == ["chosen"]
    assert tool_loop.calls[0]["tier"] == "simple"


@pytest.mark.asyncio
async def test_run_turn_returns_prompt_scope_without_shared_builder_state():
    scope = SimpleNamespace(generation=7, kind="private_conversation")
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)

    async def build_prompt():
        return PromptBuildResult([], [], model_context_scope=scope)

    async def reply_callback(**kwargs):
        pass

    result = await engine._run_turn(
        _TurnRequest(
            chat_id="chat",
            sender_id="user",
            is_group=False,
            reply_to="message",
            route_text="hello",
            prompt_factory=build_prompt,
            reply_callback=reply_callback,
        )
    )

    assert result.model_context_scope is scope


@pytest.mark.asyncio
async def test_run_turn_accepts_tool_loop_usage_elapsed_ms():
    scope = SimpleNamespace(generation=7, kind="private_conversation")
    received = []
    service = object()

    async def report_usage(recorded_scope, usage, recorded_service, model_name):
        received.append((recorded_scope, usage, recorded_service, model_name))

    async def emit_usage(kwargs):
        await kwargs["model_context_usage_callback"](
            {"prompt_tokens": 1}, service, "test", 12.5
        )

    engine = make_engine(FakeToolLoop(on_run=emit_usage))

    async def build_prompt():
        return PromptBuildResult([], [], model_context_scope=scope)

    async def reply_callback(**kwargs):
        pass

    await engine._run_turn(
        _TurnRequest(
            chat_id="chat",
            sender_id="user",
            is_group=False,
            reply_to="message",
            route_text="hello",
            prompt_factory=build_prompt,
            reply_callback=reply_callback,
            model_context_usage_callback=report_usage,
        )
    )

    assert received == [(scope, {"prompt_tokens": 1}, service, "test")]


@pytest.mark.asyncio
async def test_model_context_scope_fails_closed_for_direct_tasks_without_lifecycle_binding():
    engine = make_engine(FakeToolLoop())
    engine.model_context_enabled = True
    pending = PendingInbound(
        InputMessage(
            "task-message",
            "user",
            "chat",
            "do work",
            True,
            task_correlation_id="task-1",
        ),
        "do work",
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )

    assert await engine._model_context_scope(pending, pending.message) is None


def _direct_task_pending(
    *,
    message_id="task-message",
    sender_id="user",
    chat_id="chat",
    task_correlation_id="task-1",
):
    message = InputMessage(
        message_id,
        sender_id,
        chat_id,
        "do work",
        True,
        task_correlation_id=task_correlation_id,
    )
    return PendingInbound(
        message,
        message.content,
        InboundIntent.DIRECT_TASK,
        AdmissionOrigin.USER_MESSAGE,
    )


async def _active_direct_task_engine(pending):
    engine = make_engine(FakeToolLoop())
    engine.model_context_enabled = True
    roles = {pending.message.sender_id: "trusted"}
    engine.scheduler = ConversationScheduler(
        engine.session_manager,
        user_role=roles.__getitem__,
        role_at_least=lambda candidate, required: candidate == required,
    )
    result = await engine.scheduler.enqueue(pending.message.chat_id, pending)
    work = await engine.scheduler.next_work(
        pending.message.chat_id, owner_token=result.consumer_token
    )
    assert work is not None
    await engine.scheduler.start_turn(
        work, turn_id=pending.message.id, principal_id=pending.message.sender_id
    )
    return engine


@pytest.mark.asyncio
async def test_model_context_scope_allows_matching_active_direct_task():
    pending = _direct_task_pending()
    engine = await _active_direct_task_engine(pending)

    scope = await engine._model_context_scope(
        pending,
        pending.message,
        capabilities=engine._turn_capabilities(
            pending.intent,
            chat_id=pending.message.chat_id,
            sender_id=pending.message.sender_id,
            reply_to=pending.message.id,
        ),
    )

    assert scope is not None
    assert scope.task_correlation_id == "task-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"task_correlation_id": "task-2"},
        {"chat_id": "other-chat"},
        {"sender_id": "other-user"},
    ],
)
async def test_model_context_scope_rejects_unrelated_direct_task_identity(overrides):
    active = _direct_task_pending()
    engine = await _active_direct_task_engine(active)
    candidate = _direct_task_pending(message_id=active.message.id, **overrides)
    capabilities = engine._turn_capabilities(
        candidate.intent,
        chat_id=candidate.message.chat_id,
        sender_id=candidate.message.sender_id,
        reply_to=candidate.message.id,
    )

    assert (
        await engine._model_context_scope(
            candidate, candidate.message, capabilities=capabilities
        )
        is None
    )


@pytest.mark.asyncio
async def test_model_context_scope_rejects_completed_direct_task():
    pending = _direct_task_pending()
    engine = await _active_direct_task_engine(pending)
    active = await engine.scheduler.get_turn(pending.message.id)
    finalizing = await engine.scheduler.transition_turn(
        pending.message.id,
        expected_revision=active.revision,
        phase=TurnPhase.FINALIZING,
    )
    completed = await engine.scheduler.transition_turn(
        pending.message.id,
        expected_revision=finalizing.revision,
        phase=TurnPhase.COMPLETED,
    )
    await engine.scheduler.drop_turn(completed.turn_id)

    capabilities = engine._turn_capabilities(
        pending.intent,
        chat_id=pending.message.chat_id,
        sender_id=pending.message.sender_id,
        reply_to=pending.message.id,
    )
    assert (
        await engine._model_context_scope(
            pending, pending.message, capabilities=capabilities
        )
        is None
    )


@pytest.mark.asyncio
async def test_model_context_scope_rejects_mixed_task_correlation_batch():
    pending = _direct_task_pending()
    engine = await _active_direct_task_engine(pending)
    other = _direct_task_pending(
        message_id="other-task-message", task_correlation_id="task-2"
    )
    capabilities = engine._turn_capabilities(
        pending.intent,
        chat_id=pending.message.chat_id,
        sender_id=pending.message.sender_id,
        reply_to=other.message.id,
    )

    assert (
        await engine._model_context_scope(
            pending,
            other.message,
            capabilities=capabilities,
            batch=(other,),
        )
        is None
    )


@pytest.mark.asyncio
async def test_model_context_scope_rejects_mixed_private_principal_batch():
    engine = make_engine(FakeToolLoop())
    engine.model_context_enabled = True
    pending = PendingInbound(
        InputMessage("private-1", "user-1", "chat", "hello", False),
        "hello",
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )
    other = PendingInbound(
        InputMessage("private-2", "user-2", "chat", "secret", False),
        "secret",
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )

    assert (
        await engine._model_context_scope(pending, other.message, batch=(other,))
        is None
    )


@pytest.mark.asyncio
async def test_run_turn_rolls_back_once_without_hiding_original_error():
    engine = make_engine(FakeToolLoop(error=RuntimeError("failed")))

    async def build_prompt():
        return [], []

    async def reply_callback(**kwargs):
        pass

    with pytest.raises(RuntimeError, match="failed"):
        await engine._run_turn(
            _TurnRequest(
                chat_id="chat",
                sender_id="user",
                is_group=False,
                reply_to="message",
                route_text="hello",
                prompt_factory=build_prompt,
                reply_callback=reply_callback,
                rollback_message_id="message",
            )
        )

    assert engine.context_manager.rollbacks == [("chat", "message")]


@pytest.mark.asyncio
async def test_run_turn_preserves_original_error_when_rollback_fails():
    engine = make_engine(FakeToolLoop(error=RuntimeError("failed")))

    async def failing_rollback(*args):
        raise RuntimeError("rollback failed")

    engine.context_manager.remove_last_user_message_if_async = failing_rollback

    async def build_prompt():
        return [], []

    async def reply_callback(**kwargs):
        pass

    with pytest.raises(RuntimeError, match="failed"):
        await engine._run_turn(
            _TurnRequest(
                chat_id="chat",
                sender_id="user",
                is_group=False,
                reply_to="message",
                route_text="hello",
                prompt_factory=build_prompt,
                reply_callback=reply_callback,
                rollback_message_id="message",
            )
        )


@pytest.mark.asyncio
async def test_run_turn_propagates_timeout_and_cancellation_without_rollback():
    pause = asyncio.Event()
    engine = make_engine(FakeToolLoop(pause=pause))

    async def build_prompt():
        return [], []

    async def reply_callback(**kwargs):
        pass

    request = _TurnRequest(
        chat_id="chat",
        sender_id="user",
        is_group=False,
        reply_to="message",
        route_text="hello",
        prompt_factory=build_prompt,
        reply_callback=reply_callback,
        rollback_message_id="message",
        timeout=0.01,
    )
    with pytest.raises(asyncio.TimeoutError):
        await engine._run_turn(request)
    assert engine.context_manager.rollbacks == [("chat", "message")]

    engine = make_engine(FakeToolLoop(error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await engine._run_turn(request)
    assert engine.context_manager.rollbacks == []


@pytest.mark.asyncio
async def test_run_turn_serializes_same_session_but_not_other_sessions():
    pause = asyncio.Event()
    tool_loop = FakeToolLoop(pause=pause)
    engine = make_engine(tool_loop)

    async def build_prompt():
        return [], []

    async def reply_callback(**kwargs):
        pass

    def request(chat_id):
        return _TurnRequest(
            chat_id=chat_id,
            sender_id="user",
            is_group=False,
            reply_to="message",
            route_text="hello",
            prompt_factory=build_prompt,
            reply_callback=reply_callback,
        )

    first = asyncio.create_task(engine._run_turn(request("same")))
    await asyncio.sleep(0)
    second = asyncio.create_task(engine._run_turn(request("same")))
    other = asyncio.create_task(engine._run_turn(request("other")))
    await asyncio.sleep(0)

    assert len(tool_loop.calls) == 2
    pause.set()
    await asyncio.gather(first, second, other)
    assert len(tool_loop.calls) == 3


@pytest.mark.asyncio
async def test_process_message_preserves_prompt_stream_and_system_event_adapters():
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)
    prompt_calls = []
    system_events = []
    engine._system_events = SimpleNamespace(
        peek_non_heartbeat=lambda chat_id: [SimpleNamespace(text="event")],
        drain_non_heartbeat=lambda chat_id, expected_events=None: system_events.append(
            (chat_id, expected_events)
        ),
    )

    async def build(**kwargs):
        prompt_calls.append(kwargs)
        return [], []

    engine.prompt_builder = SimpleNamespace(build=build)
    delivered = []

    async def reply_callback(**kwargs):
        delivered.append(kwargs)

    from core.message import InputMessage

    message = InputMessage("id", "user", "chat", "hello", True)
    await engine._process_message(
        PendingInbound(message, "hello", "agent", AdmissionOrigin.USER_MESSAGE),
        reply_callback,
        lambda sender_id: "name",
    )

    assert engine.context_manager.recorded_chat_types == [("chat", True)]
    assert prompt_calls[0]["user_nickname"] == "name"
    assert delivered[0]["message_id"] == "id"
    assert tool_loop.calls[0]["get_user_nickname"]("user") == "name"
    assert tool_loop.calls[0]["capabilities"].allows_context(
        chat_id="chat", sender_id="user", reply_to="id"
    )
    assert tool_loop.calls[0]["stream_callback"] is not None
    assert system_events[0][0] == "chat"


@pytest.mark.asyncio
async def test_process_message_requires_explicit_admission_origin():
    engine = make_engine(FakeToolLoop())

    with pytest.raises(TypeError, match="requires PendingInbound"):
        await engine._process_message(
            InputMessage("id", "system", "task:1", "control", False),
            lambda **kwargs: _none(),
            lambda _: "system",
        )


@pytest.mark.asyncio
async def test_process_message_snapshots_events_after_admission():
    from core.engine.system_events import SystemEventQueue

    engine = make_engine(FakeToolLoop())
    events = SystemEventQueue()
    engine._system_events = events
    admission_started = asyncio.Event()
    release_admission = asyncio.Event()

    async def add_user_message(*args, **kwargs):
        admission_started.set()
        await release_admission.wait()

    engine.context_manager.add_user_message_async = add_user_message

    async def build(**kwargs):
        assert [event.text for event in events.peek("chat")] == ["arrived"]
        return [], []

    engine.prompt_builder = SimpleNamespace(build=build)
    task = asyncio.create_task(
        engine._process_message(
            PendingInbound(
                InputMessage("id", "user", "chat", "hello", False),
                "hello",
                "agent",
                AdmissionOrigin.USER_MESSAGE,
            ),
            lambda **kwargs: _none(),
            lambda _: "name",
        )
    )
    await admission_started.wait()
    events.enqueue("chat", "arrived", "event")
    release_admission.set()
    await task

    assert events.peek("chat") == []


@pytest.mark.asyncio
async def test_process_message_rolls_back_once_when_prompt_building_fails():
    engine = make_engine(FakeToolLoop())

    async def build(**kwargs):
        raise RuntimeError("prompt failed")

    engine.prompt_builder = SimpleNamespace(build=build)

    from core.message import InputMessage

    message = InputMessage("id", "user", "chat", "hello", True)
    with pytest.raises(RuntimeError, match="prompt failed"):
        await engine._process_message(
            PendingInbound(message, "hello", "agent", AdmissionOrigin.USER_MESSAGE),
            lambda **kwargs: None,
            lambda _: "name",
        )

    assert engine.context_manager.rollbacks == [("chat", "id")]


@pytest.mark.asyncio
async def test_consumer_sends_friendly_error_and_requeues_message(tmp_path):
    engine = make_engine(FakeToolLoop())
    engine.delivery_controller = DeliveryController(
        DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    )
    delivered = []
    delivered_event = asyncio.Event()

    async def fail_process(*args):
        raise RuntimeError("failed")

    async def reply_callback(**kwargs):
        delivered.append(kwargs)
        delivered_event.set()

    engine._process_message = fail_process
    message = InputMessage("message", "user", "chat", "hello", True)
    pending = PendingInbound(
        message,
        "hello",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await engine.session_manager.enqueue_and_claim_consumer("chat", pending)
    consumer = asyncio.create_task(
        engine._consumer(
            "chat", reply_callback, lambda sender_id: "name", enqueued.consumer_token
        )
    )

    await delivered_event.wait()
    await consumer

    assert delivered == [
        {
            "chat_id": "chat",
            "content": "抱歉，处理您的消息时出现了问题，请稍后再试。",
            "message_id": "message",
            "is_group": True,
        }
    ]
    assert engine.session_manager.has_active_consumer("chat") is False
    assert engine.session_manager.get_queue_sizes() == {}
    assert engine.session_manager.get_message_state("chat", "message") == "failed"


@pytest.mark.asyncio
async def test_consumer_keeps_pending_message_for_tool_loop_steering():
    engine = make_engine(FakeToolLoop())
    drained = []
    drained_event = asyncio.Event()

    async def drain_steering_message(kwargs):
        lease = await engine.session_manager.claim_pending_for_steer(kwargs["chat_id"])
        steering = lease.items[0]
        drained.append(steering.message.id)
        await engine.session_manager.commit(lease, steering)
        drained_event.set()

    engine.tool_loop.on_run = drain_steering_message
    engine.prompt_builder = SimpleNamespace(build=lambda **kwargs: _empty_prompt())
    first = InputMessage("first", "user", "chat", "first", False)
    steering = InputMessage("steering", "user", "chat", "steering", False)
    first_enqueued = await engine.session_manager.enqueue_and_claim_consumer(
        "chat",
        PendingInbound(first, "first", "agent", AdmissionOrigin.USER_MESSAGE),
    )
    await engine.session_manager.enqueue_and_claim_consumer(
        "chat",
        PendingInbound(steering, "steering", "agent", AdmissionOrigin.USER_MESSAGE),
    )

    consumer = asyncio.create_task(
        engine._consumer(
            "chat",
            lambda **kwargs: _none(),
            lambda sender_id: "name",
            first_enqueued.consumer_token,
        )
    )
    await drained_event.wait()
    await consumer

    assert drained == ["steering"]
    assert len(engine.tool_loop.calls) == 1


@pytest.mark.asyncio
async def test_background_task_serializes_context_build_and_reply_result():
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)
    events = []

    async def add_user_message(*args, **kwargs):
        events.append("context")

    async def build_task_messages(**kwargs):
        events.append("prompt")
        assert kwargs["tools_allow"] == ["tool"]
        return [], []

    engine.context_manager.add_user_message_async = add_user_message
    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)

    result, error = await engine.execute_background_task(
        "task-chat",
        "do work",
        "system",
        delivery_channel="delivery",
        reply_to_message_id="original",
        tools_allow=["tool"],
    )

    assert (result, error) == ("reply", None)
    assert events == ["context", "prompt"]
    assert tool_loop.calls[0]["delivery_channel"] == "delivery"
    assert tool_loop.calls[0]["reply_to_message_id"] == "original"
    assert tool_loop.calls[0]["internal_control"] is True
    assert (
        tool_loop.calls[0]["capabilities"].intent is InboundIntent.PRIVATE_CONVERSATION
    )
    assert tool_loop.calls[0]["capabilities"].chat_id == "task-chat"
    assert tool_loop.calls[0]["turn_id"].startswith("bg_task-chat_")


@pytest.mark.asyncio
async def test_background_task_logs_metadata_without_prompt_content(caplog):
    engine = make_engine(FakeToolLoop())
    prompt = "do not write this control prompt to logs"
    caplog.set_level(logging.INFO)

    await engine.execute_background_task("task-chat", prompt, "system")

    assert prompt not in caplog.text
    assert "开始后台任务: chat_id=task-chat.." in caplog.text
    assert f"prompt_chars={len(prompt)}" in caplog.text


@pytest.mark.asyncio
async def test_background_tasks_in_same_second_use_distinct_message_ids(monkeypatch):
    engine = make_engine(FakeToolLoop())

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    monkeypatch.setattr("core.engine.agent_engine.time.time", lambda: 1_000)

    await asyncio.gather(
        engine.execute_background_task("shared", "first task", "system"),
        engine.execute_background_task("shared", "second task", "system"),
    )

    user_messages = engine.context_manager.user_messages
    assert {message[1] for message in user_messages} == {"first task", "second task"}
    assert len({message[2] for message in user_messages}) == 2


@pytest.mark.asyncio
async def test_background_task_defers_context_write_until_its_turn_lock_is_available():
    pause = asyncio.Event()
    tool_loop = FakeToolLoop(pause=pause)
    engine = make_engine(tool_loop)
    events = []

    async def build_prompt():
        return [], []

    async def reply_callback(**kwargs):
        pass

    first = asyncio.create_task(
        engine._run_turn(
            _TurnRequest(
                chat_id="shared",
                sender_id="user",
                is_group=False,
                reply_to="message",
                route_text="hello",
                prompt_factory=build_prompt,
                reply_callback=reply_callback,
            )
        )
    )
    await asyncio.sleep(0)

    async def add_user_message(*args, **kwargs):
        events.append("context")

    async def build_task_messages(**kwargs):
        events.append("prompt")
        return [], []

    engine.context_manager.add_user_message_async = add_user_message
    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    background = asyncio.create_task(
        engine.execute_background_task("shared", "do work", "system")
    )
    await asyncio.sleep(0)

    assert events == []
    pause.set()
    await asyncio.gather(first, background)
    assert events == ["context", "prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("failed"), "failed"),
        (asyncio.CancelledError(), "任务被取消"),
    ],
)
async def test_background_task_preserves_error_conversion(error, expected):
    engine = make_engine(FakeToolLoop(error=error))
    engine.prompt_builder = SimpleNamespace(
        build_task_messages=lambda **kwargs: pytest.fail("sync callback")
    )

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder.build_task_messages = build_task_messages
    result, failure = await engine.execute_background_task("task", "work", "system")

    assert result is None
    assert failure == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replies", "expected"),
    [(["first", "second"], "second"), ([], None)],
)
async def test_background_task_joins_multiple_replies_and_handles_no_reply(
    replies, expected
):
    engine = make_engine(FakeToolLoop(replies=replies))

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    result, error = await engine.execute_background_task("task", "work", "system")

    assert (result, error) == (expected, None)


@pytest.mark.asyncio
async def test_background_task_does_not_treat_captured_tool_send_as_delivered():
    engine = make_engine(FakeToolLoop(replies=["报告"]))

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    result = await engine.execute_background_task("task", "work", "system")

    assert result.tool_delivered is False


@pytest.mark.asyncio
async def test_background_task_delivers_tool_text_to_real_channel_once():
    engine = make_engine(FakeToolLoop(sends_tool_text=True))
    delivered = []

    async def build_task_messages(**kwargs):
        return [], []

    async def reply_callback(**kwargs):
        delivered.append(kwargs)

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    engine._reply_callback = reply_callback
    result = await engine.execute_background_task(
        "task", "work", "system", delivery_channel="user_001"
    )

    assert result.tool_delivered is True
    assert delivered == [
        {
            "chat_id": "user_001",
            "content": "工具消息",
            "message_id": "",
            "is_group": True,
        }
    ]


@pytest.mark.asyncio
async def test_background_task_drops_intermediate_reply_when_final_round_is_silent():
    engine = make_engine(FakeToolLoop(replies=["检测完成"], final_reply_silent=True))

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    result = await engine.execute_background_task("task", "work", "system")

    assert result.result is None
    assert result.silent is True


@pytest.mark.asyncio
async def test_background_task_keeps_context_when_turn_fails():
    engine = make_engine(FakeToolLoop(error=RuntimeError("failed")))

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    result, error = await engine.execute_background_task("task", "work", "system")

    assert (result, error) == (None, "failed")
    assert len(engine.context_manager.user_messages) == 1
    assert engine.context_manager.rollbacks == []


@pytest.mark.asyncio
async def test_background_task_uses_300_second_timeout(monkeypatch):
    engine = make_engine(FakeToolLoop())
    timeouts = []
    original_wait_for = asyncio.wait_for

    async def record_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(build_task_messages=build_task_messages)
    monkeypatch.setattr("core.engine.agent_engine.asyncio.wait_for", record_wait_for)

    await engine.execute_background_task("task", "work", "system")

    assert timeouts == [300]


@pytest.mark.asyncio
async def test_wake_turn_uses_prebuilt_prompt_and_captured_reply():
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)
    engine._reply_callback = object()
    engine.prompt_builder = SimpleNamespace(
        build=lambda **kwargs: pytest.fail("must not rebuild prebuilt wake prompt")
    )

    result = await engine.run_wake_turn(
        source="system",
        session_key="wake-chat",
        messages=[{"role": "system", "content": "wake"}],
        tools=[{"name": "heartbeat_respond"}],
    )

    assert result.captured_replies == ["reply"]
    assert result.should_notify is True
    assert tool_loop.calls[0]["messages"] == [{"role": "system", "content": "wake"}]
    assert tool_loop.calls[0]["tools"] == [{"name": "heartbeat_respond"}]


@pytest.mark.asyncio
async def test_tool_loop_steers_pending_messages_only_after_full_tool_batch(
    monkeypatch,
):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.requests = []
            self.responses = iter(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            AssistantToolCall("one", "first", "{}"),
                            AssistantToolCall("two", "second", "{}"),
                        ],
                    ),
                    AssistantMessage(content="done"),
                ]
            )

        async def chat_completion_with_tools(self, *, messages, tools):
            self.requests.append([dict(message) for message in messages])
            return next(self.responses), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ai = FakeAI()
    session_manager = SessionTaskManager()
    context = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=ai,
            max_tool_rounds=3,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    loop = ToolLoop(context, session_manager=session_manager)
    pending = PendingInbound(
        InputMessage("steer", "user", "chat", "follow up", False),
        "follow up",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )

    seen_tools = []

    async def execute(name, args, tool_ctx, permission_manager):
        seen_tools.append(name)
        if name == "first":
            await session_manager.enqueue_and_claim_consumer("chat", pending)
        return ToolResult(content=name)

    async def admit(item):
        assert seen_tools == ["first", "second"]
        return SimpleNamespace(
            prompt_message={"role": "user", "content": item.prepared_content}
        )

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", execute)
    await loop.run(
        messages=[{"role": "user", "content": "initial"}],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="reply",
        reply_callback=lambda **kwargs: _none(),
        steering_enabled=True,
        steering_admission_callback=admit,
    )

    assert [request[-1]["content"] for request in ai.requests] == [
        "initial",
        "follow up",
    ]


@pytest.mark.asyncio
async def test_tool_loop_delivers_final_text_after_steering_following_send_message(
    monkeypatch,
):
    class FakeAI:
        model = "test"

        def __init__(self):
            self.responses = iter(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            AssistantToolCall("send", "send_message", "{}"),
                            AssistantToolCall("enqueue", "enqueue_steering", "{}"),
                        ],
                    ),
                    AssistantMessage(content="follow-up final"),
                ]
            )

        async def chat_completion_with_tools(self, **kwargs):
            return next(self.responses), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    session_manager = SessionTaskManager()
    loop = ToolLoop(
        SimpleNamespace(
            ai=SimpleNamespace(
                ai_service=FakeAI(),
                max_tool_rounds=3,
                model_registry=None,
                stream_reply=False,
            ),
            mgmt=SimpleNamespace(
                permission_manager=None,
                cost_tracker=None,
                context_manager=FakeContext(),
            ),
            memory=SimpleNamespace(hindsight_memory=None),
        ),
        session_manager=session_manager,
    )
    pending = PendingInbound(
        InputMessage("steer", "user", "chat", "follow up", False),
        "follow up",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )
    delivered = []

    async def execute(name, args, tool_ctx, permission_manager):
        if name == "send_message":
            return ToolResult(
                content="sent",
                delivery_receipt=DeliveryReceipt(
                    status="accepted", logical_delivery_id="tool:chat:send"
                ),
                delivery_kind="message",
            )
        await session_manager.enqueue_and_claim_consumer("chat", pending)
        return ToolResult(content="queued")

    async def admit(item):
        return SimpleNamespace(
            prompt_message={"role": "user", "content": item.prepared_content}
        )

    async def reply_callback(**kwargs):
        delivered.append(kwargs["content"])

    monkeypatch.setattr("core.tools.tool_loop.execute_tool", execute)
    await loop.run(
        messages=[{"role": "user", "content": "initial"}],
        tools=[],
        chat_id="chat",
        is_group=False,
        reply_to="reply",
        reply_callback=reply_callback,
        steering_enabled=True,
        steering_admission_callback=admit,
    )

    assert delivered == ["follow-up final"]


@pytest.mark.asyncio
async def test_tool_loop_does_not_start_followup_for_passive_steering():
    class FakeAI:
        model = "test"

        def __init__(self):
            self.requests = 0

        async def chat_completion_with_tools(self, **kwargs):
            self.requests += 1
            return AssistantMessage(content="done"), None

    class FakeContext:
        async def add_assistant_message_async(self, *args, **kwargs):
            return None

        async def add_tool_result_async(self, *args, **kwargs):
            return None

    ai = FakeAI()
    session_manager = SessionTaskManager()
    ctx = SimpleNamespace(
        ai=SimpleNamespace(
            ai_service=ai,
            max_tool_rounds=2,
            model_registry=None,
            stream_reply=False,
        ),
        mgmt=SimpleNamespace(
            permission_manager=None,
            cost_tracker=None,
            context_manager=FakeContext(),
        ),
        memory=SimpleNamespace(hindsight_memory=None),
    )
    loop = ToolLoop(ctx, session_manager=session_manager)
    passive = PendingInbound(
        InputMessage("passive", "user", "chat", "ambient", True),
        "ambient",
        "passive",
        AdmissionOrigin.USER_MESSAGE,
    )
    await session_manager.enqueue_and_claim_consumer("chat", passive)

    admitted = []

    async def admission_callback(item):
        admitted.append(item.message.id)
        return None

    await loop._drain_steering_messages(
        chat_id="chat",
        admission_callback=admission_callback,
    )

    assert admitted == ["passive"]


@pytest.mark.asyncio
async def test_background_and_wake_turns_disable_steering():
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)
    engine._reply_callback = object()

    async def build_task_messages(**kwargs):
        return [], []

    engine.prompt_builder = SimpleNamespace(
        build_task_messages=build_task_messages,
        build_heartbeat_messages=lambda **kwargs: _empty_prompt(),
    )
    await engine.execute_background_task("background", "work", "system")
    wake_tools = [
        {"type": "function", "function": {"name": "mark_important"}},
        {"type": "function", "function": {"name": "memory"}},
    ]
    await engine.run_wake_turn(
        source="system", session_key="wake", messages=[], tools=wake_tools
    )

    assert [call["steering_enabled"] for call in tool_loop.calls] == [False, False]
    assert [call["internal_control"] for call in tool_loop.calls] == [True, True]
    assert [tool["function"]["name"] for tool in tool_loop.calls[1]["tools"]] == [
        "memory"
    ]


@pytest.mark.asyncio
async def test_internal_control_turn_filters_mark_important_from_prebuilt_tools():
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)
    internal_tools = [
        {"type": "function", "function": {"name": "mark_important"}},
        {"type": "function", "function": {"name": "memory"}},
    ]

    async def prompt_factory():
        return [], internal_tools

    await engine._run_turn(
        _TurnRequest(
            chat_id="wake",
            sender_id="system",
            is_group=False,
            reply_to="wake-message",
            route_text="system event",
            prompt_factory=prompt_factory,
            reply_callback=lambda **kwargs: _none(),
            internal_control=True,
        )
    )

    assert [tool["function"]["name"] for tool in tool_loop.calls[0]["tools"]] == [
        "memory"
    ]


@pytest.mark.asyncio
async def test_user_turn_keeps_mark_important_in_prebuilt_tools():
    tool_loop = FakeToolLoop()
    engine = make_engine(tool_loop)
    user_tools = [{"type": "function", "function": {"name": "mark_important"}}]

    async def prompt_factory():
        return [], user_tools

    await engine._run_turn(
        _TurnRequest(
            chat_id="chat",
            sender_id="user",
            is_group=False,
            reply_to="message",
            route_text="remember this",
            prompt_factory=prompt_factory,
            reply_callback=lambda **kwargs: _none(),
        )
    )

    assert tool_loop.calls[0]["tools"] == user_tools


@pytest.mark.asyncio
async def test_consumer_cancellation_after_admission_commits_lease():
    engine = make_engine(FakeToolLoop(error=asyncio.CancelledError()))
    engine.prompt_builder = SimpleNamespace(build=lambda **kwargs: _empty_prompt())
    pending = PendingInbound(
        InputMessage("cancelled", "user", "chat", "hello", False),
        "hello",
        "agent",
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await engine.session_manager.enqueue_and_claim_consumer("chat", pending)
    replies = []

    async def reply_callback(**kwargs):
        replies.append(kwargs)

    consumer = asyncio.create_task(
        engine._consumer(
            "chat", reply_callback, lambda _: "user", enqueued.consumer_token
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert engine.session_manager.get_queue_sizes() == {}
    assert engine.session_manager.get_message_state("chat", "cancelled") == "admitted"


@pytest.mark.asyncio
async def test_consumer_collects_private_same_intent_messages_into_one_turn():
    engine = make_engine(FakeToolLoop())
    engine.scheduler = ConversationScheduler(
        engine.session_manager,
        collect_idle_ms=10,
        collect_max_wait_ms=20,
        collect_max_messages=8,
        collect_max_chars=6000,
    )
    engine.prompt_builder = SimpleNamespace(build=lambda **kwargs: _empty_prompt())
    first = PendingInbound(
        InputMessage("first", "user", "chat", "first", False),
        "first",
        "private_conversation",
        AdmissionOrigin.USER_MESSAGE,
    )
    second = PendingInbound(
        InputMessage("second", "user", "chat", "second", False),
        "second",
        "private_conversation",
        AdmissionOrigin.USER_MESSAGE,
    )
    enqueued = await engine.scheduler.enqueue("chat", first)
    await engine.scheduler.enqueue("chat", second)

    consumer = asyncio.create_task(
        engine._consumer(
            "chat", lambda **kwargs: _none(), lambda _: "user", enqueued.consumer_token
        )
    )
    await consumer

    assert len(engine.tool_loop.calls) == 1
    assert [message[2] for message in engine.context_manager.user_messages] == [
        "first",
        "second",
    ]


async def _empty_prompt():
    return [], []


async def _none():
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected_mode"),
    [("interval", "minimal"), ("manual", "normal")],
)
async def test_wake_turn_uses_heartbeat_builder_for_interval_and_manual(
    source, expected_mode
):
    engine = make_engine(FakeToolLoop())
    engine._reply_callback = object()
    calls = []

    async def build_heartbeat_messages(**kwargs):
        calls.append(kwargs)
        return [], []

    engine.prompt_builder = SimpleNamespace(
        build_heartbeat_messages=build_heartbeat_messages
    )
    await engine.run_wake_turn(source=source, extra_prompt="wake")

    assert calls[0]["system_prompt_mode"] == expected_mode
    assert calls[0]["session_mode"] == "isolated"


@pytest.mark.asyncio
async def test_wake_turn_uses_normal_builder_for_system_source():
    engine = make_engine(FakeToolLoop())
    engine._reply_callback = object()
    calls = []

    async def build(**kwargs):
        calls.append(kwargs)
        return [], []

    engine.prompt_builder = SimpleNamespace(build=build)
    await engine.run_wake_turn(source="system", session_key="chat", extra_prompt="wake")

    assert calls[0]["chat_id"] == "chat"
    assert calls[0]["sender_id"] == "system"
    assert isinstance(calls[0]["timeline_snapshot"], tuple)
    assert calls[0]["protocol_snapshot"] == ()


@pytest.mark.asyncio
async def test_wake_turn_prefers_explicit_notification_and_ignores_silent_reply():
    from core.tools.impl.heartbeat import heartbeat_response

    async def record_explicit_notification(kwargs):
        heartbeat_response.get().update(
            notify=True,
            notification_text="explicit notification",
            deliver_to_user="target",
        )

    engine = make_engine(
        FakeToolLoop(replies=["NO_REPLY"], on_run=record_explicit_notification)
    )
    engine._reply_callback = object()
    result = await engine.run_wake_turn(
        source="system", session_key="chat", messages=[], tools=[]
    )

    assert result.notification_text == "explicit notification"
    assert result.should_notify is True
    assert result.deliver_to_user == "target"


@pytest.mark.asyncio
async def test_wake_turn_does_not_notify_for_silent_reply():
    engine = make_engine(FakeToolLoop(replies=["NO_REPLY"]))
    engine._reply_callback = object()
    result = await engine.run_wake_turn(
        source="system", session_key="chat", messages=[], tools=[]
    )

    assert result.should_notify is False
    assert result.notification_text == ""


@pytest.mark.asyncio
async def test_wake_turn_resets_heartbeat_context_after_error_and_cancellation():
    from core.tools.impl.heartbeat import heartbeat_response

    seen_contexts = []

    async def record_context(kwargs):
        seen_contexts.append(heartbeat_response.get())

    async def build(**kwargs):
        return [], []

    engine = make_engine(
        FakeToolLoop(error=RuntimeError("failed"), on_run=record_context)
    )
    engine._reply_callback = object()
    engine.prompt_builder = SimpleNamespace(build=build)
    result = await engine.run_wake_turn(source="system", session_key="chat")

    assert result.error == "failed"
    assert seen_contexts[0] is not None
    assert heartbeat_response.get() is None

    engine = make_engine(
        FakeToolLoop(error=asyncio.CancelledError(), on_run=record_context)
    )
    engine._reply_callback = object()
    engine.prompt_builder = SimpleNamespace(build=build)
    with pytest.raises(asyncio.CancelledError):
        await engine.run_wake_turn(source="system", session_key="chat")

    assert heartbeat_response.get() is None


@pytest.mark.asyncio
async def test_wake_turn_uses_default_and_explicit_timeouts(monkeypatch):
    engine = make_engine(FakeToolLoop())
    engine._reply_callback = object()
    timeouts = []
    original_wait_for = asyncio.wait_for

    async def record_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr("core.engine.agent_engine.asyncio.wait_for", record_wait_for)
    await engine.run_wake_turn(
        source="system", session_key="first", messages=[], tools=[]
    )
    await engine.run_wake_turn(
        source="system", session_key="second", messages=[], tools=[], timeout=7
    )

    assert timeouts == [120, 7]


@pytest.mark.asyncio
async def test_wake_turn_ids_do_not_collide_within_one_second():
    engine = make_engine(FakeToolLoop())
    engine._reply_callback = object()

    first = await engine.run_wake_turn(
        source="system", session_key="same-chat", messages=[], tools=[]
    )
    second = await engine.run_wake_turn(
        source="system", session_key="same-chat", messages=[], tools=[]
    )

    assert first.turn_id != second.turn_id


@pytest.mark.asyncio
async def test_active_ambient_turn_uses_delivery_ledger(tmp_path):
    engine = make_engine(FakeToolLoop())
    engine.group_engagement = GroupEngagementManager(
        EngagementConfig(
            group_ambient_mode="active",
            group_ambient_active_chats=("chat",),
            group_ambient_delivery_mode="automatic",
            group_ambient_cooldown_seconds=0,
            group_ambient_quiet_cooldown_seconds=0,
        )
    )
    engine.delivery_controller = DeliveryController(
        DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    )
    pending = PendingInbound(
        InputMessage("ambient-1", "user", "chat", "hello?", True),
        "hello?",
        InboundIntent.GROUP_AMBIENT,
        AdmissionOrigin.USER_MESSAGE,
    )
    decision = await engine.group_engagement.evaluate("chat", batch=[pending])
    delivered = []

    async def raw_reply(**kwargs):
        delivered.append(kwargs["content"])

    async def fake_run(request):
        await request.reply_callback(
            chat_id="chat",
            content="answer",
            message_id="ambient-1",
            is_group=True,
        )
        return _TurnResult(
            replies=("answer",),
            sent_emoji=False,
            text_committed=True,
            tool_text_delivered=False,
            final_reply_silent=False,
        )

    engine._run_turn = fake_run
    await engine._process_ambient_active(
        pending, (), decision, raw_reply, lambda sender_id: sender_id
    )

    assert delivered == ["answer"]
    record = await engine.delivery_controller.ledger.get("ambient:chat:ambient-1")
    assert record is not None
    assert record.status == "sent"


@pytest.mark.asyncio
async def test_engine_recovers_stale_ambient_delivery_with_idempotency_opt_in(
    tmp_path,
):
    engine = make_engine(FakeToolLoop())
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    engine.delivery_controller = DeliveryController(ledger)
    engine.engagement_config = EngagementConfig(delivery_recovery_after_seconds=1)
    engine.context_manager = SimpleNamespace(
        get_chat_history_async=lambda chat_id: _history_with_ambient_answer()
    )
    prepared = await ledger.prepare(
        key="ambient:chat:turn",
        chat_id="chat",
        turn_id="turn",
        reason="final_reply",
        reply_anchor_id="anchor",
        content_hash=ledger.content_hash("answer"),
    )
    conn = await ledger._ensure_open()
    conn.execute(
        "UPDATE delivery_ledger SET updated_at = 0 WHERE delivery_key = ?",
        (prepared.key,),
    )
    conn.commit()
    delivered = []

    async def reply_callback(**kwargs):
        delivered.append(kwargs)

    await engine._recover_ambient_deliveries(
        "chat", reply_callback, allow_transport_retry=True
    )

    assert delivered[0]["content"] == "answer"
    assert delivered[0]["delivery_id"] == prepared.logical_delivery_id
    recovered = await ledger.get(prepared.key)
    assert recovered is not None
    assert recovered.status == "sent"
    await ledger.close()


@pytest.mark.asyncio
async def test_engine_recovery_preserves_failed_transport_receipt(tmp_path):
    engine = make_engine(FakeToolLoop())
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    engine.delivery_controller = DeliveryController(ledger)
    engine.engagement_config = EngagementConfig(delivery_recovery_after_seconds=1)
    engine.context_manager = SimpleNamespace(
        get_chat_history_async=lambda chat_id: _history_with_ambient_answer()
    )
    prepared = await ledger.prepare(
        key="ambient:chat:turn",
        chat_id="chat",
        turn_id="turn",
        reason="final_reply",
        reply_anchor_id="anchor",
        content_hash=ledger.content_hash("answer"),
    )
    conn = await ledger._ensure_open()
    conn.execute(
        "UPDATE delivery_ledger SET updated_at = 0 WHERE delivery_key = ?",
        (prepared.key,),
    )
    conn.commit()

    async def reply_callback(**kwargs):
        return DeliveryReceipt(status="failed", logical_delivery_id="ambient:chat:turn")

    result = await engine._recover_ambient_deliveries("chat", reply_callback)

    assert result.scanned == 1
    recovered = await ledger.get(prepared.key)
    assert recovered is not None
    assert recovered.status == "unknown"
    assert recovered.reason == "recovery_requires_idempotency"
    await ledger.close()


async def _history_with_ambient_answer():
    return [
        {"role": "user", "content": "question", "message_id": "anchor"},
        {"role": "assistant", "content": "answer", "message_id": "anchor"},
    ]
