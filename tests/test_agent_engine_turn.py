import asyncio
from types import SimpleNamespace

import pytest

from core.ai.protocol import AssistantMessage, AssistantToolCall
from core.engine.agent_engine import AgentEngine, BackgroundTaskResult, _TurnRequest
from core.managers.session_manager import PendingInbound, SessionTaskManager
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
async def test_admission_archives_after_message_is_persisted():
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

    async def fake_execute(name, args, tool_ctx, permission_manager):
        if name == "send_message":
            assert tool_ctx.reply_callback is tool_callback
            await tool_ctx.reply_callback(
                chat_id=tool_ctx.chat_id,
                content="sent",
                message_id="",
                is_group=tool_ctx.is_group,
            )
            return ToolResult(content="ok", sent_text=True)
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


def test_should_dispatch_to_ai_for_reply_to_bot():
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
    await engine._process_message(message, reply_callback, lambda sender_id: "name")

    assert engine.context_manager.recorded_chat_types == [("chat", True)]
    assert prompt_calls[0]["user_nickname"] == "name"
    assert delivered[0]["message_id"] == "id"
    assert tool_loop.calls[0]["get_user_nickname"]("user") == "name"
    assert tool_loop.calls[0]["stream_callback"] is not None
    assert system_events[0][0] == "chat"


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
            InputMessage("id", "user", "chat", "hello", False),
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
        await engine._process_message(message, lambda **kwargs: None, lambda _: "name")

    assert engine.context_manager.rollbacks == [("chat", "id")]


@pytest.mark.asyncio
async def test_consumer_sends_friendly_error_and_requeues_message():
    engine = make_engine(FakeToolLoop())
    delivered = []
    delivered_event = asyncio.Event()

    async def fail_process(*args):
        raise RuntimeError("failed")

    async def reply_callback(**kwargs):
        delivered.append(kwargs)
        delivered_event.set()

    engine._process_message = fail_process
    message = InputMessage("message", "user", "chat", "hello", True)
    pending = PendingInbound(message, "hello", "agent")
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
        "chat", PendingInbound(first, "first", "agent")
    )
    await engine.session_manager.enqueue_and_claim_consumer(
        "chat", PendingInbound(steering, "steering", "agent")
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
    await engine.run_wake_turn(
        source="system", session_key="wake", messages=[], tools=[]
    )

    assert [call["steering_enabled"] for call in tool_loop.calls] == [False, False]


@pytest.mark.asyncio
async def test_consumer_cancellation_after_admission_commits_lease():
    engine = make_engine(FakeToolLoop(error=asyncio.CancelledError()))
    engine.prompt_builder = SimpleNamespace(build=lambda **kwargs: _empty_prompt())
    pending = PendingInbound(
        InputMessage("cancelled", "user", "chat", "hello", False),
        "hello",
        "agent",
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
