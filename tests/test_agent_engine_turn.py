import asyncio
from types import SimpleNamespace

import pytest

from core.engine.agent_engine import AgentEngine, _TurnRequest
from core.managers.session_manager import SessionTaskManager


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
    def __init__(self, *, error=None, pause=None, replies=None, on_run=None):
        self.calls = []
        self.error = error
        self.pause = pause
        self.replies = ["reply"] if replies is None else replies
        self.on_run = on_run

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
        return True, True


def make_engine(tool_loop, *, rule_router=None, model_registry=None):
    engine = AgentEngine.__new__(AgentEngine)
    engine.context_manager = FakeContextManager()
    engine.session_manager = SessionTaskManager()
    engine.tool_loop = tool_loop
    engine.rule_router = rule_router
    engine.model_registry = model_registry
    engine._session_binding = object()
    engine.cost_tracker = object()
    engine._system_events = None
    engine._admin_id = []
    engine._reply_callback = None
    return engine


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
        drain_non_heartbeat=lambda chat_id: system_events.append(chat_id)
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
    assert system_events == ["chat"]


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
async def test_consumer_sends_friendly_error_and_marks_session_done():
    engine = make_engine(FakeToolLoop())
    delivered = []
    delivered_event = asyncio.Event()

    async def fail_process(*args):
        raise RuntimeError("failed")

    async def reply_callback(**kwargs):
        delivered.append(kwargs)
        delivered_event.set()

    engine._process_message = fail_process
    queue = await engine.session_manager.get_queue("chat")
    await queue.put(SimpleNamespace(id="message", is_group=True))
    await engine.session_manager.try_start_consumer("chat")
    consumer = asyncio.create_task(
        engine._consumer("chat", reply_callback, lambda sender_id: "name")
    )

    await delivered_event.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
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


@pytest.mark.asyncio
async def test_consumer_leaves_steering_message_for_tool_loop_to_drain():
    engine = make_engine(FakeToolLoop())
    drained = []
    drained_event = asyncio.Event()

    async def drain_steering_message(kwargs):
        queue = await engine.session_manager.get_queue(kwargs["chat_id"])
        steering = queue.get_nowait()
        drained.append(steering.id)
        queue.task_done()
        drained_event.set()

    engine.tool_loop.on_run = drain_steering_message
    engine.prompt_builder = SimpleNamespace(build=lambda **kwargs: _empty_prompt())
    queue = await engine.session_manager.get_queue("chat")
    await queue.put(
        SimpleNamespace(
            id="first",
            chat_id="chat",
            sender_id="user",
            is_group=False,
            content="first",
            model_chain=None,
            tier=None,
        )
    )
    await queue.put(SimpleNamespace(id="steering"))
    await engine.session_manager.try_start_consumer("chat")

    consumer = asyncio.create_task(
        engine._consumer("chat", lambda **kwargs: _none(), lambda sender_id: "name")
    )
    await drained_event.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
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
    [(["first", "second"], "first\nsecond"), ([], None)],
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
