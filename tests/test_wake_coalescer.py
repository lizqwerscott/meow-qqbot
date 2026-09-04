import asyncio

import pytest

import core.tasks.wake_coalescer as coalescer


@pytest.mark.asyncio
async def test_wake_timer_is_rearmed_after_it_fires():
    calls = []

    async def handler(pending):
        calls.append(pending)
        return coalescer.WakeRunResult()

    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(session_key="chat", coalesce_ms=1)
        await asyncio.sleep(0.03)
        coalescer.request_wake(session_key="chat", coalesce_ms=1)
        await asyncio.sleep(0.03)
        assert len(calls) == 2
    finally:
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_work_plan_wakes_in_same_chat_do_not_coalesce_across_plans():
    calls = []

    async def handler(pending):
        calls.append(pending.work_plan_id)
        return coalescer.WakeRunResult()

    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(
            source=coalescer.SOURCE_TASK,
            session_key="chat",
            work_plan_id="first-plan",
            coalesce_ms=1,
        )
        coalescer.request_wake(
            source=coalescer.SOURCE_TASK,
            session_key="chat",
            work_plan_id="second-plan",
            coalesce_ms=1,
        )
        await asyncio.sleep(0.03)
        assert calls == ["first-plan", "second-plan"]
    finally:
        dispose()
        coalescer.clear_pending()

    calls = []

    async def handler(pending):
        calls.append(pending)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return coalescer.WakeRunResult()

    retry_ms = coalescer.DEFAULT_RETRY_MS
    coalescer.DEFAULT_RETRY_MS = 10
    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(session_key="chat", coalesce_ms=1)
        await asyncio.sleep(0.06)
        assert len(calls) == 2
    finally:
        coalescer.DEFAULT_RETRY_MS = retry_ms
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_retryable_skip_does_not_use_default_coalesce_delay(
    monkeypatch,
):
    calls = []

    async def handler(pending):
        calls.append(pending)
        if len(calls) == 1:
            return coalescer.WakeRunResult(
                status="skipped", skip_reason="requests-in-flight"
            )
        return coalescer.WakeRunResult()

    monkeypatch.setattr(coalescer, "DEFAULT_COALESCE_MS", 1)
    monkeypatch.setattr(coalescer, "DEFAULT_RETRY_MS", 40)
    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(session_key="chat", coalesce_ms=1)
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        await asyncio.sleep(0.04)
        assert len(calls) == 2
        assert coalescer.get_status()["retry_count"] == {}
    finally:
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_handler_exception_does_not_use_default_coalesce_delay(monkeypatch):
    calls = []

    async def handler(pending):
        calls.append(pending)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return coalescer.WakeRunResult()

    monkeypatch.setattr(coalescer, "DEFAULT_COALESCE_MS", 1)
    monkeypatch.setattr(coalescer, "DEFAULT_RETRY_MS", 40)
    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(session_key="chat", coalesce_ms=1)
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        await asyncio.sleep(0.04)
        assert len(calls) == 2
        assert coalescer.get_status()["retry_count"] == {}
    finally:
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_retry_exhaustion_preserves_extended_backoff(monkeypatch):
    calls = []

    async def handler(pending):
        calls.append(pending)
        return coalescer.WakeRunResult(
            status="skipped", skip_reason="requests-in-flight"
        )

    monkeypatch.setattr(coalescer, "DEFAULT_COALESCE_MS", 1)
    monkeypatch.setattr(coalescer, "RETRY_EXHAUSTED_MS", 40)
    monkeypatch.setattr(coalescer, "MAX_RETRY_COUNT", 0)
    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(session_key="chat", coalesce_ms=1)
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        await asyncio.sleep(0.04)
        assert len(calls) == 2
    finally:
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_wake_received_during_handler_is_processed():
    calls = []
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def handler(pending):
        calls.append(pending.session_key)
        if pending.session_key == "first":
            handler_started.set()
            await release_handler.wait()
        return coalescer.WakeRunResult()

    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(session_key="first", coalesce_ms=1)
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        coalescer.request_wake(session_key="second", coalesce_ms=1)
        release_handler.set()
        await asyncio.sleep(0.05)
        assert calls == ["first", "second"]
    finally:
        release_handler.set()
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_exec_wakes_preserve_coalesced_notifications():
    prompts = []

    async def handler(pending):
        prompts.append(pending.extra_prompt)
        return coalescer.WakeRunResult()

    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(
            source=coalescer.SOURCE_EXEC,
            session_key="chat",
            extra_prompt="first",
            coalesce_ms=1,
        )
        coalescer.request_wake(
            source=coalescer.SOURCE_EXEC,
            session_key="chat",
            extra_prompt="second",
            coalesce_ms=1,
        )
        await asyncio.sleep(0.03)
        assert prompts == ["first\n\nsecond"]
    finally:
        dispose()
        coalescer.clear_pending()


@pytest.mark.asyncio
async def test_heartbeat_task_names_survive_pending_merge():
    calls = []

    async def handler(pending):
        calls.append(pending)
        return coalescer.WakeRunResult()

    dispose = coalescer.set_wake_handler(handler)
    try:
        coalescer.request_wake(
            source=coalescer.SOURCE_INTERVAL,
            intent=coalescer.INTENT_SCHEDULED,
            session_key="heartbeat:events",
            heartbeat_task_names=("早安",),
            coalesce_ms=1,
        )
        coalescer.request_wake(
            source=coalescer.SOURCE_CRON,
            intent=coalescer.INTENT_IMMEDIATE,
            session_key="heartbeat:events",
            coalesce_ms=1,
        )
        await asyncio.sleep(0.03)
        assert len(calls) == 1
        assert calls[0].heartbeat_task_names == ("早安",)
    finally:
        dispose()
        coalescer.clear_pending()
