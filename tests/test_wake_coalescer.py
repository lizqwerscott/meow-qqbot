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
async def test_handler_exception_is_retried():
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
