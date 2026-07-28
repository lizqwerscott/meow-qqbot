import pytest

from core.managers.session_manager import SessionTaskManager


@pytest.fixture
def mgr():
    return SessionTaskManager()


@pytest.mark.asyncio
async def test_get_queue_creates_on_demand(mgr):
    q = await mgr.get_queue("chat_001")
    assert q is not None
    assert q.maxsize == 256


@pytest.mark.asyncio
async def test_get_queue_reuses(mgr):
    q1 = await mgr.get_queue("chat_001")
    q2 = await mgr.get_queue("chat_001")
    assert q1 is q2


@pytest.mark.asyncio
async def test_get_lock_creates_on_demand(mgr):
    lock = await mgr.get_lock("chat_001")
    assert lock is not None


@pytest.mark.asyncio
async def test_try_start_consumer_first(mgr):
    started = await mgr.try_start_consumer("chat_001")
    assert started is True


@pytest.mark.asyncio
async def test_try_start_consumer_duplicate(mgr):
    await mgr.try_start_consumer("chat_001")
    started = await mgr.try_start_consumer("chat_001")
    assert started is False


@pytest.mark.asyncio
async def test_mark_consumer_done_clears_running(mgr):
    await mgr.try_start_consumer("chat_001")
    await mgr.mark_consumer_done("chat_001")
    assert mgr.has_active_consumer("chat_001") is False


@pytest.mark.asyncio
async def test_mark_consumer_done_with_remaining_queue(mgr):
    q = await mgr.get_queue("chat_001")
    q.put_nowait("msg1")
    q.put_nowait("msg2")
    await mgr.try_start_consumer("chat_001")
    # should not raise, just warn
    await mgr.mark_consumer_done("chat_001")
    assert mgr.has_active_consumer("chat_001") is False


@pytest.mark.asyncio
async def test_cleanup_session(mgr):
    await mgr.get_queue("chat_001")
    await mgr.try_start_consumer("chat_001")
    await mgr.cleanup_session("chat_001")
    assert mgr.has_active_consumer("chat_001") is False
    assert "chat_001" not in mgr._queues


@pytest.mark.asyncio
async def test_cleanup_all(mgr):
    await mgr.get_queue("chat_001")
    await mgr.get_queue("chat_002")
    await mgr.cleanup_all()
    assert len(mgr._queues) == 0


def test_get_queue_sizes(mgr):
    import asyncio
    async def setup():
        q = await mgr.get_queue("chat_001")
        q.put_nowait("msg")
        sizes = mgr.get_queue_sizes()
        assert sizes.get("chat_001") == 1
    asyncio.run(setup())
