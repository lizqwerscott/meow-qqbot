"""SessionBinding + SessionBindingManager 单元测试。"""

import time

import pytest

from core.managers.session_binding import SessionBinding, SessionBindingManager


class TestSessionBinding:
    """SessionBinding dataclass 行为。"""

    def test_not_expired_within_limits(self):
        b = SessionBinding(model_name="m", tier="simple", bound_at=time.monotonic(), request_count=5)
        assert b.is_expired(budget=30, ttl=600) is False

    def test_expired_by_budget(self):
        b = SessionBinding(model_name="m", tier="simple", bound_at=time.monotonic(), request_count=30)
        assert b.is_expired(budget=30, ttl=600) is True

    def test_expired_by_budget_exceeded(self):
        b = SessionBinding(model_name="m", tier="simple", bound_at=time.monotonic(), request_count=31)
        assert b.is_expired(budget=30, ttl=600) is True

    def test_expired_by_ttl(self):
        b = SessionBinding(model_name="m", tier="simple", bound_at=time.monotonic() - 601, request_count=5)
        assert b.is_expired(budget=30, ttl=600) is True

    def test_budget_zero_means_always_expired(self):
        """budget=0 → 总是到期（等同于禁用 session 绑定）。"""
        b = SessionBinding(model_name="m", tier="simple", bound_at=time.monotonic(), request_count=0)
        assert b.is_expired(budget=0, ttl=600) is True

    def test_ttl_zero_means_always_expired(self):
        """ttl=0 → 总是到期。"""
        b = SessionBinding(model_name="m", tier="simple", bound_at=time.monotonic(), request_count=5)
        assert b.is_expired(budget=30, ttl=0) is True


class TestSessionBindingManager:
    """SessionBindingManager 生命周期。"""

    @pytest.fixture
    def mgr(self):
        return SessionBindingManager()

    @pytest.mark.asyncio
    async def test_bind_get(self, mgr):
        await mgr.bind("chat_1", "simple", "deepseek/primary")
        b = await mgr.get("chat_1", "simple")
        assert b is not None
        assert b.model_name == "deepseek/primary"
        assert b.tier == "simple"
        assert b.request_count == 0

    @pytest.mark.asyncio
    async def test_get_unbound_returns_none(self, mgr):
        b = await mgr.get("chat_nonexistent", "simple")
        assert b is None

    @pytest.mark.asyncio
    async def test_unbind(self, mgr):
        await mgr.bind("chat_1", "simple", "deepseek/primary")
        await mgr.unbind("chat_1", "simple")
        b = await mgr.get("chat_1", "simple")
        assert b is None

    @pytest.mark.asyncio
    async def test_unbind_nonexistent_no_error(self, mgr):
        await mgr.unbind("nonexistent", "simple")

    @pytest.mark.asyncio
    async def test_tick(self, mgr):
        await mgr.bind("chat_1", "simple", "deepseek/primary")
        await mgr.tick("chat_1", "simple")
        b = await mgr.get("chat_1", "simple")
        assert b.request_count == 1

    @pytest.mark.asyncio
    async def test_tick_nonexistent_no_error(self, mgr):
        await mgr.tick("nonexistent", "simple")

    @pytest.mark.asyncio
    async def test_key_isolation_chat_id(self, mgr):
        await mgr.bind("chat_A", "simple", "model_a")
        await mgr.bind("chat_B", "simple", "model_b")
        assert (await mgr.get("chat_A", "simple")).model_name == "model_a"
        assert (await mgr.get("chat_B", "simple")).model_name == "model_b"

    @pytest.mark.asyncio
    async def test_key_isolation_tier(self, mgr):
        await mgr.bind("chat_1", "simple", "cheap_model")
        await mgr.bind("chat_1", "complex", "expensive_model")
        assert (await mgr.get("chat_1", "simple")).model_name == "cheap_model"
        assert (await mgr.get("chat_1", "complex")).model_name == "expensive_model"

    @pytest.mark.asyncio
    async def test_bind_overwrites_existing(self, mgr):
        await mgr.bind("chat_1", "simple", "model_a")
        await mgr.bind("chat_1", "simple", "model_b")
        b = await mgr.get("chat_1", "simple")
        assert b.model_name == "model_b"
        assert b.request_count == 0  # 重置计数器

    @pytest.mark.asyncio
    async def test_get_all(self, mgr):
        await mgr.bind("chat_1", "simple", "ds-flash")
        await mgr.bind("chat_2", "complex", "ds-v3")
        all_b = await mgr.get_all()
        assert len(all_b) == 2
        assert "chat_1:simple" in all_b
        assert "chat_2:complex" in all_b
