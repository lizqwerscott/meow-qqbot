"""FallbackRunner session 绑定逻辑测试。

使用 mock ModelRegistry 隔离测试 try_acquire_with_binding 和 mark_success。
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from core.ai.fallback_runner import FallbackRunner
from core.managers.session_binding import SessionBindingManager


def _make_service(name="svc"):
    return MagicMock(name=name)


def _make_cooldown_manager():
    cm = MagicMock()
    cm.is_cooled_down = AsyncMock(return_value=False)
    cm.record_success = AsyncMock()
    cm.record_failure = AsyncMock()
    cm.reset = AsyncMock()
    return cm


def _make_registry(
    resolve_result=None,
    get_result=None,
    get_session_config_result=(30, 600),
    chain=None,
):
    """创建一个可配置的 mock ModelRegistry。"""
    r = MagicMock()
    r.resolve_model_chain = AsyncMock(return_value=resolve_result)
    r.get = MagicMock(return_value=get_result)
    r.get_session_config = MagicMock(return_value=get_session_config_result)
    r.cooldown_manager = _make_cooldown_manager()

    type(r).default_service = PropertyMock(return_value=None)
    # resolve_model_chain logs, so provide a get_chain too
    r.get_chain = MagicMock(return_value=chain or [])
    return r


class TestTryAcquireWithBinding:
    """try_acquire_with_binding 的绑定/解绑行为。"""

    @pytest.mark.asyncio
    async def test_no_binding_acquires_and_binds(self):
        """首次调用：无 binding → acquire() → bind()。"""
        mgr = SessionBindingManager()
        svc = _make_service()
        reg = _make_registry(
            resolve_result=("deepseek/primary", svc),
            get_result=svc,
        )
        runner = FallbackRunner(reg, ["deepseek/primary"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is True
        assert runner.current == "deepseek/primary"
        reg.resolve_model_chain.assert_awaited_once()
        # 验证 binding 已创建
        binding = await mgr.get("chat_1", "simple")
        assert binding is not None
        assert binding.model_name == "deepseek/primary"
        assert binding.request_count == 0

    @pytest.mark.asyncio
    async def test_valid_binding_skips_acquire(self):
        """已有有效 binding → 直接使用，不调 acquire()。"""
        mgr = SessionBindingManager()
        svc = _make_service()
        reg = _make_registry(get_result=svc, get_session_config_result=(30, 600))

        # 预先绑定
        await mgr.bind("chat_1", "simple", "deepseek/primary")

        runner = FallbackRunner(reg, ["deepseek/primary"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is True
        assert runner.current == "deepseek/primary"
        # acquire() 不应被调用
        reg.resolve_model_chain.assert_not_awaited()
        # binding 未被解绑
        binding = await mgr.get("chat_1", "simple")
        assert binding is not None

    @pytest.mark.asyncio
    async def test_expired_budget_unbinds(self):
        """binding 过期（budget 耗尽）→ 解绑 → 重新 acquire。"""
        mgr = SessionBindingManager()
        svc = _make_service()
        reg = _make_registry(
            resolve_result=("deepseek/primary", svc),
            get_result=svc,
            get_session_config_result=(5, 600),  # budget=5
        )

        # 预先绑定，request_count=5 → 已到期
        import time
        from core.managers.session_binding import SessionBinding
        async with mgr._lock:
            mgr._bindings["chat_1:simple"] = SessionBinding(
                model_name="deepseek/primary",
                tier="simple",
                bound_at=time.monotonic(),
                request_count=5,
            )

        runner = FallbackRunner(reg, ["deepseek/primary"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is True
        reg.resolve_model_chain.assert_awaited_once()
        # 旧的 binding 被覆盖（新 acqure 后重新 bind）
        binding = await mgr.get("chat_1", "simple")
        assert binding.request_count == 0

    @pytest.mark.asyncio
    async def test_expired_ttl_unbinds(self):
        """binding 过期（TTL 超时）→ 解绑 → 重新 acquire。"""
        mgr = SessionBindingManager()
        svc = _make_service()
        reg = _make_registry(
            resolve_result=("deepseek/primary", svc),
            get_result=svc,
            get_session_config_result=(30, 0.001),  # TTL=1ms
        )

        await mgr.bind("chat_1", "simple", "deepseek/primary")
        import asyncio
        await asyncio.sleep(0.01)

        runner = FallbackRunner(reg, ["deepseek/primary"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is True
        reg.resolve_model_chain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cooldown_model_unbinds(self):
        """binding 指向的模型在冷却中 → 解绑 → 重新 acquire。"""
        mgr = SessionBindingManager()
        svc = _make_service()
        reg = _make_registry(
            resolve_result=("model_b", svc),
            get_result=svc,
        )
        reg.cooldown_manager.is_cooled_down = AsyncMock(return_value=True)

        await mgr.bind("chat_1", "simple", "model_a")

        runner = FallbackRunner(reg, ["model_a", "model_b"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is True
        assert runner.current == "model_b"
        reg.resolve_model_chain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_acquire_all_failed(self):
        """链中全部不可用 → acqure 返回 False。"""
        mgr = SessionBindingManager()
        reg = _make_registry(resolve_result=None)

        runner = FallbackRunner(reg, ["model_a"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is False
        assert runner.current is None

    @pytest.mark.asyncio
    async def test_skips_binding_when_no_mgr(self):
        """mgr=None → 普通 acquire，不尝试绑定。"""
        reg = _make_registry(
            resolve_result=("deepseek/primary", _make_service()),
        )
        runner = FallbackRunner(reg, ["deepseek/primary"])

        ok = await runner.try_acquire_with_binding(None, "chat_1", "simple")

        assert ok is True
        assert runner.current == "deepseek/primary"

    @pytest.mark.asyncio
    async def test_skips_binding_when_no_tier(self):
        """tier=None → 普通 acquire，不尝试绑定。"""
        mgr = SessionBindingManager()
        reg = _make_registry(
            resolve_result=("deepseek/primary", _make_service()),
        )
        runner = FallbackRunner(reg, ["deepseek/primary"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", None)

        assert ok is True
        assert runner.current == "deepseek/primary"
        # 不应创建 binding
        b = await mgr.get("chat_1", "simple")
        assert b is None

    @pytest.mark.asyncio
    async def test_stale_binding_model_not_in_registry_unbinds(self):
        """binding 指向的模型已不存在（如配置变更）→ 解绑 → 重新 acquire。"""
        mgr = SessionBindingManager()
        svc = _make_service()
        reg = _make_registry(
            resolve_result=("model_b", svc),
            get_result=svc,
        )
        # get 返回 None for model_a (已从 registry 移除)
        def get_side_effect(name):
            return svc if name == "model_b" else None
        reg.get = MagicMock(side_effect=get_side_effect)

        await mgr.bind("chat_1", "simple", "model_a")

        runner = FallbackRunner(reg, ["model_b"])

        ok = await runner.try_acquire_with_binding(mgr, "chat_1", "simple")

        assert ok is True
        assert runner.current == "model_b"


class TestMarkSuccess:
    """mark_success 的计数行为。"""

    @pytest.mark.asyncio
    async def test_ticks_binding_when_mgr_present(self):
        mgr = SessionBindingManager()
        reg = _make_registry()
        runner = FallbackRunner(reg, ["m"])
        runner._current_name = "m"
        await mgr.bind("chat_1", "simple", "m")

        await runner.mark_success(mgr=mgr, chat_id="chat_1", tier="simple")

        b = await mgr.get("chat_1", "simple")
        assert b.request_count == 1

    @pytest.mark.asyncio
    async def test_noop_when_mgr_absent(self):
        reg = _make_registry()
        runner = FallbackRunner(reg, ["m"])
        runner._current_name = "m"

        # 不应抛异常
        await runner.mark_success(mgr=None, chat_id="chat_1", tier="simple")

    @pytest.mark.asyncio
    async def test_noop_when_tier_absent(self):
        mgr = SessionBindingManager()
        reg = _make_registry()
        runner = FallbackRunner(reg, ["m"])
        runner._current_name = "m"
        await mgr.bind("chat_1", "simple", "m")

        await runner.mark_success(mgr=mgr, chat_id="chat_1", tier=None)

        b = await mgr.get("chat_1", "simple")
        assert b.request_count == 0  # 未 tick

    @pytest.mark.asyncio
    async def test_clears_cooldown_regardless(self):
        cm = _make_cooldown_manager()
        reg = _make_registry()
        reg.cooldown_manager = cm
        runner = FallbackRunner(reg, ["m"])
        runner._current_name = "m"

        await runner.mark_success(mgr=None, chat_id="chat_1", tier=None)

        cm.record_success.assert_awaited_once_with("m")


class TestMarkFailure:
    """mark_failure 行为。"""

    @pytest.mark.asyncio
    async def test_adds_to_failed_and_clears_current(self):
        reg = _make_registry()
        runner = FallbackRunner(reg, ["m"])
        runner._current_name = "m"
        runner._current_svc = _make_service()

        await runner.mark_failure(record_cooldown=False)

        assert runner.current is None
        assert runner.service() is None
        assert "m" in runner._failed
        # record_cooldown=False → 不写入冷却
        reg.cooldown_manager.record_failure.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_records_cooldown_when_requested(self):
        reg = _make_registry()
        runner = FallbackRunner(reg, ["m"])
        runner._current_name = "m"

        await runner.mark_failure(record_cooldown=True)

        reg.cooldown_manager.record_failure.assert_awaited_once_with("m")

    @pytest.mark.asyncio
    async def test_noop_when_no_current_name(self):
        reg = _make_registry()
        runner = FallbackRunner(reg, ["m"])
        # _current_name is None
        await runner.mark_failure(record_cooldown=True)
        reg.cooldown_manager.record_failure.assert_not_awaited()
