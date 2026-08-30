import asyncio

import pytest

from core.engine.engagement_config import EngagementConfig
from core.engine.group_engagement import (
    EngagementPhase,
    EngagementTrigger,
    GroupEngagementManager,
)
from core.engine.group_proactive import GroupProactiveScheduler


@pytest.mark.asyncio
async def test_proactive_reservation_uses_separate_budget_and_cooldown():
    now = [100.0]
    config = EngagementConfig(
        group_ambient_mode="active",
        group_ambient_active_chats=("chat",),
        group_proactive_mode="active",
        group_proactive_active_chats=("chat",),
        group_proactive_cooldown_seconds=50,
        group_proactive_quiet_cooldown_seconds=20,
        group_proactive_window_seconds=100,
        group_proactive_max_turns_per_window=1,
    )
    manager = GroupEngagementManager(config, clock=lambda: now[0])

    decision = await manager.reserve_proactive("chat")
    assert decision.allowed is True
    assert decision.trigger is EngagementTrigger.PROACTIVE
    assert await manager.start(decision) is True
    assert await manager.complete(decision, delivered=True, silent=False) is True
    assert manager.phase("chat") is EngagementPhase.COOLDOWN

    denied = await manager.reserve_proactive("chat")
    assert denied.allowed is False
    assert denied.reason == "cooldown"

    now[0] += 51
    denied = await manager.reserve_proactive("chat")
    assert denied.allowed is False
    assert denied.reason == "proactive_budget_exhausted"


@pytest.mark.asyncio
async def test_proactive_shadow_does_not_start_provider():
    manager = GroupEngagementManager(
        EngagementConfig(
            group_proactive_mode="shadow",
            group_proactive_active_chats=("chat",),
        )
    )
    decision = await manager.reserve_proactive("chat")
    assert decision.shadow is True
    assert decision.allowed is False
    assert manager.phase("chat") is EngagementPhase.RESERVED


@pytest.mark.asyncio
async def test_scheduler_skips_busy_group_and_runs_active_group():
    calls = []
    busy = {"busy"}
    config = EngagementConfig(
        group_proactive_mode="active",
        group_proactive_active_chats=("busy", "ready"),
        group_proactive_active_hours_start="00:00",
        group_proactive_active_hours_end="00:00",
    )
    manager = GroupEngagementManager(config)

    async def run_turn(decision):
        calls.append(decision.chat_id)
        return type("Result", (), {"delivered": False, "silent": True})()

    scheduler = GroupProactiveScheduler(
        config,
        manager,
        run_turn,
        is_busy=lambda chat_id: chat_id in busy,
    )
    await scheduler.tick_once()

    assert calls == ["ready"]
    assert scheduler.snapshot_metrics()["session_busy"] == 1
    assert scheduler.snapshot_metrics()["silent"] == 1


@pytest.mark.asyncio
async def test_scheduler_consumes_proactive_budget_when_turn_starts():
    now = [100.0]
    calls = []
    config = EngagementConfig(
        group_proactive_mode="active",
        group_proactive_active_chats=("chat",),
        group_proactive_interval_seconds=1,
        group_proactive_cooldown_seconds=0,
        group_proactive_quiet_cooldown_seconds=0,
        group_proactive_window_seconds=100,
        group_proactive_max_turns_per_window=1,
        group_proactive_active_hours_start="00:00",
        group_proactive_active_hours_end="00:00",
    )
    manager = GroupEngagementManager(config, clock=lambda: now[0])

    async def run_turn(decision):
        calls.append(decision.chat_id)
        return type("Result", (), {"delivered": True, "silent": False})()

    scheduler = GroupProactiveScheduler(
        config,
        manager,
        run_turn,
        clock=lambda: now[0],
        wall_clock=lambda: 0,
    )
    await scheduler.tick_once()
    now[0] += 1
    await scheduler.tick_once()

    assert calls == ["chat"]
    assert scheduler.snapshot_metrics()["skip:proactive_budget_exhausted"] == 1


@pytest.mark.asyncio
async def test_scheduler_start_stop_is_idempotent():
    config = EngagementConfig(group_proactive_mode="off")
    manager = GroupEngagementManager(config)
    scheduler = GroupProactiveScheduler(
        config, manager, lambda _decision: asyncio.sleep(0)
    )

    await scheduler.start()
    await scheduler.start()
    assert scheduler.running is True
    await scheduler.stop()
    await scheduler.stop()
    assert scheduler.running is False


@pytest.mark.asyncio
async def test_scheduler_releases_admission_lock_after_provider_start():
    config = EngagementConfig(
        group_proactive_mode="active",
        group_proactive_active_chats=("chat",),
        group_proactive_active_hours_start="00:00",
        group_proactive_active_hours_end="00:00",
    )
    manager = GroupEngagementManager(config)
    admission_lock = asyncio.Lock()
    provider_started = asyncio.Event()
    release_turn = asyncio.Event()

    async def run_turn(_decision, provider_start_gate):
        assert await provider_start_gate() is True
        provider_started.set()
        await release_turn.wait()
        return type("Result", (), {"delivered": True, "silent": False})()

    scheduler = GroupProactiveScheduler(
        config,
        manager,
        run_turn,
        admission_lock=admission_lock,
    )
    tick = asyncio.create_task(scheduler.tick_once())
    await asyncio.wait_for(provider_started.wait(), timeout=1)

    updated = EngagementConfig(
        group_proactive_mode="off",
        group_proactive_active_chats=("chat",),
    )
    await asyncio.wait_for(
        scheduler.reconfigure(updated, admission_lock=admission_lock), timeout=1
    )
    assert scheduler.config.group_proactive_mode == "off"

    release_turn.set()
    await tick


@pytest.mark.asyncio
async def test_proactive_metric_buckets_survive_restart(tmp_path):
    from core.engine.proactive_state import ProactiveStateStore

    path = str(tmp_path / "metrics.sqlite3")
    first = ProactiveStateStore(path)
    await first.increment_metric("scheduler", "delivered", at=7200)
    await first.increment_metric("scheduler", "delivered", at=7210)
    await first.increment_metric("scheduler", "failed", at=10800)
    assert await first.metric_totals("scheduler") == {
        "delivered": 2,
        "failed": 1,
    }
    await first.close()

    second = ProactiveStateStore(path)
    assert await second.metric_totals("scheduler", since=7200) == {
        "delivered": 2,
        "failed": 1,
    }
    assert await second.metric_history("scheduler", since=7200, until=10800) == [
        {"metric": "delivered", "bucket_start": 7200, "count": 2}
    ]
    await second.close()

    now = [100.0]
    wall = [1000.0]
    config = EngagementConfig(
        group_proactive_mode="active",
        group_proactive_active_chats=("chat",),
        group_proactive_cooldown_seconds=50,
        group_proactive_quiet_cooldown_seconds=20,
        group_proactive_window_seconds=100,
        group_proactive_max_turns_per_window=2,
    )
    from core.engine.proactive_state import ProactiveStateStore

    first_store = ProactiveStateStore(str(tmp_path / "proactive.sqlite3"))
    first = GroupEngagementManager(
        config,
        clock=lambda: now[0],
        wall_clock=lambda: wall[0],
        state_store=first_store,
    )
    decision = await first.reserve_proactive("chat")
    assert await first.start(decision) is True
    assert await first.complete(decision, delivered=True, silent=False) is True
    await first_store.close()

    second_store = ProactiveStateStore(str(tmp_path / "proactive.sqlite3"))
    second = GroupEngagementManager(
        config,
        clock=lambda: now[0],
        wall_clock=lambda: wall[0],
        state_store=second_store,
    )
    denied = await second.reserve_proactive("chat")
    assert denied.reason == "cooldown"

    wall[0] += 51
    now[0] += 51
    allowed = await second.reserve_proactive("chat")
    assert allowed.allowed is True
    assert await second.start(allowed) is True
    assert (await second._state_store.get("chat")).proactive_turns_in_window == 2
    await second_store.close()


@pytest.mark.asyncio
async def test_proactive_scheduler_restores_next_due_after_restart(tmp_path):
    now = [100.0]
    wall = [1000.0]
    config = EngagementConfig(
        group_proactive_mode="active",
        group_proactive_active_chats=("chat",),
        group_proactive_interval_seconds=10,
        group_proactive_active_hours_start="00:00",
        group_proactive_active_hours_end="00:00",
    )
    from core.engine.proactive_state import ProactiveStateStore

    first_store = ProactiveStateStore(str(tmp_path / "proactive.sqlite3"))
    first_manager = GroupEngagementManager(config, state_store=first_store)
    calls = []

    async def run_turn(_decision):
        calls.append("ran")
        return type("Result", (), {"delivered": True, "silent": False})()

    first = GroupProactiveScheduler(
        config,
        first_manager,
        run_turn,
        clock=lambda: now[0],
        wall_clock=lambda: wall[0],
        state_store=first_store,
    )
    await first.tick_once()
    await first_store.close()

    wall[0] += 5
    now[0] = 0.0
    second_store = ProactiveStateStore(str(tmp_path / "proactive.sqlite3"))
    second_manager = GroupEngagementManager(config, state_store=second_store)
    second = GroupProactiveScheduler(
        config,
        second_manager,
        run_turn,
        clock=lambda: now[0],
        wall_clock=lambda: wall[0],
        state_store=second_store,
    )
    await second.tick_once()

    assert calls == ["ran"]
    assert second.next_due()["chat"] == 5.0
    await second_store.close()


@pytest.mark.asyncio
async def test_scheduler_applies_deterministic_jitter_to_next_due():
    now = [100.0]
    config = EngagementConfig(
        group_proactive_mode="active",
        group_proactive_active_chats=("chat",),
        group_proactive_interval_seconds=10,
        group_proactive_jitter_seconds=5,
        group_proactive_active_hours_start="00:00",
        group_proactive_active_hours_end="00:00",
    )
    manager = GroupEngagementManager(config, clock=lambda: now[0])
    scheduler = GroupProactiveScheduler(
        config,
        manager,
        lambda _decision: asyncio.sleep(0),
        clock=lambda: now[0],
        wall_clock=lambda: 0,
    )

    await scheduler.tick_once()

    next_due = scheduler.next_due()["chat"]
    assert 110.0 <= next_due <= 115.0


@pytest.mark.asyncio
async def test_proactive_profile_excludes_ambient_media_tools():
    from core.engine.prompt_snapshot import PromptMode
    from core.engine.turn_capabilities import TurnCapabilities
    from core.managers.session_manager import InboundIntent

    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="group_proactive",
        intent=InboundIntent.GROUP_AMBIENT,
    )

    assert capabilities.allowed_tool_names == frozenset({"send_message", "send_emoji"})
    assert capabilities.planner_actions == frozenset({"wait", "no_reply"})
