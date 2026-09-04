import time

from core.tasks.heartbeat_cooldown import HeartbeatCooldown
from core.tasks.preflight import PreflightContext, run_preflight


def test_interval_wake_does_not_use_next_cycle_for_due_check():
    cooldown = HeartbeatCooldown()
    cooldown.set_next_due(time.time() * 1000 + 60_000)
    context = PreflightContext(
        source="interval",
        intent="scheduled",
        session_key="heartbeat:events",
        cooldown=cooldown,
        active_hours=(None, None, None),
        has_system_events=False,
        has_extra_prompt=True,
        is_session_active=False,
        has_cron_jobs=False,
        source_is_interval=True,
    )

    result = run_preflight(context)

    assert result.skip_reason is None
