"""测试 BackgroundTaskRunner 的静态辅助函数。"""

from core.tasks.models import CronJob, SessionMode
from core.tasks.runner import BackgroundTaskRunner


# ── _resolve_session_id ──


def test_session_id_isolated_default():
    job = CronJob(name="test", session_mode=SessionMode.ISOLATED.value)
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "task:task_001"


def test_session_id_custom_with_id():
    job = CronJob(name="test", session_mode=SessionMode.CUSTOM.value, custom_session_id="my_session")
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "cron:my_session"


def test_session_id_custom_without_id():
    job = CronJob(name="test", session_mode=SessionMode.CUSTOM.value, custom_session_id=None)
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "task:task_001"


def test_session_id_main():
    job = CronJob(name="test", session_mode=SessionMode.MAIN.value)
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "cron:main"


def test_session_id_current_fallback():
    """session_mode='current' 应回退到 isolated。"""
    job = CronJob(name="test", session_mode="current")
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "task:task_001"


# ── _check_command_safe ──


def test_command_safe_allowed():
    assert BackgroundTaskRunner._check_command_safe("ls -la") is None


def test_command_safe_denied():
    result = BackgroundTaskRunner._check_command_safe("shutdown -h now")
    assert result is not None
    assert "禁止" in result


def test_command_safe_empty():
    result = BackgroundTaskRunner._check_command_safe("")
    assert result is not None
    assert "为空" in result


def test_command_safe_invalid_syntax():
    result = BackgroundTaskRunner._check_command_safe('echo "unclosed')
    assert result is not None
    assert "无效" in result


def test_command_safe_whitespace_only():
    result = BackgroundTaskRunner._check_command_safe("   ")
    assert result is not None


# ── _resolve_event_target ──


def test_event_target_uses_delivery_channel():
    job = CronJob(name="test", delivery_channel="group_001")
    result = BackgroundTaskRunner._resolve_event_target(job, "task_001")
    assert result == "group_001"


def test_event_target_fallback_to_session_id():
    job = CronJob(name="test", session_mode=SessionMode.MAIN.value)
    result = BackgroundTaskRunner._resolve_event_target(job, "task_001")
    assert result == "cron:main"


def test_event_target_fallback_isolated():
    job = CronJob(name="test")
    result = BackgroundTaskRunner._resolve_event_target(job, "task_002")
    assert result == "task:task_002"
