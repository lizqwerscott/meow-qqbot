"""测试 BackgroundTaskRunner 的静态辅助函数。"""

from core.tasks.delivery_policy import decide_cron_delivery
from core.tasks.models import (
    CronJob,
    DeliveryStatus,
    SessionMode,
    TaskRecord,
    TaskStatus,
)
from core.tasks.runner import BackgroundTaskRunner

# ── _resolve_session_id ──


def test_session_id_isolated_default():
    job = CronJob(name="test", session_mode=SessionMode.ISOLATED.value)
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "task:task_001"


def test_session_id_custom_with_id():
    job = CronJob(
        name="test",
        session_mode=SessionMode.CUSTOM.value,
        custom_session_id="my_session",
    )
    result = BackgroundTaskRunner._resolve_session_id(job, "task_001")
    assert result == "cron:my_session"


def test_session_id_custom_without_id():
    job = CronJob(
        name="test", session_mode=SessionMode.CUSTOM.value, custom_session_id=None
    )
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
    # 无命令黑名单（对齐 OpenClaw）：危险命令只做格式校验，合理性由创建者负责
    assert BackgroundTaskRunner._check_command_safe("shutdown -h now") is None


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


# ── session_target ──


def test_session_target_defaults_to_session_mode():
    """session_target 为空时继承 session_mode。"""
    job = CronJob(name="test", session_mode="main", session_target="")
    assert job.session_target == "main"


def test_session_target_explicit():
    """显式设置 session_target 独立于 session_mode。"""
    job = CronJob(name="test", session_mode="isolated", session_target="main")
    assert job.session_target == "main"


def test_session_target_from_dict_missing_key():
    """旧数据（无 session_target key）向下兼容。"""
    d = {"name": "test", "session_mode": "main"}
    job = CronJob.from_dict(d)
    assert job.session_target == "main"


def test_session_target_from_dict_explicit():
    """新数据保留 session_target。"""
    d = {"name": "test", "session_mode": "isolated", "session_target": "main"}
    job = CronJob.from_dict(d)
    assert job.session_target == "main"


def test_session_target_from_dict_isolated_default():
    """默认 isolated。"""
    d = {"name": "test"}
    job = CronJob.from_dict(d)
    assert job.session_target == "isolated"


def test_session_target_in_to_dict():
    """to_dict 包含 session_target。"""
    job = CronJob(name="test", session_target="main")
    d = job.to_dict()
    assert d.get("session_target") == "main"


# ── _format_result_event_text ──


def _mk_task(result=None, error=None):
    from core.tasks.models import TaskRecord

    return TaskRecord(id="t", prompt="", result=result, error=error)


def test_format_result_event_text_with_result():
    text = BackgroundTaskRunner._format_result_event_text(
        "任务 '早安'已完成", _mk_task(result="头条1\n头条2")
    )
    assert text == "任务 '早安'已完成\n\n执行结果:\n头条1\n头条2"


def test_format_result_event_text_with_error():
    text = BackgroundTaskRunner._format_result_event_text(
        "任务 '早安'执行失败", _mk_task(error="exit 1")
    )
    assert text == "任务 '早安'执行失败\n\n执行结果:\nexit 1"


def test_format_result_event_text_empty_body_keeps_prefix():
    text = BackgroundTaskRunner._format_result_event_text(
        "任务 '早安'已完成", _mk_task()
    )
    assert text == "任务 '早安'已完成"


def test_format_result_event_text_truncates_long_body():
    text = BackgroundTaskRunner._format_result_event_text(
        "P", _mk_task(result="x" * 3000)
    )
    assert text.endswith("…[已截断]")
    # 结果体本身不超过上限（split 后含一个前导换行）
    body = text.split("执行结果:", 1)[1]
    assert len(body) <= 1 + 2000 + len("…[已截断]")


def test_cron_delivery_skips_final_no_reply():
    job = CronJob(name="check", delivery_channel="user")
    task = TaskRecord(status=TaskStatus.SUCCESS, result="NO_REPLY")
    decision = decide_cron_delivery(job, task)
    assert not decision.should_deliver
    assert decision.reason == "silent_final_reply"


def test_cron_delivery_uses_only_final_report():
    job = CronJob(name="check", delivery_channel="user")
    task = TaskRecord(status=TaskStatus.SUCCESS, result="最终报告")
    decision = decide_cron_delivery(job, task)
    assert decision.should_deliver
    assert "最终报告" in decision.content
    assert "检测完成" not in decision.content


def test_cron_delivery_skips_empty_success():
    job = CronJob(name="check", delivery_channel="user")
    task = TaskRecord(status=TaskStatus.SUCCESS)
    decision = decide_cron_delivery(job, task)
    assert not decision.should_deliver


def test_cron_delivery_skips_after_tool_delivery():
    job = CronJob(name="check", delivery_channel="user")
    task = TaskRecord(status=TaskStatus.SUCCESS, result="报告")
    decision = decide_cron_delivery(job, task, tool_delivered=True)
    assert not decision.should_deliver
    assert decision.reason == "already_delivered"


def test_task_record_delivery_status_is_backward_compatible():
    task = TaskRecord.from_dict({"id": "t", "status": "success"})
    assert task.delivery_status == DeliveryStatus.UNKNOWN
    assert task.to_dict()["delivery_status"] == "unknown"
