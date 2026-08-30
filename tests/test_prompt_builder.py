"""测试 _is_dirty — PII 检测与内部模式过滤。

直接从模块导入私有函数 _is_dirty 和 _DIRTY_PATTERNS。
这是少数情况下测试私有函数是合理的，因为它是纯 PII 过滤逻辑。
"""

from types import SimpleNamespace

import pytest

from core.engine.dynamic_context.memory import _DIRTY_PATTERNS, _is_dirty
from core.engine.prompt_builder import PromptBuilder

# ── 内部模式 ──


@pytest.mark.asyncio
async def test_system_event_work_plan_consumer_uses_agent_tool_profile(monkeypatch):
    profiles = []

    def fake_build_tools(profile, context, *, deps):
        profiles.append(profile)
        return [{"function": {"name": "work_plan"}}]

    monkeypatch.setattr("core.engine.prompt_builder.build_tools", fake_build_tools)
    builder = PromptBuilder.__new__(PromptBuilder)
    builder.hindsight = None
    builder._workspace_manager = None
    builder._has_tasks = False
    builder._deps = SimpleNamespace()
    builder._system_events = None

    messages, tools = await builder.build_system_event_messages(
        "background result", work_plan_consumer=True
    )

    assert profiles == ["work_plan"]
    assert tools == [{"function": {"name": "work_plan"}}]
    assert "WorkPlan 结果消费" in messages[0]["content"]

    await builder.build_system_event_messages("cron result")
    assert profiles == ["work_plan", "cron"]


def test_internal_pattern_detectable():
    assert _is_dirty("这是一段 <available_skills 的内容")


def test_skill_tag_detectable():
    assert _is_dirty("here <skill>some skill</skill>")


def test_normal_text_not_dirty():
    assert not _is_dirty("今天天气真不错")


# ── PII 检测 ──


def test_id_card_detected():
    assert _is_dirty("身份证号 110101199001011234 请保密")


def test_phone_detected():
    assert _is_dirty("手机号 13800138000 已注册")


def test_email_detected():
    assert _is_dirty("邮箱 test@example.com 已验证")


def test_short_number_not_phone():
    assert not _is_dirty("数字 12345 不是手机号")


def test_known_dirty_prompt_patterns():
    for pattern in _DIRTY_PATTERNS:
        assert _is_dirty(pattern)


# ── 边界情况 ──


def test_empty_string():
    assert not _is_dirty("")


def test_only_whitespace():
    assert not _is_dirty("   \n\t  ")
