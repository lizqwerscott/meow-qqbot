"""测试工具选择管线（core/tools/policy.py）—— 尤其是 task tools_allow 过滤。

覆盖：未知工具名告警、空 tools_allow 回退、通配符。
防止 cron 配置静默失效问题回归。
"""

import logging

import pytest

from core.tools.catalog import CRON_ALLOWED, PROFILES
from core.tools.policy import (
    ChatContext,
    _filter_task_allow,
    build_tools,
    filter_internal_control_tools,
)


def _ctx(**kw) -> ChatContext:
    defaults = dict(
        has_hindsight=True,
        has_workspace=True,
        has_skills=True,
        has_web=True,
        is_group=False,
        has_users=True,
        has_tasks=True,
    )
    defaults.update(kw)
    return ChatContext(**defaults)


# ── tools_allow 过滤 ──


def test_filter_task_allow_unknown_name_dropped():
    # 未知工具名被忽略（告警而非崩溃），已知工具保留
    result = _filter_task_allow({"exec", "process"}, ["exec", "no_such_tool"])
    assert result == {"exec"}


def test_filter_task_allow_empty_falls_back_to_announce():
    assert _filter_task_allow({"exec", "process"}, []) == {"announce"}


def test_filter_task_allow_wildcard_keeps_all():
    names = {"exec", "process", "announce"}
    assert _filter_task_allow(names, ["*"]) == names


def test_internal_profiles_and_task_allowlist_exclude_mark_important():
    for profile in ("heartbeat", "cron", "task"):
        assert "mark_important" not in PROFILES[profile]
    assert "mark_important" not in CRON_ALLOWED


def test_internal_control_filter_removes_only_mark_important():
    tools = [
        {"type": "function", "function": {"name": "mark_important"}},
        {"type": "function", "function": {"name": "memory"}},
    ]

    assert filter_internal_control_tools(tools) == [tools[1]]


@pytest.mark.parametrize("profile", ["heartbeat", "cron", "task"])
def test_restricted_profiles_do_not_gain_attachment_read_file(profile):
    from core.tools.impl import registry

    saved = dict(registry._tools)
    try:
        registry._tools.clear()
        from core.tools._types import ToolEntry

        registry.register(
            ToolEntry(
                name="read_file",
                description="read",
                parameters={},
                handler=lambda *_: None,
            )
        )
        tools = build_tools(profile, _ctx(has_media=True, has_workspace=False))
        assert "read_file" not in {tool["function"]["name"] for tool in tools}
    finally:
        registry._tools.clear()
        registry._tools.update(saved)


def test_filter_task_allow_unknown_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _filter_task_allow({"exec"}, ["bogus_tool"])
    assert any("bogus_tool" in r.message for r in caplog.records)


def test_filter_task_allow_only_allowed_names_passed():
    # 白名单只允许 announce → 仅 announce
    assert _filter_task_allow({"exec", "process", "announce"}, ["announce"]) == {
        "announce"
    }
