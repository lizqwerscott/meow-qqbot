"""测试工具选择管线（core/tools/policy.py）—— 尤其是 task tools_allow 过滤。

覆盖：未知工具名告警、空 tools_allow 回退、通配符。
防止 cron 配置静默失效问题回归。
"""

import logging

import pytest

from core.tools.policy import ChatContext, _filter_task_allow


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


def test_filter_task_allow_unknown_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _filter_task_allow({"exec"}, ["bogus_tool"])
    assert any("bogus_tool" in r.message for r in caplog.records)


def test_filter_task_allow_only_allowed_names_passed():
    # 白名单只允许 announce → 仅 announce
    assert _filter_task_allow({"exec", "process", "announce"}, ["announce"]) == {
        "announce"
    }
