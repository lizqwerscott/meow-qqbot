"""exec 工具新审批链路端到端测试（OpenClaw 风格策略面）。

覆盖角色策略归一、allowlist 命中直跑、miss 审批/拒绝、
strictInlineEval 强制审批、security=deny 等核心行为。
后台执行走 mock process_registry，不真跑命令。
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.approval.approval_manager import ApprovalManager
from core.managers.permission_manager import PermissionManager
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.exec_process import create_exec_process_entries
from core.tools.ref import Ref

ADMIN = "TEST_ADMIN_ID"
TRUSTED = "TEST_TRUSTED_ID"


@pytest.fixture
def deps(tmp_path):
    # 审批文件指向临时目录，防止 allow-always 测试污染真实 config/
    import core.approval.approval_manager as am

    original_path = am.WHITELIST_PATH
    am.WHITELIST_PATH = str(tmp_path / "approval_whitelist.json")
    try:
        yield _build_deps()
    finally:
        am.WHITELIST_PATH = original_path


def _build_deps():
    pm = PermissionManager(path="nonexistent-allowlist.toml")
    pm._data = {
        "roles": {"admin": [ADMIN], "system": ["system"], "trusted": [TRUSTED]},
        "tools": {},
        "commands": {
            "allowed": [
                "ls",
                "grep",
                "git",
                "python3",
                "false",
                "true",
                "echo",
                "tr",
                "printf",
                "wc",
            ]
        },
        "security": {},
        "exec": {
            "mode": "ask",
            "security": "allowlist",
            "ask": "on-miss",
            "ask_fallback": "deny",
            "strict_inline_eval": True,
        },
    }
    am = ApprovalManager(api_client=MagicMock(), admin_ids=[ADMIN])
    am._whitelist = {
        "version": 2,
        "defaults": {},
        "file_paths": [],
        "exec_commands": [{"command": "sudo"}],
        "allowlist": [{"pattern": "sudo", "source": "legacy"}],
    }
    deps = ToolDeps()
    deps.permission_manager = pm
    deps.approval_manager = Ref()
    deps.approval_manager.value = am
    registry = MagicMock()
    registry.spawn = AsyncMock(return_value="session-abc")
    deps.process_registry = Ref()
    deps.process_registry.value = registry
    return deps


def _ctx(sender, is_group=False):
    return ToolContext(
        chat_id="c1",
        is_group=is_group,
        reply_to="",
        sender_id=sender,
        reply_callback=lambda *a, **k: None,
        delivery_channel="",
        reply_to_message_id="",
    )


async def _exec(deps, command, sender, is_group=False, background=True):
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": command, "background": background}, _ctx(sender, is_group)
    )
    return json.loads(result.content)


# ── 角色策略归一 ──


async def test_system_full_allows_blocked_command(deps):
    # system 固定 full：黑名单命令直接执行
    r = await _exec(deps, "sudo whoami", "system")
    assert "error" not in r


async def test_admin_allowlist_hit_runs_without_approval(deps):
    r = await _exec(deps, "ls -la", ADMIN)
    assert "error" not in r


async def test_admin_miss_fallback_deny(deps):
    # vim 不在白名单 → 审批发送失败 → ask_fallback=deny → 拒绝
    r = await _exec(deps, "vim x.txt", ADMIN)
    assert "error" in r


async def test_trusted_allowlist_hit_runs(deps):
    r = await _exec(deps, "grep foo bar", TRUSTED)
    assert "error" not in r


async def test_trusted_miss_rejected_no_approval(deps):
    # trusted 固定 allowlist+off：miss 直接拒，不弹审批
    r = await _exec(deps, "vim x.txt", TRUSTED)
    assert "error" in r


async def test_trusted_blocked_in_commands_whitelist_rejected(deps):
    # sudo 在审批白名单但不在 [commands].allowed → 安全策略拒绝
    r = await _exec(deps, "sudo whoami", TRUSTED)
    assert "error" in r


# ── strictInlineEval ──


async def test_admin_inline_eval_requires_approval_and_not_whitelisted(deps):
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "python3 -c 'print(1)'", ADMIN)
        assert "error" not in r
        # inline-eval 的 allow-once 不落白名单
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "python3" not in patterns


async def test_inline_eval_allow_always_downgraded(deps):
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "python3 -c 'print(1)'", ADMIN)
        assert "error" not in r
        # strictInlineEval：allow-always 降级为 allow-once，不落白名单
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "python3" not in patterns


# ── 审批仅限 admin 私聊 ──


async def test_admin_group_foreground_miss_rejected(deps):
    # 群聊前台不弹审批（审批卡片仅 c2c）→ miss 直接拒绝
    r = await _exec(deps, "vim x.txt", ADMIN, is_group=True, background=False)
    assert "error" in r


async def test_admin_group_background_approval_requested(deps):
    # 后台执行（background=true）对齐 OpenClaw：群聊也走审批流，
    # 审批卡投递 admin c2c，通过后 spawn，不再"审批不到直接失败"。
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN, is_group=True)
    assert "error" not in r
    assert r.get("background") is True


async def test_admin_group_background_approval_denied(deps):
    # 后台 + 群聊：审批卡投递失败 → ask_fallback=deny → 拒绝（不真跑）
    r = await _exec(deps, "vim x.txt", ADMIN, is_group=True)
    assert "error" in r


# ── security=deny ──


async def test_security_deny_blocks_all(deps):
    deps.permission_manager._data["exec"]["security"] = "deny"
    r = await _exec(deps, "ls -la", ADMIN)
    assert "error" in r and "策略禁用" in r["error"]


# ── allow-always 落白名单（非 inline）──


async def test_allow_always_persists_entry(deps):
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" not in r
        # vim 落白名单（bare-name）
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "vim" in patterns
        # 下次直接命中，无需审批
        r2 = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" not in r2


# ── mode=auto（auto-reviewer）──


async def test_mode_auto_reviewer_allows_without_approval(deps):
    deps.permission_manager._data["exec"]["mode"] = "auto"
    from core.approval.auto_reviewer import ExecAutoReviewer

    reviewed = []

    async def review_fn(plan):
        reviewed.append(plan["command"])
        return "allow"  # reviewer 判定为低风险，放行一次

    deps.exec_reviewer = Ref()
    deps.exec_reviewer.value = ExecAutoReviewer(review_fn=review_fn)

    # miss 命令 → reviewer 直接放行（不触发人工审批）
    r = await _exec(deps, "vim x.txt", ADMIN)
    assert "error" not in r
    assert reviewed == ["vim x.txt"]
    # reviewer 放行不落白名单
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "vim" not in patterns


async def test_mode_auto_reviewer_ask_falls_back_to_human(deps):
    deps.permission_manager._data["exec"]["mode"] = "auto"
    from core.approval.auto_reviewer import ExecAutoReviewer

    async def review_fn(plan):
        return "ask"  # reviewer 判定不确定 → 转人工

    deps.exec_reviewer = Ref()
    deps.exec_reviewer.value = ExecAutoReviewer(review_fn=review_fn)

    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" not in r  # 人工审批通过


async def test_mode_auto_inline_eval_skips_reviewer(deps):
    deps.permission_manager._data["exec"]["mode"] = "auto"
    from core.approval.auto_reviewer import ExecAutoReviewer

    reviewed = []

    async def review_fn(plan):
        reviewed.append(plan["command"])
        return "allow"

    deps.exec_reviewer = Ref()
    deps.exec_reviewer.value = ExecAutoReviewer(review_fn=review_fn)

    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "python3 -c 'print(1)'", ADMIN)
        assert "error" not in r
    # inline-eval 直接转人工，reviewer 不被调用
    assert reviewed == []


async def test_mode_auto_reviewer_not_injected_falls_back_to_human(deps):
    deps.permission_manager._data["exec"]["mode"] = "auto"
    # 不注入 exec_reviewer → 降级人工审批
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" not in r  # 人工审批兜底


async def test_mode_auto_group_chat_never_reviewed(deps):
    deps.permission_manager._data["exec"]["mode"] = "auto"
    from core.approval.auto_reviewer import ExecAutoReviewer

    reviewed = []

    async def review_fn(plan):
        reviewed.append(plan["command"])
        return "allow"

    deps.exec_reviewer = Ref()
    deps.exec_reviewer.value = ExecAutoReviewer(review_fn=review_fn)

    # 群聊前台：reviewer 被跳过（审批/审查仅限 c2c）→ 直接拒绝
    r = await _exec(deps, "vim x.txt", ADMIN, is_group=True, background=False)
    assert "error" in r
    assert reviewed == []


async def test_mode_auto_group_background_reviewed(deps):
    # 后台执行对齐 OpenClaw：群聊也走 auto-review（模型判定，不依赖聊天面）
    deps.permission_manager._data["exec"]["mode"] = "auto"
    from core.approval.auto_reviewer import ExecAutoReviewer

    reviewed = []

    async def review_fn(plan):
        reviewed.append(plan["command"])
        return "allow"

    deps.exec_reviewer = Ref()
    deps.exec_reviewer.value = ExecAutoReviewer(review_fn=review_fn)

    r = await _exec(deps, "vim x.txt", ADMIN, is_group=True)
    assert "error" not in r
    assert reviewed == ["vim x.txt"]


# ── durable plan 绑定（审批通过后执行前比对）──


async def test_approval_plan_mismatch_rejected(deps):
    """审批通过后 plan 被篡改 → APPROVAL_MISMATCH 拒绝执行。"""
    from core.approval.exec_policy import DECISION_ALLOW_ONCE

    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                # 模拟审批期间命令被替换（对齐 openclaw caller 篡改）
                deps.approval_manager.value._pending_plans[key] = {
                    "command": "rm -rf /",
                    "cwd": "/",
                    "resolved_path": "/bin/rm",
                }
                deps.approval_manager.value.resolve(key, DECISION_ALLOW_ONCE, ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" in r
        assert "APPROVAL_MISMATCH" in r["error"]


async def test_approval_plan_matches_executes(deps):
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" not in r  # plan 未变 → 正常执行


# ── 链式命令前台执行（分析-执行绑定）──


async def test_chain_command_foreground_runs_segmentwise(deps):
    # 非 background：走 run_plan 段级执行，&& / 管道真实生效
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": "ls -la | grep -c total"}, _ctx(ADMIN)
    )
    r = json.loads(result.content)
    assert "error" not in r
    assert r["success"] is True
    # ls 的输出被 grep 消费，最终只有 grep 的计数
    assert r["stdout"].strip().isdigit()


async def test_chain_short_circuit_foreground(deps):
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": "false && echo should_not_appear"}, _ctx(ADMIN)
    )
    r = json.loads(result.content)
    assert "should_not_appear" not in r.get("stdout", "")
    assert r["exit_code"] == 1


async def test_chain_background_rejected(deps):
    r = await _exec(deps, "ls && grep foo", ADMIN)
    assert "error" in r
    assert "后台" in r["error"]


# ── 嵌套命令（CST：$(...) 内部命令参与 allowlist）──


async def test_nested_substitution_miss_goes_to_approval(deps):
    # cat 在白名单，但 $(pwd) 的 pwd 不在 → 嵌套 miss → 审批
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        entries = create_exec_process_entries(deps)
        exec_entry = next(e for e in entries if e.name == "exec")
        result = await exec_entry.handler({"command": "cat $(pwd)/x.txt"}, _ctx(ADMIN))
        r = json.loads(result.content)
        # 审批通过 → 前台执行（cat 收到字面参数 $(pwd)/x.txt，文件不存在报错但命令跑了）
        assert "error" not in r
        assert r["success"] is False  # 文件不存在 → cat 退出非 0


async def test_shell_payload_inner_blocked(deps):
    # bash -c 'rm -rf /'：bash 不在白名单且 payload 内 rm 是黑名单 → 拒绝
    r = await _exec(deps, "bash -c 'rm -rf /'", ADMIN, is_group=True)
    assert "error" in r


async def test_payload_denied_reason_penetrates_nested(deps):
    """黑名单穿透嵌套段：bash -c 的 payload 内 rm 计入 deny_reason。"""
    # bash 在白名单（先 allow-always 一次），但 payload 内 rm 是黑名单 → 拒绝
    deps.approval_manager.value._whitelist["allowlist"].append(
        {"pattern": "bash", "source": "allow-always"}
    )
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            return True  # 不 resolve，模拟等待（但 deny 分支不应触发审批）

        FakeSender.return_value.send = fake_send
        entries = create_exec_process_entries(deps)
        exec_entry = next(e for e in entries if e.name == "exec")
        # 顶层 bash 命中 allowlist 但 deny_reason（payload rm）存在：
        # admin 群聊 → 不审批 → 拒绝
        result = await exec_entry.handler(
            {"command": "bash -c 'rm -rf /'"}, _ctx(ADMIN, is_group=True)
        )
        r = json.loads(result.content)
        assert "error" in r
