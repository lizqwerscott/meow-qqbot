"""exec 工具新审批链路端到端测试（OpenClaw 风格策略面）。

覆盖角色策略归一、allowlist 命中直跑、miss 审批/拒绝、
strictInlineEval 强制审批、security=deny 等核心行为。
后台执行走 mock process_registry，不真跑命令。
"""

import asyncio
import json
import os
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


def _ctx(sender, is_group=False, transition=None):
    return ToolContext(
        chat_id="c1",
        is_group=is_group,
        reply_to="",
        sender_id=sender,
        reply_callback=lambda *a, **k: None,
        delivery_channel="",
        reply_to_message_id="",
        turn_id="turn-1" if transition is not None else "",
        turn_revision=4,
        principal_id=sender,
        transition_turn=transition,
    )


async def _exec(
    deps,
    command,
    sender,
    is_group=False,
    background=True,
    workdir=None,
    transition=None,
):
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    args = {"command": command, "background": background}
    if workdir is not None:
        args["workdir"] = workdir
    result = await exec_entry.handler(args, _ctx(sender, is_group, transition))
    return json.loads(result.content)


# ── 角色策略归一 ──


async def test_system_full_allows_dangerous_command(deps):
    # system 固定 full：无命令黑名单（对齐 OpenClaw），危险命令直接执行，
    # 安全性由 allowlist/审批层之外的系统信任边界承担
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


async def test_approved_exec_rechecks_role_before_execution(deps):
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key in list(deps.approval_manager.value._pending):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            deps.permission_manager._data["roles"]["admin"] = []
            return True

        FakeSender.return_value.send = fake_send
        result = await _exec(deps, "python3 -c 'print(1)'", ADMIN)

    assert result["error"] == "APPROVAL_ROLE_CHANGED: 审批期间权限已降级"


async def test_exec_approval_request_exception_restores_active_turn(deps):
    transitions = []

    async def transition(**kwargs):
        transitions.append(kwargs)
        return type("State", (), {"revision": kwargs["expected_revision"] + 1})()

    deps.approval_manager.value.request_approval = AsyncMock(
        side_effect=RuntimeError("approval transport failed")
    )

    with pytest.raises(RuntimeError, match="approval transport failed"):
        await _exec(
            deps,
            "vim x.txt",
            ADMIN,
            transition=transition,
        )

    assert [item["phase"].value for item in transitions] == [
        "awaiting_approval",
        "active",
    ]


async def test_exec_approval_cancellation_cancels_turn(deps):
    transitions = []

    async def transition(**kwargs):
        transitions.append(kwargs)
        return type("State", (), {"revision": kwargs["expected_revision"] + 1})()

    deps.approval_manager.value.request_approval = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    with pytest.raises(asyncio.CancelledError):
        await _exec(
            deps,
            "vim x.txt",
            ADMIN,
            transition=transition,
        )

    assert [item["phase"].value for item in transitions] == [
        "awaiting_approval",
        "cancelled",
    ]


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
        # vim 落白名单（持久化解析后的二进制 basename，对齐 openclaw resolved 路径；
        # 如 vim → vim.basic 的机器写 vim.basic，保证条目真正可命中）
        from core.tools.exec_analysis import resolve_executable

        resolved = resolve_executable(["vim"], env=os.environ)
        assert resolved.resolved_path is not None
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert os.path.basename(resolved.resolved_path) in patterns
        # 下次直接命中，无需审批
        r2 = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" not in r2


# ── 包装器解包（2.1）──


async def test_allow_always_wrapper_persists_inner(deps):
    """timeout 5 head -5 的 allow-always 记内层 head，不记 timeout。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "timeout 5 head -5", ADMIN)
        assert "error" not in r
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "head" in patterns
        assert "timeout" not in patterns


async def test_allow_always_nested_wrapper_persists_innermost(deps):
    """两层包装器 allow-always 记最内层真实可执行，不记 timeout/nice。"""
    from core.tools.exec_analysis import resolve_executable

    resolved = resolve_executable(["vim"], env=os.environ)
    assert resolved.resolved_path is not None
    inner_name = os.path.basename(
        resolved.resolved_path
    )  # vim.basic 等 alternatives 真实名
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "timeout 5 nice -n 5 vim x.txt", ADMIN)
        assert "error" not in r
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert inner_name in patterns
        assert "timeout" not in patterns
        assert "nice" not in patterns
        assert "vim" not in patterns


async def test_allow_always_chain_persists_all_segments(deps):
    """链式命令 allow-always 持久化所有顶层段（不再只记第一条）。"""
    from core.tools.exec_analysis import resolve_executable

    # 前台执行（后台不支持链式）。两段都 miss：
    # 绝对路径 /usr/bin/ls 不匹配 bare-name 静态条目；head 不在白名单。
    # 都立即退出（不读 stdin 阻塞），且非 inline（inline 段会禁用持久化）
    ls_path = resolve_executable(["ls"], env=os.environ).resolved_path
    assert ls_path is not None
    cmd = f"{ls_path} && head -5"
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, cmd, ADMIN, background=False)
        assert "error" not in r
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert ls_path in patterns  # 绝对路径条目（非 PATH 解析）
        assert "head" in patterns
        # 下次整条链直接命中，无需审批（前台直跑）
        with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender2:
            r2 = await _exec(deps, cmd, ADMIN, background=False)
            FakeSender2.assert_not_called()  # allowlist 命中 → 不弹审批
        assert "error" not in r2


async def test_wrapper_inner_allowlist_hit_runs_without_approval(deps):
    # 白名单已有 ls（bare-name），timeout 包一层直接命中，不弹审批
    deps.approval_manager.value._whitelist["allowlist"].append(
        {"pattern": "ls", "source": "allow-always"}
    )
    r = await _exec(deps, "timeout 5 ls -la", ADMIN)
    assert "error" not in r


async def test_wrapper_inner_miss_rejected(deps):
    # timeout 5 rm -rf /tmp/x：rm 不在 allowlist → miss → 拒绝（群聊前台不弹卡）
    r = await _exec(deps, "timeout 5 rm -rf /tmp/x", ADMIN, is_group=True)
    assert "error" in r


async def test_wrapper_inline_eval_requires_approval(deps):
    # timeout 包 python3 -c：内层 inline → 强制审批（strictInlineEval）
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "timeout 5 python3 -c 'print(1)'", ADMIN)
        assert "error" not in r
        # inline 的 allow-once 不落白名单
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "python3" not in patterns
        assert "timeout" not in patterns


# ── 2.2 解释器/runtime 绑定（plan 绑定精确 argv + 唯一文件）──


async def test_interp_approval_binds_inner_file(deps, tmp_path):
    """node app.js 审批时绑定脚本 realpath 到 plan。"""
    app = tmp_path / "app.js"
    app.write_text("console.log(1)")
    plans = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                plans.append(deps.approval_manager.value._pending_plans[key])
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, f"node {app}", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
    assert len(plans) == 1
    assert plans[0]["inner_file"] == os.path.realpath(str(app))
    assert plans[0]["interp_unbound"] is False


async def test_interp_unbound_allow_always_not_persisted(deps, tmp_path):
    """node missing.js：无法绑定唯一文件 → 审批通过也不落白名单。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "node missing.js", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "node" not in patterns


async def test_interp_pnpm_exec_binds_local_bin(deps, tmp_path):
    """pnpm exec eslint 解包到 node_modules/.bin/eslint。"""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    shim = bin_dir / "eslint"
    shim.write_text("#!/bin/sh\n# shim")
    plans = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                plans.append(deps.approval_manager.value._pending_plans[key])
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "pnpm exec eslint", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
    assert len(plans) == 1
    assert plans[0]["inner_file"] == os.path.realpath(str(shim))
    assert plans[0]["interp_unbound"] is False


async def test_interp_pnpm_exec_missing_bin_requires_approval(deps, tmp_path):
    """pnpm exec 无本地 bin：interp_unbound → 审批（allow-always 不落盘）。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "pnpm exec eslint", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
        patterns = [
            e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
        ]
        assert "pnpm" not in patterns
        assert "eslint" not in patterns


async def test_interp_npx_flag_allow_always_not_persisted(deps, tmp_path):
    """npx flags 使目标歧义：仅人工一次性审批，不能持久化。"""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "eslint").write_text("#!/bin/sh\n# shim")
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "npx --yes eslint", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "npx" not in patterns
    assert "eslint" not in patterns


async def test_interp_npm_exec_without_double_dash_not_persisted(deps, tmp_path):
    """npm exec 缺少分隔符时仅人工一次性审批，不能持久化。"""
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "eslint").write_text("#!/bin/sh\n# shim")
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "npm exec eslint", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "npm" not in patterns
    assert "eslint" not in patterns


async def test_flock_command_payload_allow_always_not_persisted(deps, tmp_path):
    """flock shell payload 无唯一外层 argv，allow-always 只能一次性执行。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(
            deps,
            "flock /tmp/test.lock --command='echo hello'",
            ADMIN,
            background=False,
            workdir=str(tmp_path),
        )
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "flock" not in patterns


async def test_invalid_timeout_wrapper_requires_approval_and_not_persisted(deps):
    """外层 timeout 已授权也不能让未知 flag 绕过一次性审批。"""
    deps.approval_manager.value._whitelist["allowlist"].append(
        {"pattern": "timeout", "source": "allow-always"}
    )
    calls = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            calls.append(kw)
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "timeout --bogus 5 ls", ADMIN)
        assert "error" not in r
    assert calls
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert patterns.count("timeout") == 1


async def test_nested_flock_payload_allow_always_not_persisted(deps, tmp_path):
    """嵌套 timeout → flock payload 同样不得持久化任一包装器。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(
            deps,
            "timeout 5 flock /tmp/test.lock --command='echo hello'",
            ADMIN,
            background=False,
            workdir=str(tmp_path),
        )
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "timeout" not in patterns
    assert "flock" not in patterns


async def test_shell_c_payload_allow_always_not_persisted(deps, tmp_path):
    """bash -c shell payload 内容不可预测：allow-always 不落盘。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(
            deps,
            "bash -c 'echo hello'",
            ADMIN,
            background=False,
            workdir=str(tmp_path),
        )
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "bash" not in patterns


async def test_nested_shell_c_payload_allow_always_not_persisted(deps, tmp_path):
    """嵌套 timeout → bash -c payload 同样不得持久化任一包装器。"""
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(
            deps,
            "timeout 5 bash -c 'echo hello'",
            ADMIN,
            background=False,
            workdir=str(tmp_path),
        )
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "timeout" not in patterns
    assert "bash" not in patterns


async def test_interp_wrapper_inner_bound(deps, tmp_path):
    """timeout 包 node app.js：解释器绑定看内层（2.1 × 2.2 组合）。"""
    app = tmp_path / "app.js"
    app.write_text("console.log(1)")
    plans = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                plans.append(deps.approval_manager.value._pending_plans[key])
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, f"timeout 5 node {app}", ADMIN, workdir=str(tmp_path))
        assert "error" not in r
    assert plans[0]["inner_file"] == os.path.realpath(str(app))
    assert plans[0]["interp_unbound"] is False


async def test_interp_argv_mismatch_rejected(deps):
    """审批期间 argv 被篡改 → APPROVAL_MISMATCH（2.2 绑定精确 argv 快照）。"""
    from core.approval.exec_policy import DECISION_ALLOW_ONCE

    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                # 模拟审批期间执行方重新构造了不同 argv 的 plan（内容漂移）
                drifted = dict(deps.approval_manager.value._pending_plans[key])
                drifted["argv"] = ["rm", "-rf", "/"]  # 仅篡改 argv
                deps.approval_manager.value._pending_plans[key] = drifted
                deps.approval_manager.value.resolve(key, DECISION_ALLOW_ONCE, ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "vim x.txt", ADMIN)
        assert "error" in r
        assert "APPROVAL_MISMATCH" in r["error"]


async def test_interp_meta_command_allowlist_hit_runs(deps):
    """python3 --version 是元命令：白名单命中直跑，不发起审批。"""
    from core.tools.exec_analysis import resolve_executable

    # 静态 [commands].allowed 的 python3 是输入名；本机 realpath 后可能是
    # python3.11——用解析后的 basename 写入动态白名单（机器无关）
    resolved = resolve_executable(["python3"], env=os.environ)
    assert resolved.resolved_path is not None
    deps.approval_manager.value._whitelist["allowlist"].append(
        {
            "pattern": os.path.basename(resolved.resolved_path),
            "source": "allow-always",
        }
    )
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:
        r = await _exec(deps, "python3 --version", ADMIN)
        FakeSender.assert_not_called()  # 元命令 + allowlist 命中 → 不弹审批
    assert "error" not in r


async def test_interp_meta_command_no_unbound(deps):
    """node --version（不在白名单）走正常审批，不再因 interp_unbound 强制。"""
    plans = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                plans.append(deps.approval_manager.value._pending_plans[key])
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(deps, "node --version", ADMIN)
        assert "error" not in r  # 审批通过（元命令可放行）
    assert len(plans) == 1
    assert plans[0]["interp_unbound"] is False
    assert plans[0]["inner_file"] is None


async def test_interp_multi_script_allow_always_not_persisted(deps, tmp_path):
    """多段解释器无法用单个 inner_file 覆盖：allow-always 不落盘。"""
    (tmp_path / "a.js").write_text("console.log(1)")
    (tmp_path / "b.js").write_text("console.log(2)")
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-always", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(
            deps,
            "node a.js && node b.js",
            ADMIN,
            background=False,
            workdir=str(tmp_path),
        )
        assert "error" not in r
    patterns = [
        e["pattern"] for e in deps.approval_manager.value._whitelist["allowlist"]
    ]
    assert "node" not in patterns
    assert "a.js" not in patterns
    assert "b.js" not in patterns


async def test_interp_multi_script_plan_marks_unbound(deps, tmp_path):
    """多段解释器 plan 记录 multi_interp_target 并保持 interp_unbound。"""
    (tmp_path / "a.js").write_text("console.log(1)")
    (tmp_path / "b.js").write_text("console.log(2)")
    plans = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                plans.append(deps.approval_manager.value._pending_plans[key])
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        r = await _exec(
            deps,
            "node a.js && node b.js",
            ADMIN,
            background=False,
            workdir=str(tmp_path),
        )
        assert "error" not in r
    assert plans[0]["multi_interp_target"] is True
    assert plans[0]["inner_file"] == os.path.realpath(str(tmp_path / "a.js"))


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


async def test_shell_payload_inner_miss_rejected(deps):
    # bash -c 'rm -rf /'：bash 不在白名单 → miss → 拒绝（无黑名单，纯 allowlist 语义）
    r = await _exec(deps, "bash -c 'rm -rf /'", ADMIN, is_group=True)
    assert "error" in r


async def test_payload_inner_miss_blocks_allowlisted_outer(deps):
    """嵌套 miss 阻断外层：bash 已 allowlist，但 payload 内 rm 无条目 → 整段 miss。"""
    # bash 先 allow-always 一次，payload 内 rm 仍无条目 → 嵌套 miss → 拒绝
    deps.approval_manager.value._whitelist["allowlist"].append(
        {"pattern": "bash", "source": "allow-always"}
    )
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            return True  # 不 resolve，模拟等待（但 miss 分支不应触发审批）

        FakeSender.return_value.send = fake_send
        entries = create_exec_process_entries(deps)
        exec_entry = next(e for e in entries if e.name == "exec")
        # 顶层 bash 命中 allowlist 但嵌套 rm miss：
        # admin 群聊 → 不审批 → 拒绝
        result = await exec_entry.handler(
            {"command": "bash -c 'rm -rf /'"}, _ctx(ADMIN, is_group=True)
        )
        r = json.loads(result.content)
        assert "error" in r


# ── exec env 参数（对齐 OpenClaw：模型直接传环境变量，免 `bash -c 'export ...'`）──


async def test_env_param_injected_and_runs(deps):
    # env 参数传给命令（用 python3 读环境变量验证注入）。python3 -c 是 inline
    # eval → 走审批路径（mock 自动 allow-once），环境变量需真实注入子进程。
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        entries = create_exec_process_entries(deps)
        exec_entry = next(e for e in entries if e.name == "exec")
        pycmd = "import os,sys; sys.stdout.write(os.environ.get('FRESHRSS_URL',''))"
        result = await exec_entry.handler(
            {
                "command": "python3 -c " + repr(pycmd),
                "env": {"FRESHRSS_URL": "http://192.168.100.203:8050"},
            },
            _ctx(ADMIN),
        )
        r = json.loads(result.content)
        assert "error" not in r
        assert "192.168.100.203" in r["stdout"]


async def test_env_param_blocked_dangerous_key(deps):
    # 危险键（PYTHONPATH）→ Security Violation：即使命令 allowlist 命中也拒绝
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": "ls -la", "env": {"PYTHONPATH": "/tmp/x"}},
        _ctx(ADMIN),
    )
    r = json.loads(result.content)
    assert "error" in r
    assert "Security Violation" in r["error"]


async def test_env_param_blocked_path(deps):
    # PATH 覆盖 → 拒绝
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": "ls -la", "env": {"PATH": "/evil:/bin"}},
        _ctx(ADMIN),
    )
    r = json.loads(result.content)
    assert "error" in r
    assert "Security Violation" in r["error"]


async def test_env_param_blocked_prefix(deps):
    # LD_ 前缀 → 拒绝
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": "ls -la", "env": {"LD_PRELOAD": "/tmp/x.so"}},
        _ctx(ADMIN),
    )
    r = json.loads(result.content)
    assert "error" in r
    assert "Security Violation" in r["error"]


async def test_env_param_invalid_key(deps):
    # 非法键名 → 拒绝
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {"command": "ls -la", "env": {"1BAD KEY": "x"}},
        _ctx(ADMIN),
    )
    r = json.loads(result.content)
    assert "error" in r
    assert "Security Violation" in r["error"]


async def test_env_param_bound_in_plan(deps):
    # 审批（admin miss 走卡）时 env 进入 plan 绑定，approval 通过后执行比对一致
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(deps.approval_manager.value._pending.items()):
                deps.approval_manager.value.resolve(key, "allow-once", ADMIN)
            return True

        FakeSender.return_value.send = fake_send
        entries = create_exec_process_entries(deps)
        exec_entry = next(e for e in entries if e.name == "exec")
        # vim 不在白名单 → 审批；env 进入 plan
        result = await exec_entry.handler(
            {
                "command": "python3 -c 'print(1)'",
                "env": {"MY_TOOL_KEY": "secret"},
            },
            _ctx(ADMIN),
        )
        r = json.loads(result.content)
        # python3 -c 是 inline → 审批路径；允许通过后 MY_TOOL_KEY 环境已生效
        assert "error" not in r


async def test_env_param_background_reaches_spawn(deps):
    # 后台执行：模型 env 覆盖必须透传到 spawn 的子进程环境。
    # 用 allowlist 命中的 ls（非 inline）→ 直跑，不走审批；断言 spawn 收到 env。
    entries = create_exec_process_entries(deps)
    exec_entry = next(e for e in entries if e.name == "exec")
    result = await exec_entry.handler(
        {
            "command": "ls -la",
            "env": {"FRESHRSS_URL": "http://bg:8050"},
            "background": True,
        },
        _ctx(ADMIN),
    )
    r = json.loads(result.content)
    assert "error" not in r
    assert r.get("background") is True
    # spawn 被调用且 env 含覆盖子集（且是 str 归一后的值）
    spawn_env = deps.process_registry.value.spawn.call_args.kwargs["env"]
    assert spawn_env.get("FRESHRSS_URL") == "http://bg:8050"
    # 危险键仍被拒绝：即使后台也不放行
    result2 = await exec_entry.handler(
        {"command": "ls -la", "env": {"NODE_OPTIONS": "--evil"}, "background": True},
        _ctx(ADMIN),
    )
    assert "Security Violation" in json.loads(result2.content)["error"]


# ── env_override_policy 模块级单测（集中验证安全边界，锁住黑名单行为）──


async def test_validate_env_override_allows_benign_and_coerces_numeric(deps):
    from core.tools.env_override_policy import validate_env_override

    overrides, errors = validate_env_override(
        {"FRESHRSS_URL": "http://x:8050", "MY_PORT": 8080, "MY_FLAG": True}
    )
    # 数字被接收并强转 str；布尔 True 不是 str/int/float → 拒绝并报错
    assert "FRESHRSS_URL" in overrides
    assert "MY_PORT" in overrides and overrides["MY_PORT"] == "8080"
    assert "MY_FLAG" not in overrides
    assert any("MY_FLAG" in e for e in errors)


async def test_validate_env_override_rejects_invalid_and_blocked(deps):
    from core.tools.env_override_policy import validate_env_override

    overrides, errors = validate_env_override(
        {"1BAD K": "x", "PATH": "/evil", "PYTHONPATH": "/p", "LD_PRELOAD": "/x"}
    )
    assert overrides == {}  # 全部被拒，无一注入
    joined = "; ".join(errors)
    assert "非法环境变量键名" in joined
    assert "PATH" in joined
    assert "PYTHONPATH" in joined
    assert "LD_PRELOAD" in joined


async def test_validate_env_override_none_and_non_dict(deps):
    from core.tools.env_override_policy import validate_env_override

    assert validate_env_override(None) == ({}, [])
    assert validate_env_override({}) == ({}, [])
    overrides, errors = validate_env_override("not-a-dict")
    assert overrides == {}
    assert errors == ["env 参数必须是 {KEY: value} 字典"]


async def test_validate_env_override_case_insensitive_block(deps):
    from core.tools.env_override_policy import validate_env_override

    # 键名大小写不敏感：小写 path 也拦（对齐 openclaw 的 uppercase 归一）
    overrides, errors = validate_env_override({"path": "/evil", "paths": "/ok"})
    assert "path" not in overrides
    assert "非法" not in "; ".join(errors)  # 不因小写是"合法标识符"而放行
    assert "paths" in overrides  # 非关键键正常放行
