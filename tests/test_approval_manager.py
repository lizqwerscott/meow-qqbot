import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.approval.approval_manager import (
    WHITELIST_PATH,
    ApprovalManager,
    _parse_target,
)

# ── fixtures ──


@pytest.fixture
def tmp_whitelist(tmp_path):
    """将 WHITELIST_PATH 指向临时目录。"""
    path = tmp_path / "approval_whitelist.json"
    import core.approval.approval_manager as am

    original = am.WHITELIST_PATH
    am.WHITELIST_PATH = str(path)
    yield str(path)
    am.WHITELIST_PATH = original


@pytest.fixture
def am(tmp_whitelist):
    mgr = ApprovalManager(api_client=MagicMock(), admin_ids=["admin_001"])
    return mgr


# ── check_whitelist ──


def test_check_whitelist_file_path_match(am):
    am._whitelist = {
        "file_paths": [{"path": "/home/user/project"}],
        "exec_commands": [],
    }
    assert am.check_whitelist("read_file", "/home/user/project/main.py") is True


def test_check_whitelist_file_path_exact(am):
    am._whitelist = {"file_paths": [{"path": "/data/config.toml"}], "exec_commands": []}
    assert am.check_whitelist("write_file", "/data/config.toml") is True


def test_check_whitelist_file_path_no_match(am):
    am._whitelist = {
        "file_paths": [{"path": "/home/user/project"}],
        "exec_commands": [],
    }
    assert am.check_whitelist("read_file", "/etc/passwd") is False


def test_check_whitelist_empty_target(am):
    assert am.check_whitelist("read_file", "") is False


def test_check_whitelist_exec_match(am):
    am._whitelist = {"file_paths": [], "exec_commands": [{"command": "ls"}]}
    assert am.check_whitelist("exec", "ls -la /tmp") is True


def test_check_whitelist_exec_no_match(am):
    am._whitelist = {"file_paths": [], "exec_commands": [{"command": "ls"}]}
    assert am.check_whitelist("exec", "rm -rf /") is False


def test_check_whitelist_unknown_tool(am):
    assert am.check_whitelist("unknown_tool", "something") is False


# ── add_to_whitelist ──


def test_add_to_whitelist_file_path(am):
    am.add_to_whitelist("read_file", "/data/logs")
    assert any(e["path"] == "/data/logs" for e in am._whitelist["file_paths"])


def test_add_to_whitelist_file_path_dedup(am):
    am.add_to_whitelist("read_file", "/data/logs")
    am.add_to_whitelist("read_file", "/data/logs")
    assert len(am._whitelist["file_paths"]) == 1


def test_add_to_whitelist_empty_target(am):
    am.add_to_whitelist("read_file", "")
    assert len(am._whitelist.get("file_paths", [])) == 0


def test_add_to_whitelist_exec(am):
    am.add_to_whitelist("exec", "ls -la")
    assert any(e["command"] == "ls" for e in am._whitelist["exec_commands"])


def test_add_to_whitelist_exec_dedup(am):
    am.add_to_whitelist("exec", "ls -la")
    am.add_to_whitelist("exec", "ls /tmp")
    # 命令名相同，去重
    assert len(am._whitelist["exec_commands"]) == 1


def test_add_to_whitelist_exec_empty(am):
    am.add_to_whitelist("exec", "")
    assert len(am._whitelist.get("exec_commands", [])) == 0


# ── resolve ──


@pytest.mark.asyncio
async def test_resolve_admin_approves(am):
    future = asyncio.get_running_loop().create_future()
    session_key = "approval:chat:tool:abc123"
    am._pending[session_key] = future
    assert am.resolve(session_key, "allow", "admin_001") is True
    assert future.result() == "allow"


@pytest.mark.asyncio
async def test_resolve_non_admin_rejected(am):
    future = asyncio.get_running_loop().create_future()
    session_key = "approval:chat:tool:abc123"
    am._pending[session_key] = future
    assert am.resolve(session_key, "allow", "unknown_user") is False
    assert future.done() is False


# ── request_approval (无 admin) ──


@pytest.mark.asyncio
async def test_request_approval_no_admin(am):
    am._admin_ids = set()
    result = await am.request_approval("chat_001", "exec", "需要执行命令")
    assert result == "deny"


# ── 白名单持久化 ──


def test_whitelist_persistence(am, tmp_whitelist):
    am.add_to_whitelist("read_file", "/data")
    am.add_to_whitelist("exec", "python")
    saved = json.loads(Path(tmp_whitelist).read_text())
    assert len(saved["file_paths"]) == 1
    assert len(saved["exec_commands"]) == 1


def test_load_whitelist_from_file(tmp_path, monkeypatch):
    path = tmp_path / "approval_whitelist.json"
    path.write_text(
        json.dumps(
            {
                "file_paths": [{"path": "/data", "approved_at": "2024-01-01"}],
                "exec_commands": [{"command": "ls", "approved_at": "2024-01-01"}],
            }
        )
    )
    import core.approval.approval_manager as am

    monkeypatch.setattr(am, "WHITELIST_PATH", str(path))
    mgr = ApprovalManager(api_client=MagicMock(), admin_ids=[])
    assert mgr.check_whitelist("read_file", "/data/config.toml") is True
    assert mgr.check_whitelist("exec", "ls -la") is True


# ── v2 schema（OpenClaw 风格 allowlist + defaults）──


def test_load_v1_migrates_to_v2(tmp_path, monkeypatch):
    path = tmp_path / "approval_whitelist.json"
    path.write_text(
        json.dumps(
            {
                "file_paths": [],
                "exec_commands": [
                    {"command": "sudo", "approved_at": "2026-07-21T21:59:19"},
                    {"command": "ddgr", "approved_at": "2026-07-24T14:04:56"},
                ],
            }
        )
    )
    import core.approval.approval_manager as am

    monkeypatch.setattr(am, "WHITELIST_PATH", str(path))
    mgr = ApprovalManager(api_client=MagicMock(), admin_ids=[])
    # v1 条目迁移为 allowlist，且原镜像保留
    patterns = [e["pattern"] for e in mgr._whitelist["allowlist"]]
    assert patterns == ["sudo", "ddgr"]
    assert all(e["source"] == "legacy" for e in mgr._whitelist["allowlist"])
    assert mgr._whitelist["version"] == 2
    # 兼容旧检查：v1 迁移后 sudo 仍命中
    assert mgr.check_whitelist("exec", "sudo apt update") is True


def test_get_host_policy_defaults(am):
    policy = am.get_host_policy()
    assert policy.security == "allowlist"
    assert policy.ask == "on-miss"
    assert policy.ask_fallback == "deny"


def test_get_host_policy_from_defaults(am):
    am._whitelist["defaults"] = {"security": "allowlist", "ask": "always"}
    policy = am.get_host_policy()
    assert policy.ask == "always"


def test_get_allowlist_entries(am):
    am._whitelist["allowlist"] = [
        {"pattern": "git", "source": "allow-always"},
        {"pattern": "python3", "arg_pattern": "^safe\\.py$", "source": "manual"},
        {"pattern": ""},  # 空 pattern 跳过
    ]
    entries = am.get_allowlist_entries()
    assert [e.pattern for e in entries] == ["git", "python3"]
    assert entries[1].arg_pattern == "^safe\\.py$"
    assert entries[0].source == "allow-always"


def test_add_to_whitelist_exec_writes_both_schemas(am):
    am.add_to_whitelist("exec", "python3 script.py")
    # v1 镜像
    assert any(e["command"] == "python3" for e in am._whitelist["exec_commands"])
    # v2 allowlist（bare-name pattern）
    entry = next(e for e in am._whitelist["allowlist"] if e["pattern"] == "python3")
    assert entry["source"] == "allow-always"
    assert entry["last_used_command"] == "python3 script.py"


def test_check_whitelist_v2_entry(am):
    am._whitelist["allowlist"] = [{"pattern": "rg", "source": "allow-always"}]
    assert am.check_whitelist("exec", "rg -n TODO") is True


# ── 包装器解包持久化（2.1：allow-always 记内层可执行路径）──


def test_add_to_whitelist_exec_uses_plan_persist_pattern(am):
    # timeout 5 python3 x.py → allow-always 记 python3，不记 timeout
    am.add_to_whitelist(
        "exec", "timeout 5 python3 x.py", plan={"persist_pattern": "python3"}
    )
    patterns = [e["pattern"] for e in am._whitelist["allowlist"]]
    assert "python3" in patterns
    assert "timeout" not in patterns
    # v1 镜像同步写内层命令名
    assert any(e["command"] == "python3" for e in am._whitelist["exec_commands"])


def test_add_to_whitelist_exec_absolute_persist_pattern(am):
    # 内层是非 PATH 二进制 → 绝对路径条目（路径 glob 分支命中）
    am.add_to_whitelist(
        "exec", "timeout 5 /opt/bin/tool", plan={"persist_pattern": "/opt/bin/tool"}
    )
    patterns = [e["pattern"] for e in am._whitelist["allowlist"]]
    assert "/opt/bin/tool" in patterns
    assert any(e["command"] == "tool" for e in am._whitelist["exec_commands"])


def test_add_to_whitelist_exec_no_plan_keeps_legacy_behavior(am):
    # 无 plan（旧调用方）→ 保持现状：裸命令名
    am.add_to_whitelist("exec", "vim x.txt")
    assert any(e["pattern"] == "vim" for e in am._whitelist["allowlist"])
    assert any(e["command"] == "vim" for e in am._whitelist["exec_commands"])


# ── ask_fallback ──


def test_apply_fallback_default_deny(am):
    assert am._apply_fallback("deny", "anything") == "deny"


def test_apply_fallback_full(am):
    assert am._apply_fallback("full", "anything") == "allow"


def test_apply_fallback_allowlist(am):
    am._whitelist["allowlist"] = [{"pattern": "ls", "source": "legacy"}]
    assert am._apply_fallback("allowlist", "ls -la") == "allow"
    assert am._apply_fallback("allowlist", "vim x") == "deny"


@pytest.mark.asyncio
async def test_request_approval_fallback_deny(am):
    # 无 admin → 直接 deny（原行为）
    am._admin_ids = set()
    result = await am.request_approval("chat_001", "exec", "原因")
    assert result == "deny"


# ── 2.3 文本兜底：pending 列表 / 前缀匹配 / 多目标转发 / 超时通知 ──


@pytest.mark.asyncio
async def test_list_pending_empty(am):
    assert am.list_pending() == []


@pytest.mark.asyncio
async def test_request_approval_registers_pending_info(am):
    seen = {}
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(am._pending.items()):
                # 卡已发出但未审批：pending 列表可见（含元信息）
                pend = am.list_pending()
                assert len(pend) == 1
                p = pend[0]
                assert p["session_key"] == key
                assert p["tool_name"] == "exec"
                assert p["details"] == "ls -la"
                assert p["remaining_secs"] <= 60
                seen["key"] = key
                am.resolve(key, "allow-once", "admin_001")
            return True

        FakeSender.return_value.send = fake_send
        result = await am.request_approval(
            "chat_001",
            "exec",
            "命令不在允许列表中",
            details="ls -la",
            timeout=60,
            plan={"command": "ls -la", "cwd": "/"},
            return_session_key=True,
        )
    decision, session_key = result
    assert decision == "allow-once"
    assert session_key == seen["key"]
    # 审批后 pending 清空
    assert am.list_pending() == []


@pytest.mark.asyncio
async def test_request_approval_preserves_runtime_session_key(am):
    session_key = "approval:turn-1:exec:stable-plan"
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            assert session_key in am._pending
            am.resolve(session_key, "allow-once", "admin_001")
            return True

        FakeSender.return_value.send = fake_send
        result = await am.request_approval(
            "chat_001",
            "exec",
            "reason",
            plan={"command": "ls"},
            return_session_key=True,
            session_key=session_key,
        )

    assert result == ("allow-once", session_key)
    future = asyncio.get_running_loop().create_future()
    am._pending["approval:chat:exec:abc12345"] = future
    # 短 id 前缀匹配
    assert am.resolve("approval:chat:exec:abc", "allow", "admin_001") is True
    assert future.result() == "allow"


@pytest.mark.asyncio
async def test_request_approval_does_not_replace_existing_session(am):
    existing = asyncio.get_running_loop().create_future()
    session_key = "approval:turn-1:exec:stable-plan"
    am._pending[session_key] = existing

    result = await am.request_approval(
        "chat_001",
        "exec",
        "reason",
        plan={"command": "ls"},
        return_session_key=True,
        session_key=session_key,
    )

    assert result == ("deny", "")
    assert am._pending[session_key] is existing
    assert not existing.done()


@pytest.mark.asyncio
async def test_resolve_prefix_ambiguous_fails(am):
    f1 = asyncio.get_running_loop().create_future()
    f2 = asyncio.get_running_loop().create_future()
    am._pending["approval:chat:exec:abc111"] = f1
    am._pending["approval:chat:exec:abc222"] = f2
    # 前缀命中多个 → 不处理
    assert am.resolve("approval:chat:exec:abc", "allow", "admin_001") is False
    assert f1.done() is False
    assert f2.done() is False


@pytest.mark.asyncio
async def test_forward_to_sends_to_all_targets(am):
    am._forward_to = ["group:g_001", "c2c:other_admin", "bare_target"]
    sent_to = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            sent_to.append((kw["chat_type"], kw["chat_id"]))
            for key, future in list(am._pending.items()):
                am.resolve(key, "allow-once", "admin_001")
            return True

        FakeSender.return_value.send = fake_send
        result = await am.request_approval("chat_001", "exec", "原因", details="ls")
    assert result == "allow-once"  # 主目标发送成功 + fake_send resolve 结果
    # 主 admin c2c + 3 个转发目标
    assert ("c2c", "admin_001") in sent_to
    assert ("group", "g_001") in sent_to
    assert ("c2c", "other_admin") in sent_to
    assert ("c2c", "bare_target") in sent_to
    assert len(sent_to) == 4


@pytest.mark.asyncio
async def test_forward_all_fail_falls_back(am):
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:
        FakeSender.return_value.send = AsyncMock(return_value=False)
        result = await am.request_approval(
            "chat_001", "exec", "原因", details="vim x", ask_fallback="deny"
        )
    assert result == "deny"
    # fallback 后 pending 清理
    assert am.list_pending() == []


@pytest.mark.asyncio
async def test_timeout_sends_admin_text_notice(am):
    am._api.send_text = AsyncMock()
    # 直接构造 pending（不等真实 60s 超时）
    session_key = "approval:chat:exec:deadbeef"
    future = asyncio.get_running_loop().create_future()
    am._pending[session_key] = future
    am._pending_info[session_key] = {
        "tool_name": "exec",
        "details": "ls -la",
        "created_at": time.time(),
        "expires_at": time.time() + 1,
    }
    am._on_timeout(session_key, fallback="deny", details="ls -la")
    assert future.result() == "deny"  # 超时按 fallback 处理
    await asyncio.sleep(0)  # 让通知 task 跑
    # 文本通知发到 admin c2c，且包含 session key 与审批提示
    assert am._api.send_text.called
    text = am._api.send_text.call_args.args[-1]
    assert session_key in text
    assert "审批" in text
    assert am.list_pending() == []


# ── 2.4 审批管理：删除条目 / 使用计数 / 统计 ──


def test_remove_allowlist_entry(am):
    am.add_to_whitelist("exec", "vim x.txt")
    assert any(e["pattern"] == "vim" for e in am._whitelist["allowlist"])
    assert am.remove_allowlist_entry("vim") is True
    assert not any(e["pattern"] == "vim" for e in am._whitelist["allowlist"])
    # v1 镜像同步清理
    assert not any(e["command"] == "vim" for e in am._whitelist["exec_commands"])


def test_remove_allowlist_entry_missing(am):
    assert am.remove_allowlist_entry("nope") is False


def test_remove_allowlist_entry_source_limited(am):
    am._whitelist["allowlist"] = [
        {"pattern": "git", "source": "manual"},
        {"pattern": "git", "source": "allow-always"},
    ]
    assert am.remove_allowlist_entry("git", source="manual") is True
    remaining = [e for e in am._whitelist["allowlist"] if e["pattern"] == "git"]
    assert [e["source"] for e in remaining] == ["allow-always"]


def test_record_use_counts_and_updates_last_used(am):
    am._whitelist["allowlist"] = [{"pattern": "ls", "source": "allow-always"}]
    am.record_use("ls")
    am.record_use("ls")
    entry = next(e for e in am._whitelist["allowlist"] if e["pattern"] == "ls")
    assert entry["uses"] == 2
    assert entry["last_used_at"] > 0
    # 未注册的 pattern 不报错
    am.record_use("unknown")


def test_record_use_persists_on_next_save(am, tmp_whitelist):
    am._whitelist["allowlist"] = [{"pattern": "ls", "source": "allow-always"}]
    am.record_use("ls")
    # 下一次落盘操作（如 add）携带 uses
    am.add_to_whitelist("exec", "grep foo")
    saved = json.loads(Path(tmp_whitelist).read_text())
    entry = next(e for e in saved["allowlist"] if e["pattern"] == "ls")
    assert entry["uses"] == 1


def test_record_use_flush_writes_pending_count(am, tmp_whitelist):
    # 进程关闭路径：10s 防抖未到也落盘（对齐 nickname flush_save）
    am._whitelist["allowlist"] = [{"pattern": "ls", "source": "allow-always"}]
    am.record_use("ls")
    am.flush()
    saved = json.loads(Path(tmp_whitelist).read_text())
    entry = next(e for e in saved["allowlist"] if e["pattern"] == "ls")
    assert entry["uses"] == 1
    assert entry["last_used_at"] > 0


def test_record_use_flush_idempotent(am, tmp_whitelist):
    am._whitelist["allowlist"] = [{"pattern": "ls", "source": "allow-always"}]
    am.record_use("ls")
    am.flush()
    am.flush()  # dirty 已清，重复 flush 不重复写
    saved = json.loads(Path(tmp_whitelist).read_text())
    entry = next(e for e in saved["allowlist"] if e["pattern"] == "ls")
    assert entry["uses"] == 1


def test_record_use_flush_without_dirty_no_write(am, tmp_whitelist):
    # 无计数 → flush 不创建/不写文件
    am._whitelist["allowlist"] = [{"pattern": "ls", "source": "allow-always"}]
    am.flush()
    assert Path(tmp_whitelist).exists() is False


def test_whitelist_stats(am):
    am._whitelist["allowlist"] = [
        {
            "pattern": "git",
            "source": "allow-always",
            "approved_at": "2026-01-01T00:00:00",
        },
        {"pattern": "ls", "source": "legacy", "approved_at": "2026-02-01T00:00:00"},
        {"pattern": ""},
    ]
    stats = am.whitelist_stats()
    assert stats["count"] == 2
    assert stats["last_allow_always_at"] == "2026-02-01T00:00:00"


def test_whitelist_stats_empty(am):
    stats = am.whitelist_stats()
    assert stats["count"] == 0
    assert stats["last_allow_always_at"] == ""


def test_whitelist_stats_int_timestamp_safe(am):
    # 手改/旧 v1 的 int 时间戳不炸，返回 ISO 字符串
    am._whitelist["allowlist"] = [
        {"pattern": "git", "source": "allow-always", "approved_at": 1700000000}
    ]
    stats = am.whitelist_stats()
    assert stats["count"] == 1
    assert stats["last_allow_always_at"].startswith("2023-")


# ── 修复项：转发目标校验 / host 覆盖接线 / plan 深拷贝 / 链式持久化 ──


def test_parse_target_valid():
    assert _parse_target("group:g_001") == ("group", "g_001")
    assert _parse_target("c2c:other_admin") == ("c2c", "other_admin")
    assert _parse_target("bare_target") == ("c2c", "bare_target")


def test_parse_target_invalid_rejected():
    # 畸形目标（非法前缀 / 空 id / 空串）→ None，不再静默当 chat id
    assert _parse_target("foo:bar") is None
    assert _parse_target("c2c:") is None
    assert _parse_target("group:") is None
    assert _parse_target("") is None
    assert _parse_target("   ") is None


@pytest.mark.asyncio
async def test_forward_skips_invalid_targets(am):
    am._forward_to = ["group:g_001", "foo:bar", "", "c2c:"]
    sent_to = []
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            sent_to.append((kw["chat_type"], kw["chat_id"]))
            for key, future in list(am._pending.items()):
                am.resolve(key, "allow-once", "admin_001")
            return True

        FakeSender.return_value.send = fake_send
        await am.request_approval("chat_001", "exec", "原因", details="ls")
    # 只发主目标 + 合法转发目标
    assert sent_to == [("c2c", "admin_001"), ("group", "g_001")]


def test_get_host_policy_reads_safe_bins_and_timeout(am):
    am._whitelist["defaults"] = {
        "security": "allowlist",
        "safe_bins": ["wc"],
        "safe_bin_profiles": {"wc": {"max_positional": 0}},
        "approval_timeout": 180,
    }
    host = am.get_host_policy()
    assert host.safe_bins == ("wc",)
    assert host.safe_bin_profiles == {"wc": {"max_positional": 0}}
    assert host.approval_timeout == 180


def test_get_host_policy_unset_timeout_is_none(am):
    host = am.get_host_policy()
    assert host.approval_timeout is None


@pytest.mark.asyncio
async def test_pending_plan_stored_deepcopy(am):
    """plan 存储时深拷贝：审批后修改原 plan 不污染 stored（漂移检测真实生效）。"""
    plan = {"command": "ls -la", "cwd": "/", "argv": ["ls", "-la"]}
    with patch("qqbot_agent_sdk.ApprovalSender") as FakeSender:

        async def fake_send(**kw):
            for key, future in list(am._pending.items()):
                am.resolve(key, "allow-once", "admin_001")
            return True

        FakeSender.return_value.send = fake_send
        result = await am.request_approval(
            "chat_001",
            "exec",
            "原因",
            details="ls -la",
            plan=plan,
            return_session_key=True,
        )
    _, session_key = result
    # 审批期间调用方改了自己的 plan 对象（模拟内容漂移）
    plan["command"] = "rm -rf /"
    stored = am.take_pending_plan(session_key)
    assert stored["command"] == "ls -la"  # stored 是独立副本


def test_add_to_whitelist_plan_persist_pattern_list(am):
    # 链式命令：allow-always 持久化所有顶层段
    am.add_to_whitelist(
        "exec", "vim x && head -5", plan={"persist_pattern": ["vim", "head"]}
    )
    patterns = [e["pattern"] for e in am._whitelist["allowlist"]]
    assert "vim" in patterns
    assert "head" in patterns
    assert any(e["command"] == "vim" for e in am._whitelist["exec_commands"])
    assert any(e["command"] == "head" for e in am._whitelist["exec_commands"])
