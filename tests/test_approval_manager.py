import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.approval.approval_manager import ApprovalManager, WHITELIST_PATH


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
    am._whitelist = {"file_paths": [{"path": "/home/user/project"}], "exec_commands": []}
    assert am.check_whitelist("read_file", "/home/user/project/main.py") is True


def test_check_whitelist_file_path_exact(am):
    am._whitelist = {"file_paths": [{"path": "/data/config.toml"}], "exec_commands": []}
    assert am.check_whitelist("write_file", "/data/config.toml") is True


def test_check_whitelist_file_path_no_match(am):
    am._whitelist = {"file_paths": [{"path": "/home/user/project"}], "exec_commands": []}
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
    path.write_text(json.dumps({
        "file_paths": [{"path": "/data", "approved_at": "2024-01-01"}],
        "exec_commands": [{"command": "ls", "approved_at": "2024-01-01"}],
    }))
    import core.approval.approval_manager as am
    monkeypatch.setattr(am, "WHITELIST_PATH", str(path))
    mgr = ApprovalManager(api_client=MagicMock(), admin_ids=[])
    assert mgr.check_whitelist("read_file", "/data/config.toml") is True
    assert mgr.check_whitelist("exec", "ls -la") is True
