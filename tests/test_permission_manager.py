from pathlib import Path

import pytest

from core.managers.permission_manager import PermissionManager

# ── fixtures ──


def _write_toml(path: str, content: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


_VALID_ALLOWLIST_TOML = """\
[roles]
admin = ["admin_001"]
trusted = ["trusted_001"]

[tools]
read_file = "all"
write_file = "trusted"
"group:file" = "all"
execute_command = "admin"

[commands]
allowed = ["ls", "cat", "echo", "python"]

[security]
deny_substitution = true
deny_chaining = true
deny_pipe = false
deny_redirect = true
max_command_length = 2000
"""


@pytest.fixture
def default_allowlist(tmp_path):
    path = str(tmp_path / "allowlist.toml")
    _write_toml(path, _VALID_ALLOWLIST_TOML)
    return path


@pytest.fixture
def pm(default_allowlist):
    return PermissionManager(path=default_allowlist)


# ── 角色解析 ──


def test_get_user_role_admin(pm):
    assert pm.get_user_role("admin_001") == "admin"


def test_get_user_role_trusted(pm):
    assert pm.get_user_role("trusted_001") == "trusted"


def test_get_user_role_default(pm):
    assert pm.get_user_role("unknown_user") == "default"


def test_is_admin_role():
    from core.managers.permission_manager import ROLE_LEVEL
    pm = PermissionManager(path="/tmp/nonexistent.toml")
    assert pm.is_admin_role("admin") is True
    assert pm.is_admin_role("trusted") is False
    assert pm.is_admin_role("default") is False


# ── 工具权限 ──


def test_can_use_tool_exact_match(pm):
    assert pm.can_use_tool("read_file", "default") is True
    assert pm.can_use_tool("write_file", "default") is False
    assert pm.can_use_tool("write_file", "trusted") is True


def test_can_use_tool_group_match(pm):
    # file group contains file-related tools
    assert pm.can_use_tool("read_file", "default") is True


def test_can_use_tool_unknown_defaults_admin(pm):
    assert pm.can_use_tool("unknown_tool", "default") is False
    assert pm.can_use_tool("unknown_tool", "admin") is True


def test_can_use_tool_admin_always(pm):
    assert pm.can_use_tool("execute_command", "admin") is True


# ── 命令安全策略 ──


def test_admin_bypasses_all_checks(pm):
    assert pm.check_command_allowed("rm -rf /", ["rm", "-rf", "/"], "admin") is None


def test_empty_command(pm):
    result = pm.check_command_allowed("", [], "default")
    assert result is not None
    assert "命令为空" in result


def test_substitution_dollar_paren(pm):
    result = pm.check_command_allowed("echo $(ls)", ["echo", "$(ls)"], "default")
    assert result is not None
    assert "命令替换" in result


def test_substitution_backtick(pm):
    result = pm.check_command_allowed("echo `ls`", ["echo", "`ls`"], "default")
    assert result is not None
    assert "命令替换" in result


def test_chaining_semicolon(pm):
    result = pm.check_command_allowed("ls; rm -rf /", ["ls;", "rm", "-rf", "/"], "default")
    assert result is not None
    assert "命令串联" in result


def test_chaining_double_ampersand(pm):
    result = pm.check_command_allowed("ls && rm -rf /", ["ls", "&&", "rm"], "default")
    assert result is not None
    assert "命令串联" in result


def test_chaining_pipe_not_chaining(pm):
    """管道 | 本身不是串联符（pipe flow check still applies）。"""
    # cat 在白名单中，所以管道可以通过 flow check
    assert pm.check_command_allowed("echo hello | cat", ["echo", "hello", "|", "cat"], "default") is None


def test_chaining_pipe_inside_segment(pm):
    """管道分段内不允许串联。"""
    result = pm.check_command_allowed("ls; cat | head", ["ls;", "cat", "|", "head"], "default")
    assert result is not None
    assert "命令串联" in result


def test_redirect_denied(pm):
    result = pm.check_command_allowed("echo hello > /tmp/x", ["echo", "hello", ">", "/tmp/x"], "default")
    assert result is not None
    assert "重定向" in result


def test_command_length_exceeded(pm):
    long_cmd = "echo " + "x" * 2000
    result = pm.check_command_allowed(long_cmd, ["echo", "x" * 2000], "default")
    assert result is not None
    assert "过长" in result


def test_command_not_in_whitelist(pm):
    result = pm.check_command_allowed("rm -rf /", ["rm", "-rf", "/"], "default")
    assert result is not None
    assert "白名单" in result


def test_command_in_whitelist(pm):
    assert pm.check_command_allowed("ls -la", ["ls", "-la"], "default") is None


def test_pipe_segment_not_in_whitelist(pm):
    result = pm.check_command_allowed("ls | rm -rf", ["ls", "|", "rm", "-rf"], "default")
    assert result is not None
    assert "管道" in result or "白名单" in result


# ── 缺失配置文件 ──


def test_missing_config_file(tmp_path):
    pm = PermissionManager(path=str(tmp_path / "nonexistent.toml"))
    assert pm.get_user_role("any") == "default"
    assert pm.can_use_tool("anything", "default") is False
    assert pm.check_command_allowed("ls", ["ls"], "default") is not None  # 无白名单


# ── 超时 ──


def test_default_timeout(pm):
    assert pm.get_default_timeout() == 60


def test_max_timeout(pm):
    assert pm.get_max_timeout() == 300


def test_get_role_ids(pm):
    assert "admin_001" in pm.get_role_ids("admin")
    assert "trusted_001" in pm.get_role_ids("trusted")
    assert pm.get_role_ids("unknown_role") == []
