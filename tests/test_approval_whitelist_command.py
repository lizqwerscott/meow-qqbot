"""审批白名单管理命令测试（2.4）。"""

from unittest.mock import MagicMock

import pytest

from core.approval.approval_manager import ApprovalManager
from core.command_handlers.approval_whitelist import ApprovalWhitelistCommand
from core.message import InputMessage


def _msg(sender, content):
    return InputMessage(
        id="m1",
        sender_id=sender,
        chat_id="c1",
        content=content,
        is_group=False,
    )


@pytest.fixture
def am(tmp_path, monkeypatch):
    import core.approval.approval_manager as m

    monkeypatch.setattr(m, "WHITELIST_PATH", str(tmp_path / "wl.json"))
    return ApprovalManager(api_client=MagicMock(), admin_ids=["admin_001"])


def test_list_empty(am):
    cmd = ApprovalWhitelistCommand(approval_manager=am)

    async def run():
        return await cmd.execute(_msg("admin_001", "/审批白名单"), "")

    import asyncio

    replies = asyncio.run(run())
    assert "白名单为空" in replies[0]["content"]


def test_list_entries_with_uses(am):
    am.add_to_whitelist("exec", "python3 script.py")
    am._whitelist["allowlist"][-1]["uses"] = 3
    cmd = ApprovalWhitelistCommand(approval_manager=am)

    import asyncio

    replies = asyncio.run(cmd.execute(_msg("admin_001", "/审批白名单"), ""))
    assert "python3" in replies[0]["content"]
    assert "使用 3 次" in replies[0]["content"]


def test_delete_entry(am):
    am.add_to_whitelist("exec", "vim x.txt")
    cmd = ApprovalWhitelistCommand(approval_manager=am)

    import asyncio

    replies = asyncio.run(
        cmd.execute(_msg("admin_001", "/审批白名单 删除 vim"), "删除 vim")
    )
    assert "已删除" in replies[0]["content"]
    assert not any(e["pattern"] == "vim" for e in am._whitelist["allowlist"])


def test_delete_missing(am):
    cmd = ApprovalWhitelistCommand(approval_manager=am)

    import asyncio

    replies = asyncio.run(
        cmd.execute(_msg("admin_001", "/审批白名单 删除 nope"), "删除 nope")
    )
    assert "未找到" in replies[0]["content"]


def test_not_initialized():
    cmd = ApprovalWhitelistCommand(approval_manager=None)

    import asyncio

    replies = asyncio.run(cmd.execute(_msg("admin_001", "/审批白名单"), ""))
    assert "未初始化" in replies[0]["content"]
