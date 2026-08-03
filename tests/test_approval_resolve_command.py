"""审批文本兜底命令测试（2.3）。"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from core.approval.approval_manager import ApprovalManager
from core.command_handlers.approval_resolve import (
    ApprovalListCommand,
    ApprovalResolveCommand,
)
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
def mgr():
    return ApprovalManager(api_client=MagicMock(), admin_ids=["admin_001"])


@pytest.fixture
def bot_engine(mgr):
    eng = MagicMock()
    eng.approval_manager = mgr
    return eng


@pytest.mark.asyncio
async def test_resolve_command_short_id(bot_engine, mgr):
    cmd = ApprovalResolveCommand(bot_engine=bot_engine)
    future = asyncio.get_running_loop().create_future()
    mgr._pending["approval:chat:exec:abc12345"] = future
    replies = await cmd.execute(
        _msg("admin_001", "/审批 abc allow-once"), "abc allow-once"
    )
    assert "已处理" in replies[0]["content"]
    assert future.result() == "allow-once"


@pytest.mark.asyncio
async def test_resolve_command_bad_decision(bot_engine, mgr):
    cmd = ApprovalResolveCommand(bot_engine=bot_engine)
    replies = await cmd.execute(
        _msg("admin_001", "/审批 abc maybe"), "abc maybe"
    )
    assert "无效决策" in replies[0]["content"]


@pytest.mark.asyncio
async def test_resolve_command_unknown_id(bot_engine, mgr):
    cmd = ApprovalResolveCommand(bot_engine=bot_engine)
    replies = await cmd.execute(
        _msg("admin_001", "/审批 nope deny"), "nope deny"
    )
    assert "不存在或已超时" in replies[0]["content"]


@pytest.mark.asyncio
async def test_list_command_empty(bot_engine, mgr):
    cmd = ApprovalListCommand(bot_engine=bot_engine)
    replies = await cmd.execute(_msg("admin_001", "/审批列表"), "")
    assert "没有待审批" in replies[0]["content"]


@pytest.mark.asyncio
async def test_list_command_with_pending(bot_engine, mgr):
    cmd = ApprovalListCommand(bot_engine=bot_engine)
    future = asyncio.get_running_loop().create_future()
    key = "approval:chat:exec:abc12345"
    mgr._pending[key] = future
    mgr._pending_info[key] = {
        "tool_name": "exec",
        "details": "vim x.txt",
        "created_at": 0,
        "expires_at": 10**9,
    }
    replies = await cmd.execute(_msg("admin_001", "/审批列表"), "")
    assert key in replies[0]["content"]
    assert "vim x.txt" in replies[0]["content"]


@pytest.mark.asyncio
async def test_command_not_initialized():
    cmd = ApprovalResolveCommand(bot_engine=None)
    replies = await cmd.execute(_msg("admin_001", "/审批 x allow"), "x allow")
    assert "未初始化" in replies[0]["content"]
