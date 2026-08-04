"""审批白名单管理命令（2.4）。

- `审批白名单`：列出全部 allowlist 条目（pattern/arg_pattern/source/使用次数）
- `审批白名单 删除 <pattern>`：移除条目（防误授权）
数据仍存 config/approval_whitelist.json（单机 JSON 足够，不迁移 SQLite）。
"""

import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(
    name="审批白名单",
    aliases=["whitelist", "approval-whitelist"],
    permission="admin",
    description="查看/删除审批白名单条目：审批白名单 [删除 <pattern>]",
)
class ApprovalWhitelistCommand:
    def __init__(self, approval_manager=None):
        self.approval_manager = approval_manager

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if self.approval_manager is None:
            return make_reply(input_message, "❌ 审批系统未初始化")

        parts = args.split()
        if parts and parts[0] in ("删除", "del", "remove"):
            if len(parts) < 2:
                return make_reply(input_message, "用法：审批白名单 删除 <pattern>")
            pattern = parts[1]
            ok = self.approval_manager.remove_allowlist_entry(pattern)
            if ok:
                _log.info(
                    "审批白名单已删除: %s (by %s..)",
                    pattern,
                    input_message.sender_id[:12],
                )
                return make_reply(input_message, f"🗑️ 已删除白名单条目: `{pattern}`")
            return make_reply(
                input_message, f"⚠️ 未找到条目: `{pattern}`（可用 审批白名单 查看）"
            )

        entries = self.approval_manager.get_allowlist_entries()
        if not entries:
            return make_reply(input_message, "📭 审批白名单为空")
        lines = [f"📋 审批白名单（{len(entries)} 条）："]
        for e in entries:
            arg = f" | arg=`{e.arg_pattern}`" if e.arg_pattern else ""
            lines.append(f"- `{e.pattern}`{arg} | {e.source} | 使用 {e.uses} 次")
        lines.append("\n删除：审批白名单 删除 <pattern>")
        return make_reply(input_message, "\n".join(lines))
