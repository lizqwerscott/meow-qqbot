"""审批文本兜底命令（2.3）— 卡片失败/超时后的人工处理入口。

- `审批 <id> allow-once|allow-always|deny`：处理待审批请求（支持唯一前缀匹配）
- `审批列表`：查看所有待审批请求（含剩余超时秒数）
"""

import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)

_VALID_DECISIONS = ("allow-once", "allow-always", "deny")


@command(
    name="审批",
    aliases=["approve", "approval-resolve"],
    permission="admin",
    description="处理待审批请求：审批 <id> allow-once|allow-always|deny",
)
class ApprovalResolveCommand:
    def __init__(self, bot_engine=None):
        self.bot_engine = bot_engine

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if self.bot_engine is None or self.bot_engine.approval_manager is None:
            return make_reply(input_message, "❌ 审批系统未初始化")

        parts = args.split()
        if len(parts) < 2:
            return make_reply(
                input_message,
                "用法：审批 <id> allow-once|allow-always|deny\n"
                "（id 可用 审批列表 查看，支持唯一前缀）",
            )
        session_key, decision = parts[0], parts[1].lower()
        if decision == "allow":
            decision = "allow-once"  # 宽容输入：allow ≈ allow-once
        if decision not in _VALID_DECISIONS:
            return make_reply(
                input_message,
                f"❌ 无效决策: {decision}（支持 allow-once / allow-always / deny）",
            )

        mgr = self.bot_engine.approval_manager
        ok = mgr.resolve(session_key, decision, input_message.sender_id)
        if ok:
            _log.info(
                "文本审批已处理: %s.. decision=%s by %s..",
                session_key[:20],
                decision,
                input_message.sender_id[:12],
            )
            return make_reply(
                input_message, f"✅ 已处理审批 {session_key}: {decision}"
            )
        return make_reply(
            input_message,
            f"⚠️ 审批 {session_key} 不存在或已超时（可先执行 审批列表 查看）",
        )


@command(
    name="审批列表",
    aliases=["pending-approvals", "approvals"],
    permission="admin",
    description="列出所有待审批请求",
)
class ApprovalListCommand:
    def __init__(self, bot_engine=None):
        self.bot_engine = bot_engine

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if self.bot_engine is None or self.bot_engine.approval_manager is None:
            return make_reply(input_message, "❌ 审批系统未初始化")

        pending = self.bot_engine.approval_manager.list_pending()
        if not pending:
            return make_reply(input_message, "📭 当前没有待审批的请求")

        lines = [f"📋 共 {len(pending)} 个待审批请求："]
        for p in pending:
            remaining = p.get("remaining_secs")
            remain_text = f"{remaining}s" if remaining is not None else "?"
            details = (p.get("details") or "")[:50]
            lines.append(
                f"- `{p['session_key']}` | {p.get('tool_name', '')} | "
                f"{details} | 剩余 {remain_text}"
            )
        lines.append("\n处理：审批 <id> allow-once|allow-always|deny")
        return make_reply(input_message, "\n".join(lines))
