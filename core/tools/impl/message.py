"""send_message 工具 — 向用户发送文本消息"""

import logging

from core.engine.delivery_ledger import DeliveryReceipt
from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)

_SYNTHETIC_ID_PREFIXES = ("wake_", "hb_", "bg_", "subagent:")


def create_message_entries(deps: ToolDeps) -> list[ToolEntry]:

    SEND_MESSAGE_PARAMS = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "要发送给用户的文本内容",
            },
        },
        "required": ["text"],
    }

    async def _send_message(args: dict, ctx: ToolContext) -> ToolResult:
        text = args.get("text", "").strip()
        if not text:
            return ToolResult(content="消息为空")

        reply_to = ctx.reply_to
        if reply_to and reply_to.startswith(_SYNTHETIC_ID_PREFIXES):
            reply_to = ""

        try:
            receipt = await ctx.reply_callback(
                chat_id=ctx.chat_id,
                content=text,
                message_id=reply_to,
                is_group=ctx.is_group,
            )
        except Exception as exc:
            _log.warning(
                "send_message transport failed [%s..]: %s", ctx.chat_id[:12], exc
            )
            receipt = DeliveryReceipt(
                status="failed",
                logical_delivery_id=(
                    f"tool:{ctx.turn_id}:send_message" if ctx.turn_id else ""
                ),
                error_code="transport_exception",
                retryable=True,
            )
        if isinstance(receipt, DeliveryReceipt) and receipt.status not in {
            "accepted",
            "partial",
        }:
            return ToolResult(
                content="消息发送未确认",
                delivery_receipt=receipt,
            )
        _log.info(
            "send_message 已投递 [%s..]: %s",
            ctx.chat_id[:12],
            text[:60],
        )
        return ToolResult(
            content="消息已发送",
            delivery_receipt=(
                receipt if isinstance(receipt, DeliveryReceipt) else None
            ),
        )

    return [
        ToolEntry(
            name="send_message",
            section="message",
            description="向用户发送一条文本消息。当需要回复用户时使用此工具。",
            parameters=SEND_MESSAGE_PARAMS,
            handler=_send_message,
            delivery_kind="message",
        ),
    ]
