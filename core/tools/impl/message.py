"""send_message 工具 — 向用户发送文本消息"""

import logging

from core.tools._types import ToolEntry, ToolContext, ToolResult

_log = logging.getLogger(__name__)

_SYNTHETIC_ID_PREFIXES = ("wake_", "hb_", "bg_", "subagent:")

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
        return ToolResult(content="消息为空", sent_text=False)

    # 自动判断回复还是主动：只有真实 QQ 消息 ID 才回复，合成 ID 回退到主动消息
    reply_to = ctx.reply_to
    if reply_to and reply_to.startswith(_SYNTHETIC_ID_PREFIXES):
        reply_to = ""

    await ctx.reply_callback(
        chat_id=ctx.chat_id,
        content=text,
        message_id=reply_to,
        is_group=ctx.is_group,
    )
    _log.info(
        "send_message 已投递 [%s..]: %s",
        ctx.chat_id[:12], text[:60],
    )
    return ToolResult(content="消息已发送", sent_text=True)


def _register_all(register):
    register(ToolEntry(
        name="send_message",
        section="message",
        description="向用户发送一条文本消息。当需要回复用户时使用此工具。",
        parameters=SEND_MESSAGE_PARAMS,
        handler=_send_message,
    ))
