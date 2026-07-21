import json
import logging

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import _HEARTBEAT_RESPONSE

_log = logging.getLogger(__name__)


async def _heartbeat_respond(args: dict, ctx: ToolContext) -> ToolResult:
    _HEARTBEAT_RESPONSE.update({
        "notify": bool(args.get("notify", False)),
        "notification_text": (args.get("notification_text") or "").strip(),
    })
    _log.info(
        f"心跳响应: notify={_HEARTBEAT_RESPONSE['notify']} "
        f"text={_HEARTBEAT_RESPONSE['notification_text'][:80]!r}"
    )
    return ToolResult(content=json.dumps({
        "success": True,
        "acknowledged": True,
    }, ensure_ascii=False))


HEARTBEAT_PARAMS = {
    "type": "object",
    "properties": {
        "notify": {
            "type": "boolean",
            "description": "是否发送通知。false=无需关注，true=需要提醒",
        },
        "notification_text": {
            "type": "string",
            "description": "提醒文本，不超过 300 字。仅在 notify=true 时需要",
        },
    },
    "required": ["notify"],
}


def _register_all(register):
    register(ToolEntry(
        name="heartbeat_respond",
        section="heartbeat",
        description="回应心跳检查。notify=false 表示本次心跳无需要关注的事项；notify=true 时附带提醒内容。",
        parameters=HEARTBEAT_PARAMS,
        handler=_heartbeat_respond,
    ))
