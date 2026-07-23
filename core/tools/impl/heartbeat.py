import json
import logging

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import get_dep

_log = logging.getLogger(__name__)


async def _heartbeat_respond(args: dict, ctx: ToolContext) -> ToolResult:
    hb_resp = get_dep("_heartbeat_response")
    notify = bool(args.get("notify", False))
    notification_text = (args.get("notification_text") or "").strip()
    if hb_resp is not None:
        if hb_resp.get("recorded"):
            _log.warning("heartbeat_respond 在同一轮中被重复调用，忽略")
            return ToolResult(
                content=json.dumps({"success": False, "error": "already recorded for this turn"}),
                no_reply=not notify,
            )
        hb_resp["notify"] = notify
        hb_resp["notification_text"] = notification_text
        hb_resp["outcome"] = args.get("outcome", "")
        hb_resp["summary"] = args.get("summary", "")
        hb_resp["priority"] = args.get("priority", "normal")
        hb_resp["next_check"] = args.get("next_check", "")
        hb_resp["recorded"] = True
    else:
        _log.warning(
            "heartbeat_respond 在非心跳上下文中被调用，响应将被丢弃: "
            f"notify={notify} text={notification_text[:80]!r}"
        )
    _log.info(
        "心跳响应: notify=%s outcome=%s priority=%s text=%s",
        notify, args.get("outcome", ""), args.get("priority", "normal"),
        notification_text[:80],
    )
    return ToolResult(
        content=json.dumps({"success": True, "acknowledged": True}, ensure_ascii=False),
        no_reply=not notify,
    )


HEARTBEAT_PARAMS = {
    "type": "object",
    "properties": {
        "notify": {
            "type": "boolean",
            "description": "是否需要发送通知。false=无需关注，true=需要提醒",
        },
        "notification_text": {
            "type": "string",
            "description": "通知文本，不超过 300 字。仅在 notify=true 时需要",
        },
        "outcome": {
            "type": "string",
            "enum": ["no_change", "progress", "done", "blocked", "needs_attention"],
            "description": "本轮检查的结果状态",
        },
        "summary": {
            "type": "string",
            "description": "本轮检查的简要描述，1-2 句话",
        },
        "priority": {
            "type": "string",
            "enum": ["low", "normal", "high"],
            "description": "通知优先级，默认 normal",
        },
        "next_check": {
            "type": "string",
            "description": "建议下次检查的时间，如 '30m'、'1h' 或自然语言描述",
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
