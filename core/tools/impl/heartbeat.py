import json
import logging
from contextvars import ContextVar

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)

heartbeat_response: ContextVar[dict | None] = ContextVar(
    "heartbeat_response", default=None
)


def create_heartbeat_entries(deps: ToolDeps) -> list[ToolEntry]:

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
            "deliver_to_user": {
                "type": "string",
                "description": "投递目标 chat_id。设置后通知发到该用户的聊天而不是管理员 DM。仅当 notify=true 时生效",
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

    async def _heartbeat_respond(args: dict, ctx: ToolContext) -> ToolResult:
        hb_resp = heartbeat_response.get()
        notify = bool(args.get("notify", False))
        notification_text = (args.get("notification_text") or "").strip()
        if hb_resp is not None:
            if hb_resp.get("recorded"):
                _log.warning("heartbeat_respond 在同一轮中被重复调用，忽略")
                return ToolResult(
                    content=json.dumps(
                        {"success": False, "error": "already recorded for this turn"}
                    ),
                    no_reply=not notify,
                )
            hb_resp["notify"] = notify
            hb_resp["notification_text"] = notification_text
            hb_resp["deliver_to_user"] = args.get("deliver_to_user", "")
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
            notify,
            args.get("outcome", ""),
            args.get("priority", "normal"),
            notification_text[:80],
        )
        return ToolResult(
            content=json.dumps(
                {"success": True, "acknowledged": True}, ensure_ascii=False
            ),
            no_reply=not notify,
        )

    return [
        ToolEntry(
            name="heartbeat_respond",
            section="heartbeat",
            description="回应心跳/系统事件检查。notify=false 表示无需关注；notify=true 时附带提醒内容。如需将结果直接告知用户，设置 deliver_to_user 为目标 chat_id。",
            parameters=HEARTBEAT_PARAMS,
            handler=_heartbeat_respond,
        ),
    ]
