import json
import logging

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import _DEPS

_log = logging.getLogger(__name__)


async def _spawn_subagent(args: dict, ctx: ToolContext) -> ToolResult:
    sub_agent_manager = _DEPS.get("sub_agent_manager")
    if not sub_agent_manager:
        return ToolResult(content=json.dumps(
            {"error": "子智能体系统未就绪"}, ensure_ascii=False,
        ))
    task = (args.get("task") or "").strip()
    if not task:
        return ToolResult(content=json.dumps(
            {"error": "任务指令不能为空"}, ensure_ascii=False,
        ))
    context = args.get("context", "isolated")
    if context not in ("isolated",):
        context = "isolated"
    result = await sub_agent_manager.spawn(
        parent_chat_id=ctx.chat_id, task=task, context=context,
    )
    return ToolResult(content=json.dumps(result, ensure_ascii=False))


async def _subagents(args: dict, ctx: ToolContext) -> ToolResult:
    sub_agent_manager = _DEPS.get("sub_agent_manager")
    if not sub_agent_manager:
        return ToolResult(content=json.dumps(
            {"error": "子智能体系统未就绪"}, ensure_ascii=False,
        ))
    action = (args.get("action") or "list").strip()
    if action == "cancel":
        subagent_id = (args.get("subagent_id") or "").strip()
        if not subagent_id:
            return ToolResult(content=json.dumps(
                {"error": "subagent_id 不能为空"}, ensure_ascii=False,
            ))
        result = await sub_agent_manager.cancel(subagent_id)
        return ToolResult(content=json.dumps(result, ensure_ascii=False))
    status = (args.get("status") or "").strip() or None
    records = await sub_agent_manager.get_records(
        parent_chat_id=ctx.chat_id, status=status,
    )
    return ToolResult(content=json.dumps({
        "action": "list", "subagents": records, "total": len(records),
    }, ensure_ascii=False))


async def _announce(args: dict, ctx: ToolContext) -> ToolResult:
    sub_agent_manager = _DEPS.get("sub_agent_manager")
    system_events = _DEPS.get("system_events")
    if not sub_agent_manager or not system_events:
        return ToolResult(content=json.dumps(
            {"error": "通知系统未就绪"}, ensure_ascii=False,
        ))
    if not ctx.chat_id.startswith("subagent:"):
        return ToolResult(content=json.dumps(
            {"error": "只有子智能体可以使用此工具"}, ensure_ascii=False,
        ))
    message = (args.get("message") or "").strip()
    if not message:
        return ToolResult(content=json.dumps(
            {"error": "消息不能为空"}, ensure_ascii=False,
        ))
    sub_id = ctx.chat_id.split("subagent:", 1)[1]
    record = await sub_agent_manager.get_record_by_id(sub_id)
    if not record:
        return ToolResult(content=json.dumps(
            {"error": "未找到子智能体记录"}, ensure_ascii=False,
        ))
    parent_chat_id = record.get("parent_chat_id")
    if not parent_chat_id:
        return ToolResult(content=json.dumps(
            {"error": "未找到父会话"}, ensure_ascii=False,
        ))
    system_events.enqueue(
        session_key=parent_chat_id,
        text=f"子智能体 [{sub_id[:8]}..] 进度: {message}",
        context_key=f"subagent_announce:{sub_id}",
        replace=True,
    )
    return ToolResult(content=json.dumps({"success": True}, ensure_ascii=False))


SPAWN_SUBAGENT_PARAMS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "子智能体要执行的详细任务指令。越详细越好，AI 会根据指令独立完成任务。",
        },
        "context": {
            "type": "string",
            "enum": ["isolated"],
            "description": "上下文模式。isolated（默认）使用全新隔离上下文。",
        },
    },
    "required": ["task"],
}

SUBAGENTS_PARAMS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "cancel"],
            "description": "操作类型。list（默认）列出子智能体，cancel 取消子智能体。",
        },
        "status": {
            "type": "string",
            "enum": ["running", "completed", "failed", "timeout", "cancelled"],
            "description": "仅 action=list 时有效。按状态过滤（可选）。不传则返回全部。",
        },
        "subagent_id": {
            "type": "string",
            "description": "仅 action=cancel 时必填。要取消的子智能体 ID。",
        },
    },
}

ANNOUNCE_PARAMS = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": "要报告给父会话的消息内容，建议简洁明了",
        },
    },
    "required": ["message"],
}


def _register_all(register):
    register(ToolEntry(
        name="spawn_subagent",
        section="sub_agent",
        description=(
            "创建一个子智能体在后台独立执行任务，不阻塞当前对话。"
            "子智能体有独立的会话和工具，执行完成后结果会通过系统事件通知你。"
            "适合执行耗时的研究、批量查询、文件处理等任务。"
        ),
        parameters=SPAWN_SUBAGENT_PARAMS,
        handler=_spawn_subagent,
    ))
    register(ToolEntry(
        name="subagents",
        section="sub_agent",
        description="列出或取消子智能体。action=list（默认）列出当前会话的所有子智能体状态；action=cancel 按 subagent_id 取消指定的子智能体。",
        parameters=SUBAGENTS_PARAMS,
        handler=_subagents,
    ))
    register(ToolEntry(
        name="announce",
        section="sub_agent",
        description="向父会话报告当前子智能体的进度或中间结果。父 AI 在下一轮对话时会看到这条消息。适合汇报阶段性进展、发现的异常或请求帮助。",
        parameters=ANNOUNCE_PARAMS,
        handler=_announce,
    ))
