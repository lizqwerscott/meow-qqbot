import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import _DEPS

_log = logging.getLogger(__name__)

_CRON_ALLOWED: frozenset = frozenset({
    "announce", "search_user",
    "memory", "mark_important",
    "read_file", "write_file", "edit_file", "apply_patch",
    "execute_command",
    "view_skill", "execute_skill", "rescan_skills",
})


def _parse_iso_datetime(s: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return None


def _find_cron_job(job_id: str):
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return None
    job = cron_job_manager.get_job(job_id)
    if job is None:
        matched = cron_job_manager.find_jobs_by_name(job_id)
        if matched:
            job = matched[0]
    return job


# ── Cron actions ──

async def _cron_add(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps(
            {"error": "定时任务系统未就绪"}, ensure_ascii=False,
        ))

    name = (args.get("name") or "").strip()
    cron_expression = (args.get("cron_expression") or "").strip()
    at_str = (args.get("at") or "").strip()
    prompt = (args.get("prompt") or "").strip()
    session_mode = (args.get("session_mode") or "isolated").strip()
    session_id = (args.get("session_id") or "").strip()
    payload_type = (args.get("payload_type") or "message").strip()
    payload_command = (args.get("command") or "").strip()
    payload_model = (args.get("model") or "").strip() or None
    payload_thinking = (args.get("thinking") or "").strip() or None
    enable_notify = bool(args.get("enable_notify", True))

    tools_allow = args.get("tools_allow")
    if tools_allow is not None:
        if not isinstance(tools_allow, list):
            return ToolResult(content=json.dumps(
                {"error": "tools_allow 必须是字符串数组"}, ensure_ascii=False,
            ))
        invalid_names = [n for n in tools_allow if n not in _CRON_ALLOWED and n != "*"]
        if invalid_names:
            return ToolResult(content=json.dumps({
                "error": (
                    f"以下工具不允许用于定时任务: {', '.join(invalid_names)}。"
                    f"可用工具: {', '.join(sorted(_CRON_ALLOWED))}"
                ),
            }, ensure_ascii=False))
        if "*" in tools_allow:
            tools_allow = ["*"]
        else:
            tools_allow = [n for n in tools_allow if n != "*"]

    if not name:
        return ToolResult(content=json.dumps({"error": "name 不能为空"}, ensure_ascii=False))
    if not prompt and payload_type not in ("command", "system_event"):
        return ToolResult(content=json.dumps({"error": "prompt 不能为空"}, ensure_ascii=False))
    if not cron_expression and not at_str:
        return ToolResult(content=json.dumps(
            {"error": "cron_expression 和 at 必须至少提供一个"}, ensure_ascii=False,
        ))

    at_ts = None
    if at_str:
        at_ts = _parse_iso_datetime(at_str)
        if at_ts is None:
            return ToolResult(content=json.dumps(
                {"error": f"时间格式无法解析: {at_str}"}, ensure_ascii=False,
            ))

    valid_modes = {"isolated", "custom", "main"}
    if session_mode not in valid_modes:
        session_mode = "isolated"
    custom_session_id = session_id if session_mode == "custom" else None

    valid_payloads = {"message", "command", "system_event"}
    if payload_type not in valid_payloads:
        payload_type = "message"
    if payload_type == "command" and not payload_command:
        return ToolResult(content=json.dumps(
            {"error": "payload_type=command 时 command 不能为空"}, ensure_ascii=False,
        ))

    if payload_type == "message":
        payload_command = ""
    elif payload_type == "command":
        prompt = ""
    elif payload_type == "system_event":
        payload_command = ""

    job = await cron_job_manager.create_job(
        name=name, cron_expression=cron_expression, prompt=prompt,
        at=at_ts, delivery_channel=ctx.chat_id, is_group=ctx.is_group,
        session_mode=session_mode, custom_session_id=custom_session_id,
        payload_type=payload_type, command=payload_command,
        model=payload_model, thinking=payload_thinking,
        enable_notify=enable_notify, tools_allow=tools_allow,
    )

    payload_labels = {
        "message": "AI 消息", "command": "Shell 命令", "system_event": "系统事件",
    }
    if job.is_one_shot:
        desc = f"🕐 一次性{payload_labels.get(payload_type, '任务')}「{name}」已创建！将在 {at_str} 执行。"
    else:
        desc = f"定时{payload_labels.get(payload_type, '任务')}「{name}」已创建！"
    if payload_type == "command":
        desc += f"\n命令: `{payload_command[:80]}`"
    if tools_allow is not None and payload_type == "message":
        if tools_allow == ["*"]:
            desc += "\n工具权限: 所有 cron 允许的工具"
        else:
            desc += f"\n工具权限: {', '.join(tools_allow)}"
    mode_desc = {
        "isolated": "每次执行使用全新隔离 session",
        "custom": f"在命名 session cron:{custom_session_id} 中执行（跨运行保留上下文）",
        "main": "在专用通道 cron:main 中执行",
    }.get(session_mode, "")
    if mode_desc:
        desc += f"\nSession 模式: {mode_desc}"

    return ToolResult(content=json.dumps({
        "success": True, "job_id": job.id[:16], "name": job.name,
        "cron_expression": job.cron_expression or "", "at": at_str,
        "session_mode": session_mode, "session_id": job.custom_session_id or "",
        "payload_type": payload_type, "command": payload_command or "",
        "model": payload_model or "", "thinking": payload_thinking or "",
        "tools_allow": tools_allow, "message": desc,
    }, ensure_ascii=False))


async def _cron_update(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps(
            {"error": "定时任务系统未就绪"}, ensure_ascii=False,
        ))
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return ToolResult(content=json.dumps({"error": "job_id 不能为空"}, ensure_ascii=False))
    job = _find_cron_job(job_id)
    if job is None:
        return ToolResult(content=json.dumps({"error": f"未找到定时任务: {job_id}"}, ensure_ascii=False))
    old_name = job.name
    changed = []

    for field, setter in _UPDATERS.items():
        if field in args:
            setter(job, args[field], changed)

    if not changed:
        return ToolResult(content=json.dumps({"error": "未提供要修改的字段"}, ensure_ascii=False))
    await cron_job_manager.update_job(job)
    return ToolResult(content=json.dumps({
        "success": True, "job_id": job.id[:16], "name": job.name,
        "changed": changed, "tools_allow": job.tools_allow,
        "message": f"定时任务「{old_name}」已更新: {', '.join(changed)}",
    }, ensure_ascii=False))


async def _cron_remove(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps({"error": "定时任务系统未就绪"}, ensure_ascii=False))
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return ToolResult(content=json.dumps({"error": "job_id 不能为空"}, ensure_ascii=False))
    job = _find_cron_job(job_id)
    if job is None:
        return ToolResult(content=json.dumps({"error": f"未找到定时任务: {job_id}"}, ensure_ascii=False))
    name = job.name
    await cron_job_manager.delete_job(job.id)
    return ToolResult(content=json.dumps({
        "success": True, "job_id": job.id[:16], "name": name,
        "message": f"定时任务「{name}」已删除",
    }, ensure_ascii=False))


async def _cron_enable(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps({"error": "定时任务系统未就绪"}, ensure_ascii=False))
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return ToolResult(content=json.dumps({"error": "job_id 不能为空"}, ensure_ascii=False))
    job = _find_cron_job(job_id)
    if job is None:
        return ToolResult(content=json.dumps({"error": f"未找到定时任务: {job_id}"}, ensure_ascii=False))
    if job.enabled:
        return ToolResult(content=json.dumps({
            "success": True, "job_id": job.id[:16], "name": job.name,
            "message": f"定时任务「{job.name}」已是启用状态",
        }, ensure_ascii=False))
    success = await cron_job_manager.enable_job(job.id)
    return ToolResult(content=json.dumps({
        "success": success, "job_id": job.id[:16], "name": job.name,
        "message": f"定时任务「{job.name}」已启用",
    }, ensure_ascii=False))


async def _cron_disable(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps({"error": "定时任务系统未就绪"}, ensure_ascii=False))
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return ToolResult(content=json.dumps({"error": "job_id 不能为空"}, ensure_ascii=False))
    job = _find_cron_job(job_id)
    if job is None:
        return ToolResult(content=json.dumps({"error": f"未找到定时任务: {job_id}"}, ensure_ascii=False))
    if not job.enabled:
        return ToolResult(content=json.dumps({
            "success": True, "job_id": job.id[:16], "name": job.name,
            "message": f"定时任务「{job.name}」已是暂停状态",
        }, ensure_ascii=False))
    success = await cron_job_manager.disable_job(job.id)
    return ToolResult(content=json.dumps({
        "success": success, "job_id": job.id[:16], "name": job.name,
        "message": f"定时任务「{job.name}」已暂停",
    }, ensure_ascii=False))


async def _cron_list(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps({"error": "定时任务系统未就绪"}, ensure_ascii=False))
    jobs = cron_job_manager.list_jobs()
    if not jobs:
        return ToolResult(content=json.dumps({"jobs": [], "message": "暂无定时任务"}, ensure_ascii=False))
    result = [{
        "id": j.id, "name": j.name, "cron_expression": j.cron_expression or "",
        "at": j.at, "enabled": j.enabled, "next_run_at": j.next_run_at,
        "is_one_shot": j.is_one_shot, "session_mode": j.session_mode,
        "custom_session_id": j.custom_session_id or "",
        "payload_type": j.payload_type,
        "prompt": j.prompt[:100] if j.prompt else "",
        "command": j.command[:100] if j.command else "",
        "tools_allow": j.tools_allow,
    } for j in jobs]
    return ToolResult(content=json.dumps({"jobs": result, "total": len(result)}, ensure_ascii=False))


async def _cron_get(args: dict, ctx: ToolContext) -> ToolResult:
    cron_job_manager = _DEPS.get("cron_job_manager")
    if not cron_job_manager:
        return ToolResult(content=json.dumps({"error": "定时任务系统未就绪"}, ensure_ascii=False))
    job_id = (args.get("job_id") or "").strip()
    if not job_id:
        return ToolResult(content=json.dumps({"error": "job_id 不能为空"}, ensure_ascii=False))
    job = _find_cron_job(job_id)
    if job is None:
        return ToolResult(content=json.dumps({"error": f"未找到定时任务: {job_id}"}, ensure_ascii=False))
    return ToolResult(content=json.dumps({
        "id": job.id, "name": job.name,
        "cron_expression": job.cron_expression or "",
        "at": job.at, "enabled": job.enabled,
        "next_run_at": job.next_run_at,
        "is_one_shot": job.is_one_shot,
        "session_mode": job.session_mode,
        "custom_session_id": job.custom_session_id or "",
        "payload_type": job.payload_type,
        "prompt": job.prompt if job.prompt else "",
        "command": job.command if job.command else "",
        "tools_allow": job.tools_allow,
    }, ensure_ascii=False))


async def _cron(args: dict, ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").strip()
    match action:
        case "add":     return await _cron_add(args, ctx)
        case "update":  return await _cron_update(args, ctx)
        case "remove":  return await _cron_remove(args, ctx)
        case "enable":  return await _cron_enable(args, ctx)
        case "disable": return await _cron_disable(args, ctx)
        case "list":    return await _cron_list(args, ctx)
        case "get":     return await _cron_get(args, ctx)
        case _:
            return ToolResult(content=json.dumps(
                {"error": f"未知 action: {action}，可用: add, update, remove, enable, disable, list, get"},
                ensure_ascii=False,
            ))


# ── Task actions ──

async def _task_cancel(args: dict, ctx: ToolContext) -> ToolResult:
    task_manager = _DEPS.get("task_manager")
    system_events = _DEPS.get("system_events")
    if not task_manager:
        return ToolResult(content=json.dumps(
            {"error": "任务系统未就绪"}, ensure_ascii=False,
        ))
    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return ToolResult(content=json.dumps({"error": "task_id 不能为空"}, ensure_ascii=False))
    task = task_manager.get_task(task_id)
    if task is None:
        tasks = task_manager.list_tasks(limit=50)
        matched = [t for t in tasks if t.id.startswith(task_id)]
        if not matched:
            return ToolResult(content=json.dumps({"error": f"未找到任务: {task_id}"}, ensure_ascii=False))
        task_id = matched[0].id
    success = await task_manager.cancel_task(task_id)
    if success:
        if system_events:
            system_events.enqueue(
                session_key=ctx.chat_id, text="任务已取消", context_key=f"task:{task_id}",
            )
        return ToolResult(content=json.dumps({
            "success": True, "task_id": task_id[:16],
            "message": f"任务 {task_id[:12]}.. 已取消。",
        }, ensure_ascii=False))
    return ToolResult(content=json.dumps({"error": f"无法取消任务 {task_id[:12]}.."}, ensure_ascii=False))


async def _task_list(args: dict, ctx: ToolContext) -> ToolResult:
    task_manager = _DEPS.get("task_manager")
    if not task_manager:
        return ToolResult(content=json.dumps({"error": "任务系统未就绪"}, ensure_ascii=False))
    status_str = (args.get("status") or "").strip().lower()
    limit = min(args.get("limit") or 20, 50)
    if not isinstance(limit, int) or limit < 1:
        limit = 20
    status_filter = None
    if status_str:
        from core.tasks.models import TaskStatus as TS
        try:
            status_filter = TS(status_str)
        except ValueError:
            pass
    tasks = task_manager.list_tasks(limit=limit, status=status_filter)
    if not tasks:
        return ToolResult(content=json.dumps({"tasks": [], "message": "暂无任务记录"}, ensure_ascii=False))
    result = [{
        "id": t.id, "type": t.type, "status": t.status.value,
        "created_at": t.created_at, "started_at": t.started_at,
        "finished_at": t.finished_at, "job_id": t.job_id,
        "prompt": t.prompt[:100] if t.prompt else "",
        "result": t.result[:200] if t.result else None,
        "error": t.error[:200] if t.error else None,
    } for t in tasks]
    return ToolResult(content=json.dumps({"tasks": result, "total": len(result)}, ensure_ascii=False))


async def _task(args: dict, ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").strip()
    match action:
        case "list":   return await _task_list(args, ctx)
        case "cancel": return await _task_cancel(args, ctx)
        case _:
            return ToolResult(content=json.dumps(
                {"error": f"未知 action: {action}，可用: list, cancel"},
                ensure_ascii=False,
            ))


# ── Updaters for cron update ──

def _updater_name(job, val, changed):
    job.name = (val or "").strip()
    changed.append("name")

def _updater_cron(job, val, changed):
    job.cron_expression = (val or "").strip()
    job.at = None
    changed.append("cron_expression")

def _updater_at(job, val, changed):
    at_str = (val or "").strip()
    if at_str:
        at_ts = _parse_iso_datetime(at_str)
        if at_ts is None:
            return
        job.at = at_ts
        job.cron_expression = ""
        changed.append("at")

def _updater_prompt(job, val, changed):
    job.prompt = (val or "").strip()
    changed.append("prompt")

def _updater_enabled(job, val, changed):
    job.enabled = bool(val)
    changed.append("enabled")

def _updater_session_mode(job, val, changed):
    mode = (val or "").strip()
    if mode in {"isolated", "custom", "main"}:
        job.session_mode = mode
        changed.append("session_mode")

def _updater_session_id(job, val, changed):
    sid = (val or "").strip()
    if job.session_mode == "custom":
        job.custom_session_id = sid
        changed.append("session_id")

def _updater_payload_type(job, val, changed):
    pt = (val or "").strip()
    if pt in {"message", "command", "system_event"}:
        job.payload_type = pt
        changed.append("payload_type")

def _updater_command(job, val, changed):
    job.command = (val or "").strip()
    changed.append("command")

def _updater_model(job, val, changed):
    job.model = (val or "").strip() or None
    changed.append("model")

def _updater_thinking(job, val, changed):
    job.thinking = (val or "").strip() or None
    changed.append("thinking")

def _updater_enable_notify(job, val, changed):
    job.enable_notify = bool(val)
    changed.append("enable_notify")

def _updater_tools_allow(job, val, changed):
    if val is None:
        job.tools_allow = None
        changed.append("tools_allow")
    elif isinstance(val, list):
        invalid_names = [n for n in val if n not in _CRON_ALLOWED and n != "*"]
        if invalid_names:
            return
        job.tools_allow = ["*"] if "*" in val else [n for n in val if n != "*"]
        changed.append("tools_allow")

_UPDATERS = {
    "name": _updater_name,
    "cron_expression": _updater_cron,
    "at": _updater_at,
    "prompt": _updater_prompt,
    "enabled": _updater_enabled,
    "session_mode": _updater_session_mode,
    "session_id": _updater_session_id,
    "payload_type": _updater_payload_type,
    "command": _updater_command,
    "model": _updater_model,
    "thinking": _updater_thinking,
    "enable_notify": _updater_enable_notify,
    "tools_allow": _updater_tools_allow,
}


# ── Shared job fields schema ──

_CRON_JOB_FIELDS = {
    "name": {"type": "string", "description": "任务的名字，方便管理和查找，如'早安提醒'、'新年提醒'"},
    "cron_expression": {"type": "string", "description": "周期性 cron 表达式（北京时间 CST/UTC+8，与 at 二选一）。例如：'0 8 * * *' 表示北京时间每天早上8点"},
    "at": {"type": "string", "description": "一次性执行时间，ISO 8601 格式（北京时间 CST/UTC+8）。例如：'2027-01-01T08:00:00+08:00'"},
    "prompt": {"type": "string", "description": "AI 要执行的指令。仅在 payload_type=message 时有效且必填。"},
    "session_mode": {"type": "string", "enum": ["isolated", "custom", "main"], "description": "任务执行所在的 session 模式。默认为 isolated。"},
    "session_id": {"type": "string", "description": "custom 模式下使用的命名 session ID。"},
    "payload_type": {"type": "string", "enum": ["message", "command", "system_event"], "description": "任务载荷类型。默认为 message。"},
    "command": {"type": "string", "description": "shell 命令。仅在 payload_type=command 时有效且必填。"},
    "model": {"type": "string", "description": "AI 模型覆盖，仅对 message 载荷有效。"},
    "thinking": {"type": "string", "enum": ["off", "low", "medium", "high"], "description": "AI 思考级别覆盖。"},
    "enable_notify": {"type": "boolean", "description": "是否投递执行结果到频道。默认为 true。"},
    "tools_allow": {
        "type": ["array", "null"], "items": {"type": "string"},
        "description": "指定该定时任务可用的工具列表。设置为 ['*'] 可使用全部 cron 允许的工具。",
    },
}

CRON_PARAMS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string", "enum": ["add", "update", "remove", "enable", "disable", "list", "get"],
            "description": "操作类型：add 创建 | update 修改 | remove 删除 | enable 启用 | disable 暂停 | list 列出所有 | get 查看单个",
        },
        "job_id": {
            "type": "string",
            "description": "定时任务 ID（update/remove/enable/disable/get 时必填，支持名称前缀模糊匹配）",
        },
        **_CRON_JOB_FIELDS,
    },
    "required": ["action"],
    "description": (
        "定时/一次性任务管理工具。所有时间均为北京时间 (CST/UTC+8)。\n\n"
        "ACTION 说明：\n"
        "- add: 创建新定时任务，需提供 name + (cron_expression 或 at) + prompt\n"
        "- update: 修改已有任务，提供 job_id + 要改的字段\n"
        "- remove: 删除任务，提供 job_id\n"
        "- enable: 启用已暂停的任务，提供 job_id\n"
        "- disable: 暂停任务，提供 job_id\n"
        "- list: 列出所有定时任务，无需额外参数\n"
        "- get: 查看单个任务详情，提供 job_id\n"
    ),
}

TASK_PARAMS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string", "enum": ["list", "cancel"],
            "description": "操作类型：list 列出后台任务执行记录 | cancel 取消正在运行或等待中的任务",
        },
        "task_id": {
            "type": "string",
            "description": "要取消的任务 ID（完整 ID 或前 12 位短 ID，仅在 action=cancel 时必填）",
        },
        "status": {
            "type": "string",
            "description": "按状态过滤（仅在 action=list 时生效）。可选: pending, running, success, failed, cancelled, timeout",
        },
        "limit": {
            "type": "integer",
            "description": "返回数量上限，默认 20（仅在 action=list 时生效）",
        },
    },
    "required": ["action"],
}


def _register_all(register):
    register(ToolEntry(
        name="cron",
        section="cron",
        description="定时/一次性任务管理：创建、修改、删除、启用、暂停、列出、查看单个定时任务。所有时间均为北京时间 (CST/UTC+8)。",
        parameters=CRON_PARAMS,
        handler=_cron,
    ))
    register(ToolEntry(
        name="task",
        section="task",
        description="后台任务管理：列出后台任务执行记录、取消正在运行或等待中的任务。",
        parameters=TASK_PARAMS,
        handler=_task,
    ))
