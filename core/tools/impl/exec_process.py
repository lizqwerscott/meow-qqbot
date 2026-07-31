"""exec + process 工具"""

import json
import logging

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps
from core.tools.impl.file import is_admin_private
from core.tools.security import check_command_denied, parse_command_safe
from core.tools.shell_env import build_exec_env_for

_log = logging.getLogger(__name__)


def create_exec_process_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _exec(args: dict, ctx: ToolContext) -> ToolResult:
        process_registry = deps.process_registry.value
        perm = deps.permission_manager
        if not process_registry:
            return ToolResult(
                content=json.dumps(
                    {"error": "进程系统未就绪"},
                    ensure_ascii=False,
                )
            )

        command = (args.get("command") or "").strip()
        if not command:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供要执行的命令"},
                    ensure_ascii=False,
                )
            )

        timeout = args.get("timeout")
        workdir = args.get("workdir")
        user_provided_workdir = workdir is not None
        background = args.get("background", False)
        delivery_channel = args.get("delivery_channel")
        role = perm.get_user_role(ctx.sender_id) if perm else "admin"

        if workdir is None:
            ws_mgr = deps.workspace_manager
            if ws_mgr:
                if is_admin_private(ctx, deps):
                    workdir = str(ws_mgr.root_dir().resolve())
                else:
                    workdir = str(
                        ws_mgr.sandbox_dir(ctx.is_group, ctx.chat_id).resolve()
                    )

        if user_provided_workdir and not (perm and perm.is_admin_role(role)):
            ws_mgr = deps.workspace_manager
            if ws_mgr:
                try:
                    safe_path = ws_mgr.resolve_safe_path(
                        ctx.is_group, ctx.chat_id, workdir
                    )
                    workdir = str(safe_path)
                except ValueError:
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"工作目录不在允许范围内: {workdir}"},
                            ensure_ascii=False,
                        )
                    )
            else:
                workdir = None

        parts = parse_command_safe(command)
        if parts is None:
            _log.warning("exec 命令格式无效: %s", command[:80])
            return ToolResult(
                content=json.dumps(
                    {"error": f"命令格式无效（引号不匹配等）: {command[:80]}"},
                    ensure_ascii=False,
                )
            )

        approval_mgr = deps.approval_manager.value

        if approval_mgr and approval_mgr.check_whitelist("exec", command):
            _log.info("exec 命令命中审批白名单: %s", command[:80])
        else:
            reason = check_command_denied(parts)
            if reason:
                _log.warning("exec 被拒绝: %s", reason)
                if approval_mgr and role == "admin" and not ctx.is_group:
                    result = await approval_mgr.request_approval(
                        chat_id=ctx.chat_id,
                        tool_name="exec",
                        reason=reason,
                        details=command,
                    )
                    if result == "deny":
                        return ToolResult(
                            content=json.dumps(
                                {"error": f"审批已拒绝: {reason}"},
                                ensure_ascii=False,
                            )
                        )
                    if result == "timeout":
                        return ToolResult(
                            content=json.dumps(
                                {"error": f"审批超时: {reason}"},
                                ensure_ascii=False,
                            )
                        )
                else:
                    return ToolResult(
                        content=json.dumps(
                            {"error": reason},
                            ensure_ascii=False,
                        )
                    )

            if role != "admin" and perm:
                reason = perm.check_command_allowed(command, parts, role)
                if reason:
                    _log.warning("exec 白名单拒绝: %s", reason)
                    return ToolResult(
                        content=json.dumps(
                            {"error": reason},
                            ensure_ascii=False,
                        )
                    )

        if background:
            try:
                effective_timeout = min(timeout or 120, 300)
                env = await build_exec_env_for(perm)
                session_id = await process_registry.spawn(
                    command=command,
                    parts=parts,
                    workdir=workdir,
                    chat_id=ctx.chat_id,
                    delivery_channel=delivery_channel,
                    timeout=effective_timeout,
                    env=env,
                )
                return ToolResult(
                    content=json.dumps(
                        {
                            "background": True,
                            "session_id": session_id,
                            "message": f"进程已在后台启动 (session_id: {session_id[:8]}..)",
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as e:
                _log.warning("后台进程启动失败: %s", e)
                return ToolResult(
                    content=json.dumps(
                        {"error": f"后台进程启动失败: {e}"},
                        ensure_ascii=False,
                    )
                )

        import asyncio
        import subprocess

        effective_timeout = min(timeout or 60, 120)
        try:
            env = await build_exec_env_for(perm)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    parts,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    env=env,
                    cwd=workdir,
                ),
                timeout=effective_timeout + 5,
            )
            stdout = (
                result.stdout[-100000:]
                if len(result.stdout) > 100000
                else result.stdout
            )
            stderr = (
                result.stderr[-100000:]
                if len(result.stderr) > 100000
                else result.stderr
            )
            return ToolResult(
                content=json.dumps(
                    {
                        "success": result.returncode == 0,
                        "exit_code": result.returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                        "truncated": {
                            "stdout": len(result.stdout) > 100000,
                            "stderr": len(result.stderr) > 100000,
                        },
                    },
                    ensure_ascii=False,
                )
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content=json.dumps(
                    {"error": f"命令执行超时 ({effective_timeout}秒)"},
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            return ToolResult(
                content=json.dumps(
                    {"error": str(e)},
                    ensure_ascii=False,
                )
            )

    async def _process(args: dict, ctx: ToolContext) -> ToolResult:
        process_registry = deps.process_registry.value
        if not process_registry:
            return ToolResult(
                content=json.dumps(
                    {"error": "进程系统未就绪"},
                    ensure_ascii=False,
                )
            )

        action = args.get("action", "")
        session_id = args.get("session_id", "")

        if action == "list":
            sessions = await process_registry.list_sessions()
            return ToolResult(
                content=json.dumps(
                    {
                        "sessions": sessions,
                        "count": len(sessions),
                    },
                    ensure_ascii=False,
                )
            )

        if not session_id:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供 session_id（list 除外）"},
                    ensure_ascii=False,
                )
            )

        if action == "poll":
            timeout = args.get("timeout", 30.0)
            result = await process_registry.poll(session_id, timeout=timeout)
            if result is None:
                return ToolResult(
                    content=json.dumps(
                        {"error": "会话不存在"},
                        ensure_ascii=False,
                    )
                )
            return ToolResult(content=json.dumps(result, ensure_ascii=False))

        if action == "log":
            offset = args.get("offset", 0)
            limit = args.get("limit", 200)
            result = await process_registry.get_log(
                session_id, offset=offset, limit=limit
            )
            if result is None:
                return ToolResult(
                    content=json.dumps(
                        {"error": "会话不存在"},
                        ensure_ascii=False,
                    )
                )
            return ToolResult(content=json.dumps(result, ensure_ascii=False))

        if action == "write":
            data = (args.get("data") or "").strip()
            if not data:
                return ToolResult(
                    content=json.dumps(
                        {"error": "请提供要写入的数据"},
                        ensure_ascii=False,
                    )
                )
            error = await process_registry.write_stdin(session_id, data)
            if error:
                return ToolResult(
                    content=json.dumps(
                        {"error": error},
                        ensure_ascii=False,
                    )
                )
            return ToolResult(content=json.dumps({"success": True}))

        if action == "kill":
            error = await process_registry.kill(session_id)
            if error:
                return ToolResult(
                    content=json.dumps(
                        {"error": error},
                        ensure_ascii=False,
                    )
                )
            return ToolResult(content=json.dumps({"success": True}))

        if action == "remove":
            error = await process_registry.remove(session_id)
            if error:
                return ToolResult(
                    content=json.dumps(
                        {"error": error},
                        ensure_ascii=False,
                    )
                )
            return ToolResult(content=json.dumps({"success": True}))

        return ToolResult(
            content=json.dumps(
                {"error": f"未知操作: {action}"},
                ensure_ascii=False,
            )
        )

    EXEC_PARAMS = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令",
            },
            "timeout": {
                "type": "integer",
                "description": "前台执行超时（秒），默认 60，最大 120",
            },
            "workdir": {
                "type": "string",
                "description": "工作目录（可选，默认项目根目录）",
            },
            "background": {
                "type": "boolean",
                "description": "是否后台运行。长时间任务设为 true，会立即返回 session_id",
            },
            "delivery_channel": {
                "type": "string",
                "description": "后台进程退出时投递通知的聊天 ID（可选）",
            },
        },
        "required": ["command"],
    }

    PROCESS_PARAMS = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "write", "kill", "remove"],
                "description": "操作类型: list=列出所有会话, poll=等待新输出, log=查看日志, write=写 stdin, kill=终止, remove=移除",
            },
            "session_id": {
                "type": "string",
                "description": "会话 ID（list 操作不需要）",
            },
            "data": {
                "type": "string",
                "description": "要写入 stdin 的内容（action=write 时使用）",
            },
            "timeout": {
                "type": "number",
                "description": "poll 等待最大时间（秒，最大 30）",
            },
            "offset": {
                "type": "integer",
                "description": "日志起始行（action=log 时使用）",
            },
            "limit": {
                "type": "integer",
                "description": "日志返回行数（action=log 时使用，默认 200）",
            },
        },
        "required": ["action"],
    }

    return [
        ToolEntry(
            name="exec",
            section="exec",
            description=(
                "执行 shell 命令，默认前台运行并等待结果。\n"
                "对于长时间运行的任务（构建服务器、文件监视器等），"
                "设置 background=true 使其后台运行，然后使用 process 工具管理。\n"
                "安全限制：命令受 allowlist 限制，不在白名单中的命令会被拒绝。\n"
                "注意：文件读取/编辑用 read_file/edit_file/write_file，"
                "内容搜索用 search_content，文件查找用 find_files，"
                "目录列表用 list_dir——不要用 exec 包装这些操作。"
                "二进制文件或超大文件（>1MB）的操作请使用本工具。"
            ),
            parameters=EXEC_PARAMS,
            handler=_exec,
        ),
        ToolEntry(
            name="process",
            section="exec",
            description=(
                "管理后台进程会话（由 exec 的 background=true 创建）。\n"
                "list: 列出所有后台会话（每个会话有 session_id、pid、command）\n"
                "poll: 等待新输出或进程退出（最长 timeout 秒）\n"
                "log: 查看会话的完整日志，支持 offset/limit 分页\n"
                "write: 写入数据到进程的 stdin\n"
                "kill: 终止进程\n"
                "remove: 终止并移除会话"
            ),
            parameters=PROCESS_PARAMS,
            handler=_process,
        ),
    ]
