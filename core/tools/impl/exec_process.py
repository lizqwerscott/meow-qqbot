"""exec + process 工具"""

import json
import logging
import os

from core.approval.allowlist import AllowlistEntry, match_allowlist, merge_allowlists
from core.approval.exec_policy import (
    ALLOW_DECISIONS,
    DECISION_ALLOW,
    DECISION_ALLOW_ONCE,
    DECISION_DENY,
    ExecPolicy,
    config_to_policy,
    effective_policy,
    policy_for_role,
    requires_approval,
    resolve_mode_from_policy,
)
from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps
from core.tools.exec_analysis import analyze_command, iter_all_segments
from core.tools.exec_runner import build_argv, run_plan
from core.tools.impl.file import is_admin_private
from core.tools.security import check_command_denied, parse_command_safe
from core.tools.shell_env import build_exec_env_for

_log = logging.getLogger(__name__)


def _build_exec_policy(perm, approval_mgr):
    """策略面：config [exec] × host 审批文件取更严（对齐 openclaw）。

    Returns:
        (effective, host_policy) 二元组——host_policy 供 policy_for_role
        对固定角色（trusted/default/system）继续收紧。
    """
    if perm:
        config_policy = config_to_policy(perm.get_exec_policy())
    else:
        config_policy = ExecPolicy()
    host_policy = approval_mgr.get_host_policy() if approval_mgr else ExecPolicy()
    return effective_policy(config_policy, host_policy), host_policy


def _plan_mismatch(stored: dict, current: dict) -> bool:
    """比对审批时绑定的 plan 与当前执行计划（对齐 openclaw approval mismatch）。"""
    for key in ("command", "cwd"):
        if stored.get(key) != current.get(key):
            return True
    if stored.get("resolved_path") != current.get("resolved_path"):
        return True
    return False


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
        reviewer = deps.exec_reviewer.value if deps.exec_reviewer else None

        # ── 1. 策略面（OpenClaw 风格：config × host 取更严，再按角色归一）──
        effective, host_policy = _build_exec_policy(perm, approval_mgr)
        policy = policy_for_role(role, effective, host_policy)
        mode = resolve_mode_from_policy(policy)

        if policy.security == "deny":
            return ToolResult(
                content=json.dumps(
                    {"error": "exec 已被策略禁用 (security=deny)"},
                    ensure_ascii=False,
                )
            )

        # ── 2. 命令分析：切段 + 真实路径解析 + inline-eval 检测 ──
        segments = analyze_command(command, env=os.environ, cwd=workdir)
        if not segments:
            _log.warning("exec 命令格式无效: %s", command[:80])
            return ToolResult(
                content=json.dumps(
                    {"error": f"命令格式无效（引号不匹配等）: {command[:80]}"},
                    ensure_ascii=False,
                )
            )

        # 硬拦截原因（DENIED_COMMANDS 黑名单 + 危险重定向目标），穿透嵌套段：
        # bash -c 'rm ...' 的 payload 内 rm 也计入（对齐 openclaw wrapper 分析）。
        # allowlist 命中时保持原语义直接放行（allowlist 覆盖黑名单，如 sudo）。
        deny_reason = check_command_denied(parts)
        if not deny_reason:
            for seg in iter_all_segments(segments):
                reason = check_command_denied(seg.argv)
                if reason:
                    deny_reason = reason
                    break

        # ── 3. allowlist 匹配（静态 [commands] + 运行时审批白名单，逐段）──
        static_entries = [
            AllowlistEntry(pattern=cmd, source="manual")
            for cmd in (perm.get_allowed_commands() if perm else [])
            if cmd
        ]
        dynamic_entries = approval_mgr.get_allowlist_entries() if approval_mgr else []
        allowlist_satisfied, _ = match_allowlist(
            segments,
            merge_allowlists(static_entries, dynamic_entries),
        )
        analysis_ok = all(
            seg.resolution and seg.resolution.resolved_path
            for seg in iter_all_segments(segments)
        )
        inline_hit = policy.strict_inline_eval and any(
            seg.inline_eval for seg in iter_all_segments(segments)
        )

        effective_allow = allowlist_satisfied and not inline_hit and analysis_ok
        needs_ask = requires_approval(
            ask=policy.ask,
            security=policy.security,
            analysis_ok=analysis_ok,
            allowlist_satisfied=allowlist_satisfied and not inline_hit,
            durable_satisfied=effective_allow,
        )

        plan = {
            "command": command,
            "argv": parts,
            "cwd": workdir,
            "resolved_path": (
                segments[0].resolution.resolved_path
                if segments and segments[0].resolution
                else None
            ),
            "role": role,
        }

        if needs_ask:
            decision: str | None = None
            session_key: str | None = None
            # 审批可达性：前台仅 admin 私聊弹卡（群聊前台不放行）；
            # 后台执行（background=true）对齐 OpenClaw——interactive chat 中的
            # background exec 走同一审批流，审批卡投递 admin c2c，通过后才 spawn，
            # 避免"审批不到直接失败"。reviewer 是模型判定，不依赖聊天面，同样放行。
            can_approve_in_ctx = not ctx.is_group or background
            if (
                mode == "auto"
                and can_approve_in_ctx
                and not inline_hit
                and not deny_reason
                and reviewer
                and reviewer.available
            ):
                if await reviewer.review(plan) == DECISION_ALLOW:
                    decision = DECISION_ALLOW_ONCE
            if decision is None:
                if approval_mgr and role == "admin" and can_approve_in_ctx:
                    result = await approval_mgr.request_approval(
                        chat_id=ctx.chat_id,
                        tool_name="exec",
                        reason=deny_reason or "命令不在允许列表中",
                        details=command,
                        plan=plan,
                        ask_fallback=policy.ask_fallback,
                        # strictInlineEval：inline 命令的 allow-always 不落白名单
                        persist=not inline_hit,
                        return_session_key=True,
                    )
                    decision, session_key = result
                    if decision == DECISION_ALLOW:
                        decision = DECISION_ALLOW_ONCE
                else:
                    decision = DECISION_DENY
            if decision not in ALLOW_DECISIONS:
                reason_text = deny_reason or "命令不在允许列表中"
                _log.warning("exec 未获审批: %s", reason_text)
                return ToolResult(
                    content=json.dumps(
                        {"error": reason_text},
                        ensure_ascii=False,
                    )
                )
            # durable 比对（对齐 openclaw approval mismatch）：审批通过后执行前，
            # 校验当前命令与审批时绑定的 canonical plan 一致，防止内容漂移。
            if approval_mgr and session_key:
                stored = approval_mgr.take_pending_plan(session_key)
                if stored and _plan_mismatch(stored, plan):
                    _log.warning("exec 审批计划漂移，拒绝执行: %s", command[:80])
                    return ToolResult(
                        content=json.dumps(
                            {"error": "APPROVAL_MISMATCH: 命令与已审批内容不一致"},
                            ensure_ascii=False,
                        )
                    )
        else:
            # 无审批路径：security=full 全放行；allowlist 模式必须命中
            if policy.security != "full" and not effective_allow:
                _log.warning("exec allowlist 拒绝: %s", command[:80])
                return ToolResult(
                    content=json.dumps(
                        {"error": "命令不在允许列表中"},
                        ensure_ascii=False,
                    )
                )

        # ── 4. 非 admin 叠加安全策略（替换/串联/重定向/长度/命令名）──
        if role not in ("admin", "system") and perm:
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
            if len(segments) > 1:
                return ToolResult(
                    content=json.dumps(
                        {"error": "链式/管道命令暂不支持后台执行，请用前台执行"},
                        ensure_ascii=False,
                    )
                )
            try:
                effective_timeout = min(timeout or 120, 300)
                env = await build_exec_env_for(perm)
                # 后台同样绑定解析后的可执行路径（pin executable）
                bg_parts = build_argv(segments[0]) if segments else parts
                session_id = await process_registry.spawn(
                    command=command,
                    parts=bg_parts,
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

        # 前台：段级执行（分析-执行绑定，支持 && || ; | 语义）
        effective_timeout = min(timeout or 60, 120)
        try:
            env = await build_exec_env_for(perm)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_plan,
                    segments,
                    env=env,
                    cwd=workdir,
                    timeout=effective_timeout,
                ),
                timeout=effective_timeout + 5,
            )
            return ToolResult(content=json.dumps(result, ensure_ascii=False))
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
