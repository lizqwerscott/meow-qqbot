"""exec + process 工具"""

import json
import logging
import os
from typing import Optional

from core.approval.allowlist import (
    AllowlistEntry,
    match_allowlist,
    match_safe_bins,
    merge_allowlists,
)
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
from core.tools.env_override_policy import validate_env_override
from core.tools.exec_analysis import (
    INTERPRETER_BINS,
    analyze_command,
    extract_wrapper_payloads,
    iter_all_segments,
    resolve_interpreter_target,
)
from core.tools.exec_runner import build_argv, run_plan
from core.tools.impl.file import is_admin_private
from core.tools.security import parse_command_safe
from core.tools.shell_env import build_exec_env_for

_log = logging.getLogger(__name__)

# ── exec 工具 env 参数（对齐 OpenClaw bash-tools.exec.ts 的 env override）──
#
# 模型可经 exec 的 env 参数传环境变量（如给 freshrss 脚本传 FRESHRSS_URL），
# 从而不再需要包一层 `bash -c 'export ... && ...'`（那会触发 strictInlineEval 门禁）。
#
# 安全语义（对齐 openclaw sanitizeHostExecEnvWithDiagnostics）：键名必须合法；
# PATH 覆盖禁止；危险键/前缀硬拒绝（Security Violation，非静默忽略）。策略表
# 集中维护在 core/tools/env_override_policy.py，评审/更新只改一处。


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
    # 2.2：绑定精确 argv 快照 + 解释器目标文件（防内容漂移）
    if stored.get("argv") != current.get("argv"):
        return True
    if stored.get("inner_file") != current.get("inner_file"):
        return True
    # env 绑定（对齐 openclaw args.env）：绑定的是模型传入的覆盖子集
    # （plan['env'] = env_overrides），非整份合并基础环境——基础环境含 login-shell
    # PATH/HOME 等非确定值，比对会导致误报；覆盖子集是确定性输入。
    if stored.get("env") != current.get("env"):
        return True
    return False


def _persist_target(seg) -> str:
    """allow-always 持久化目标（2.1 包装器解包）。

    优先取内层（最内层）可执行：PATH 解析命中 → bare-name；非 PATH 解析
    → 绝对路径条目（路径 glob 分支命中）。内层无法解析 / 非包装器 →
    回退现状（外层命令 basename）。
    """
    inner = seg.inner_resolution if seg.inner_argv else None
    res = inner if (inner and inner.resolved_path) else seg.resolution
    if res and res.resolved_path:
        if res.found_in_path:
            return os.path.basename(res.resolved_path)
        return res.resolved_path
    return os.path.basename(seg.argv[0])


def _has_payload_wrapper(seg) -> bool:
    """递归判断是否含 shell-payload 包装器；这类命令不可安全持久化。"""
    return bool(
        extract_wrapper_payloads(seg.argv)
        or (seg.inner_argv and extract_wrapper_payloads(seg.inner_argv))
        or any(_has_payload_wrapper(nested) for nested in seg.nested_segments)
    )


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

        # ── env 参数（对齐 OpenClaw exec 工具）：模型可传环境变量，免去
        #    `bash -c 'export ...'` 包装（那会触发 strictInlineEval 门禁）。
        #    危险键 / PATH / 非法键名 → 硬拒绝（对齐 openclaw Security Violation）。
        #    注意：plan 只绑定模型传入的覆盖子集 env_overrides（确定性），不绑定
        #    合并后的整份 base 环境（login-shell PATH/HOME 等非确定，比对会误报
        #    APPROVAL_MISMATCH）。
        base_env = await build_exec_env_for(perm)
        env_overrides, env_errors = validate_env_override(args.get("env"))
        if env_errors:
            detail = "; ".join(env_errors)
            _log.warning("exec 环境变量校验失败: %s", detail)
            return ToolResult(
                content=json.dumps(
                    {"error": f"Security Violation: {detail}"},
                    ensure_ascii=False,
                )
            )
        exec_env = dict(base_env)
        exec_env.update(env_overrides)
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

        # 无命令黑名单（对齐 OpenClaw）：危险命令由 allowlist 覆盖率决定——
        # rm 等不在 allowlist → miss → 审批/拒绝。auto-reviewer 的审查 prompt
        # 将 rm/mv/docker 等列为高风险，不会自动放行。

        # ── 3. allowlist 匹配（静态 [commands] + 运行时审批白名单，逐段）──
        static_entries = [
            AllowlistEntry(pattern=cmd, source="manual")
            for cmd in (perm.get_allowed_commands() if perm else [])
            if cmd
        ]
        dynamic_entries = approval_mgr.get_allowlist_entries() if approval_mgr else []
        _, allow_matches = match_allowlist(
            segments,
            merge_allowlists(static_entries, dynamic_entries),
        )
        # 2.4 使用计数：动态条目（allow-always/legacy）命中时记录（manual 静态条目不记）
        if approval_mgr:
            for m in allow_matches:
                if m and m.source != "manual":
                    approval_mgr.record_use(m.pattern)
        # safe bins：窄过滤器自动放行（对齐 openclaw tools.exec.safeBins）
        _, safe_matches = match_safe_bins(
            segments, policy.safe_bins, policy.safe_bin_profiles
        )
        # 逐段语义：每段命中 allowlist **或** safe-bin 即视为满足
        # （对齐 openclaw：shell chaining 时每个顶层段满足 allowlist 即可，
        # 混合链如 `ls | head -5` 由 ls 命中 allowlist + head 命中 safe-bin 组合满足）
        per_segment_ok = bool(segments) and all(
            (allow_matches[i] is not None) or (safe_matches[i] is not None)
            for i in range(len(segments))
        )
        analysis_ok = all(
            seg.resolution and seg.resolution.resolved_path
            for seg in iter_all_segments(segments)
        )
        # 2.2 解释器绑定：解释器/runtime 命令（python3/node/pnpm/npm/npx）必须能
        # 绑定到唯一具体本地文件；无法唯一确定（eval/模块/多文件形态、bin 缺失）
        # → analysis_ok=False 强制审批，且 allow-always 不落盘（不声称覆盖）。
        # 包装器段（timeout 5 node app.js）看内层（2.1 × 2.2 组合）。
        inner_file: Optional[str] = None
        interp_unbound = False
        multi_interp_target = False
        for seg in iter_all_segments(segments):
            target_argv = seg.inner_argv or seg.argv
            if target_argv and os.path.basename(target_argv[0]) in INTERPRETER_BINS:
                target, unique = resolve_interpreter_target(target_argv, cwd=workdir)
                if unique:
                    if inner_file is None:
                        inner_file = target
                    elif target != inner_file:
                        multi_interp_target = True
                else:
                    interp_unbound = True
        analysis_ok = analysis_ok and not interp_unbound and not multi_interp_target
        inline_hit = policy.strict_inline_eval and any(
            seg.inline_eval for seg in iter_all_segments(segments)
        )
        # heredoc（<<EOF）：对齐 openclaw reason: "heredoc" 独立审批触发点。
        # shell=False 下 heredoc 本就不生效（token 当参数），且可嵌入任意多行
        # 脚本内容，因此即使 allowlist 命中也要走审批。
        heredoc_hit = any(seg.heredoc for seg in iter_all_segments(segments))
        wrapper_invalid = any(
            seg.wrapper_invalid for seg in iter_all_segments(segments)
        )
        # flock -c/--command 的实际可执行内容来自 shell payload；外层 segment
        # 无法提供可持久化的唯一 argv，因此 allow-always 降级为一次性审批。
        payload_wrapper_hit = any(_has_payload_wrapper(seg) for seg in segments)

        effective_allow = (
            per_segment_ok
            and not inline_hit
            and not heredoc_hit
            and not wrapper_invalid
            and analysis_ok
        )
        needs_ask = requires_approval(
            ask=policy.ask,
            security=policy.security,
            analysis_ok=analysis_ok,
            allowlist_satisfied=(
                per_segment_ok
                and not inline_hit
                and not heredoc_hit
                and not wrapper_invalid
            ),
            durable_satisfied=effective_allow,
        )

        plan = {
            "command": command,
            "argv": parts,
            "cwd": workdir,
            # 绑定覆盖子集（非整份合并环境），对齐 openclaw 绑定 requestedEnv
            "env": env_overrides,
            "resolved_path": (
                segments[0].resolution.resolved_path
                if segments and segments[0].resolution
                else None
            ),
            "role": role,
            # 2.1 包装器解包：allow-always 持久化内层可执行路径；
            # 链式命令持久化所有顶层段（每段独立解析）
            "persist_pattern": (
                [_persist_target(seg) for seg in segments] if segments else None
            ),
            # 2.2 解释器绑定：目标脚本文件（唯一确定时）+ 是否无法声称覆盖
            "inner_file": inner_file,
            "interp_unbound": interp_unbound,
            "multi_interp_target": multi_interp_target,
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
                and not interp_unbound  # 2.2：无法绑定的解释器直接转人工，不自动审查
                and not multi_interp_target
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
                        reason="命令不在允许列表中",
                        details=command,
                        plan=plan,
                        ask_fallback=policy.ask_fallback,
                        # strictInlineEval：inline 命令的 allow-always 不落白名单；
                        # 2.2：interp_unbound（无法绑定唯一文件）同样不落白名单
                        persist=(
                            not inline_hit
                            and not interp_unbound
                            and not multi_interp_target
                            and not wrapper_invalid
                            and not payload_wrapper_hit
                        ),
                        timeout=policy.approval_timeout or 300,
                        return_session_key=True,
                    )
                    decision, session_key = result
                    if decision == DECISION_ALLOW:
                        decision = DECISION_ALLOW_ONCE
                else:
                    decision = DECISION_DENY
            if decision not in ALLOW_DECISIONS:
                reason_text = "命令不在允许列表中"
                _log.warning("exec 未获审批: %s", reason_text)
                if background and delivery_channel:
                    # 对齐 openclaw：审批超时/拒绝后回主会话 followup，
                    # 避免后台任务以为命令已在运行。
                    try:
                        import core.tasks.wake_coalescer as _coalescer

                        _coalescer.request_wake(
                            source="exec-event",
                            intent="event",
                            session_key=delivery_channel,
                            delivery_target=delivery_channel,
                            extra_prompt=(
                                f"后台命令未执行（审批未通过/超时）：{command[:100]}\n"
                                f"原因: {reason_text}"
                            ),
                            reason=f"exec 审批未通过: {command[:80]}",
                        )
                    except Exception as e:
                        _log.warning("exec 审批 followup 通知失败: %s", e)
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
                # 后台同样绑定解析后的可执行路径（pin executable）
                bg_parts = build_argv(segments[0]) if segments else parts
                session_id = await process_registry.spawn(
                    command=command,
                    parts=bg_parts,
                    workdir=workdir,
                    chat_id=ctx.chat_id,
                    delivery_channel=delivery_channel,
                    timeout=effective_timeout,
                    env=exec_env,
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
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_plan,
                    segments,
                    env=exec_env,
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
            "env": {
                "type": "object",
                "additionalProperties": {"type": ["string", "number"]},
                "description": (
                    "传递环境变量（可选）。键值对会在 exec 时注入子进程环境，"
                    "这样就不需要包一层 `bash -c 'export K=V && ...'`。"
                    "值可为字符串或数字（数字会转字符串）。"
                    "注意：PATH / 语言运行时 / 凭据等危险变量会被拒绝（Security Violation），"
                    "不能覆盖这些键。"
                ),
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
