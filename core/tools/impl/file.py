import asyncio
import json
import logging
from pathlib import Path

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps
from core.tools.patch_parser import (
    ActionType,
    DiffError,
    apply_update_hunks,
    parse_patch_text,
)

_log = logging.getLogger(__name__)


def is_admin_private(ctx: ToolContext, deps: ToolDeps) -> bool:
    if ctx.is_group:
        return False
    perm = deps.permission_manager
    admin_ids = deps.admin_ids
    if perm:
        role = perm.get_user_role(ctx.sender_id)
        return perm.is_admin_role(role)
    return ctx.sender_id in admin_ids


async def _approve_path_access(
    ctx: ToolContext, file_path: str, tool_name: str, reason: str, deps: ToolDeps
):
    if not is_admin_private(ctx, deps):
        return None
    mgr = deps.approval_manager.value
    if not mgr:
        return None
    try:
        resolved = sandbox_target(
            ctx.is_group, ctx.chat_id, file_path, deps, admin_override=True
        )
    except ValueError:
        resolved = Path(file_path).resolve()
    if mgr.check_whitelist(tool_name, str(resolved)):
        return resolved
    result = await mgr.request_approval(ctx.chat_id, tool_name, reason, str(resolved))
    if result in ("allow-once", "allow-always"):
        return resolved
    return None


def sandbox_target(
    is_group: bool,
    chat_id: str,
    rel_path: str,
    deps: ToolDeps,
    admin_override: bool = False,
) -> Path:
    wm = deps.workspace_manager
    if not wm:
        raise ValueError("工作区未就绪")
    if rel_path in ("", "."):
        return wm.root_dir() if admin_override else wm.sandbox_dir(is_group, chat_id)
    return wm.resolve_safe_path(
        is_group, chat_id, rel_path, admin_override=admin_override
    )


def create_file_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _apply_patch(args: dict, ctx: ToolContext) -> ToolResult:
        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )
        raw_input = (args.get("input") or "").strip()
        if not raw_input:
            return ToolResult(
                content=json.dumps({"error": "请提供 patch 输入"}, ensure_ascii=False)
            )
        try:
            actions = parse_patch_text(raw_input)
        except DiffError as e:
            return ToolResult(
                content=json.dumps(
                    {"error": f"Patch 格式错误: {e}"}, ensure_ascii=False
                )
            )
        if not actions:
            return ToolResult(
                content=json.dumps(
                    {"error": "Patch 中没有有效的操作"}, ensure_ascii=False
                )
            )
        admin_override = is_admin_private(ctx, deps)

        # Phase 1: Validate all paths before writing
        resolved_targets: dict[str, Path] = {}
        for path, action in actions.items():
            try:
                target = sandbox_target(
                    ctx.is_group, ctx.chat_id, path, deps, admin_override=admin_override
                )
            except ValueError as e:
                approved = await _approve_path_access(
                    ctx, path, "apply_patch", str(e), deps
                )
                if approved is not None:
                    target = approved
                else:
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"路径非法 ({path}): {e}"}, ensure_ascii=False
                        )
                    )
            resolved_targets[path] = target
            if action.move_path:
                try:
                    dest = sandbox_target(
                        ctx.is_group,
                        ctx.chat_id,
                        action.move_path,
                        deps,
                        admin_override=admin_override,
                    )
                except ValueError as e:
                    approved = await _approve_path_access(
                        ctx, action.move_path, "apply_patch", str(e), deps
                    )
                    if approved is not None:
                        dest = approved
                    else:
                        return ToolResult(
                            content=json.dumps(
                                {"error": f"移动路径非法 ({action.move_path}): {e}"},
                                ensure_ascii=False,
                            )
                        )
                resolved_targets[f"__dest:{path}"] = dest

        # Phase 2: Execute
        summary = {"added": [], "modified": [], "deleted": [], "moved": []}
        for path, action in actions.items():
            target = resolved_targets[path]
            if action.type == ActionType.ADD:
                target.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    target.write_text, action.content, encoding="utf-8"
                )
                summary["added"].append(path)
            elif action.type == ActionType.DELETE:
                if not target.exists():
                    summary["deleted"].append(f"{path} (不存在，跳过)")
                    continue
                if not target.is_file():
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"不是文件，无法删除: {path}"}, ensure_ascii=False
                        )
                    )
                await asyncio.to_thread(target.unlink)
                summary["deleted"].append(path)
            elif action.type == ActionType.UPDATE:
                if not target.exists():
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"文件不存在: {path}"}, ensure_ascii=False
                        )
                    )
                if not target.is_file():
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"路径不是文件: {path}"}, ensure_ascii=False
                        )
                    )
                try:
                    current = await asyncio.to_thread(
                        target.read_text, encoding="utf-8"
                    )
                except Exception as e:
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"读取失败 ({path}): {e}"}, ensure_ascii=False
                        )
                    )
                if not action.hunks:
                    continue
                try:
                    new_content = apply_update_hunks(current, action.hunks)
                except DiffError as e:
                    return ToolResult(
                        content=json.dumps(
                            {"error": f"Patch 应用失败 ({path}): {e}"},
                            ensure_ascii=False,
                        )
                    )
                if new_content == current:
                    continue
                dest = target
                if action.move_path:
                    dest = resolved_targets[f"__dest:{path}"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    summary["moved"].append(f"{path} → {action.move_path}")
                await asyncio.to_thread(dest.write_text, new_content, encoding="utf-8")
                if dest != target:
                    await asyncio.to_thread(target.unlink)
                summary["modified"].append(path)
        lines = ["Patch 应用成功。更新了以下文件："]
        for f in summary["added"]:
            lines.append(f"A {f}")
        for f in summary["modified"]:
            lines.append(f"M {f}")
        for f in summary["deleted"]:
            lines.append(f"D {f}")
        for m in summary["moved"]:
            lines.append(f"MV {m}")
        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "summary": summary,
                    "message": "\n".join(lines),
                },
                ensure_ascii=False,
            )
        )

    async def _read_file(args: dict, ctx: ToolContext) -> ToolResult:
        media_uri = (args.get("media_uri") or "").strip()
        if media_uri:
            service = deps.media_service
            if not service:
                return ToolResult(
                    content=json.dumps({"error": "媒体服务未就绪"}, ensure_ascii=False)
                )
            result = await service.read_file(
                chat_id=ctx.chat_id,
                media_uri=media_uri,
                max_chars=args.get("max_chars"),
            )
            return ToolResult(content=json.dumps(result.as_dict(), ensure_ascii=False))
        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )
        file_path = (args.get("file_path") or "").strip()
        if not file_path:
            return ToolResult(
                content=json.dumps({"error": "请提供 file_path"}, ensure_ascii=False)
            )
        admin_override = is_admin_private(ctx, deps)
        try:
            target = sandbox_target(
                ctx.is_group,
                ctx.chat_id,
                file_path,
                deps,
                admin_override=admin_override,
            )
        except ValueError as e:
            approved = await _approve_path_access(
                ctx, file_path, "read_file", str(e), deps
            )
            if approved is not None:
                target = approved
            else:
                return ToolResult(
                    content=json.dumps({"error": str(e)}, ensure_ascii=False)
                )
        if not target.exists():
            return ToolResult(
                content=json.dumps(
                    {"error": f"路径不存在: {file_path}"}, ensure_ascii=False
                )
            )

        if target.is_dir():
            return ToolResult(
                content=json.dumps(
                    {
                        "error": f"'{file_path}' 是目录，请使用 list_dir 工具列出目录内容"
                    },
                    ensure_ascii=False,
                )
            )

        if not target.is_file():
            return ToolResult(
                content=json.dumps(
                    {"error": f"路径不是文件: {file_path}"}, ensure_ascii=False
                )
            )
        _MAX_FILE_SIZE = 1024 * 1024
        file_size = target.stat().st_size
        if file_size > _MAX_FILE_SIZE:
            return ToolResult(
                content=json.dumps(
                    {
                        "error": f"文件过大（{file_size} bytes），超过 1MB 限制。请使用 exec 命令切片读取",
                    },
                    ensure_ascii=False,
                )
            )
        try:
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                content=json.dumps(
                    {
                        "error": f"无法读取二进制文件（{file_size} bytes），请使用 exec 命令处理",
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            return ToolResult(
                content=json.dumps({"error": f"读取失败: {e}"}, ensure_ascii=False)
            )
        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "content": content,
                    "path": file_path,
                },
                ensure_ascii=False,
            )
        )

    async def _write_file(args: dict, ctx: ToolContext) -> ToolResult:
        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )
        file_path = (args.get("file_path") or "").strip()
        content = args.get("content") or ""
        if not file_path:
            return ToolResult(
                content=json.dumps({"error": "请提供 file_path"}, ensure_ascii=False)
            )
        admin_override = is_admin_private(ctx, deps)
        try:
            target = sandbox_target(
                ctx.is_group,
                ctx.chat_id,
                file_path,
                deps,
                admin_override=admin_override,
            )
        except ValueError as e:
            approved = await _approve_path_access(
                ctx, file_path, "write_file", str(e), deps
            )
            if approved is not None:
                target = approved
            else:
                return ToolResult(
                    content=json.dumps({"error": str(e)}, ensure_ascii=False)
                )
        if target.exists() and target.is_dir():
            return ToolResult(
                content=json.dumps(
                    {"error": f"'{file_path}' 是目录，无法写入"}, ensure_ascii=False
                )
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        except Exception as e:
            return ToolResult(
                content=json.dumps({"error": f"写入失败: {e}"}, ensure_ascii=False)
            )
        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "path": file_path,
                    "size": len(content),
                },
                ensure_ascii=False,
            )
        )

    async def _list_dir(args: dict, ctx: ToolContext) -> ToolResult:
        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )
        dir_path = (args.get("path") or ".").strip()
        limit = max(1, min(5000, int(args.get("limit", 500))))
        admin_override = is_admin_private(ctx, deps)
        approved_outside_sandbox = False
        try:
            target = sandbox_target(
                ctx.is_group, ctx.chat_id, dir_path, deps, admin_override=admin_override
            )
        except ValueError as e:
            approved = await _approve_path_access(
                ctx, dir_path, "list_dir", str(e), deps
            )
            if approved is not None:
                target = approved
                approved_outside_sandbox = True
            else:
                return ToolResult(
                    content=json.dumps({"error": str(e)}, ensure_ascii=False)
                )
        target = target.resolve()
        if not target.exists():
            return ToolResult(
                content=json.dumps(
                    {"error": f"路径不存在: {dir_path}"}, ensure_ascii=False
                )
            )
        if not target.is_dir():
            return ToolResult(
                content=json.dumps(
                    {"error": f"不是目录: {dir_path}"}, ensure_ascii=False
                )
            )
        try:
            items = sorted(
                target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return ToolResult(
                content=json.dumps({"error": "无权限访问该目录"}, ensure_ascii=False)
            )
        files = []
        dirs = []
        count = 0
        fillable = 0
        for item in items:
            if approved_outside_sandbox:
                rel = str(item)
            else:
                sandbox = (
                    wm.root_dir().resolve()
                    if admin_override
                    else wm.sandbox_dir(ctx.is_group, ctx.chat_id).resolve()
                )
                try:
                    rel = str(item.relative_to(sandbox))
                except ValueError:
                    continue
            fillable += 1
            if count >= limit:
                continue
            if item.is_dir():
                dirs.append(rel + "/")
            else:
                size = item.stat().st_size if item.is_file() else 0
                files.append({"path": rel, "size": size})
            count += 1
        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "path": dir_path,
                    "is_dir": True,
                    "directories": dirs,
                    "files": files,
                    "displayed": len(dirs) + len(files),
                    "total": fillable,
                    "limit_reached": fillable > len(dirs) + len(files),
                },
                ensure_ascii=False,
            )
        )

    async def _edit_file(args: dict, ctx: ToolContext) -> ToolResult:
        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )
        file_path = (args.get("file_path") or "").strip()
        old_string = args.get("old_string") or ""
        new_string = args.get("new_string") or ""
        replace_all = args.get("replace_all", False)
        if not file_path:
            return ToolResult(
                content=json.dumps({"error": "请提供 file_path"}, ensure_ascii=False)
            )
        if not old_string:
            return ToolResult(
                content=json.dumps({"error": "请提供 old_string"}, ensure_ascii=False)
            )
        admin_override = is_admin_private(ctx, deps)
        try:
            target = sandbox_target(
                ctx.is_group,
                ctx.chat_id,
                file_path,
                deps,
                admin_override=admin_override,
            )
        except ValueError as e:
            approved = await _approve_path_access(
                ctx, file_path, "edit_file", str(e), deps
            )
            if approved is not None:
                target = approved
            else:
                return ToolResult(
                    content=json.dumps({"error": str(e)}, ensure_ascii=False)
                )
        if not target.exists():
            return ToolResult(
                content=json.dumps(
                    {"error": f"文件不存在: {file_path}"}, ensure_ascii=False
                )
            )
        _MAX_FILE_SIZE = 1024 * 1024
        if target.stat().st_size > _MAX_FILE_SIZE:
            return ToolResult(
                content=json.dumps(
                    {
                        "error": f"文件过大（{target.stat().st_size} bytes），超过 1MB 限制，请使用 exec 命令处理",
                    },
                    ensure_ascii=False,
                )
            )
        try:
            current = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except Exception as e:
            return ToolResult(
                content=json.dumps({"error": f"读取失败: {e}"}, ensure_ascii=False)
            )
        if replace_all:
            if old_string not in current:
                return ToolResult(
                    content=json.dumps(
                        {"error": f"未找到匹配: {old_string[:60]}"}, ensure_ascii=False
                    )
                )
            new_content = current.replace(old_string, new_string)
        else:
            count = current.count(old_string)
            if count == 0:
                return ToolResult(
                    content=json.dumps(
                        {"error": f"未找到匹配: {old_string[:60]}"}, ensure_ascii=False
                    )
                )
            if count > 1:
                return ToolResult(
                    content=json.dumps(
                        {
                            "error": f"找到 {count} 处匹配，请提供更多上下文或使用 replaceAll"
                        },
                        ensure_ascii=False,
                    )
                )
            new_content = current.replace(old_string, new_string, 1)
        if new_content == current:
            return ToolResult(
                content=json.dumps(
                    {
                        "success": True,
                        "path": file_path,
                        "replaced": False,
                        "message": "内容未变化（old_string 与 new_string 相同）",
                    },
                    ensure_ascii=False,
                )
            )
        try:
            await asyncio.to_thread(target.write_text, new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(
                content=json.dumps({"error": f"写入失败: {e}"}, ensure_ascii=False)
            )
        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "path": file_path,
                    "replaced": True,
                },
                ensure_ascii=False,
            )
        )

    APPLY_PATCH_PARAMS = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "patch 完整内容，包含 *** Begin Patch 和 *** End Patch 包围",
            },
        },
        "required": ["input"],
    }

    FILE_READ_PARAMS = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'。如果是目录请使用 list_dir 工具。",
            },
            "media_uri": {
                "type": "string",
                "description": "当前会话中的受控附件引用，例如 media://inbound/abc123。",
            },
            "max_chars": {
                "type": "integer",
                "description": "读取附件时的最大字符数，默认使用服务限制。",
            },
        },
        "anyOf": [{"required": ["file_path"]}, {"required": ["media_uri"]}],
    }

    LIST_DIR_PARAMS = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目录相对路径，默认当前目录"},
            "limit": {
                "type": "integer",
                "description": "最大条目数，默认 500，最大 5000",
            },
        },
    }

    FILE_WRITE_PARAMS = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'",
            },
            "content": {"type": "string", "description": "要写入的文件内容"},
        },
        "required": ["file_path", "content"],
    }

    FILE_EDIT_PARAMS = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'",
            },
            "old_string": {
                "type": "string",
                "description": "要被替换的旧文本，必须完全匹配（包括空格和换行）",
            },
            "new_string": {"type": "string", "description": "替换后的新文本"},
            "replace_all": {"type": "boolean", "description": "是否替换所有匹配"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    return [
        ToolEntry(
            name="apply_patch",
            section="file",
            description=(
                "批量修改工作区文件，一个 patch 可同时新建(ADD)、删除(DELETE)、"
                "更新(UPDATE)、移动(MOVE)多个文件。"
                "Patch 格式需包含 *** Begin Patch 和 *** End Patch 包围。"
                "若只改单个文件的一小段，优先使用 edit_file 而非本工具。"
                "默认不支持路径穿越(..)，管理员可通过审批访问越界路径。"
            ),
            parameters=APPLY_PATCH_PARAMS,
            handler=_apply_patch,
        ),
        ToolEntry(
            name="read_file",
            section="file",
            description=(
                "读取工作区文本文件或当前会话授权的文本附件。仅支持 1MB 以下的工作区文本文件。"
                "不支持读取目录（使用 list_dir）。"
                "优先使用本工具读取文件，不要使用 exec + cat/head/tail。"
                "默认不支持路径穿越(..)，管理员可通过审批访问越界路径。"
            ),
            parameters=FILE_READ_PARAMS,
            handler=_read_file,
        ),
        ToolEntry(
            name="list_dir",
            section="file",
            description=(
                "列出工作区目录内容。目录以 / 结尾，文件附带大小。"
                "优先使用本工具查看目录，不要使用 exec + ls/dir。"
                "默认不支持路径穿越(..)，管理员可通过审批访问越界路径。"
            ),
            parameters=LIST_DIR_PARAMS,
            handler=_list_dir,
        ),
        ToolEntry(
            name="write_file",
            section="file",
            description=(
                "写入文件到工作区，父目录自动创建，已存在文件会被覆盖。"
                "优先使用本工具创建/覆写文件，不要使用 exec + echo/cat/tee。"
                "默认不支持路径穿越(..)，管理员可通过审批访问越界路径。"
            ),
            parameters=FILE_WRITE_PARAMS,
            handler=_write_file,
        ),
        ToolEntry(
            name="edit_file",
            section="file",
            description=(
                "编辑工作区文件，进行精确字符串替换。"
                "old_string 必须完全匹配原文（含空格换行）。"
                "若文件有多处相同文本设 replace_all=true 替换全部。"
                "不确定唯一性时先 read_file 查看上下文确认。"
                "仅支持 1MB 以下的文本文件。"
                "优先使用本工具编辑文件，不要使用 exec + sed/awk。"
                "默认不支持路径穿越(..)，管理员可通过审批访问越界路径。"
            ),
            parameters=FILE_EDIT_PARAMS,
            handler=_edit_file,
        ),
    ]
