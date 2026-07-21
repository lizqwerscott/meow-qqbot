import asyncio
import json
import logging
import subprocess
from pathlib import Path

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import _DEPS
from core.tools.patch_parser import parse_patch_text, apply_update_hunks, ActionType, DiffError

_log = logging.getLogger(__name__)


def _is_admin_private(ctx: ToolContext) -> bool:
    if ctx.is_group:
        return False
    perm = _DEPS.get("permission_manager")
    admin_ids = _DEPS.get("admin_ids", [])
    if perm:
        role = perm.get_user_role(ctx.sender_id)
        return perm._role_level(role) >= 3
    return ctx.sender_id in admin_ids


def _sandbox_target(is_group: bool, chat_id: str, rel_path: str, admin_override: bool = False) -> Path:
    wm = _DEPS.get("workspace_manager")
    if not wm:
        raise ValueError("工作区未就绪")
    if rel_path in ("", "."):
        return wm.root_dir() if admin_override else wm.sandbox_dir(is_group, chat_id)
    return wm.resolve_safe_path(is_group, chat_id, rel_path, admin_override=admin_override)


async def _apply_patch(args: dict, ctx: ToolContext) -> ToolResult:
    wm = _DEPS.get("workspace_manager")
    if not wm:
        return ToolResult(content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False))
    raw_input = (args.get("input") or "").strip()
    if not raw_input:
        return ToolResult(content=json.dumps({"error": "请提供 patch 输入"}, ensure_ascii=False))
    try:
        actions = parse_patch_text(raw_input)
    except DiffError as e:
        return ToolResult(content=json.dumps({"error": f"Patch 格式错误: {e}"}, ensure_ascii=False))
    if not actions:
        return ToolResult(content=json.dumps({"error": "Patch 中没有有效的操作"}, ensure_ascii=False))
    admin_override = _is_admin_private(ctx)
    summary = {"added": [], "modified": [], "deleted": [], "moved": []}
    for path, action in actions.items():
        try:
            target = wm.resolve_safe_path(ctx.is_group, ctx.chat_id, path, admin_override=admin_override)
        except ValueError as e:
            return ToolResult(content=json.dumps({"error": f"路径非法 ({path}): {e}"}, ensure_ascii=False))
        if action.type == ActionType.ADD:
            target.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(target.write_text, action.content, encoding="utf-8")
            summary["added"].append(path)
        elif action.type == ActionType.DELETE:
            if not target.exists():
                summary["deleted"].append(f"{path} (不存在，跳过)")
                continue
            if not target.is_file():
                return ToolResult(content=json.dumps({"error": f"不是文件，无法删除: {path}"}, ensure_ascii=False))
            await asyncio.to_thread(target.unlink)
            summary["deleted"].append(path)
        elif action.type == ActionType.UPDATE:
            if not target.exists():
                return ToolResult(content=json.dumps({"error": f"文件不存在: {path}"}, ensure_ascii=False))
            if not target.is_file():
                return ToolResult(content=json.dumps({"error": f"路径不是文件: {path}"}, ensure_ascii=False))
            try:
                current = await asyncio.to_thread(target.read_text, encoding="utf-8")
            except Exception as e:
                return ToolResult(content=json.dumps({"error": f"读取失败 ({path}): {e}"}, ensure_ascii=False))
            if not action.hunks:
                continue
            try:
                new_content = apply_update_hunks(current, action.hunks)
            except DiffError as e:
                return ToolResult(content=json.dumps({"error": f"Patch 应用失败 ({path}): {e}"}, ensure_ascii=False))
            if new_content == current:
                continue
            dest = target
            if action.move_path:
                try:
                    dest = wm.resolve_safe_path(ctx.is_group, ctx.chat_id, action.move_path, admin_override=admin_override)
                except ValueError as e:
                    return ToolResult(content=json.dumps({"error": f"移动路径非法 ({action.move_path}): {e}"}, ensure_ascii=False))
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
    return ToolResult(content=json.dumps({
        "success": True, "summary": summary, "message": "\n".join(lines),
    }, ensure_ascii=False))


async def _read_file(args: dict, ctx: ToolContext) -> ToolResult:
    wm = _DEPS.get("workspace_manager")
    if not wm:
        return ToolResult(content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False))
    file_path = (args.get("file_path") or "").strip()
    if not file_path:
        return ToolResult(content=json.dumps({"error": "请提供 file_path"}, ensure_ascii=False))
    admin_override = _is_admin_private(ctx)
    try:
        target = wm.resolve_safe_path(ctx.is_group, ctx.chat_id, file_path, admin_override=admin_override)
    except ValueError as e:
        return ToolResult(content=json.dumps({"error": str(e)}, ensure_ascii=False))
    if not target.exists():
        return ToolResult(content=json.dumps({"error": f"路径不存在: {file_path}"}, ensure_ascii=False))

    # 如果路径是目录，返回文件列表
    if target.is_dir():
        try:
            items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return ToolResult(content=json.dumps({"error": "无权限访问该目录"}, ensure_ascii=False))
        sandbox = wm.root_dir() if admin_override else wm.sandbox_dir(ctx.is_group, ctx.chat_id)
        files = []
        dirs = []
        for item in items:
            try:
                rel = str(item.relative_to(sandbox))
            except ValueError:
                continue
            if item.is_dir():
                dirs.append(rel + "/")
            else:
                size = item.stat().st_size if item.is_file() else 0
                files.append({"path": rel, "size": size})
        return ToolResult(content=json.dumps({
            "success": True, "path": file_path, "is_dir": True,
            "directories": dirs, "files": files,
            "total": len(dirs) + len(files),
        }, ensure_ascii=False))

    # 如果是文件，读取内容
    if not target.is_file():
        return ToolResult(content=json.dumps({"error": f"路径不是文件: {file_path}"}, ensure_ascii=False))
    try:
        content = await asyncio.to_thread(target.read_text, encoding="utf-8")
    except Exception as e:
        return ToolResult(content=json.dumps({"error": f"读取失败: {e}"}, ensure_ascii=False))
    return ToolResult(content=json.dumps({
        "success": True, "content": content, "path": file_path, "is_dir": False,
    }, ensure_ascii=False))


async def _write_file(args: dict, ctx: ToolContext) -> ToolResult:
    wm = _DEPS.get("workspace_manager")
    if not wm:
        return ToolResult(content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False))
    file_path = (args.get("file_path") or "").strip()
    content = (args.get("content") or "")
    if not file_path:
        return ToolResult(content=json.dumps({"error": "请提供 file_path"}, ensure_ascii=False))
    admin_override = _is_admin_private(ctx)
    try:
        target = wm.resolve_safe_path(ctx.is_group, ctx.chat_id, file_path, admin_override=admin_override)
    except ValueError as e:
        return ToolResult(content=json.dumps({"error": str(e)}, ensure_ascii=False))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(target.write_text, content, encoding="utf-8")
    except Exception as e:
        return ToolResult(content=json.dumps({"error": f"写入失败: {e}"}, ensure_ascii=False))
    return ToolResult(content=json.dumps({
        "success": True, "path": file_path, "size": len(content),
    }, ensure_ascii=False))


async def _edit_file(args: dict, ctx: ToolContext) -> ToolResult:
    wm = _DEPS.get("workspace_manager")
    if not wm:
        return ToolResult(content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False))
    file_path = (args.get("file_path") or "").strip()
    old_string = (args.get("old_string") or "")
    new_string = (args.get("new_string") or "")
    replace_all = args.get("replace_all", False)
    if not file_path:
        return ToolResult(content=json.dumps({"error": "请提供 file_path"}, ensure_ascii=False))
    if not old_string:
        return ToolResult(content=json.dumps({"error": "请提供 old_string"}, ensure_ascii=False))
    admin_override = _is_admin_private(ctx)
    try:
        target = wm.resolve_safe_path(ctx.is_group, ctx.chat_id, file_path, admin_override=admin_override)
    except ValueError as e:
        return ToolResult(content=json.dumps({"error": str(e)}, ensure_ascii=False))
    if not target.exists():
        return ToolResult(content=json.dumps({"error": f"文件不存在: {file_path}"}, ensure_ascii=False))
    try:
        current = await asyncio.to_thread(target.read_text, encoding="utf-8")
    except Exception as e:
        return ToolResult(content=json.dumps({"error": f"读取失败: {e}"}, ensure_ascii=False))
    if replace_all:
        if old_string not in current:
            return ToolResult(content=json.dumps({"error": f"未找到匹配: {old_string[:60]}"}, ensure_ascii=False))
        new_content = current.replace(old_string, new_string)
    else:
        count = current.count(old_string)
        if count == 0:
            return ToolResult(content=json.dumps({"error": f"未找到匹配: {old_string[:60]}"}, ensure_ascii=False))
        if count > 1:
            return ToolResult(content=json.dumps(
                {"error": f"找到 {count} 处匹配，请提供更多上下文或使用 replaceAll"}, ensure_ascii=False,
            ))
        new_content = current.replace(old_string, new_string, 1)
    try:
        await asyncio.to_thread(target.write_text, new_content, encoding="utf-8")
    except Exception as e:
        return ToolResult(content=json.dumps({"error": f"写入失败: {e}"}, ensure_ascii=False))
    return ToolResult(content=json.dumps({
        "success": True, "path": file_path, "replaced": not replace_all,
    }, ensure_ascii=False))





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
        "file_path": {"type": "string", "description": "文件/目录相对路径。如果是目录则列出文件列表，如果是文件则读取内容。例如 'note.txt' 或 'dir/'。"},
    },
    "required": ["file_path"],
}

FILE_WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'"},
        "content": {"type": "string", "description": "要写入的文件内容"},
    },
    "required": ["file_path", "content"],
}

FILE_EDIT_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'"},
        "old_string": {"type": "string", "description": "要被替换的旧文本，必须完全匹配（包括空格和换行）"},
        "new_string": {"type": "string", "description": "替换后的新文本"},
        "replace_all": {"type": "boolean", "description": "是否替换所有匹配"},
    },
    "required": ["file_path", "old_string", "new_string"],
}


def _register_all(register):
    register(ToolEntry(
        name="apply_patch",
        section="file",
        description="批量修改工作区文件。一个 patch 可以同时新建、删除、更新、移动多个文件。",
        parameters=APPLY_PATCH_PARAMS,
        handler=_apply_patch,
    ))
    register(ToolEntry(
        name="read_file",
        section="file",
        description="读取工作区文件内容或列出目录。File 路径为目录时返回文件列表；路径为文件时返回文件内容。不支持路径穿越(..)。搜索文件内容请使用 execute_command + rg。",
        parameters=FILE_READ_PARAMS,
        handler=_read_file,
    ))
    register(ToolEntry(
        name="write_file",
        section="file",
        description="写入文件到工作区。路径相对于当前聊天的 files/ 目录，父目录自动创建。已存在的文件会被覆盖。",
        parameters=FILE_WRITE_PARAMS,
        handler=_write_file,
    ))
    register(ToolEntry(
        name="edit_file",
        section="file",
        description="编辑工作区内的文件，进行精确的字符串替换。路径相对于当前聊天的 files/ 目录。比使用 sed 更可靠。",
        parameters=FILE_EDIT_PARAMS,
        handler=_edit_file,
    ))
