import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps
from core.tools.impl.file import _approve_path_access, is_admin_private, sandbox_target
from core.tools.shell_env import build_exec_env_for

_log = logging.getLogger(__name__)


def create_search_entries(deps: ToolDeps) -> list[ToolEntry]:

    rg_path: str | None = None
    fd_path: str | None = None

    def _ensure_rg() -> str | None:
        nonlocal rg_path
        if rg_path is None:
            rg_path = shutil.which("rg")
        return rg_path

    def _ensure_fd() -> str | None:
        nonlocal fd_path
        if fd_path is None:
            fd_path = shutil.which("fd")
        return fd_path

    async def _run_subprocess(
        cmd: list[str],
        timeout: int = 30,
    ) -> tuple[int, str, str] | ToolResult:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=await build_exec_env_for(deps.permission_manager),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            assert proc.returncode is not None
            return (
                proc.returncode,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return ToolResult(
                content=json.dumps(
                    {"error": f"搜索超时（{timeout}秒）"}, ensure_ascii=False
                )
            )
        except Exception as e:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return ToolResult(
                content=json.dumps({"error": f"搜索失败: {e}"}, ensure_ascii=False)
            )

    async def _search_content(args: dict, ctx: ToolContext) -> ToolResult:
        pattern = (args.get("pattern") or "").strip()
        search_path = (args.get("path") or ".").strip()
        file_glob = (args.get("glob") or "").strip() or None
        ignore_case = args.get("ignore_case", False)
        literal = args.get("literal", False)
        limit = max(1, min(500, int(args.get("limit", 50))))

        if not pattern:
            _log.warning("search_content called without pattern")
            return ToolResult(
                content=json.dumps({"error": "请提供 pattern"}, ensure_ascii=False)
            )

        rg = _ensure_rg()
        if not rg:
            _log.warning("rg not found, can't search_content")
            return ToolResult(
                content=json.dumps(
                    {"error": "rg (ripgrep) 未安装，无法搜索文件内容"},
                    ensure_ascii=False,
                )
            )

        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )

        admin = is_admin_private(ctx, deps)
        try:
            target = sandbox_target(
                ctx.is_group, ctx.chat_id, search_path, deps, admin_override=admin
            )
        except ValueError as e:
            approved = await _approve_path_access(
                ctx, search_path, "search_content", str(e), deps
            )
            if approved is not None:
                target = approved
            else:
                _log.warning("search_content path rejected: %s", e)
                return ToolResult(
                    content=json.dumps({"error": str(e)}, ensure_ascii=False)
                )

        if not target.exists():
            return ToolResult(
                content=json.dumps(
                    {"error": f"路径不存在: {search_path}"}, ensure_ascii=False
                )
            )

        base_path = str(wm.sandbox_dir(ctx.is_group, ctx.chat_id))
        if admin:
            base_path = str(wm.root_dir())

        cmd = [rg, "--json", "--line-number", "--color=never", "--hidden"]
        cmd.extend(["--max-count", str(max(100, limit * 2))])
        if ignore_case:
            cmd.append("--ignore-case")
        if literal:
            cmd.append("--fixed-strings")
        if file_glob:
            cmd.extend(["--glob", file_glob])
        cmd.extend(["--", pattern, str(target)])
        _log.info(
            "search_content: %s %s in %s",
            pattern[:60],
            " (literal)" if literal else "",
            search_path,
        )

        SAFETY_LIMIT = 10000
        matches: list[str] = []
        match_count = 0
        total_matches = 0
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=await build_exec_env_for(deps.permission_manager),
            )
            assert proc.stdout is not None
            assert proc.stderr is not None

            while True:
                try:
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=30
                    )
                except asyncio.TimeoutError:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    return ToolResult(
                        content=json.dumps(
                            {"error": "搜索超时（30秒）"}, ensure_ascii=False
                        )
                    )

                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "match":
                    match_count += 1
                    if len(matches) < limit and match_count <= SAFETY_LIMIT:
                        data = event.get("data", {})
                        abs_path = (data.get("path", {}) or {}).get("text", "?")
                        rel_path = (
                            os.path.relpath(abs_path, base_path)
                            if abs_path != "?"
                            else "?"
                        )
                        line_number = data.get("line_number", "?")
                        line_text = (data.get("lines", {}) or {}).get("text", "")
                        matches.append(
                            f"{rel_path}:{line_number}: {line_text.rstrip()}"
                        )

                    if match_count > SAFETY_LIMIT:
                        proc.kill()
                        await proc.wait()
                        result_data = {
                            "success": True,
                            "matches": matches,
                            "total": match_count,
                            "limit_reached": True,
                            "truncated": True,
                        }
                        _log.info(
                            "search_content: %d+ matches for %s (safety limit)",
                            match_count,
                            pattern[:60],
                        )
                        return ToolResult(
                            content=json.dumps(result_data, ensure_ascii=False)
                        )

                elif event.get("type") == "summary":
                    stats = event.get("data", {}).get("stats", {})
                    total_matches = stats.get("matches", match_count)

            stderr_bytes = await proc.stderr.read()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            await proc.wait()

            if proc.returncode not in (0, 1) and stderr_text:
                _log.warning("search_content rg error: %s", stderr_text[:200])
                return ToolResult(
                    content=json.dumps({"error": stderr_text}, ensure_ascii=False)
                )

        except Exception as e:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            _log.warning("search_content rg exception: %s", e)
            return ToolResult(
                content=json.dumps({"error": f"搜索失败: {e}"}, ensure_ascii=False)
            )

        _log.info("search_content: %d matches for %s", match_count, pattern[:60])
        result_data = {
            "success": True,
            "matches": matches,
            "total": total_matches,
            "limit_reached": total_matches > len(matches),
            "truncated": False,
        }
        return ToolResult(content=json.dumps(result_data, ensure_ascii=False))

    async def _find_files(args: dict, ctx: ToolContext) -> ToolResult:
        raw_pattern = (args.get("pattern") or "").strip()
        search_path = (args.get("path") or ".").strip()
        limit = max(1, min(2000, int(args.get("limit", 200))))

        if not raw_pattern:
            _log.warning("find_files called without pattern")
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供 pattern（glob 模式）"}, ensure_ascii=False
                )
            )

        fd = _ensure_fd()
        if not fd:
            _log.warning("fd not found, can't find_files")
            return ToolResult(
                content=json.dumps(
                    {"error": "fd 未安装，无法搜索文件"}, ensure_ascii=False
                )
            )

        wm = deps.workspace_manager
        if not wm:
            return ToolResult(
                content=json.dumps({"error": "工作区未就绪"}, ensure_ascii=False)
            )

        admin = is_admin_private(ctx, deps)
        try:
            target = sandbox_target(
                ctx.is_group, ctx.chat_id, search_path, deps, admin_override=admin
            )
        except ValueError as e:
            approved = await _approve_path_access(
                ctx, search_path, "find_files", str(e), deps
            )
            if approved is not None:
                target = approved
            else:
                _log.warning("find_files path rejected: %s", e)
                return ToolResult(
                    content=json.dumps({"error": str(e)}, ensure_ascii=False)
                )

        if not target.exists():
            return ToolResult(
                content=json.dumps(
                    {"error": f"路径不存在: {search_path}"}, ensure_ascii=False
                )
            )
        if not target.is_dir():
            return ToolResult(
                content=json.dumps(
                    {"error": f"'{search_path}' 不是目录"}, ensure_ascii=False
                )
            )

        def _is_inside_git_repo(path: Path) -> bool:
            for p in [path, *path.parents]:
                if (p / ".git").exists():
                    return True
            return False

        base_path = str(wm.sandbox_dir(ctx.is_group, ctx.chat_id))
        if admin:
            base_path = str(wm.root_dir())

        fd_glob = raw_pattern
        cmd = [fd, "--glob", "--color=never", "--hidden"]
        resolved = str(target)
        if not _is_inside_git_repo(target):
            cmd.append("--no-require-git")
        if "/" in fd_glob and not fd_glob.startswith("**/"):
            cmd.append("--full-path")
            fd_glob = f"**/{fd_glob}"
        fd_limit = limit + 1
        cmd.extend(["--max-results", str(fd_limit), "--", fd_glob, resolved])
        _log.info("find_files: glob=%s in %s", fd_glob[:80], search_path)

        result = await _run_subprocess(cmd)
        if isinstance(result, ToolResult):
            return result
        returncode, stdout, stderr_text = result

        if returncode != 0 and stderr_text.strip():
            _log.warning("find_files fd error: %s", stderr_text[:200])
            return ToolResult(
                content=json.dumps({"error": stderr_text.strip()}, ensure_ascii=False)
            )

        relativized = []
        for line in stdout.splitlines():
            p = line.strip().rstrip("/\\")
            if not p:
                continue
            rel = os.path.relpath(p, base_path)
            if rel and rel != ".":
                relativized.append(rel)

        _log.info("find_files: %d files for glob=%s", len(relativized), fd_glob[:80])
        total = len(relativized)
        result_data = {
            "success": True,
            "files": relativized[:limit],
            "total": total,
            "limit_reached": total > limit,
        }
        return ToolResult(content=json.dumps(result_data, ensure_ascii=False))

    SEARCH_CONTENT_PARAMS = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "搜索模式（支持正则表达式）"},
            "path": {
                "type": "string",
                "description": "工作区内的相对路径，默认当前目录",
            },
            "glob": {"type": "string", "description": "文件名过滤 glob，如 *.py"},
            "ignore_case": {"type": "boolean", "description": "忽略大小写，默认 false"},
            "literal": {
                "type": "boolean",
                "description": "字面匹配（非正则），默认 false",
            },
            "limit": {
                "type": "integer",
                "description": "最大结果数，默认 50，最大 500",
            },
        },
        "required": ["pattern"],
    }

    FIND_FILES_PARAMS = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "glob 模式，如 **/*.py 或 src/**/*.ts",
            },
            "path": {
                "type": "string",
                "description": "工作区内的相对路径，默认当前目录",
            },
            "limit": {
                "type": "integer",
                "description": "最大结果数，默认 200，最大 2000",
            },
        },
        "required": ["pattern"],
    }

    return [
        ToolEntry(
            name="search_content",
            section="search",
            description=(
                "搜索文件内容（基于 ripgrep）。"
                "默认正则搜索，设 literal=true 做字面搜索（特殊字符无需转义）。"
                "设 glob=*.py 限定文件类型，设 ignore_case=true 忽略大小写。"
                "自动忽略 .gitignore。"
                "优先使用本工具搜索内容，不要使用 exec + grep/rg。"
                "默认限工作区内使用，管理员可通过审批访问越界路径。"
            ),
            parameters=SEARCH_CONTENT_PARAMS,
            handler=_search_content,
        ),
        ToolEntry(
            name="find_files",
            section="search",
            description=(
                "按 glob 模式搜索文件名（基于 fd），如 **/*.py 或 src/**/*.ts。"
                "自动忽略 .gitignore。"
                "优先使用本工具搜索文件，不要使用 exec + find/fd。"
                "默认限工作区内使用，管理员可通过审批访问越界路径。"
            ),
            parameters=FIND_FILES_PARAMS,
            handler=_find_files,
        ),
    ]
