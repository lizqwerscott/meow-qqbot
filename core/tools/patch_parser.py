"""Patch 解析与执行 — 解析 OpenAI/OpenClaw 风格的 patch 信封格式并应用到工作区文件。"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


class DiffError(ValueError):
    pass


@dataclass
class Hunk:
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    context_line: str = ""  # @@ skip-ahead anchor（不混入 old_lines）


@dataclass
class PatchAction:
    type: ActionType
    content: str = ""
    hunks: list[Hunk] = field(default_factory=list)
    move_path: str = ""


# ═══════════════════════════════════════════════════════════
# 解析器
# ═══════════════════════════════════════════════════════════

def parse_patch_text(text: str) -> dict[str, PatchAction]:
    """解析 patch 文本，返回 path → PatchAction 的映射。

    Raises DiffError 如果格式非法。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = text.split("\n")

    # 兼容 <<EOF / EOF 包裹（LLM 常见行为）
    if len(lines) >= 4 and (lines[0].startswith("<<") and lines[-1].strip() == "EOF"):
        lines = lines[1:-1]

    if len(lines) < 2:
        raise DiffError("Patch text too short")
    if not lines[0].startswith("*** Begin Patch"):
        raise DiffError("First line must be '*** Begin Patch'")
    if not lines[-1].startswith("*** End Patch"):
        raise DiffError("Last line must be '*** End Patch'")

    actions: dict[str, PatchAction] = {}
    idx = 1  # skip begin marker

    while idx < len(lines) - 1:  # stop before end marker
        line = lines[idx].rstrip()

        # 空行跳过
        if not line:
            idx += 1
            continue

        if line.startswith("*** Update File: "):
            path = line[len("*** Update File: "):]
            if path in actions:
                raise DiffError(f"Duplicate path in patch: {path}")
            idx += 1
            move_path = ""
            if idx < len(lines) - 1 and lines[idx].startswith("*** Move to: "):
                move_path = lines[idx][len("*** Move to: "):]
                idx += 1
            action, idx = _parse_update_hunks(lines, idx)
            action.move_path = move_path
            actions[path] = action

        elif line.startswith("*** Delete File: "):
            path = line[len("*** Delete File: "):]
            if path in actions:
                raise DiffError(f"Duplicate path in patch: {path}")
            actions[path] = PatchAction(type=ActionType.DELETE)
            idx += 1

        elif line.startswith("*** Add File: "):
            path = line[len("*** Add File: "):]
            if path in actions:
                raise DiffError(f"Duplicate path in patch: {path}")
            action, idx = _parse_add(lines, idx)
            actions[path] = action

        else:
            raise DiffError(f"Unexpected line {idx + 1}: {line[:80]}")

    return actions


def _parse_add(lines: list[str], idx: int) -> tuple[PatchAction, int]:
    """从 idx 开始解析一个 Add File hunk。"""
    idx += 1
    content_lines: list[str] = []
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("***"):
            break
        if not line:
            idx += 1
            continue
        if not line.startswith("+"):
            raise DiffError(f"Add File lines must start with '+', got: {line[:60]}")
        content_lines.append(line[1:])
        idx += 1
    return PatchAction(type=ActionType.ADD, content="\n".join(content_lines)), idx


def _parse_update_hunks(lines: list[str], idx: int) -> tuple[PatchAction, int]:
    """从 idx 开始解析一个 Update File 的 hunks。

    每个 `@@ 文本` 行作为 skip-ahead 锚点（对齐 OpenAI 官方实现）。
    文本存在 Hunk.context_line 中，仅在 apply 阶段用于定位，不混入 old_lines。
    """
    action = PatchAction(type=ActionType.UPDATE)
    while idx < len(lines):
        line = lines[idx].rstrip()

        # 检测下一个 hunk 开始或 patch 结束
        if line.startswith("*** End Patch") or line.startswith("*** Update File: "):
            break
        if line.startswith("*** Delete File: ") or line.startswith("*** Add File: "):
            break
        if not line:
            idx += 1
            continue

        # @@ 上下文标记 — 存储为定位锚点
        context_line = ""
        if line.startswith("@@ ") or line == "@@":
            context_line = line[2:].strip()  # @@ 之后的内容（若无则为空）
            idx += 1

        hunk, idx = _parse_one_hunk(lines, idx, context_line=context_line)
        if hunk is not None:
            action.hunks.append(hunk)

        # 检查是否到达 *** End of File
        if idx < len(lines) and lines[idx].rstrip() == "*** End of File":
            idx += 1
            break

    if not action.hunks:
        raise DiffError("Update hunk has no content")

    return action, idx


def _parse_one_hunk(
    lines: list[str], idx: int, context_line: str = "",
) -> tuple[Optional[Hunk], int]:
    """解析单个 Hunk（+/-/ 行）。

    context_line 不混入 old_lines，仅在 apply 时用作 skip-ahead 锚点。
    这样避免 @@ 内容与 - 行重复时导致匹配失败。
    """
    old_lines: list[str] = []
    new_lines: list[str] = []
    started = False

    while idx < len(lines):
        line = lines[idx]
        raw = line

        # 检查终止条件
        if raw.startswith((
            "@@",
            "*** End Patch",
            "*** Update File:",
            "*** Delete File:",
            "*** Add File:",
            "*** End of File",
        )):
            break
        if raw == "***":
            idx += 1
            break
        if raw.startswith("***"):
            raise DiffError(f"Invalid line in hunk: {raw[:60]}")

        idx += 1

        if not raw:
            raw = " "

        marker = raw[0]
        content = raw[1:]

        if marker == " ":
            started = True
            old_lines.append(content)
            new_lines.append(content)
        elif marker == "-":
            started = True
            old_lines.append(content)
        elif marker == "+":
            started = True
            new_lines.append(content)
        else:
            raise DiffError(f"Invalid line marker '{marker}' in hunk: {raw[:60]}")

    if not started:
        return None, idx
    return Hunk(old_lines=old_lines, new_lines=new_lines, context_line=context_line), idx


# ═══════════════════════════════════════════════════════════
# Update 匹配引擎（三级降级，对齐 OpenAI）
# ═══════════════════════════════════════════════════════════

def _normalizers() -> list[Callable[[str], str]]:
    return [
        lambda s: s,
        lambda s: s.rstrip(),
        lambda s: s.strip(),
    ]


def _find_context(
    lines: list[str], old_lines: list[str], start: int, eof: bool,
) -> int:
    """在 lines 中搜索 old_lines 序列，返回匹配起始行号。

    三级降级匹配：exact → rstrip → strip
    EOF 模式（对齐 OpenAI）：先试文件末尾单点，失败则回退 start 正向搜索。
    """
    if not old_lines or start >= len(lines):
        return -1
    if len(old_lines) > len(lines):
        return -1

    max_start = len(lines) - len(old_lines)
    nz = _normalizers()

    if eof:
        # 先精确试末尾位置（len(lines) - len(old_lines)）
        for norm in nz:
            if max_start >= start:
                ok = True
                for j, expected in enumerate(old_lines):
                    if norm(lines[max_start + j]) != norm(expected):
                        ok = False
                        break
                if ok:
                    return max_start
        # 回退到从 start 正向搜索（加惩罚但不影响 return）
        return _find_context(lines, old_lines, start, False)

    for norm in nz:
        for i in range(start, max_start + 1):
            ok = True
            for j, expected in enumerate(old_lines):
                if norm(lines[i + j]) != norm(expected):
                    ok = False
                    break
            if ok:
                return i
    return -1


def apply_update_hunks(content: str, hunks: list[Hunk]) -> str:
    """对文件内容应用所有 hunks，返回新内容。"""
    lines = content.split("\n")
    # 去掉尾随空行
    if lines and lines[-1] == "":
        lines = lines[:-1]

    replacements: list[tuple[int, int, list[str]]] = []
    search_start = 0

    for hunk in hunks:
        # @@ skip-ahead 锚点：先正向搜索，找不到再 EOF 兜底
        if hunk.context_line:
            ctx_idx = _find_context(lines, [hunk.context_line], search_start, False)
            if ctx_idx == -1:
                ctx_idx = _find_context(lines, [hunk.context_line], search_start, True)
            if ctx_idx != -1:
                search_start = ctx_idx if hunk.old_lines else ctx_idx + 1

        if not hunk.old_lines:
            # 纯新增 — 插入到当前 search_start 位置
            eof = search_start >= len(lines)
            idx = len(lines) if eof else search_start
            replacements.append((idx, 0, hunk.new_lines))
            continue

        idx = _find_context(lines, hunk.old_lines, search_start, False)
        if idx == -1:
            idx = _find_context(lines, hunk.old_lines, search_start, True)
        if idx == -1:
            raise DiffError(
                "Failed to find expected lines:\n" + "\n".join(hunk.old_lines)
            )

        replacements.append((idx, len(hunk.old_lines), hunk.new_lines))
        search_start = idx + len(hunk.old_lines)

    # 从后往前应用（保持索引稳定）
    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, count, new_lines in replacements:
        lines[start:start + count] = list(new_lines)

    return "\n".join(lines)
