"""Allowlist 匹配 — 移植 OpenClaw exec-approvals-allowlist 模型。

条目形态（对齐 openclaw）：
- bare name（不含 /）：如 ``rg``，只匹配"通过 PATH 解析出的二进制 basename"。
  不匹配 ``./rg`` 或 ``/tmp/rg``（那两类必须用路径 glob 显式信任）。
- 路径 glob（含 /）：如 ``~/Projects/**/bin/rg``、``/opt/homebrew/bin/rg``，
  匹配解析后的绝对路径；支持 ``**`` 多级通配与 ``~`` 展开。
- arg_pattern：可选 ECMAScript 风格正则（Python re），对 argv[1:] 空格拼接
  做 search 匹配；省略 = 仅路径匹配。示例 ``^safe\\.py$`` 只允许
  ``python3 safe.py``。
- 链式/管道命令：每个顶层 segment 都必须命中，全部命中才 satisfied。
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.tools.exec_analysis import ExecSegment


@dataclass
class AllowlistEntry:
    """单条 allowlist 规则。"""

    pattern: str
    arg_pattern: Optional[str] = None
    source: str = "manual"  # manual | allow-always
    id: str = ""
    last_used_at: int = 0
    last_used_command: str = ""
    last_resolved_path: str = ""


def _expand_home(pattern: str) -> str:
    if pattern == "~" or pattern.startswith("~/"):
        home = os.path.expanduser("~")
        return os.path.join(home, pattern[2:]) if len(pattern) > 2 else home
    return pattern


def _glob_to_regex(pattern: str) -> re.Pattern:
    """路径 glob → 正则。** 跨目录，* 单目录内，? 单字符。"""
    i = 0
    out = []
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                # 吃掉后面的 /
                if i < n and pattern[i] == "/":
                    i += 1
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                cls = pattern[i + 1 : j]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                out.append("[" + cls.replace("\\", "\\\\") + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def entry_matches(
    entry: AllowlistEntry,
    argv: List[str],
    resolved_path: Optional[str],
    found_in_path: bool = False,
) -> bool:
    """单条规则是否命中某段命令。

    bare-name 规则（不含 /）只匹配"通过 PATH 解析"出的二进制
    （found_in_path=True），不匹配 ./x 或 /tmp/x。
    """
    if not resolved_path:
        return False
    pattern = _expand_home(entry.pattern)

    if "/" not in pattern:
        # bare name：只匹配 PATH 解析出的 basename
        if not found_in_path:
            return False
        if not fnmatch.fnmatch(os.path.basename(resolved_path), pattern):
            return False
    else:
        if not _glob_to_regex(pattern).fullmatch(resolved_path):
            return False

    if entry.arg_pattern:
        args_text = " ".join(argv[1:])
        try:
            if not re.search(entry.arg_pattern, args_text):
                return False
        except re.error:
            return False
    return True


def _match_one(
    seg: ExecSegment,
    entries: List[AllowlistEntry],
) -> Optional[AllowlistEntry]:
    """匹配单个段（含嵌套段递归：内部命令 miss 则整段 miss）。"""
    hit: Optional[AllowlistEntry] = None
    for entry in entries:
        if entry_matches(
            entry,
            seg.argv,
            seg.resolution.resolved_path,
            found_in_path=seg.resolution.found_in_path,
        ):
            hit = entry
            break
    if hit is not None and seg.nested_segments:
        nested_ok = all(
            _match_one(nested, entries) is not None for nested in seg.nested_segments
        )
        if not nested_ok:
            hit = None
    return hit


def match_allowlist(
    segments: List[ExecSegment],
    entries: List[AllowlistEntry],
) -> Tuple[bool, List[Optional[AllowlistEntry]]]:
    """对命令所有 segment 做 allowlist 匹配（含嵌套内部命令）。

    Returns:
        (satisfied, matches)：satisfied 为 True 当且仅当每个 segment 及其
        嵌套段都命中至少一条规则；matches[i] 为 segments[i] 命中的条目
        （miss 为 None）。
    """
    matches: List[Optional[AllowlistEntry]] = [
        _match_one(seg, entries) for seg in segments
    ]
    satisfied = bool(segments) and all(m is not None for m in matches)
    return satisfied, matches


def merge_allowlists(*groups: List[AllowlistEntry]) -> List[AllowlistEntry]:
    """合并多组 allowlist（静态 [commands].allowed + 运行时审批白名单）。"""
    seen: set = set()
    merged: List[AllowlistEntry] = []
    for group in groups:
        for entry in group:
            key = (entry.pattern, entry.arg_pattern)
            if key not in seen:
                seen.add(key)
                merged.append(entry)
    return merged
