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
    source: str = "manual"  # manual | allow-always | legacy
    id: str = ""
    last_used_at: int = 0
    last_used_command: str = ""
    last_resolved_path: str = ""
    uses: int = 0  # 2.4 使用计数（命中次数）


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
    """匹配单个段（含嵌套段递归：内部命令 miss 则整段 miss）。

    包装器段（seg.inner_argv）：**只认内层命令**——allow-always 持久化的是
    内层可执行路径（timeout 5 python3 x.py 命中 python3 条目），授权 timeout
    不代表授权其内层任意命令。
    """
    hit: Optional[AllowlistEntry] = None
    if seg.inner_argv and seg.inner_resolution and seg.inner_resolution.resolved_path:
        # 包装器段（2.1）：只认内层命令——授权 timeout 不代表授权其内层任意命令，
        # 外层单独命中不算（对齐 openclaw "persist the inner executable path"）。
        for entry in entries:
            if entry_matches(
                entry,
                seg.inner_argv,
                seg.inner_resolution.resolved_path,
                found_in_path=seg.inner_resolution.found_in_path,
            ):
                hit = entry
                break
    else:
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


# ── Safe bins（对齐 openclaw tools.exec.safeBins + safeBinProfiles）──
#
# 预信任的窄 stdin 过滤器：命中且满足 profile 的段视为 allowlist 满足，
# 无需显式白名单条目。profile 字段（对齐 openclaw）：
#   max_positional:      最多允许的位置参数数（默认 0——窄过滤器通常只吃 stdin）
#   allowed_value_flags: 允许的带值 flag（如 -n、--lines），值本身不校验
#   allowed_flags:       允许的布尔 flag（如 -q、--quiet）
#   denied_flags:        禁止的 flag（如 -f/--follow 会挂起）

# 内置默认 profiles（用户可通过 safe_bin_profiles 覆盖）
DEFAULT_SAFE_BIN_PROFILES: Dict[str, dict] = {
    "head": {
        "max_positional": 0,
        "allowed_value_flags": ["-n", "--lines", "-c", "--bytes"],
        "allowed_flags": ["-q", "--quiet", "-v", "--verbose"],
        "denied_flags": [],
    },
    "tail": {
        "max_positional": 0,
        "allowed_value_flags": ["-n", "--lines", "-c", "--bytes"],
        "allowed_flags": ["-q", "--quiet", "-v", "--verbose"],
        # -f/--follow 会挂起（前台超时兜底，但无意义地占用），拒绝
        "denied_flags": ["-f", "--follow"],
    },
    "wc": {
        "max_positional": 0,
        # wc 的 -l/-w/-c/-m/-L 都是布尔 flag（无参数）；
        # 不开放 --files0-from（从文件读文件名列表，违反"不读文件"边界）
        "allowed_value_flags": [],
        "allowed_flags": [
            "-l",
            "--lines",
            "-w",
            "--words",
            "-c",
            "--bytes",
            "-m",
            "--chars",
            "-L",
            "--max-line-length",
        ],
        "denied_flags": [],
    },
    "tr": {
        "max_positional": 2,  # tr SET1 [SET2]
        "allowed_value_flags": [],
        "allowed_flags": [
            "-d",
            "--delete",
            "-s",
            "--squeeze-repeats",
            "-c",
            "-C",
            "--complement",
        ],
        "denied_flags": [],
    },
}


def _split_flag(arg: str):
    """拆出 flag 与其内联值。

    ``--flag=value`` → ("--flag", "value")；短 flag（含 -abc 组合、-n5 粘连）
    原样返回 (arg, None)，组合/粘连展开由调用方（_check_safe_bin_argv）处理；
    非 flag 返回 (None, None)——调用方已前置过滤，此分支实际不可达。
    """
    if arg.startswith("--"):
        if "=" in arg:
            flag, _, value = arg.partition("=")
            return flag, value
        return arg, None
    return arg, None


def _check_safe_bin_argv(bin_name: str, argv: List[str], profile: dict) -> bool:
    """校验 argv[1:] 是否符合 safe-bin profile。"""
    max_positional = int(profile.get("max_positional", 0))
    allowed_value = set(profile.get("allowed_value_flags", []) or [])
    allowed_bool = set(profile.get("allowed_flags", []) or [])
    denied = set(profile.get("denied_flags", []) or [])

    positional: List[str] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            positional.extend(argv[i + 1 :])
            break
        if not arg.startswith("-") or arg == "-":
            positional.append(arg)
            i += 1
            continue
        flag, value = _split_flag(arg)
        # 纯数字短参数（-5）：等价 -n 5（head/tail 老式语法），安全放行
        if re.fullmatch(r"-\d+", flag):
            i += 1
            continue
        if flag in denied:
            return False
        if flag in allowed_value:
            # 带值 flag：--lines=5 已带值；否则吃掉下一个参数作为值
            if value is None and i + 1 < len(argv):
                i += 1
            i += 1
            continue
        if flag in allowed_bool:
            i += 1
            continue
        # 未声明的 flag：检查短组合（-qv → -q -v）；含未声明字符则拒绝
        if flag.startswith("-") and not flag.startswith("--") and len(flag) > 2:
            # 兼容 -n5（短值 flag 粘连，-n5 ≈ -n 5）
            short_value_flags = {f[1] for f in allowed_value if len(f) == 2}
            if flag[1] in short_value_flags:
                i += 1
                continue
            chars = flag[1:]
            ok = True
            for ch in chars:
                f = f"-{ch}"
                if f in denied:
                    return False
                if f in allowed_bool:
                    continue
                ok = False
                break
            if ok:
                i += 1
                continue
        return False

    if len(positional) > max_positional:
        return False
    return True


def match_safe_bins(
    segments: List[ExecSegment],
    safe_bins: tuple,
    profiles: Optional[Dict[str, dict]] = None,
) -> Tuple[bool, List[Optional[dict]]]:
    """safe-bin 自动放行匹配：每个顶层段 argv[0] 在 safe_bins 且 argv 满足 profile。

    Returns:
        (satisfied, matches)：satisfied 为 True 当且仅当每个段（含嵌套段）
        都命中 safe-bin（嵌套段不满足则整段视为 miss）。
    """
    if not safe_bins:
        return False, [None] * len(segments)
    bins = set(safe_bins)
    merged_profiles = dict(DEFAULT_SAFE_BIN_PROFILES)
    if profiles:
        merged_profiles.update(profiles)

    def _match_one(seg: ExecSegment) -> Optional[dict]:
        if not seg.argv:
            return None
        # 包装器段：外层 + 内层（最内层命令）都参与 safe-bin 判定
        candidates = [(seg.argv, seg.resolution)]
        if seg.inner_argv and seg.inner_resolution:
            candidates.append((seg.inner_argv, seg.inner_resolution))
        for argv, res in candidates:
            if not argv or not res.resolved_path or not res.found_in_path:
                continue
            bin_name = os.path.basename(argv[0])
            if bin_name not in bins:
                continue
            profile = merged_profiles.get(bin_name, {})
            if not _check_safe_bin_argv(bin_name, argv, profile):
                continue
            if seg.nested_segments:
                if not all(_match_one(n) is not None for n in seg.nested_segments):
                    continue
            return {"bin": bin_name}
        return None

    matches: List[Optional[dict]] = [_match_one(seg) for seg in segments]
    satisfied = bool(segments) and all(m is not None for m in matches)
    return satisfied, matches
