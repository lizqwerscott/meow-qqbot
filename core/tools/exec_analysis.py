"""Shell 命令分析 — 移植 OpenClaw exec-approvals-analysis 的思路（Python 版）。

OpenClaw 的三个关键差异，本模块逐一实现：
1. 按 shell 链操作符切段，每段独立做 argv 解析 + 可执行文件解析 + allowlist 匹配
2. allowlist 匹配的是"解析后的真实二进制路径"，不是命令名 basename
3. 内联求值形式（python -c / node -e / osascript -e 等）即使二进制在
   allowlist 里也要强制审批（strictInlineEval 纵深防御）

切段基于 tree-sitter-bash CST（core/tools/bash_cst.py）：按 ``&& || ; | &``
切段，``$(...)``/反引号/``<(...)`` 内部命令与 shell wrapper ``-c`` payload
递归分析（深度 2）。段内 token 仍用 shlex（保留重定向等原始 token，执行
语义与 subprocess.run(shell=False) 一致）。解析失败 fail-closed 返回空列表。
"""

from __future__ import annotations

import logging
import os
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

# 内联求值表单（对齐 openclaw strictInlineEval 列举的形态）
_INLINE_EVAL_ARGS: Dict[str, frozenset] = {
    "python": frozenset({"-c"}),
    "python2": frozenset({"-c"}),
    "python3": frozenset({"-c"}),
    "node": frozenset({"-e", "--eval", "-p"}),
    "ruby": frozenset({"-e"}),
    "perl": frozenset({"-e", "-E"}),
    "php": frozenset({"-r"}),
    "lua": frozenset({"-e"}),
    "lua5.1": frozenset({"-e"}),
    "lua5.3": frozenset({"-e"}),
    "lua5.4": frozenset({"-e"}),
    "osascript": frozenset({"-e"}),
    "bash": frozenset({"-c"}),
    "sh": frozenset({"-c"}),
    "dash": frozenset({"-c"}),
    "zsh": frozenset({"-c"}),
    "ksh": frozenset({"-c"}),
}

# 无参数形式直接视为内联求值（program 是位置参数 / 读 stdin 执行）
_ALWAYS_INLINE_EVAL = frozenset({"awk", "xargs", "make"})

# shell wrapper：-c payload 内部命令需递归分析（对齐 openclaw wrapper payload）
_SHELL_WRAPPERS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})


def extract_shell_payloads(argv: List[str]) -> List[str]:
    """提取 shell wrapper 的 -c payload（如 bash -c 'rm -rf /' → ['rm -rf /']）。"""
    if len(argv) >= 3 and os.path.basename(argv[0]) in _SHELL_WRAPPERS:
        if argv[1] == "-c":
            return [argv[2]]
    return []


@dataclass
class ExecutableResolution:
    """单段命令的可执行文件解析结果。"""

    resolved_path: Optional[str] = None  # realpath 后的绝对路径
    found_in_path: bool = False  # 是否通过 PATH 解析（bare name 允许匹配的前提）
    reason: str = ""


@dataclass
class ExecSegment:
    """shell 链中的一段命令。"""

    raw: str
    argv: List[str]
    resolution: ExecutableResolution = field(default_factory=ExecutableResolution)
    inline_eval: bool = False
    shell_chain: bool = False  # 命令含多个 segment（链式/管道）
    op: str = ""  # 与上一段之间的连接操作符（首段为空串；&& | ; ||）
    nested: bool = (
        False  # 是否为嵌套段（command_substitution / shell wrapper payload 内部）
    )
    is_compound: bool = False  # 复合命令（for/if/...，shell=False 下无可执行文件）
    nested_segments: List["ExecSegment"] = field(default_factory=list)  # 内部命令段


def iter_all_segments(segments: List[ExecSegment]):
    """递归产出顶层 + 所有嵌套段（allowlist/inline 判定用）。"""
    for seg in segments:
        yield seg
        yield from iter_all_segments(seg.nested_segments)


def _tokenize_shell(command: str) -> List[str]:
    """引号感知的 shell token 化（失败返回空列表）。"""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";|&")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return []


def split_shell_command(command: str) -> List[tuple[str, List[str]]]:
    """按链操作符切成 ``(op, argv)`` 对（CST 精确切段，段内 shlex token）。

    基于 tree-sitter-bash：正确识别 `$(...)`、管道优先级、复合命令边界，
    且引号内操作符不切。返回空列表表示解析失败（fail-closed）。
    """
    from core.tools.bash_cst import parse_shell_command

    cst = parse_shell_command(command)
    if cst is None:
        return []
    pairs: List[tuple[str, List[str]]] = []
    for cseg in cst:
        argv = _tokenize_shell(cseg.text)
        if argv:
            pairs.append((cseg.op, argv))
    return pairs


def split_shell_segments(command: str) -> List[List[str]]:
    """兼容层：只返回 argv 段列表（丢弃操作符）。"""
    return [argv for _, argv in split_shell_command(command)]


def resolve_executable(
    argv: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> ExecutableResolution:
    """解析 argv[0] 的真实可执行文件路径（PATH 查找 + realpath）。"""
    if not argv:
        return ExecutableResolution(reason="empty argv")
    command = argv[0]
    env = env if env is not None else os.environ
    cwd = cwd if cwd is not None else os.getcwd()

    def _is_executable(p: str) -> bool:
        try:
            return os.path.isfile(p) and os.access(p, os.X_OK)
        except OSError:
            return False

    if os.path.isabs(command):
        resolved = os.path.realpath(command)
        if _is_executable(resolved):
            return ExecutableResolution(resolved_path=resolved)
        return ExecutableResolution(reason=f"not executable: {command}")

    if "/" in command:
        # 相对路径（./x、subdir/x）
        resolved = os.path.realpath(os.path.join(cwd, command))
        if _is_executable(resolved):
            return ExecutableResolution(resolved_path=resolved)
        return ExecutableResolution(reason=f"not executable: {command}")

    # bare name → PATH 查找
    path_value = env.get("PATH", "") or ""
    for d in path_value.split(os.pathsep):
        if not d:
            continue
        cand = os.path.realpath(os.path.join(d, command))
        if _is_executable(cand):
            return ExecutableResolution(resolved_path=cand, found_in_path=True)
    return ExecutableResolution(reason=f"not found in PATH: {command}")


def detect_inline_eval(argv: List[str]) -> bool:
    """检测内联求值形式（strictInlineEval 用）。"""
    if not argv:
        return False
    cmd = os.path.basename(argv[0])
    if cmd in _ALWAYS_INLINE_EVAL:
        return True
    flags = _INLINE_EVAL_ARGS.get(cmd)
    if flags:
        for arg in argv[1:]:
            if arg in flags:
                return True
    if cmd == "find":
        return "-exec" in argv[1:] or "-execdir" in argv[1:]
    return False


def analyze_command(
    command: str,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    _depth: int = 0,
) -> List[ExecSegment]:
    """完整分析一条命令：CST 切段（含操作符）→ 每段解析可执行文件 +
    内联求值检测 + 嵌套命令分析（$(...) / 反引号 / shell wrapper payload）。

    Args:
        command: 原始命令字符串
        env: 子进程环境（PATH 解析用）
        cwd: 工作目录
        _depth: 内部递归深度（嵌套分析上限 2，对齐 openclaw wrapper depth）

    Returns:
        ExecSegment 列表；空列表表示解析失败（fail-closed）。
    """
    from core.tools.bash_cst import parse_shell_command

    cst = parse_shell_command(command)
    if cst is None:
        return []
    if not cst:
        return []
    is_chain = len(cst) > 1
    segments: List[ExecSegment] = []
    for cseg in cst:
        argv = _tokenize_shell(cseg.text)
        if not argv:
            # 段内 token 化失败：fail-closed（与 CST 失败同等对待）
            return []
        resolution = resolve_executable(argv, env=env, cwd=cwd)
        seg = ExecSegment(
            raw=cseg.text,
            argv=argv,
            resolution=resolution,
            inline_eval=detect_inline_eval(argv),
            shell_chain=is_chain,
            op=cseg.op,
            is_compound=cseg.is_compound,
        )
        if _depth < 2:
            nested: List[ExecSegment] = []
            # 1) command_substitution / 反引号内部命令
            for sub in cseg.substitutions:
                nested.extend(analyze_command(sub, env=env, cwd=cwd, _depth=_depth + 1))
            # 2) shell wrapper -c payload 内部命令
            for payload in extract_shell_payloads(argv):
                nested.extend(
                    analyze_command(payload, env=env, cwd=cwd, _depth=_depth + 1)
                )
            for n in nested:
                n.nested = True
                n.shell_chain = True
            seg.nested_segments = nested
        segments.append(seg)
    return segments
