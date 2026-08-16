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

# 转发包装器（对齐 openclaw wrapper unwrapping）：allow-always 持久化**内层**
# 可执行路径（timeout 10 python3 x.py 记住的是 python3），包装器本身无需授权
WRAPPER_BINS = frozenset({"env", "flock", "nice", "nohup", "stdbuf", "timeout"})
# 多调用二进制：busybox/toybox 按 applet 名解包
MULTICALL_BINS = frozenset({"busybox", "toybox"})

# wrapper 带值 flags（flag → 需额外跳过一个后续 token；未列出 = 布尔 flag）
_WRAPPER_VALUE_FLAGS: Dict[str, frozenset] = {
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    # -c/--command 的值为命令字符串（payload），由 extract_wrapper_payloads 另分析
    "flock": frozenset({"-E", "--conflict-exit-code", "-w", "--wait"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "nohup": frozenset(),
    "stdbuf": frozenset({"-i", "--input", "-o", "--output", "-e", "--error"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
}


def _wrapper_inner_start(argv: List[str]) -> Optional[int]:
    """定位 wrapper 内层命令的 argv 起点（逐 wrapper 跳位）。

    - timeout：flags 后第一个位置参数是时长，命令从第二个位置参数开始
    - flock：  flags 后第一个位置参数是锁文件，命令从第二个位置参数开始
    - env：    跳过 flags 与 ``VAR=VAL`` 赋值
    - nice/nohup/stdbuf：跳过 flags，第一个位置参数即命令

    返回 None 表示无法解包（无内层命令）。
    """
    if not argv:
        return None
    cmd = os.path.basename(argv[0])
    value_flags = _WRAPPER_VALUE_FLAGS.get(cmd, frozenset())
    i = 1
    n = len(argv)
    while i < n:
        arg = argv[i]
        if cmd == "env" and "=" in arg and not arg.startswith("-"):
            i += 1
            continue
        if arg == "--":
            return i + 1
        if arg.startswith("-") and arg != "-":
            flag = arg.split("=", 1)[0]
            if flag in value_flags and "=" not in arg:
                i += 2  # flag + 值
            else:
                i += 1
            continue
        break
    if cmd in ("timeout", "flock"):
        start = i + 1  # 跳过时长/锁文件
        # flock -c 'cmd' 形态：命令是 -c 的 flag 值，不做位置解包
        # （payload 由 extract_wrapper_payloads 另走嵌套分析）
        if cmd == "flock" and start < len(argv) and argv[start] in ("-c", "--command"):
            return None
        return start
    return i


def unwrap_wrapper(argv: List[str], _depth: int = 0) -> Optional[List[str]]:
    """解包转发包装器，返回**最内层**命令 argv（递归深度 2）。

    - env/flock/nice/nohup/stdbuf/timeout：跳过 flags 与前置位置参数
    - busybox/toybox：argv[1] 为 applet 名（--list 等元操作不解包）

    非 wrapper / 无法解包 → None（调用方回退外层语义）。
    注意：解包只影响分析/匹配/持久化，**不影响执行**（仍跑外层命令）。
    """
    if not argv or _depth >= 2:
        return None
    cmd = os.path.basename(argv[0])
    if cmd in MULTICALL_BINS:
        if len(argv) >= 2 and not argv[1].startswith("-"):
            inner = [argv[1], *argv[2:]]
        else:
            return None
    elif cmd in WRAPPER_BINS:
        start = _wrapper_inner_start(argv)
        if start is None or start >= len(argv):
            return None
        inner = list(argv[start:])
    else:
        return None
    deeper = unwrap_wrapper(inner, _depth + 1)
    return deeper if deeper is not None else inner


def extract_wrapper_payloads(argv: List[str]) -> List[str]:
    """提取 wrapper 的 -c 型 payload（flock -c 'cmd'，与 shell wrapper 同构）。

    flock 的 ``-c`` 值是一条 shell 命令字符串（flock 内部经 sh 执行），
    需递归分析内部命令（对齐 openclaw wrapper payload 处理）。
    """
    if len(argv) >= 3 and os.path.basename(argv[0]) == "flock":
        for j, arg in enumerate(argv[1:-1], start=1):
            if arg in ("-c", "--command"):
                return [argv[j + 1]]
    return []


# ── 2.2 解释器/runtime 命令绑定（对齐 openclaw：审批绑定唯一具体本地文件）──
#
# 解释器/运行时命令（python3/node）审批时绑定精确 argv 快照 + 目标脚本文件；
# pnpm/npm/npx exec 解包到 node_modules/.bin 的唯一具体文件；
# 无法唯一确定（eval 形态、模块形态、多文件/loader 链、bin 缺失）→
# unique=False → 调用方置 analysis_ok=False 强制审批，且 allow-always 不落盘
# （对齐 openclaw "denied instead of claiming semantic coverage"）。

INTERPRETER_BINS = frozenset({"python", "python3", "node", "pnpm", "npm", "npx"})

# 包管理器 exec 形态（解包到本地 node_modules/.bin）
_PKG_EXEC_BINS = frozenset({"pnpm", "npm", "npx"})

# python 解释器：布尔 flags（跳过）；带值 flags（跳过 flag+值）；终止形态（-c/-m）
_PY_BOOL_FLAGS = frozenset(
    {
        "-B",
        "-E",
        "-i",
        "-I",
        "-O",
        "-OO",
        "-P",
        "-q",
        "-s",
        "-S",
        "-u",
        "-v",
        "-x",
        "-d",
    }
)
_PY_VALUE_FLAGS = frozenset({"-W", "-X", "-Q"})
_PY_TERMINAL_FLAGS = frozenset({"-c", "-m"})  # inline / 模块形态：不声称覆盖

# node：多文件/求值形态直接不绑定；其余 - 开头 token 保守跳过
_NODE_TERMINAL_FLAGS = frozenset(
    {
        "-e",
        "--eval",
        "-p",
        "--print",
        "-r",
        "--require",
        "--import",
        "-i",
        "--input-type",
        "--loader",
        "--experimental-loader",
    }
)


# 元命令 flags：只打印版本/帮助，不执行任何用户代码（无需绑定，可放行）。
# 注意 python 的 -v 是 verbose（启动解释器），不是版本——不列入
_META_FLAGS: Dict[str, frozenset] = {
    "python": frozenset({"-V", "--version", "-h", "--help"}),
    "python3": frozenset({"-V", "--version", "-h", "--help"}),
    "node": frozenset({"-v", "--version", "-h", "--help"}),
    "pnpm": frozenset({"-v", "--version", "-h", "--help"}),
    "npm": frozenset({"-v", "--version", "-h", "--help"}),
    "npx": frozenset({"-v", "--version", "-h", "--help"}),
}


def _is_meta_command(argv: List[str]) -> bool:
    """纯元命令形态：参数全是版本/帮助 flag（python3 --version / npm -h）。

    混合形态（--version extra / -V script.py）不算——有位置参数时按脚本
    或 fail-closed 处理，不在此放行。
    """
    if not argv or len(argv) < 2:
        return False
    metas = _META_FLAGS.get(os.path.basename(argv[0]), frozenset())
    return bool(metas) and all(a in metas for a in argv[1:])


def _interpreter_script(argv: List[str]) -> Optional[str]:
    """提取解释器命令的目标脚本参数（跳过 flags）；非脚本形态返回 None。"""
    if not argv:
        return None
    cmd = os.path.basename(argv[0])
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            i += 1
            break
        if not arg.startswith("-") or arg == "-":
            break
        flag = arg.split("=", 1)[0]
        if cmd.startswith("python"):
            if flag in _PY_TERMINAL_FLAGS:
                return None
            if flag in _PY_VALUE_FLAGS and "=" not in arg:
                i += 2
                continue
            if flag in _PY_VALUE_FLAGS:
                i += 1
                continue
            if flag in _PY_BOOL_FLAGS:
                i += 1
                continue
            # 未识别的 python flag：保守跳过（fail-closed 由文件存在性兜底）
            i += 1
            continue
        # node：多文件/求值形态不绑定；其余 flag 跳过
        if flag in _NODE_TERMINAL_FLAGS:
            return None
        i += 1
        continue
    if i >= len(argv):
        return None
    return argv[i]


def _resolve_package_bin_path(base: str, bin_path: object) -> Optional[str]:
    """解析 package.json#bin 的单一路径；非字符串或非普通文件视为未绑定。"""
    if not isinstance(bin_path, str) or not bin_path:
        return None
    candidate = os.path.realpath(os.path.join(base, bin_path))
    return candidate if os.path.isfile(candidate) else None


def _find_local_bin(cwd: Optional[str], bin_name: str) -> Optional[str]:
    """向上逐级查 node_modules/.bin/<bin>；未命中查 package.json#bin 唯一条目。"""
    base = os.path.abspath(cwd or os.getcwd())
    d = base if os.path.isdir(base) else os.path.dirname(base)
    while True:
        p = os.path.join(d, "node_modules", ".bin", bin_name)
        if os.path.isfile(p):
            return os.path.realpath(p)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    pkg = os.path.join(base, "package.json")
    if os.path.isfile(pkg):
        try:
            with open(pkg, encoding="utf-8") as f:
                import json

                data = json.load(f)
        except Exception:
            return None
        bins = data.get("bin")
        if isinstance(bins, str):
            return _resolve_package_bin_path(base, bins)
        if isinstance(bins, dict) and len(bins) == 1:
            return _resolve_package_bin_path(base, next(iter(bins.values())))
    return None


def _pkg_exec_bin(argv: List[str], cwd: Optional[str]) -> Optional[str]:
    """pnpm/npm/npx exec 形态 → 解包到唯一本地文件；否则 None。"""
    cmd = os.path.basename(argv[0])
    if cmd == "pnpm":
        if len(argv) < 3 or argv[1] != "exec" or argv[2].startswith("-"):
            return None
        return _find_local_bin(cwd, argv[2])
    if cmd == "npm":
        if len(argv) < 3 or argv[1] != "exec":
            return None
        j = 2
        if argv[j] == "--":
            j += 1
        if j >= len(argv) or argv[j].startswith("-"):
            return None
        return _find_local_bin(cwd, argv[j])
    # npx
    if len(argv) < 2 or argv[1].startswith("-"):
        return None
    return _find_local_bin(cwd, argv[1])


def resolve_interpreter_target(
    argv: List[str], cwd: Optional[str] = None
) -> tuple[Optional[str], bool]:
    """解释器/runtime 命令绑定到唯一具体本地文件（对齐 openclaw）。

    Returns:
        (realpath, unique)：
        - realpath: 绑定的文件绝对路径（无法绑定为 None）
        - unique:   True=可唯一确定并绑定；False=无法声称覆盖
          （eval/模块/多文件形态、目标文件不存在、bin 缺失）→ 调用方
          强制走审批且 allow-always 不落白名单。
    """
    if not argv:
        return None, False
    cmd = os.path.basename(argv[0])
    if cmd not in INTERPRETER_BINS:
        return None, False
    if cmd in _PKG_EXEC_BINS:
        target = _pkg_exec_bin(argv, cwd)
        if target is not None:
            return target, True
        # 元命令（npm --version / npx -h）：不执行用户代码，无需绑定即可放行
        if _is_meta_command(argv):
            return None, True
        return None, False
    script = _interpreter_script(argv)
    if script is None:
        # 元命令形态（python3 --version）与 eval/模块形态区分：
        # 前者不执行用户代码 → 可放行；后者（-c/-m 等）→ 不声称覆盖
        if _is_meta_command(argv):
            return None, True
        return None, False
    if os.path.isfile(script):
        return os.path.realpath(script), True
    if os.path.isabs(script) or os.path.isdir(cwd or os.getcwd()):
        joined = os.path.join(cwd or os.getcwd(), script)
        if os.path.isfile(joined):
            return os.path.realpath(joined), True
    return None, False


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
    heredoc: bool = (
        False  # 段内含 heredoc（<<EOF，对齐 openclaw reason: heredoc 审批触发）
    )
    nested_segments: List["ExecSegment"] = field(default_factory=list)  # 内部命令段
    # 包装器解包（2.1）：argv[0] 为转发包装器（timeout/env/...）时，
    # inner_argv 为**最内层**命令 argv，inner_resolution 为其可执行文件解析。
    # 仅影响分析/匹配/持久化；执行仍跑外层命令。
    inner_argv: List[str] = field(default_factory=list)
    inner_resolution: ExecutableResolution = field(default_factory=ExecutableResolution)


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
        inner = unwrap_wrapper(argv)
        seg = ExecSegment(
            raw=cseg.text,
            argv=argv,
            resolution=resolution,
            inline_eval=detect_inline_eval(argv),
            shell_chain=is_chain,
            op=cseg.op,
            is_compound=cseg.is_compound,
            heredoc=cseg.has_heredoc,
        )
        if inner is not None:
            # 包装器解包：内层命令解析 + inline-eval 穿透（timeout 5 python3 -c ...）
            seg.inner_argv = inner
            seg.inner_resolution = resolve_executable(inner, env=env, cwd=cwd)
            seg.inline_eval = seg.inline_eval or detect_inline_eval(inner)
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
            # 3) wrapper -c 型 payload（flock -c 'cmd'，与 2 同构）
            for payload in extract_wrapper_payloads(argv):
                nested.extend(
                    analyze_command(payload, env=env, cwd=cwd, _depth=_depth + 1)
                )
            # 4) 内层命令自身的 shell payload（timeout 5 bash -c '...' / flock -c '...'）
            if inner is not None:
                for payload in extract_shell_payloads(inner):
                    nested.extend(
                        analyze_command(payload, env=env, cwd=cwd, _depth=_depth + 1)
                    )
                for payload in extract_wrapper_payloads(inner):
                    nested.extend(
                        analyze_command(payload, env=env, cwd=cwd, _depth=_depth + 1)
                    )
            for n in nested:
                n.nested = True
                n.shell_chain = True
            seg.nested_segments = nested
        segments.append(seg)
    return segments
