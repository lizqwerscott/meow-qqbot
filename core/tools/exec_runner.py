"""段级执行引擎 — 分析-执行绑定（对齐 openclaw 的 plan 重建执行）。

现状问题：exec 分析层按 ``&& || ; |`` 切段逐段放行，但执行仍是
``subprocess.run(parts, shell=False)`` 整条跑，链式命令会把 ``&&`` 当
参数传给第一个命令而失败。

本模块解决：按分析结果逐段执行，保持 shell=False（不经过 shell，防注入），
实现 ``&&``/``||`` 短路、``;`` 顺序、``|`` 管道（PIPE 级联），且每段的
argv[0] 用解析后的真实二进制路径（pin executable，对齐 openclaw）。

从左到右模型对常见组合与 bash 一致：``a | b && c`` ≈ ``(a|b) && c``，
``a && b | c`` 中 ``c`` 只接收 ``b`` 的 stdout（与 bash 的 a && (b|c) 等价）。
"""

from __future__ import annotations

import subprocess
import time
from typing import Dict, List, Optional

from core.tools.exec_analysis import ExecSegment

# 输出聚合上限（与 exec 工具一致）
_MAX_OUTPUT_BYTES = 100_000


def build_argv(seg: ExecSegment) -> List[str]:
    """执行用 argv：argv[0] 用解析后的真实路径（找不到时回退原命令）。"""
    if seg.resolution and seg.resolution.resolved_path:
        return [seg.resolution.resolved_path, *seg.argv[1:]]
    return list(seg.argv)


def run_plan(
    segments: List[ExecSegment],
    *,
    env: Dict[str, str],
    cwd: Optional[str] = None,
    timeout: float = 60,
) -> dict:
    """逐段执行 shell 计划，返回与 exec 工具同构的结果 dict。

    Returns:
        {"success", "exit_code", "stdout", "stderr", "truncated"}
        success=整体退出码为 0（最后实际执行的段）。
    """
    if not segments:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "命令为空",
            "truncated": {"stdout": False, "stderr": False},
        }

    deadline = time.monotonic() + max(timeout, 1)
    prev_rc = 0
    prev_stdout: Optional[bytes] = None
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []
    last_rc = 0
    skip = False

    for index, seg in enumerate(segments):
        # seg.op 是该段与上一段的连接符（首段为 ''）：
        # && / || 短路判定、| 管道输入都基于它。
        if index > 0:
            if seg.op == "&&" and prev_rc != 0:
                skip = True
            elif seg.op == "||" and prev_rc == 0:
                skip = True
            else:
                skip = False
        if skip:
            last_rc = prev_rc
            continue

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "success": False,
                "exit_code": 124,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks) + "\n命令执行超时",
                "truncated": {"stdout": False, "stderr": False},
            }

        argv = build_argv(seg)
        stdin_input = prev_stdout if (index > 0 and seg.op == "|") else None
        try:
            result = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                input=stdin_input,
                timeout=max(remaining, 1),
                env=env,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exit_code": 124,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks)
                + f"\n段 {index + 1} 执行超时: {' '.join(seg.argv[:4])}",
                "truncated": {"stdout": False, "stderr": False},
            }
        except FileNotFoundError:
            return {
                "success": False,
                "exit_code": 127,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks) + f"\n命令不存在: {seg.argv[0]}",
                "truncated": {"stdout": False, "stderr": False},
            }
        except OSError as e:
            return {
                "success": False,
                "exit_code": 126,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks) + f"\n执行失败: {e}",
                "truncated": {"stdout": False, "stderr": False},
            }

        prev_rc = result.returncode
        prev_stdout = result.stdout
        last_rc = result.returncode
        # 管道语义：本段 stdout 被下一段消费（下一段连接符为 |）时不进最终输出；
        # stderr 不进管道，全部聚合。
        piped_away = index + 1 < len(segments) and segments[index + 1].op == "|"
        if result.stdout and not piped_away:
            stdout_chunks.append(result.stdout.decode("utf-8", errors="replace"))
        if result.stderr:
            stderr_chunks.append(result.stderr.decode("utf-8", errors="replace"))

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    return {
        "success": last_rc == 0,
        "exit_code": last_rc,
        "stdout": (
            stdout[-_MAX_OUTPUT_BYTES:] if len(stdout) > _MAX_OUTPUT_BYTES else stdout
        ),
        "stderr": (
            stderr[-_MAX_OUTPUT_BYTES:] if len(stderr) > _MAX_OUTPUT_BYTES else stderr
        ),
        "truncated": {
            "stdout": len(stdout) > _MAX_OUTPUT_BYTES,
            "stderr": len(stderr) > _MAX_OUTPUT_BYTES,
        },
    }
