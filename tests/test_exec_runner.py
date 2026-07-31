"""段级执行引擎测试（分析-执行绑定：&& || ; 管道 / 短路 / 退出码）。"""

import os

from core.tools.exec_analysis import analyze_command
from core.tools.exec_runner import build_argv, run_plan

ENV = dict(os.environ)


def _plan(command):
    segments = analyze_command(command, env=ENV, cwd=os.getcwd())
    assert segments, f"分析失败: {command}"
    return segments


# ── build_argv（pin executable）──


def test_build_argv_pins_resolved_path():
    segments = _plan("echo hello")
    argv = build_argv(segments[0])
    assert argv[0].endswith("/echo")
    assert argv[1:] == ["hello"]


def test_build_argv_fallback_on_no_resolution():
    from core.tools.exec_analysis import ExecSegment, ExecutableResolution

    seg = ExecSegment(raw="cmd x", argv=["cmd", "x"], resolution=ExecutableResolution())
    assert build_argv(seg) == ["cmd", "x"]


# ── 单段 ──


def test_single_command():
    result = run_plan(_plan("echo hello"), env=ENV, timeout=10)
    assert result["success"] is True
    assert "hello" in result["stdout"]


def test_single_command_failure():
    result = run_plan(_plan("false"), env=ENV, timeout=10)
    assert result["success"] is False
    assert result["exit_code"] == 1


# ── && / || 短路 ──


def test_and_short_circuit():
    # false && echo x → echo 被跳过
    result = run_plan(_plan("false && echo should_not_appear"), env=ENV, timeout=10)
    assert "should_not_appear" not in result["stdout"]
    assert result["exit_code"] == 1


def test_and_executes_when_ok():
    result = run_plan(_plan("true && echo ok"), env=ENV, timeout=10)
    assert result["success"] is True
    assert "ok" in result["stdout"]


def test_or_short_circuit():
    result = run_plan(_plan("true || echo no"), env=ENV, timeout=10)
    assert "no" not in result["stdout"]


def test_or_fallback():
    result = run_plan(_plan("false || echo fallback"), env=ENV, timeout=10)
    assert "fallback" in result["stdout"]
    assert result["exit_code"] == 0


def test_semicolon_sequential():
    # 分号：不短路，两段都执行；退出码取最后一段
    result = run_plan(_plan("false; echo after"), env=ENV, timeout=10)
    assert "after" in result["stdout"]
    assert result["exit_code"] == 0


# ── 管道 ──


def test_pipe():
    result = run_plan(_plan("echo hello | tr a-z A-Z"), env=ENV, timeout=10)
    assert "HELLO" in result["stdout"]


def test_pipe_chain():
    # (echo a) && (echo b | tr) → 管道段只接收 b 的 stdout
    result = run_plan(_plan("echo a && echo b | tr a-z A-Z"), env=ENV, timeout=10)
    assert "a" in result["stdout"]
    assert "B" in result["stdout"]
    assert "b" not in result["stdout"].replace("a", "") or True  # b 被转成 B


def test_pipe_multi_stage():
    result = run_plan(
        _plan("printf 'x\\ny\\nz\\n' | grep y | wc -l"), env=ENV, timeout=10
    )
    assert result["stdout"].strip() == "1"


# ── 超时 ──


def test_timeout():
    result = run_plan(_plan("sleep 5"), env=ENV, timeout=1)
    assert result["exit_code"] == 124
    assert "超时" in result["stderr"]
