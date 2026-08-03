"""Heredoc 检测测试（对齐 openclaw reason: heredoc 审批触发点）。"""

import os

from core.tools.exec_analysis import analyze_command, iter_all_segments

ENV = dict(os.environ)


def _has_heredoc(command):
    segs = analyze_command(command, env=ENV, cwd=os.getcwd())
    assert segs, f"分析失败: {command}"
    return any(s.heredoc for s in iter_all_segments(segs))


def test_heredoc_detected():
    assert _has_heredoc("cat <<EOF\nhello\nEOF") is True
    assert _has_heredoc("python3 <<'PY'\nprint(1)\nPY") is True
    assert _has_heredoc("cat <<-EOF\n  indented\nEOF") is True


def test_heredoc_with_chain():
    # 链式命令中某段含 heredoc
    assert _has_heredoc("echo start && cat <<EOF\nx\nEOF") is True


def test_heredoc_not_detected_normal():
    assert _has_heredoc("echo hello") is False
    assert _has_heredoc("ls -la | head -5") is False
    assert _has_heredoc("cat file.txt") is False
    assert _has_heredoc("echo 'a<<b'") is False  # 字符串中的 << 不算


def test_heredoc_analysis_still_segments():
    # heredoc 命令仍能正常切段（不会因为标记而破坏分析）
    segs = analyze_command("cat <<EOF\nx\nEOF", env=ENV, cwd=os.getcwd())
    assert segs
    assert segs[0].heredoc is True
