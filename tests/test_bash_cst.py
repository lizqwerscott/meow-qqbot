"""tree-sitter-bash CST 切段提取器测试。"""

import os

import pytest

from core.tools.bash_cst import parse_shell_command

ENV = dict(os.environ)


def _ops(command):
    segs = parse_shell_command(command)
    assert segs is not None, f"解析失败: {command}"
    return [(s.op, s.text) for s in segs]


# ── 切段 ──


def test_chain_ops():
    assert _ops("a && b || c; d | e") == [
        ("", "a"),
        ("&&", "b"),
        ("||", "c"),
        (";", "d"),
        ("|", "e"),
    ]


def test_no_space_chain():
    assert _ops("a&&b") == [("", "a"), ("&&", "b")]
    assert _ops("a;b") == [("", "a"), (";", "b")]


def test_operator_inside_quotes_not_split():
    assert _ops('echo "a && b"') == [("", 'echo "a && b"')]


def test_redirect_stays_in_segment():
    assert _ops("ls -la > out.txt") == [("", "ls -la > out.txt")]
    assert _ops("cmd 2>&1 | tee log") == [
        ("", "cmd 2>&1"),
        ("|", "tee log"),
    ]


def test_pipeline_priority():
    # a && b | c：管道优先于 &&（CST 中 b|c 是 pipeline）
    assert _ops("a && b | c") == [("", "a"), ("&&", "b"), ("|", "c")]


# ── substitution 提取 ──


def test_command_substitution_extracted():
    segs = parse_shell_command("cat $(pwd)/x.txt")
    assert segs is not None
    assert segs[0].substitutions == ["pwd"]


def test_backtick_substitution_extracted():
    segs = parse_shell_command("echo `date`")
    assert segs is not None
    assert segs[0].substitutions == ["date"]


# ── 复合命令 ──


def test_compound_marked():
    segs = parse_shell_command("for i in 1 2; do echo $i; done")
    assert segs is not None
    assert segs[0].is_compound is True
    assert segs[0].op == ""


# ── fail-closed ──


def test_syntax_error_returns_none():
    assert parse_shell_command("echo 'unclosed") is None
    assert parse_shell_command("if then fi") is None


def test_empty_command():
    segs = parse_shell_command("")
    assert segs == []


# ── 尾随重定向（CST 提升包裹链的回归修复）──


def test_trailing_redirect_does_not_collapse_chain():
    # a && b > f：重定向提升为包裹整个链的 redirected_statement，
    # 必须展开内部链，重定向挂到最后一段（不能塌缩成单段）
    assert _ops("a && b > f") == [("", "a"), ("&&", "b > f")]
    assert _ops("a | b > f") == [("", "a"), ("|", "b > f")]
    assert _ops("a && b | c > f") == [("", "a"), ("&&", "b"), ("|", "c > f")]


def test_ampersand_chain_token():
    # & 是链操作符（后台/并列），CST 切段
    assert _ops("a & b") == [("", "a"), ("&", "b")]
