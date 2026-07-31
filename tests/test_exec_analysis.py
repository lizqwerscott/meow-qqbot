"""命令分析器测试（切段 / 真实路径解析 / inline-eval 检测）。"""

import os

import pytest

from core.tools.exec_analysis import (
    analyze_command,
    detect_inline_eval,
    resolve_executable,
    split_shell_segments,
)

# ── split_shell_segments ──


def test_single_command():
    assert split_shell_segments("ls -la") == [["ls", "-la"]]


def test_chain_and_pipe():
    assert split_shell_segments("a && b || c; d | e") == [
        ["a"],
        ["b"],
        ["c"],
        ["d"],
        ["e"],
    ]


def test_operator_inside_quotes_not_split():
    assert split_shell_segments('echo "a && b"') == [["echo", "a && b"]]


def test_redirect_fd_ampersand_not_split():
    # 2>&1 中的 & 是重定向的一部分（`&` 不是切段符，`&&` 才是）
    assert split_shell_segments("cmd 2>&1") == [["cmd", "2>", "&", "1"]]


def test_no_space_chain():
    # 紧贴单词也能切段（&& 由两个 & 合并还原）
    assert split_shell_segments("a;b") == [["a"], ["b"]]
    assert split_shell_segments("a||b") == [["a"], ["b"]]
    assert split_shell_segments("a&&b") == [["a"], ["b"]]
    assert split_shell_segments("a && b") == [["a"], ["b"]]


def test_unbalanced_quotes_returns_empty():
    assert split_shell_segments('echo "abc') == []


def test_empty_input():
    assert split_shell_segments("") == []


# ── resolve_executable ──


def test_resolve_absolute_path():
    r = resolve_executable(["/bin/echo"])
    assert r.resolved_path is not None
    assert os.path.isabs(r.resolved_path)


def test_resolve_bare_name_via_path():
    r = resolve_executable(["ls"], env={"PATH": "/usr/bin:/bin"})
    assert r.resolved_path is not None
    assert r.found_in_path is True
    assert os.path.basename(r.resolved_path) == "ls"


def test_resolve_not_found():
    r = resolve_executable(["definitely-not-a-real-cmd-xyz"], env={"PATH": "/usr/bin"})
    assert r.resolved_path is None
    assert "not found" in r.reason


def test_resolve_relative_path():
    r = resolve_executable(["./tool"], env={"PATH": ""}, cwd="/nonexistent-dir")
    assert r.resolved_path is None  # 目录不存在


def test_resolve_empty_argv():
    assert resolve_executable([]).resolved_path is None


# ── detect_inline_eval ──


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-c", "print(1)"],
        ["python3", "-c", "print(1)"],
        ["node", "-e", "console.log(1)"],
        ["node", "--eval", "1"],
        ["ruby", "-e", "puts 1"],
        ["perl", "-e", "print 1"],
        ["php", "-r", "echo 1;"],
        ["lua", "-e", "print(1)"],
        ["osascript", "-e", "say hi"],
        ["bash", "-c", "ls"],
        ["sh", "-c", "ls"],
        ["awk", "{print $1}"],
        ["xargs", "echo"],
        ["make"],
        ["find", ".", "-exec", "rm", "{}", ";"],
    ],
)
def test_inline_eval_detected(argv):
    assert detect_inline_eval(argv) is True, argv


@pytest.mark.parametrize(
    "argv",
    [
        ["ls", "-la"],
        ["git", "status"],
        ["python", "script.py"],
        ["node", "app.js"],
        ["sed", "s/a/b/", "file"],  # 无 -e/-f，sed 位置参数是文件
        ["find", ".", "-name", "*.py"],
        ["grep", "-r", "foo"],
    ],
)
def test_inline_eval_not_detected(argv):
    assert detect_inline_eval(argv) is False, argv


# ── analyze_command ──


def test_analyze_chain_marks_all_segments():
    segments = analyze_command("ls && grep foo", env={"PATH": "/bin"})
    assert len(segments) == 2
    assert all(s.shell_chain for s in segments)


def test_analyze_invalid_command():
    assert analyze_command('echo "x', env={}) == []


def test_analyze_segment_ops():
    segments = analyze_command("ls && grep foo | wc", env={"PATH": "/bin"})
    assert [s.op for s in segments] == ["", "&&", "|"]
    assert segments[0].shell_chain is True


def test_analyze_single_segment_op_empty():
    segments = analyze_command("ls -la", env={"PATH": "/bin"})
    assert len(segments) == 1
    assert segments[0].op == ""


# ── 嵌套分析（CST：command_substitution / shell wrapper payload）──


def test_nested_substitution_segment():
    segments = analyze_command("cat $(pwd)/x.txt", env=os.environ)
    assert len(segments) == 1
    assert segments[0].op == ""
    nested = segments[0].nested_segments
    assert len(nested) == 1
    assert nested[0].nested is True
    assert nested[0].argv == ["pwd"]


def test_nested_shell_payload_segment():
    segments = analyze_command("bash -c 'rm -rf /tmp/x'", env=os.environ)
    assert len(segments) == 1
    nested = segments[0].nested_segments
    assert len(nested) == 1
    assert nested[0].argv[0] == "rm"


def test_nested_inline_eval_detected():
    # payload 内 python -c 也要被 inline 检测到
    segments = analyze_command("bash -c \"python3 -c 'print(1)'\"", env=os.environ)
    assert any(s.inline_eval for s in segments[0].nested_segments)


def test_nested_depth_limit():
    # 两层 wrapper 嵌套展开，第三层截断（对齐 openclaw depth 2）
    segments = analyze_command("bash -c 'sh -c \"echo hi\"'", env=os.environ)
    assert len(segments) == 1
    inner = segments[0].nested_segments  # sh -c 展开
    assert len(inner) == 1
    inner2 = inner[0].nested_segments  # sh 的 payload 展开
    assert len(inner2) == 1
    assert inner2[0].argv == ["echo", "hi"]
    # 第三层不再展开
    assert inner2[0].nested_segments == []


def test_compound_segment_marked():
    segments = analyze_command("for i in 1 2; do echo $i; done", env=os.environ)
    assert segments[0].is_compound is True


def test_syntax_error_fail_closed():
    assert analyze_command('echo "unclosed', env=os.environ) == []


def test_parser_init_failure_fail_closed(monkeypatch):
    # tree-sitter 初始化失败 → fail-closed 空列表
    import core.tools.bash_cst as bc

    monkeypatch.setattr(bc, "_get_parser", lambda: None)
    assert analyze_command("ls -la", env=os.environ) == []


def test_parser_parse_exception_fail_closed(monkeypatch):
    import core.tools.bash_cst as bc

    class _BrokenParser:
        def parse(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(bc, "_get_parser", lambda: _BrokenParser())
    assert analyze_command("ls -la", env=os.environ) == []


def test_trailing_redirect_chain_segments():
    # 回归：尾随重定向不再塌缩整链
    segments = analyze_command("echo hi | rm -rf / > /dev/null", env=os.environ)
    assert len(segments) == 2
    assert segments[1].argv[0] == "rm"  # rm 段参与 allowlist
