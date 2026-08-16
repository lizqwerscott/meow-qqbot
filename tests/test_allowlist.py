"""Allowlist 匹配测试（路径 glob / bare name / argPattern / 链式逐段）。"""

import pytest

from core.approval.allowlist import (
    AllowlistEntry,
    entry_matches,
    match_allowlist,
    merge_allowlists,
)
from core.tools.exec_analysis import ExecSegment, ExecutableResolution

# /usr/local/bin/git 与 /usr/bin/grep 作为已知解析结果


def seg(argv, resolved, found_in_path=False):
    return ExecSegment(
        raw=" ".join(argv),
        argv=argv,
        resolution=ExecutableResolution(
            resolved_path=resolved, found_in_path=found_in_path
        ),
    )


# ── entry_matches ──


def test_bare_name_matches_path_basename():
    entry = AllowlistEntry(pattern="git")
    assert (
        entry_matches(
            entry, ["git", "status"], "/usr/local/bin/git", found_in_path=True
        )
        is True
    )


def test_bare_name_requires_found_in_path():
    # ./git 或 /tmp/git 不是 PATH 解析结果 → bare name 不匹配
    entry = AllowlistEntry(pattern="git")
    assert (
        entry_matches(entry, ["./git", "status"], "/tmp/git", found_in_path=False)
        is False
    )


def test_bare_name_does_not_match_other_binary():
    entry = AllowlistEntry(pattern="git")
    assert (
        entry_matches(entry, ["git", "status"], "/tmp/git-copy", found_in_path=True)
        is False
    )


def test_bare_name_no_resolved_path_fails():
    assert entry_matches(AllowlistEntry(pattern="git"), ["git"], None) is False


def test_path_glob_matches_resolved_path():
    entry = AllowlistEntry(pattern="/usr/local/bin/git")
    assert entry_matches(entry, ["git"], "/usr/local/bin/git") is True


def test_path_glob_star():
    entry = AllowlistEntry(pattern="/usr/*/bin/git")
    assert entry_matches(entry, ["git"], "/usr/local/bin/git") is True


def test_path_glob_double_star():
    import os

    home = os.path.expanduser("~")
    entry = AllowlistEntry(pattern="~/Projects/**/bin/tool")
    expanded = entry_matches(
        entry,
        ["tool"],
        f"{home}/Projects/a/b/bin/tool",
    )
    assert expanded is True


def test_arg_pattern_restricts():
    entry = AllowlistEntry(pattern="python3", arg_pattern=r"^safe\.py$")
    assert (
        entry_matches(
            entry, ["python3", "safe.py"], "/usr/bin/python3", found_in_path=True
        )
        is True
    )
    assert (
        entry_matches(
            entry, ["python3", "other.py"], "/usr/bin/python3", found_in_path=True
        )
        is False
    )


def test_arg_pattern_no_args_fails():
    entry = AllowlistEntry(pattern="python3", arg_pattern=r"^safe\.py$")
    assert entry_matches(entry, ["python3"], "/usr/bin/python3") is False


# ── match_allowlist ──


def test_all_segments_satisfied():
    entries = [
        AllowlistEntry(pattern="ls"),
        AllowlistEntry(pattern="grep"),
    ]
    segments = [
        seg(["ls", "-la"], "/bin/ls", found_in_path=True),
        seg(["grep", "foo"], "/bin/grep", found_in_path=True),
    ]
    satisfied, matches = match_allowlist(segments, entries)
    assert satisfied is True
    assert all(m is not None for m in matches)


def test_one_segment_miss():
    entries = [AllowlistEntry(pattern="ls")]
    segments = [
        seg(["ls"], "/bin/ls", found_in_path=True),
        seg(["vim"], "/usr/bin/vim", found_in_path=True),
    ]
    satisfied, matches = match_allowlist(segments, entries)
    assert satisfied is False
    assert matches[0] is not None
    assert matches[1] is None


def test_chain_with_relative_segment_misses():
    # 管道中 ./grep（相对路径，非 PATH 解析）不能被 bare name 覆盖
    entries = [AllowlistEntry(pattern="ls"), AllowlistEntry(pattern="grep")]
    segments = [
        seg(["ls"], "/bin/ls", found_in_path=True),
        seg(["./grep", "foo"], "/tmp/workdir/grep", found_in_path=False),
    ]
    satisfied, _ = match_allowlist(segments, entries)
    assert satisfied is False


def test_empty_segments_not_satisfied():
    satisfied, _ = match_allowlist([], [AllowlistEntry(pattern="ls")])
    assert satisfied is False


# ── merge_allowlists ──


def test_merge_dedup():
    a = [AllowlistEntry(pattern="git"), AllowlistEntry(pattern="ls")]
    b = [AllowlistEntry(pattern="git"), AllowlistEntry(pattern="grep")]
    merged = merge_allowlists(a, b)
    patterns = [e.pattern for e in merged]
    assert patterns == ["git", "ls", "grep"]


# ── 嵌套段（command_substitution / payload 内部命令）──


def test_nested_segment_miss_blocks_segment():
    from core.tools.exec_analysis import analyze_command

    entries = [AllowlistEntry(pattern="cat")]
    segments = analyze_command("cat $(rm -rf /tmp)/x", env=None, cwd="/")
    # cat 命中，但嵌套 rm 无对应条目 → 整段 miss
    satisfied, matches = match_allowlist(segments, entries)
    assert satisfied is False
    assert matches[0] is None


def test_nested_segment_hit_requires_all():
    from core.tools.exec_analysis import analyze_command

    entries = [
        AllowlistEntry(pattern="cat"),
        AllowlistEntry(pattern="pwd"),
    ]
    segments = analyze_command("cat $(pwd)/x.txt", env=None, cwd="/")
    satisfied, matches = match_allowlist(segments, entries)
    assert satisfied is True
    assert matches[0] is not None


def test_shell_payload_inner_requires_allowlist():
    from core.tools.exec_analysis import analyze_command

    entries = [AllowlistEntry(pattern="bash")]
    segments = analyze_command("bash -c 'rm -rf /'", env=None, cwd="/")
    satisfied, _ = match_allowlist(segments, entries)
    assert satisfied is False  # payload 内 rm 无条目


# ── 包装器段（2.1：内层命令参与 allowlist 匹配）──


def seg_wrapped(argv, inner, resolved_outer, resolved_inner, found_in_path=True):
    return ExecSegment(
        raw=" ".join(argv),
        argv=argv,
        resolution=ExecutableResolution(
            resolved_path=resolved_outer, found_in_path=found_in_path
        ),
        inner_argv=inner,
        inner_resolution=ExecutableResolution(
            resolved_path=resolved_inner, found_in_path=found_in_path
        ),
    )


def test_wrapper_segment_matches_inner_bare_name():
    # timeout 外层无条目，但内层 python3 命中 → 整段满足
    entries = [AllowlistEntry(pattern="python3")]
    segments = [
        seg_wrapped(
            ["timeout", "5", "python3", "x.py"],
            ["python3", "x.py"],
            "/usr/bin/timeout",
            "/usr/bin/python3",
        )
    ]
    satisfied, matches = match_allowlist(segments, entries)
    assert satisfied is True
    assert matches[0] is not None


def test_wrapper_segment_miss_when_inner_unlisted():
    # 只有外层 timeout 的条目 → 内层 python3 无条目 → miss
    entries = [AllowlistEntry(pattern="timeout")]
    segments = [
        seg_wrapped(
            ["timeout", "5", "python3", "x.py"],
            ["python3", "x.py"],
            "/usr/bin/timeout",
            "/usr/bin/python3",
        )
    ]
    satisfied, _ = match_allowlist(segments, entries)
    assert satisfied is False


def test_wrapper_arg_pattern_applies_to_inner_args():
    entries = [AllowlistEntry(pattern="python3", arg_pattern=r"^x\.py$")]
    segments = [
        seg_wrapped(
            ["timeout", "5", "python3", "x.py"],
            ["python3", "x.py"],
            "/usr/bin/timeout",
            "/usr/bin/python3",
        )
    ]
    satisfied, _ = match_allowlist(segments, entries)
    assert satisfied is True
    # 内层参数变成 y.py → miss
    segs2 = [
        seg_wrapped(
            ["timeout", "5", "python3", "y.py"],
            ["python3", "y.py"],
            "/usr/bin/timeout",
            "/usr/bin/python3",
        )
    ]
    satisfied2, _ = match_allowlist(segs2, entries)
    assert satisfied2 is False


def test_analyzed_wrapper_matches_inner_entry():
    from core.tools.exec_analysis import analyze_command

    segments = analyze_command(
        "timeout 5 ls -la", env={"PATH": "/bin:/usr/bin"}, cwd="/"
    )
    entries = [AllowlistEntry(pattern="ls")]
    satisfied, _ = match_allowlist(segments, entries)
    assert satisfied is True


def test_wrapped_payload_requires_inner_and_nested():
    """嵌套包装器 + payload：内层 flock 与 payload 内命令都必须命中。"""
    from core.tools.exec_analysis import analyze_command

    command = "timeout 5 flock /tmp/l -c 'ls -la'"
    entries = [AllowlistEntry(pattern="flock"), AllowlistEntry(pattern="ls")]
    segments = analyze_command(command, env={"PATH": "/bin:/usr/bin"}, cwd="/")
    satisfied, _ = match_allowlist(segments, entries)
    assert satisfied is True

    only_outer = [AllowlistEntry(pattern="flock")]
    segments2 = analyze_command(command, env={"PATH": "/bin:/usr/bin"}, cwd="/")
    satisfied2, _ = match_allowlist(segments2, only_outer)
    assert satisfied2 is False  # payload 内 ls 未授权 → miss


def test_wrapper_inner_misses_safe_bin_outer_only():
    """safe-bin 也应看内层：timeout 5 head -5 命中 head profile。"""
    from core.approval.allowlist import match_safe_bins

    segments = [
        seg_wrapped(
            ["timeout", "5", "head", "-5"],
            ["head", "-5"],
            "/usr/bin/timeout",
            "/usr/bin/head",
        )
    ]
    satisfied, matches = match_safe_bins(segments, ("head",))
    assert satisfied is True
    assert matches[0]["bin"] == "head"


def test_wrapper_inner_not_safe_bin_miss():
    from core.approval.allowlist import match_safe_bins

    segments = [
        seg_wrapped(
            ["timeout", "5", "vim", "x.txt"],
            ["vim", "x.txt"],
            "/usr/bin/timeout",
            "/usr/bin/vim",
        )
    ]
    satisfied, _ = match_safe_bins(segments, ("head",))
    assert satisfied is False
