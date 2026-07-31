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
