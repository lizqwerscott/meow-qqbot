"""命令分析器测试（切段 / 真实路径解析 / inline-eval 检测）。"""

import json
import os

import pytest

from core.tools.exec_analysis import (
    analyze_command,
    detect_inline_eval,
    resolve_executable,
    resolve_interpreter_target,
    split_shell_segments,
    unwrap_wrapper,
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


# ── 包装器解包（2.1：allow-always 持久化内层可执行路径）──


def test_unwrap_timeout_skips_duration():
    assert unwrap_wrapper(["timeout", "10", "python3", "x.py"]) == [
        "python3",
        "x.py",
    ]


def test_unwrap_timeout_with_flags():
    assert unwrap_wrapper(["timeout", "-s", "KILL", "10", "python3", "x.py"]) == [
        "python3",
        "x.py",
    ]
    assert unwrap_wrapper(["timeout", "--signal=KILL", "--", "10", "ls"]) == ["ls"]
    assert unwrap_wrapper(["timeout", "--bogus", "10", "ls"]) is None
    assert unwrap_wrapper(["timeout", "-s"]) is None


def test_unwrap_env_skips_assignments_and_flags():
    assert unwrap_wrapper(["env", "FOO=1", "BAR=2", "ls", "-la"]) == ["ls", "-la"]
    assert unwrap_wrapper(["env", "-u", "HOME", "ls"]) == ["ls"]
    assert unwrap_wrapper(["env", "--unset=HOME", "--", "ls", "-la"]) == [
        "ls",
        "-la",
    ]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["env", "-S", "ls -la"], ["ls", "-la"]),
        (["env", "-S", "FOO=1 python3 x.py"], ["python3", "x.py"]),
        (["env", "--split-string=ls -la"], ["ls", "-la"]),
        (["env", "-S"], None),
        (["env", "-S", "'unclosed", "ls"], None),
        (["env", "-Sls", "target"], None),
        (["env", "--bogus", "ls"], None),
    ],
)
def test_unwrap_env_split_string(argv, expected):
    assert unwrap_wrapper(argv) == expected


def test_unwrap_flock_skips_lockfile():
    assert unwrap_wrapper(["flock", "/tmp/lock", "python3", "x.py"]) == [
        "python3",
        "x.py",
    ]


def test_unwrap_nice_nohup_stdbuf():
    assert unwrap_wrapper(["nice", "-n", "5", "ls"]) == ["ls"]
    assert unwrap_wrapper(["nohup", "python3", "x.py"]) == ["python3", "x.py"]
    assert unwrap_wrapper(["stdbuf", "-oL", "grep", "x"]) == ["grep", "x"]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["nice", "-n", "5", "--", "ls"], ["ls"]),
        (["nohup", "--", "python3", "x.py"], ["python3", "x.py"]),
        (["stdbuf", "-i0", "-o0", "--", "grep", "x"], ["grep", "x"]),
        (
            ["stdbuf", "--input=0", "--output=0", "--", "grep", "x"],
            ["grep", "x"],
        ),
        (["busybox", "sh", "-c", "ls"], ["sh", "-c", "ls"]),
        (["nice", "--bogus", "5", "ls"], None),
        (["stdbuf", "-oL", "--bogus", "grep", "x"], None),
        (["nohup", "--bogus", "ls"], None),
    ],
)
def test_unwrap_wrapper_combination_matrix(argv, expected):
    assert unwrap_wrapper(argv) == expected


def test_unwrap_busybox_applet():
    assert unwrap_wrapper(["busybox", "rm", "-rf", "/"]) == ["rm", "-rf", "/"]
    # busybox 元操作不解包
    assert unwrap_wrapper(["busybox", "--list"]) is None


def test_unwrap_nested_recursive():
    assert unwrap_wrapper(["timeout", "5", "nohup", "python3", "x.py"]) == [
        "python3",
        "x.py",
    ]


def test_unwrap_not_wrapper_returns_none():
    assert unwrap_wrapper(["ls", "-la"]) is None
    assert unwrap_wrapper(["python3", "x.py"]) is None
    assert unwrap_wrapper([]) is None


def test_unwrap_fail_closed():
    # 无内层命令（只有时长/锁文件）→ 不解包
    assert unwrap_wrapper(["timeout"]) is None
    assert unwrap_wrapper(["timeout", "5"]) is None
    assert unwrap_wrapper(["flock", "/tmp/lock"]) is None
    # flock -c 形态（命令是 flag 值）不解包，payload 另走嵌套分析
    assert unwrap_wrapper(["flock", "/tmp/lock", "-c", "echo hi"]) is None


def test_analyze_wrapper_sets_inner_fields():
    segments = analyze_command("timeout 5 python3 x.py", env=os.environ)
    seg = segments[0]
    assert seg.inner_argv == ["python3", "x.py"]
    assert seg.inner_resolution.resolved_path is not None
    assert seg.inner_resolution.found_in_path is True


def test_analyze_wrapper_inline_penetration():
    # timeout 包 python3 -c：内层 inline 必须被检测（strictInlineEval 纵深防御）
    segments = analyze_command("timeout 5 python3 -c 'print(1)'", env=os.environ)
    assert segments[0].inline_eval is True


def test_analyze_wrapper_shell_payload_nested():
    # timeout 5 bash -c 'rm ...'：payload 内 rm 进嵌套段（黑名单穿透）
    segments = analyze_command("timeout 5 bash -c 'rm -rf /tmp/x'", env=os.environ)
    nested = segments[0].nested_segments
    assert any(s.argv[0] == "rm" for s in nested)


def test_analyze_flock_c_payload_nested():
    # flock -c 'cmd'：command 字符串与 shell wrapper -c payload 同构
    segments = analyze_command("flock /tmp/l -c 'python3 x.py'", env=os.environ)
    nested = segments[0].nested_segments
    assert any(s.argv == ["python3", "x.py"] for s in nested)


def test_analyze_flock_command_equals_payload_nested():
    segments = analyze_command("flock /tmp/l --command='python3 x.py'", env=os.environ)
    assert segments[0].inner_argv == []
    nested = segments[0].nested_segments
    assert any(s.argv == ["python3", "x.py"] for s in nested)


def test_analyze_wrapped_flock_c_payload_nested():
    # 包装 flock -c：timeout 5 flock /tmp/l -c 'rm ...' → payload 内 rm 进嵌套段
    # （黑名单已移除，rm 必须参与 allowlist 匹配，否则 flock 命中即直跑）
    segments = analyze_command(
        "timeout 5 flock /tmp/l -c 'rm -rf /tmp/x'", env=os.environ
    )
    nested = segments[0].nested_segments
    assert any(s.argv[0] == "rm" for s in nested)


# ── 2.2 解释器/runtime 绑定（唯一具体文件，否则不声称覆盖）──


def test_interp_python_script_bound(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("print(1)")
    target, unique = resolve_interpreter_target(["python3", str(f)], cwd=str(tmp_path))
    assert unique is True
    assert target == os.path.realpath(str(f))


def test_interp_python_relative_script_bound(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("print(1)")
    target, unique = resolve_interpreter_target(
        ["python3", "script.py"], cwd=str(tmp_path)
    )
    assert unique is True
    assert target == os.path.realpath(str(f))


def test_interp_python_skips_boolean_flags(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("x")
    target, unique = resolve_interpreter_target(
        ["python3", "-O", "-B", "script.py"], cwd=str(tmp_path)
    )
    assert unique is True
    assert target == os.path.realpath(str(f))


def test_interp_python_c_inline_not_bound(tmp_path):
    target, unique = resolve_interpreter_target(
        ["python3", "-c", "print(1)"], cwd=str(tmp_path)
    )
    assert unique is False
    assert target is None


def test_interp_python_m_module_not_bound(tmp_path):
    target, unique = resolve_interpreter_target(
        ["python3", "-m", "http.server"], cwd=str(tmp_path)
    )
    assert unique is False


def test_interp_python_unknown_flag_not_bound(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("print(1)")

    target, unique = resolve_interpreter_target(
        ["python3", "--unknown", "script.py"], cwd=str(tmp_path)
    )

    assert unique is False
    assert target is None


def test_interp_python_missing_file_not_bound(tmp_path):
    target, unique = resolve_interpreter_target(
        ["python3", "nope.py"], cwd=str(tmp_path)
    )
    assert unique is False


# ── 元命令形态（--version/-h 等：不执行用户代码，无需绑定即可放行）──


@pytest.mark.parametrize(
    "argv",
    [
        ["python3", "--version"],
        ["python3", "-V"],
        ["python3", "--help"],
        ["node", "--version"],
        ["node", "-v"],
        ["node", "-h"],
        ["npm", "--version"],
        ["npm", "-v"],
        ["pnpm", "--version"],
        ["npx", "-h"],
    ],
)
def test_interp_meta_command_ok_to_run(argv):
    # 元命令：可放行（unique=True，无绑定文件）——不再强制审批
    target, unique = resolve_interpreter_target(argv, cwd="/tmp")
    assert unique is True
    assert target is None


@pytest.mark.parametrize(
    "argv",
    [
        ["python3", "-c", "print(1)"],  # inline 仍强制审批
        ["node", "-e", "1"],
        ["python3", "-m", "http.server"],
        ["npm", "install"],  # 非 exec 非 meta → 不覆盖
        ["pnpm", "exec", "eslint"],  # exec 无本地 bin → 不覆盖
        ["node", "--require", "./dep.js", "app.js"],
        ["python3", "--version", "extra"],  # 混合形态 → fail-closed
    ],
)
def test_interp_not_meta_keeps_behavior(argv):
    target, unique = resolve_interpreter_target(argv, cwd="/tmp")
    assert unique is False
    assert target is None


def test_interp_node_script_bound(tmp_path):
    f = tmp_path / "app.js"
    f.write_text("console.log(1)")
    target, unique = resolve_interpreter_target(["node", "app.js"], cwd=str(tmp_path))
    assert unique is True
    assert target == os.path.realpath(str(f))


def test_interp_node_eval_not_bound(tmp_path):
    target, unique = resolve_interpreter_target(["node", "-e", "1"], cwd=str(tmp_path))
    assert unique is False


def test_interp_node_multi_file_form_not_bound(tmp_path):
    # --require/--loader 等多文件形态：不声称覆盖
    target, unique = resolve_interpreter_target(
        ["node", "--require", "./dep.js", "app.js"], cwd=str(tmp_path)
    )
    assert unique is False


def test_interp_non_interpreter_not_bound(tmp_path):
    target, unique = resolve_interpreter_target(["ls", "-la"], cwd=str(tmp_path))
    assert unique is False
    assert target is None


def test_interp_pnpm_exec_binds_local_bin(tmp_path):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    shim = bin_dir / "eslint"
    shim.write_text("#!/bin/sh\necho eslint")
    target, unique = resolve_interpreter_target(
        ["pnpm", "exec", "eslint", "--fix"], cwd=str(tmp_path)
    )
    assert unique is True
    assert target == os.path.realpath(str(shim))


def test_interp_pnpm_exec_missing_bin_not_bound(tmp_path):
    target, unique = resolve_interpreter_target(
        ["pnpm", "exec", "eslint"], cwd=str(tmp_path)
    )
    assert unique is False


def test_interp_npm_exec_double_dash_binds_bin(tmp_path):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    shim = bin_dir / "prettier"
    shim.write_text("#!/bin/sh\n# shim")
    target, unique = resolve_interpreter_target(
        ["npm", "exec", "--", "prettier", "--write", "src"], cwd=str(tmp_path)
    )
    assert unique is True
    assert target == os.path.realpath(str(shim))


def test_interp_npm_exec_without_double_dash_not_bound(tmp_path):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "eslint").write_text("#!/bin/sh\n# shim")

    target, unique = resolve_interpreter_target(
        ["npm", "exec", "eslint"], cwd=str(tmp_path)
    )

    assert unique is False
    assert target is None


def test_interp_npx_binds_local_bin(tmp_path):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    shim = bin_dir / "tsc"
    shim.write_text("#!/bin/sh\n# shim")
    target, unique = resolve_interpreter_target(
        ["npx", "tsc", "--noEmit"], cwd=str(tmp_path)
    )
    assert unique is True
    assert target == os.path.realpath(str(shim))


@pytest.mark.parametrize(
    "argv",
    [
        ["pnpm", "exec", "--package", "eslint", "eslint"],
        ["pnpm", "exec", "--", "eslint"],
        ["npm", "exec", "eslint"],
        ["npm", "exec", "--package", "eslint", "--", "eslint"],
        ["npm", "exec", "--package=eslint", "--", "eslint"],
        ["npm", "exec", "--"],
        ["npx", "--yes", "eslint"],
        ["npx", "--package", "eslint", "eslint"],
        ["npx", "eslint@9"],
        ["pnpm", "exec", "eslint@9"],
        ["npm", "exec", "--", "eslint@9"],
        ["npx", "../eslint"],
    ],
)
def test_interp_package_exec_ambiguous_forms_not_bound(tmp_path, argv):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "eslint").write_text("#!/bin/sh\n# shim")

    target, unique = resolve_interpreter_target(argv, cwd=str(tmp_path))

    assert unique is False
    assert target is None


def test_interp_npx_flag_fail_closed(tmp_path):
    # npx -y 等 flags：不声称覆盖 → 审批
    target, unique = resolve_interpreter_target(
        ["npx", "-y", "eslint"], cwd=str(tmp_path)
    )
    assert unique is False


@pytest.mark.parametrize(
    "package_config",
    [
        {"name": "cli", "bin": "./bin/cli.js"},
        {"bin": {"cli": "./bin/cli.js"}},
        {"bin": {"cli": "./bin/cli.js", "other": "./bin/other.js"}},
    ],
)
def test_interp_package_json_bin_fallback(tmp_path, package_config):
    (tmp_path / "package.json").write_text(json.dumps(package_config))
    cli = tmp_path / "bin" / "cli.js"
    cli.parent.mkdir()
    cli.write_text("#!/usr/bin/env node\n")
    target, unique = resolve_interpreter_target(["npx", "cli"], cwd=str(tmp_path))
    assert unique is True
    assert target == os.path.realpath(str(cli))


@pytest.mark.parametrize(
    "bin_config, create_directory",
    [
        ("./bin/missing.js", False),
        ("./bin", True),
        ({"cli": "./bin/missing.js"}, False),
        ({"cli": "./bin"}, True),
        ("", False),
        (123, False),
        (["./bin/cli.js"], False),
        ({"cli": None}, False),
        ([], False),
        ({"cli": "./bin/cli.js", "other": "./bin/other.js"}, False),
    ],
)
def test_interp_package_json_bin_requires_existing_regular_file(
    tmp_path, bin_config, create_directory
):
    package_config = {"bin": bin_config}
    if isinstance(bin_config, str):
        package_config["name"] = "cli"
    (tmp_path / "package.json").write_text(json.dumps(package_config))
    if create_directory:
        (tmp_path / "bin").mkdir()

    target, unique = resolve_interpreter_target(["npx", "cli"], cwd=str(tmp_path))

    assert unique is False
    assert target is None


def test_interp_package_json_root_not_object_not_bound(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps([]))

    target, unique = resolve_interpreter_target(["npx", "cli"], cwd=str(tmp_path))

    assert unique is False
    assert target is None


def test_interp_package_json_bin_unknown_name_not_bound(tmp_path):
    cli = tmp_path / "bin" / "cli.js"
    cli.parent.mkdir()
    cli.write_text("#!/usr/bin/env node\n")
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "cli", "bin": "./bin/cli.js"})
    )

    target, unique = resolve_interpreter_target(["npx", "unknown"], cwd=str(tmp_path))

    assert unique is False
    assert target is None


def test_interp_local_bin_symlink_outside_package_not_bound(tmp_path):
    outside = tmp_path.parent / "outside-eslint"
    outside.write_text("#!/bin/sh\n# shim")
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "eslint").symlink_to(outside)

    target, unique = resolve_interpreter_target(["npx", "eslint"], cwd=str(tmp_path))

    assert unique is False
    assert target is None


def test_interp_package_exec_path_like_bin_not_bound(tmp_path):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (tmp_path / "node_modules" / "eslint").write_text("#!/bin/sh\n# shim")

    target, unique = resolve_interpreter_target(["npx", "../eslint"], cwd=str(tmp_path))

    assert unique is False
    assert target is None


@pytest.mark.parametrize("bin_path", ["../outside-cli.js", "./bin/cli.js"])
def test_interp_package_json_bin_outside_package_not_bound(tmp_path, bin_path):
    outside = tmp_path.parent / "outside-cli.js"
    outside.write_text("#!/usr/bin/env node\n")
    if bin_path == "./bin/cli.js":
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "cli.js").symlink_to(outside)
    (tmp_path / "package.json").write_text(json.dumps({"name": "cli", "bin": bin_path}))

    target, unique = resolve_interpreter_target(["npx", "cli"], cwd=str(tmp_path))

    assert unique is False
    assert target is None
