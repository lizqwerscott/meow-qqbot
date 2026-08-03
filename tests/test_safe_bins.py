"""Safe bins 匹配测试（对齐 openclaw tools.exec.safeBins + safeBinProfiles）。"""

from core.approval.allowlist import (
    DEFAULT_SAFE_BIN_PROFILES,
    match_safe_bins,
)
from core.tools.exec_analysis import ExecSegment, ExecutableResolution


def _seg(argv, found_in_path=True):
    """构造段；默认视为 PATH 解析出的二进制（found_in_path=True）。"""
    return ExecSegment(
        raw=" ".join(argv),
        argv=argv,
        resolution=ExecutableResolution(
            resolved_path=f"/usr/bin/{argv[0]}", found_in_path=found_in_path
        ),
    )


# ── 基础命中 ──


def test_safe_bin_hit_basic():
    segs = [_seg(["head", "-5"])]
    ok, matches = match_safe_bins(segs, ("head", "tail", "wc", "tr"))
    assert ok is True
    assert matches[0] == {"bin": "head"}


def test_safe_bin_stdin_only_no_positional():
    # 位置参数（读文件）拒绝
    segs = [_seg(["head", "/etc/passwd"])]
    ok, _ = match_safe_bins(segs, ("head",))
    assert ok is False

    # 纯 stdin 过滤放行
    segs = [_seg(["head"])]
    ok, _ = match_safe_bins(segs, ("head",))
    assert ok is True


def test_safe_bin_value_flags():
    ok, _ = match_safe_bins([_seg(["head", "-n", "5"])], ("head",))
    assert ok is True
    ok, _ = match_safe_bins([_seg(["head", "--lines=5"])], ("head",))
    assert ok is True
    # 短粘连 -n5
    ok, _ = match_safe_bins([_seg(["head", "-n5"])], ("head",))
    assert ok is True


def test_safe_bin_denied_flags():
    # tail -f 挂起 → 拒绝
    ok, _ = match_safe_bins([_seg(["tail", "-f", "log"])], ("tail",))
    assert ok is False
    ok, _ = match_safe_bins([_seg(["tail", "--follow"])], ("tail",))
    assert ok is False


def test_safe_bin_unknown_flag_rejected():
    ok, _ = match_safe_bins([_seg(["head", "--evil"])], ("head",))
    assert ok is False
    ok, _ = match_safe_bins([_seg(["wc", "-x"])], ("wc",))
    assert ok is False


def test_safe_bin_tr_positional():
    # tr 允许 SET1 [SET2] 位置参数
    ok, _ = match_safe_bins([_seg(["tr", "a-z", "A-Z"])], ("tr",))
    assert ok is True
    ok, _ = match_safe_bins([_seg(["tr", "-d", "a-z"])], ("tr",))
    assert ok is True
    # 超过 2 个位置参数拒绝
    ok, _ = match_safe_bins([_seg(["tr", "a", "b", "c"])], ("tr",))
    assert ok is False


def test_safe_bin_wc_flags():
    ok, _ = match_safe_bins([_seg(["wc", "-l"])], ("wc",))
    assert ok is True
    ok, _ = match_safe_bins([_seg(["wc", "-l", "-w"])], ("wc",))
    assert ok is True


def test_safe_bin_short_flag_combos():
    # -lw → -l -w 组合
    ok, _ = match_safe_bins([_seg(["wc", "-lw"])], ("wc",))
    assert ok is True
    # 含未声明字符拒绝
    ok, _ = match_safe_bins([_seg(["wc", "-lx"])], ("wc",))
    assert ok is False


def test_safe_bin_not_in_list():
    ok, _ = match_safe_bins([_seg(["cat", "x"])], ("head", "tail"))
    assert ok is False
    # 空 safe_bins → 永不满足
    ok, _ = match_safe_bins([_seg(["head"])], ())
    assert ok is False


def test_safe_bin_custom_profile_overrides():
    custom = {
        "myfilter": {
            "max_positional": 0,
            "allowed_value_flags": ["-n", "--limit"],
            "allowed_flags": [],
            "denied_flags": ["-c", "--command"],
        }
    }
    ok, _ = match_safe_bins([_seg(["myfilter", "-n", "10"])], ("myfilter",), custom)
    assert ok is True
    ok, _ = match_safe_bins([_seg(["myfilter", "--command", "x"])], ("myfilter",), custom)
    assert ok is False


def test_safe_bin_nested_segments():
    from core.tools.exec_analysis import ExecSegment as S

    # 外层段本身命中 safe-bin，内层段也必须命中才放行
    inner = _seg(["head", "-5"])
    outer = _seg(["head", "-5"])
    outer.nested_segments = [inner]
    ok, _ = match_safe_bins([outer], ("head",))
    assert ok is True

    # 内层 miss（读文件）→ 整体 miss
    bad_inner = _seg(["head", "/etc/passwd"])
    outer2 = _seg(["head", "-5"])
    outer2.nested_segments = [bad_inner]
    ok, _ = match_safe_bins([outer2], ("head",))
    assert ok is False

    # argv[0] 不是 safe bin（如 bash -c ...）→ 不命中（正确：bash 非窄过滤器）
    segs = [_seg(["bash", "-c", "head -5"])]
    ok, _ = match_safe_bins(segs, ("head",))
    assert ok is False


def test_default_profiles_shape():
    for bin_name in ("head", "tail", "wc", "tr"):
        assert bin_name in DEFAULT_SAFE_BIN_PROFILES
        p = DEFAULT_SAFE_BIN_PROFILES[bin_name]
        assert set(p) >= {"max_positional", "allowed_value_flags", "allowed_flags", "denied_flags"}


# ── 修复回归：found_in_path 要求 / 文件读取 flag / 策略透传 ──


def test_safe_bin_requires_found_in_path():
    from core.tools.exec_analysis import ExecSegment, ExecutableResolution

    # ./head（本地文件，非 PATH 解析）→ 拒绝
    seg = ExecSegment(
        raw="./head -5",
        argv=["./head", "-5"],
        resolution=ExecutableResolution(resolved_path="/tmp/head", found_in_path=False),
    )
    ok, _ = match_safe_bins([seg], ("head",))
    assert ok is False

    # PATH 解析的 head → 放行
    seg2 = ExecSegment(
        raw="head -5",
        argv=["head", "-5"],
        resolution=ExecutableResolution(resolved_path="/usr/bin/head", found_in_path=True),
    )
    ok, _ = match_safe_bins([seg2], ("head",))
    assert ok is True


def test_safe_bin_wc_files0_from_rejected():
    # --files0-from 从文件读文件名列表（读文件原语）→ 拒绝
    ok, _ = match_safe_bins([_seg(["wc", "--files0-from", "/etc/passwd"])], ("wc",))
    assert ok is False
