"""Exec 策略模型测试（对齐 openclaw 判定逻辑）。"""

import pytest

from core.approval.exec_policy import (
    ExecPolicy,
    effective_policy,
    policy_for_role,
    requires_approval,
    resolve_mode_from_policy,
)

# ── resolve_mode_from_policy ──


@pytest.mark.parametrize(
    "security,ask,expected",
    [
        ("deny", "off", "deny"),
        ("deny", "always", "deny"),
        ("allowlist", "off", "allowlist"),
        ("allowlist", "on-miss", "ask"),
        ("allowlist", "always", "ask"),
        ("full", "off", "full"),
        ("full", "on-miss", "full"),
        ("full", "always", "ask"),
    ],
)
def test_resolve_mode(security, ask, expected):
    policy = ExecPolicy(security=security, ask=ask)
    assert resolve_mode_from_policy(policy) == expected


# ── requires_approval 真值表 ──


def test_ask_always_requires_approval():
    assert (
        requires_approval(
            ask="always",
            security="allowlist",
            analysis_ok=True,
            allowlist_satisfied=True,
        )
        is True
    )


def test_durable_satisfied_skips_approval():
    assert (
        requires_approval(
            ask="on-miss",
            security="allowlist",
            analysis_ok=False,
            allowlist_satisfied=False,
            durable_satisfied=True,
        )
        is False
    )


def test_allowlist_hit_no_approval():
    assert (
        requires_approval(
            ask="on-miss",
            security="allowlist",
            analysis_ok=True,
            allowlist_satisfied=True,
        )
        is False
    )


def test_allowlist_miss_requires_approval():
    assert (
        requires_approval(
            ask="on-miss",
            security="allowlist",
            analysis_ok=True,
            allowlist_satisfied=False,
        )
        is True
    )


def test_analysis_fail_requires_approval():
    assert (
        requires_approval(
            ask="on-miss",
            security="allowlist",
            analysis_ok=False,
            allowlist_satisfied=True,
        )
        is True
    )


def test_ask_off_never_approves():
    assert (
        requires_approval(
            ask="off", security="allowlist", analysis_ok=True, allowlist_satisfied=False
        )
        is False
    )


def test_security_full_never_approves():
    assert (
        requires_approval(
            ask="on-miss", security="full", analysis_ok=True, allowlist_satisfied=False
        )
        is False
    )


# ── effective_policy（host 只能收紧）──


def test_effective_host_tightens_security():
    requested = ExecPolicy(security="full", ask="off")
    host = ExecPolicy(security="allowlist", ask="on-miss")
    eff = effective_policy(requested, host)
    assert eff.security == "allowlist"
    assert eff.ask == "on-miss"


def test_effective_keeps_looser_config():
    requested = ExecPolicy(security="allowlist", ask="on-miss")
    host = ExecPolicy(security="allowlist", ask="off")
    eff = effective_policy(requested, host)
    assert eff.security == "allowlist"
    assert eff.ask == "on-miss"


def test_effective_ask_fallback_tightest_wins():
    requested = ExecPolicy(ask_fallback="full")
    host = ExecPolicy(ask_fallback="deny")
    assert effective_policy(requested, host).ask_fallback == "deny"


# ── policy_for_role ──


def test_role_system_full():
    p = policy_for_role("system", ExecPolicy())
    assert p.security == "full"
    assert p.ask == "off"


def test_role_trusted_allowlist_off():
    p = policy_for_role("trusted", ExecPolicy())
    assert p.security == "allowlist"
    assert p.ask == "off"


def test_role_admin_uses_config():
    config = ExecPolicy(security="allowlist", ask="on-miss")
    p = policy_for_role("admin", config)
    assert p.security == "allowlist"
    assert p.ask == "on-miss"


# ── policy_for_role × host 收紧 ──


def test_role_fixed_policy_tightened_by_host():
    # trusted 固定 allowlist+off，host security=deny 仍可收紧（host 只能收紧）
    host = ExecPolicy(security="deny")
    p = policy_for_role("trusted", ExecPolicy(), host)
    assert p.security == "deny"


def test_role_fixed_policy_no_host_unchanged():
    p = policy_for_role("trusted", ExecPolicy())
    assert p.security == "allowlist"
    assert p.ask == "off"


def test_system_tightened_by_host_deny():
    host = ExecPolicy(security="deny")
    p = policy_for_role("system", ExecPolicy(), host)
    assert p.security == "deny"


def test_admin_mode_propagates_through_host():
    cfg = ExecPolicy(mode="auto", security="allowlist", ask="on-miss")
    p = policy_for_role("admin", cfg, ExecPolicy())
    assert p.mode == "auto"
    assert resolve_mode_from_policy(p) == "auto"
