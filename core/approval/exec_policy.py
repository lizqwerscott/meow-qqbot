"""Exec 审批策略模型 — 移植 OpenClaw 的 exec-policy / exec-approvals 判定。

OpenClaw 风格核心：
- security: ``deny | allowlist | full`` — 命令执行基础开关
- ask: ``off | on-miss | always`` — 何时弹审批
- mode: ``deny | allowlist | ask | auto | full`` — 归一化策略面
- 判定函数 requires_approval：allowlist 命中直跑，miss 才问
- 两层策略堆叠（requested config × host 审批文件）取更严，host 只能收紧

审批是 operator 通道，不是权限边界：非 admin 角色（trusted/default）
固定为 allowlist + ask=off（miss 直接拒，不弹卡）；system（cron/心跳）
固定为 full；admin 才走可审批的 config 策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

SECURITY_LEVEL: Dict[str, int] = {"deny": 0, "allowlist": 1, "full": 2}
ASK_LEVEL: Dict[str, int] = {"off": 0, "on-miss": 1, "always": 2}

_VALID_SECURITY = frozenset(SECURITY_LEVEL)
_VALID_ASK = frozenset(ASK_LEVEL)
_VALID_MODE = frozenset({"deny", "allowlist", "ask", "auto", "full"})

# 审批决策常量（跨模块传递，避免裸字符串）
DECISION_ALLOW = "allow"
DECISION_ALLOW_ONCE = "allow-once"
DECISION_ALLOW_ALWAYS = "allow-always"
DECISION_DENY = "deny"
ALLOW_DECISIONS = frozenset(
    {DECISION_ALLOW, DECISION_ALLOW_ONCE, DECISION_ALLOW_ALWAYS}
)

# 角色 → 固定策略（None = 使用 config [exec] 段）
ROLE_POLICY: Dict[str, Optional[dict]] = {
    "system": {"security": "full", "ask": "off", "ask_fallback": "full"},
    "admin": None,
    "trusted": {"security": "allowlist", "ask": "off", "ask_fallback": "deny"},
    "default": {"security": "allowlist", "ask": "off", "ask_fallback": "deny"},
}


@dataclass(frozen=True)
class ExecPolicy:
    """归一化后的 exec 策略面。

    mode 为 config 显式声明的模式（"" = 未声明，由 security/ask 推导）；
    auto 模式仅在基础策略为 allowlist+on-miss 时由 resolve_mode_from_policy 产出。
    safe_bins：预信任的窄过滤器工具（对齐 openclaw tools.exec.safeBins），
    命中且满足 profile 的段视为 allowlist 满足，无需白名单条目。
    approval_timeout：审批卡超时秒数（对齐 openclaw pending exec approval 过期）。
    """

    security: str = "allowlist"
    ask: str = "on-miss"
    ask_fallback: str = "deny"
    strict_inline_eval: bool = True
    mode: str = ""
    safe_bins: tuple = ()  # 预信任窄过滤器工具名（如 head/tail/wc/tr）
    safe_bin_profiles: dict = field(default_factory=dict)  # 工具名 → profile
    # 审批卡超时（秒）。None = 未声明（host 侧），effective 时由 requested 兜底，
    # 消费方用 ``policy.approval_timeout or 300``；None 消除"显式 300 vs 未配置"
    # 的魔法数字哨兵（对齐 openclaw pending exec approval 过期）
    approval_timeout: Optional[int] = None

    def __post_init__(self):
        if self.security not in _VALID_SECURITY:
            raise ValueError(f"无效 security: {self.security}")
        if self.ask not in _VALID_ASK:
            raise ValueError(f"无效 ask: {self.ask}")
        if self.ask_fallback not in _VALID_SECURITY:
            raise ValueError(f"无效 ask_fallback: {self.ask_fallback}")
        if self.mode and self.mode not in _VALID_MODE:
            raise ValueError(f"无效 mode: {self.mode}")


def config_to_policy(cfg: dict) -> ExecPolicy:
    """[exec] 配置 dict → ExecPolicy 工厂（对齐 openclaw tools.exec.*）。"""
    return ExecPolicy(
        security=cfg.get("security", "allowlist"),
        ask=cfg.get("ask", "on-miss"),
        ask_fallback=cfg.get("ask_fallback", "deny"),
        strict_inline_eval=cfg.get("strict_inline_eval", True),
        mode=cfg.get("mode", ""),
        safe_bins=tuple(cfg.get("safe_bins") or ()),
        safe_bin_profiles=dict(cfg.get("safe_bin_profiles") or {}),
        approval_timeout=cfg.get("approval_timeout"),
    )


def resolve_mode_from_policy(policy: ExecPolicy) -> str:
    """把 (security, ask, mode) 归一化为单一 mode（对齐 openclaw resolveExecModeFromPolicy）。

    auto 仅当显式声明 mode=auto 且基础策略仍为 allowlist+on-miss：host 收紧
    到 ask=always / security=deny 时条件不满足，自然退化为 ask / deny。
    """
    if (
        policy.mode == "auto"
        and policy.security == "allowlist"
        and policy.ask == "on-miss"
    ):
        return "auto"
    if policy.security == "deny":
        return "deny"
    if policy.security == "allowlist" and policy.ask == "off":
        return "allowlist"
    if policy.security == "full" and policy.ask != "always":
        return "full"
    return "ask"


def requires_approval(
    *,
    ask: str,
    security: str,
    analysis_ok: bool,
    allowlist_satisfied: bool,
    durable_satisfied: bool = False,
) -> bool:
    """对齐 openclaw requiresExecApproval：是否弹审批。

    - ask=always → 总是审批
    - durable_satisfied（持久信任命中）→ 不审批
    - 其余：仅当 ask=on-miss 且 security=allowlist 且分析失败或 allowlist miss
    """
    if ask == "always":
        return True
    if durable_satisfied:
        return False
    return (
        ask == "on-miss"
        and security == "allowlist"
        and (not analysis_ok or not allowlist_satisfied)
    )


def _stricter(a: str, b: str, level: Dict[str, int], direction: str) -> str:
    """按等级表取更严值。direction="low" 取更低等级（security/ask_fallback，deny 最严），
    direction="high" 取更高等级（ask，always 最严）。"""
    if direction == "low":
        return a if level[a] <= level[b] else b
    return a if level[a] >= level[b] else b


def effective_policy(requested: ExecPolicy, host: ExecPolicy) -> ExecPolicy:
    """两层策略堆叠取更严（对齐 openclaw：host 文件只能收紧 config）。

    security / ask_fallback 取更严（deny 最严）；ask 取更严（always 最严）。
    mode 仅来自 config（host defaults 不声明 mode）；host 收紧后 auto 会因
    security/ask 条件不满足而在 resolve_mode_from_policy 中自然退化。
    safe_bins / safe_bin_profiles / approval_timeout 是非收紧性配置：
    host 显式定义了才覆盖（host 是收紧方，未定义时沿用 requested）。
    """
    safe_bins = (
        host.safe_bins
        if host.safe_bins is not None and len(host.safe_bins) > 0
        else requested.safe_bins
    )
    safe_bin_profiles = (
        host.safe_bin_profiles
        if host.safe_bin_profiles
        else requested.safe_bin_profiles
    )
    return ExecPolicy(
        security=_stricter(requested.security, host.security, SECURITY_LEVEL, "low"),
        ask=_stricter(requested.ask, host.ask, ASK_LEVEL, "high"),
        ask_fallback=_stricter(
            requested.ask_fallback, host.ask_fallback, SECURITY_LEVEL, "low"
        ),
        strict_inline_eval=requested.strict_inline_eval or host.strict_inline_eval,
        mode=requested.mode,
        safe_bins=safe_bins,
        safe_bin_profiles=safe_bin_profiles,
        approval_timeout=(
            host.approval_timeout
            if host.approval_timeout is not None
            else requested.approval_timeout
        ),
    )


def policy_for_role(
    role: str,
    effective_config: ExecPolicy,
    host: Optional[ExecPolicy] = None,
) -> ExecPolicy:
    """角色 → 最终策略。admin/未知角色直接使用 config×host 合并结果。

    固定角色（trusted/default/system）使用 ROLE_POLICY 的基础值，但 host
    （审批文件 defaults）仍可收紧：host security=deny 时这些角色同样被禁用
    （对齐 docstring "host 只能收紧"）。
    """
    fixed = ROLE_POLICY.get(role)
    if fixed is None:
        return effective_config
    base = ExecPolicy(
        security=fixed["security"],
        ask=fixed["ask"],
        ask_fallback=fixed.get("ask_fallback", "deny"),
        strict_inline_eval=effective_config.strict_inline_eval,
        mode=effective_config.mode if fixed["ask"] == "on-miss" else "",
        # 非收紧性配置：固定角色沿用 config×host 合并结果
        safe_bins=effective_config.safe_bins,
        safe_bin_profiles=effective_config.safe_bin_profiles,
        approval_timeout=effective_config.approval_timeout,
    )
    if host is not None:
        base = effective_policy(base, host)
    return base
