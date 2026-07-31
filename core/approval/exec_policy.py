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

from dataclasses import dataclass
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
    """

    security: str = "allowlist"
    ask: str = "on-miss"
    ask_fallback: str = "deny"
    strict_inline_eval: bool = True
    mode: str = ""

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
    """
    return ExecPolicy(
        security=_stricter(requested.security, host.security, SECURITY_LEVEL, "low"),
        ask=_stricter(requested.ask, host.ask, ASK_LEVEL, "high"),
        ask_fallback=_stricter(
            requested.ask_fallback, host.ask_fallback, SECURITY_LEVEL, "low"
        ),
        strict_inline_eval=requested.strict_inline_eval or host.strict_inline_eval,
        mode=requested.mode,
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
    )
    if host is not None:
        base = effective_policy(base, host)
    return base
