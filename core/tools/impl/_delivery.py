from __future__ import annotations

from core.tools._types import ToolContext

_INTERNAL_PREFIXES = (
    "agent:",
    "cron:",
    "heartbeat:",
    "task:",
    "work-plan:",
    "workplan:",
    "subagent:",
    "system:",
    "exec:",
)


def resolve_transport_target(ctx: ToolContext, bot_engine) -> tuple[str, bool]:
    if ctx.delivery_target is not None:
        return ctx.delivery_target.target_id, bool(ctx.internal_control)
    delivery_value = ctx.delivery_channel or ctx.chat_id
    is_background = bool(ctx.delivery_channel and ctx.internal_control)
    resolver = getattr(bot_engine, "resolve_delivery_target", None)
    if callable(resolver):
        target = resolver(delivery_value, is_group=ctx.is_group)
        if target is not None:
            return target.target_id, is_background
    if delivery_value.startswith(_INTERNAL_PREFIXES):
        return "", False
    return delivery_value, is_background
