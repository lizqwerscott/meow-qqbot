"""Delivery policy for optional ambient participation.

The policy is deliberately pure. Transport adapters and the tool loop remain
responsible for actually sending a message and recording the outcome.
"""

from dataclasses import dataclass

_SILENT_REPLIES = frozenset({"NO_REPLY", "HEARTBEAT_OK"})


@dataclass(frozen=True)
class AmbientDeliveryDecision:
    should_deliver: bool
    content: str = ""
    reason: str = ""
    reply_anchor_id: str = ""


def decide_ambient_delivery(
    content: str | None,
    *,
    delivery_mode: str,
    tool_delivered: bool = False,
    reply_anchor_id: str = "",
) -> AmbientDeliveryDecision:
    """Decide whether an ambient turn may use automatic final-text delivery."""
    if tool_delivered:
        return AmbientDeliveryDecision(
            False, reason="already_delivered", reply_anchor_id=reply_anchor_id
        )
    normalized = (content or "").strip()
    if not normalized or normalized.upper() in _SILENT_REPLIES:
        return AmbientDeliveryDecision(
            False, reason="silent_final_reply", reply_anchor_id=reply_anchor_id
        )
    if delivery_mode == "message_tool_only":
        return AmbientDeliveryDecision(
            False,
            reason="automatic_delivery_disabled",
            reply_anchor_id=reply_anchor_id,
        )
    if delivery_mode != "automatic":
        return AmbientDeliveryDecision(
            False, reason="unknown_delivery_mode", reply_anchor_id=reply_anchor_id
        )
    return AmbientDeliveryDecision(
        True,
        content=normalized,
        reason="final_reply",
        reply_anchor_id=reply_anchor_id,
    )
