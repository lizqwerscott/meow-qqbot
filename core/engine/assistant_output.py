"""Trusted classification for a completed assistant response."""

from dataclasses import dataclass
from typing import Sequence

from core.engine.turn_capabilities import TurnCapabilities
from core.tools.stream_delivery import is_silent_reply_text


@dataclass(frozen=True)
class AssistantOutputDecision:
    """Whether a completed assistant text may be automatically delivered."""

    should_deliver: bool
    reason: str


def decide_assistant_output(
    content: str | None,
    tool_calls: Sequence[object],
    *,
    capabilities: TurnCapabilities | None,
    explicit_delivery_already_sent: bool,
    suppress_reply: bool,
) -> AssistantOutputDecision:
    """Classify final text without trusting model-provided delivery intent."""
    if tool_calls:
        return AssistantOutputDecision(False, "tool_calls_present")
    if explicit_delivery_already_sent:
        return AssistantOutputDecision(False, "explicit_delivery_already_sent")
    if suppress_reply or is_silent_reply_text(content or ""):
        return AssistantOutputDecision(False, "silent_reply")
    if capabilities is not None and not capabilities.allow_automatic_reply:
        return AssistantOutputDecision(False, "automatic_delivery_not_allowed")
    return AssistantOutputDecision(True, "final_text")
