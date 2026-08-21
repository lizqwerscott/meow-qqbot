"""Admission side-effect eligibility.

The admission origin is a runtime fact. This module derives which durable
observers may receive an admitted message so callers do not infer eligibility
from sender IDs or session naming conventions.
"""

from core.managers.session_manager import AdmissionOrigin, PendingInbound
from core.message import MessageType

_EFFECT_TYPES = ("hindsight", "learner")


def effect_types_for(pending: PendingInbound) -> tuple[str, ...]:
    """Return durable observer effects allowed for an admitted message."""
    if pending.origin is not AdmissionOrigin.USER_MESSAGE:
        return ()
    if pending.message.msg_type is MessageType.CARD:
        return ()
    return _EFFECT_TYPES
