from core.engine.admission_effect_policy import effect_types_for
from core.managers.session_manager import AdmissionOrigin, PendingInbound
from core.message import InputMessage, MessageType


def _pending(*, origin: AdmissionOrigin, msg_type: MessageType = MessageType.TEXT):
    return PendingInbound(
        InputMessage("message", "user", "chat", "hello", False, msg_type=msg_type),
        "hello",
        "agent",
        origin,
    )


def test_user_message_retains_hindsight_and_learner():
    assert effect_types_for(_pending(origin=AdmissionOrigin.USER_MESSAGE)) == (
        "hindsight",
        "learner",
    )


def test_internal_control_has_no_durable_effects():
    assert effect_types_for(_pending(origin=AdmissionOrigin.INTERNAL_CONTROL)) == ()


def test_card_has_no_durable_effects_regardless_of_origin():
    for origin in AdmissionOrigin:
        assert (
            effect_types_for(_pending(origin=origin, msg_type=MessageType.CARD)) == ()
        )
