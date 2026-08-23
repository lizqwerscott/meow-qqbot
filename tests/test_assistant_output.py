from core.engine.assistant_output import decide_assistant_output
from core.engine.turn_capabilities import TurnCapabilities
from core.managers.session_manager import InboundIntent


def test_tool_bearing_assistant_text_is_never_automatically_delivered():
    decision = decide_assistant_output(
        "I will check it.",
        [object()],
        capabilities=TurnCapabilities.for_intent(InboundIntent.DIRECT_TASK),
        explicit_delivery_already_sent=False,
        suppress_reply=False,
    )

    assert decision.should_deliver is False
    assert decision.reason == "tool_calls_present"


def test_final_text_is_delivered_for_automatic_turn():
    decision = decide_assistant_output(
        "The result is ready.",
        [],
        capabilities=TurnCapabilities.for_intent(InboundIntent.PRIVATE_CONVERSATION),
        explicit_delivery_already_sent=False,
        suppress_reply=False,
    )

    assert decision.should_deliver is True
    assert decision.reason == "final_text"


def test_ambient_turn_cannot_automatically_deliver_final_text():
    decision = decide_assistant_output(
        "Useful group answer",
        [],
        capabilities=TurnCapabilities.for_intent(InboundIntent.GROUP_AMBIENT),
        explicit_delivery_already_sent=False,
        suppress_reply=False,
    )

    assert decision.should_deliver is False
    assert decision.reason == "automatic_delivery_not_allowed"


def test_silent_and_explicitly_delivered_responses_are_suppressed():
    silent = decide_assistant_output(
        "NO_REPLY",
        [],
        capabilities=None,
        explicit_delivery_already_sent=False,
        suppress_reply=False,
    )
    already_sent = decide_assistant_output(
        "Final text",
        [],
        capabilities=None,
        explicit_delivery_already_sent=True,
        suppress_reply=False,
    )

    assert silent.reason == "silent_reply"
    assert already_sent.reason == "explicit_delivery_already_sent"
