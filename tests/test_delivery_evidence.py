import pytest

from core.engine.delivery_ledger import DeliveryReceipt
from core.tools.delivery_evidence import DeliveryEvidence


@pytest.mark.parametrize("kind", ["emoji", "image", "voice"])
def test_media_delivery_does_not_complete_text_reply(kind):
    evidence = DeliveryEvidence()
    evidence.record(
        kind=kind,
        target_chat_id="chat",
        receipt=DeliveryReceipt(status="accepted"),
    )

    assert evidence.has_completed_text_reply("chat") is False


@pytest.mark.parametrize(
    "status, expected", [("accepted", True), ("partial", True), ("failed", False)]
)
def test_only_successful_message_delivery_completes_text_reply(status, expected):
    evidence = DeliveryEvidence()
    evidence.record(
        kind="message",
        target_chat_id="chat",
        receipt=DeliveryReceipt(status=status),
    )

    assert evidence.has_completed_text_reply("chat") is expected


def test_text_delivery_for_another_target_does_not_complete_current_reply():
    evidence = DeliveryEvidence()
    evidence.record(
        kind="message",
        target_chat_id="other-chat",
        receipt=DeliveryReceipt(status="accepted"),
    )

    assert evidence.has_completed_text_reply("chat") is False


def test_reset_forgets_previous_text_delivery():
    evidence = DeliveryEvidence()
    evidence.record(
        kind="message",
        target_chat_id="chat",
        receipt=DeliveryReceipt(status="accepted"),
    )
    evidence.reset()

    assert evidence.has_completed_text_reply("chat") is False
