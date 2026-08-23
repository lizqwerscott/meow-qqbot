from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
)
from core.engine.duplicate_reply import DuplicateReplyDetector
from core.message import InputMessage


class FakeContext:
    def __init__(self):
        self.history = ["same", "same"]
        self.assistant_messages = []

    async def get_recent_user_contents_async(self, _chat_id):
        return self.history

    async def add_assistant_message_async(self, chat_id, content, message_id):
        self.assistant_messages.append((chat_id, content, message_id))


def duplicate_message():
    return InputMessage("message-2", "user", "chat", "same", True)


async def callback(**_kwargs):
    return DeliveryReceipt(status="accepted", platform_message_id="qq-1")


async def failed_callback(**_kwargs):
    return DeliveryReceipt(
        status="failed",
        error_code="timeout",
        retryable=True,
    )


async def test_duplicate_reply_uses_delivery_ledger(tmp_path):
    context = FakeContext()
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    detector = DuplicateReplyDetector(context, lambda: controller)

    assert await detector.handle_message(duplicate_message(), callback, lambda _: "")
    record = await controller.ledger.get("external:dupe_message-2")

    assert record is not None
    assert record.status == "sent"
    assert context.assistant_messages == [("chat", "same", "dupe_message-2")]
    await controller.ledger.close()


async def test_duplicate_reply_does_not_project_failed_delivery(tmp_path):
    context = FakeContext()
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    detector = DuplicateReplyDetector(context, lambda: controller)

    assert not await detector.handle_message(
        duplicate_message(), failed_callback, lambda _: ""
    )
    record = await controller.ledger.get("external:dupe_message-2")

    assert record is not None
    assert record.status == "failed"
    assert context.assistant_messages == []
    await controller.ledger.close()
