from types import SimpleNamespace

import pytest

from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
)
from core.tasks.delivery_strategy import ChatReplyDeliveryStrategy


@pytest.mark.asyncio
async def test_chat_reply_strategy_settles_wake_delivery_receipt(tmp_path):
    sent = []

    async def callback(**kwargs):
        sent.append(kwargs)
        return DeliveryReceipt(
            status="accepted",
            logical_delivery_id="transport-1",
            platform_message_id="qq-1",
        )

    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    strategy = ChatReplyDeliveryStrategy(
        callback,
        delivery_controller=controller,
    )
    result = SimpleNamespace(
        should_notify=True,
        notification_text="wake result",
        deliver_to_user="",
        captured_replies=[],
        turn_id="wake-1",
    )

    await strategy.deliver(result, delivery_target="chat")

    assert sent[0]["content"] == "wake result"
    record = await controller.ledger.get("external:wake:wake-1:notification")
    assert record is not None
    assert record.status == "sent"
    assert record.receipt_status == "accepted"
    assert record.platform_message_id == "qq-1"
    await controller.ledger.close()


@pytest.mark.asyncio
async def test_heartbeat_admin_delivery_is_ledgered_and_idempotent(tmp_path):
    class Heartbeat:
        _cooldown_hours = 1
        admin_delivery_target = "admin"

        def __init__(self):
            self.sent = []
            self.notifications = []

        def should_suppress(self, text):
            return False

        def record_delivery_start(self):
            pass

        def record_notification(self, text):
            self.notifications.append(text)

        async def deliver_to_admin_receipt(self, text):
            self.sent.append(text)
            return DeliveryReceipt(
                status="accepted",
                platform_message_id="admin-message-1",
            )

        async def deliver_to_admin(self, text):
            raise AssertionError("ledgered receipt path should be used")

    from core.tasks.delivery_strategy import HeartbeatDeliveryStrategy

    heartbeat = Heartbeat()
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    strategy = HeartbeatDeliveryStrategy(
        heartbeat,
        show_alerts=True,
        delivery_controller=controller,
    )
    result = SimpleNamespace(
        should_notify=True,
        notification_text="disk usage is high",
        deliver_to_user="",
        turn_id="heartbeat-1",
    )

    await strategy.deliver(result)
    await strategy.deliver(result)

    assert heartbeat.sent == ["disk usage is high"]
    record = await controller.ledger.get(
        "external:heartbeat:heartbeat-1:notification:admin"
    )
    assert record is not None
    assert record.status == "sent"
    assert record.platform_message_id == "admin-message-1"
    await controller.ledger.close()

    async def callback(**kwargs):
        return DeliveryReceipt(
            status="unknown",
            logical_delivery_id="transport-unknown",
            error_code="timeout",
        )

    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    strategy = ChatReplyDeliveryStrategy(callback, delivery_controller=controller)
    result = SimpleNamespace(
        should_notify=True,
        notification_text="wake result",
        deliver_to_user="",
        captured_replies=[],
        turn_id="wake-unknown",
    )

    await strategy.deliver(result, delivery_target="chat")

    record = await controller.ledger.get("external:wake:wake-unknown:notification")
    assert record is not None
    assert record.status == "unknown"
    assert record.error_code == "timeout"
    await controller.ledger.close()
