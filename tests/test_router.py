import pytest

from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
)
from core.engine.router import Router
from core.message import InputMessage


class CommandManager:
    async def process_message(self, _message):
        return [
            {
                "chat_id": "chat",
                "content": "command result",
                "message_id": "reply-1",
                "is_group": True,
            }
        ]


def message():
    return InputMessage("command-1", "user", "chat", "猫猫状态", True)


@pytest.mark.asyncio
async def test_router_records_command_delivery_without_timeline(tmp_path):
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    sent = []

    async def callback(**kwargs):
        sent.append(kwargs)
        return DeliveryReceipt(status="accepted", platform_message_id="qq-1")

    agent = type("Agent", (), {"_get_delivery_controller": lambda self: controller})()
    router = Router(agent)
    router.command_manager = CommandManager()

    await router.route(message(), callback, lambda _: "")

    record = await controller.ledger.get("external:command:chat:command-1:0")
    assert record is not None
    assert record.status == "sent"
    assert sent[0]["content"] == "command result"
    await controller.ledger.close()


@pytest.mark.asyncio
async def test_router_keeps_legacy_command_delivery_without_controller():
    sent = []

    async def callback(**kwargs):
        sent.append(kwargs)

    agent = type("Agent", (), {})()
    router = Router(agent)
    router.command_manager = CommandManager()

    await router.route(message(), callback, lambda _: "")

    assert sent[0]["content"] == "command result"
