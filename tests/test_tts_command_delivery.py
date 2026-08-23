import pytest

from core.command_handlers.tts import TtsCommand
from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
)
from core.message import InputMessage


class FakeTts:
    async def synthesize(self, text):
        return b"audio"

    def save_temp_audio(self, audio):
        return "/tmp/tts.wav"


class FakeUploader:
    async def upload(self, **kwargs):
        return "file-info"


class FakeBot:
    def __init__(self, receipt):
        self.media_uploader = FakeUploader()
        self.receipt = receipt
        self.calls = []

    async def send_reply(self, **kwargs):
        self.calls.append(kwargs)
        return self.receipt


@pytest.mark.asyncio
async def test_tts_command_uses_stable_ledger_delivery_id(tmp_path):
    bot = FakeBot(DeliveryReceipt(status="accepted", logical_delivery_id="transport"))
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    command = TtsCommand(bot, FakeTts(), controller)
    message = InputMessage("message-1", "user", "chat", "/tts", False)

    assert await command.execute(message, "hello") == []
    assert await command.execute(message, "hello") == []
    assert len(bot.calls) == 1

    record = await controller.ledger.get("external:command:chat:message-1:tts")
    assert record is not None
    assert record.status == "sent"
    assert record.reason == "command_media"


@pytest.mark.asyncio
async def test_tts_command_reports_unconfirmed_delivery(tmp_path):
    bot = FakeBot(
        DeliveryReceipt(
            status="unknown",
            logical_delivery_id="transport",
            error_code="timeout",
        )
    )
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    command = TtsCommand(bot, FakeTts(), controller)
    message = InputMessage("message-1", "user", "chat", "/tts", False)

    replies = await command.execute(message, "hello")

    assert replies[0]["content"] == "语音发送未确认，请稍后重试"
