import pytest

from core.engine.delivery_ledger import DeliveryReceipt
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.message import create_message_entries


@pytest.mark.asyncio
async def test_send_message_preserves_accepted_delivery_receipt():
    async def reply_callback(**kwargs):
        return DeliveryReceipt(
            status="accepted",
            logical_delivery_id="reply:chat:message",
            platform_message_id="qq-1",
        )

    tool = create_message_entries(ToolDeps())[0]
    result = await tool.handler(
        {"text": "hello"},
        ToolContext(
            chat_id="chat",
            is_group=True,
            reply_to="message",
            sender_id="user",
            reply_callback=reply_callback,
        ),
    )

    assert result.delivery_receipt is not None
    assert result.delivery_receipt.status == "accepted"


@pytest.mark.asyncio
async def test_send_message_does_not_claim_failed_delivery():
    async def reply_callback(**kwargs):
        return DeliveryReceipt(
            status="failed",
            logical_delivery_id="reply:chat:message",
            error_code="timeout",
            retryable=True,
        )

    tool = create_message_entries(ToolDeps())[0]
    result = await tool.handler(
        {"text": "hello"},
        ToolContext(
            chat_id="chat",
            is_group=True,
            reply_to="message",
            sender_id="user",
            reply_callback=reply_callback,
        ),
    )


@pytest.mark.asyncio
async def test_send_message_converts_transport_exception_to_failed_receipt():
    async def reply_callback(**kwargs):
        raise TimeoutError("network timeout")

    tool = create_message_entries(ToolDeps())[0]
    result = await tool.handler(
        {"text": "hello"},
        ToolContext(
            chat_id="chat",
            is_group=True,
            reply_to="message",
            sender_id="user",
            reply_callback=reply_callback,
            turn_id="turn-1",
        ),
    )

    assert result.delivery_receipt is not None
    assert result.delivery_receipt.status == "failed"
    assert result.delivery_receipt.retryable is True
    assert result.delivery_receipt.error_code == "transport_exception"
