from unittest.mock import AsyncMock

import pytest

from core.tasks.heartbeat import HeartbeatManager


def make_manager(api):
    manager = HeartbeatManager(
        {"enabled": True},
        api_client=api,
        admin_ids=["admin"],
    )
    manager._running = True
    return manager


@pytest.mark.asyncio
async def test_heartbeat_receipt_preserves_platform_message_id():
    api = type("Api", (), {})()
    api.send_text = AsyncMock(return_value={"message_id": "hb-1"})

    receipt = await make_manager(api).deliver_to_admin_receipt("alert")

    assert receipt.status == "accepted"
    assert receipt.transport_id == "hb-1"
    assert receipt.platform_message_id == "hb-1"


@pytest.mark.asyncio
async def test_heartbeat_timeout_is_unknown_and_not_retryable():
    api = type("Api", (), {})()
    api.send_text = AsyncMock(side_effect=TimeoutError("timeout"))

    receipt = await make_manager(api).deliver_to_admin_receipt("alert")

    assert receipt.status == "unknown"
    assert receipt.retryable is False
    assert receipt.error_code == "TimeoutError"
