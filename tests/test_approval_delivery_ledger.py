from unittest.mock import MagicMock, patch

import pytest

from core.approval.approval_manager import ApprovalManager
from core.engine.delivery_ledger import DeliveryController, DeliveryLedger


@pytest.mark.asyncio
async def test_approval_card_delivery_is_recorded_without_timeline_projection(tmp_path):
    manager = ApprovalManager(
        api_client=MagicMock(),
        admin_ids=["admin"],
        delivery_controller=DeliveryController(
            DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
        ),
    )

    async def send_card(**_kwargs):
        for future in manager._pending.values():
            future.set_result("allow-once")
        return True

    with patch("qqbot_agent_sdk.ApprovalSender") as sender:
        sender.return_value.send = send_card
        result = await manager.request_approval(
            "chat",
            "exec",
            "需要审批",
            details="ls",
            session_key="approval:turn-1:exec:stable",
        )

    assert result == "allow-once"
    record = await manager._delivery_controller.ledger.get(
        "external:approval-card:approval:turn-1:exec:stable:c2c:admin"
    )
    assert record is not None
    assert record.status == "sent"
    assert record.reason == "approval_card"
    assert manager._delivery_controller.timeline is None
    await manager._delivery_controller.ledger.close()
