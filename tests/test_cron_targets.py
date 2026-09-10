from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.command_handlers.cron import CronCommand
from core.message import InputMessage
from core.session_identity import DeliveryTarget, DeliveryTargetCatalog
from core.tasks.models import CronJob


@pytest.mark.asyncio
async def test_cron_targets_lists_catalogued_qq_targets(tmp_path):
    catalog = DeliveryTargetCatalog(tmp_path / "targets.sqlite3")
    catalog.observe(DeliveryTarget("qq", "default", "group", "123"))
    command = CronCommand(
        cron_job_manager=SimpleNamespace(),
        delivery_target_catalog=catalog,
    )

    replies = await command.execute(
        InputMessage("m1", "admin", "123", "", True), "targets"
    )

    assert replies[0]["chat_id"] == "123"
    assert "qq/default/group" in replies[0]["content"]
    assert "`123`" in replies[0]["content"]
    catalog.close()


@pytest.mark.asyncio
async def test_cron_create_can_select_observed_delivery_target(tmp_path):
    catalog = DeliveryTargetCatalog(tmp_path / "targets.sqlite3")
    catalog.observe(DeliveryTarget("qq", "default", "direct", "456"))
    manager = SimpleNamespace(
        create_job=AsyncMock(return_value=CronJob(id="job-1", name="提醒"))
    )
    command = CronCommand(
        cron_job_manager=manager,
        delivery_target_catalog=catalog,
    )

    replies = await command.execute(
        InputMessage("m1", "admin", "123", "", True),
        'create 提醒 "0 8 * * *" 早安 --delivery-target 456',
    )

    assert manager.create_job.await_args.kwargs["delivery_channel"] == "456"
    assert manager.create_job.await_args.kwargs["is_group"] is False
    assert "已创建" in replies[0]["content"]
    catalog.close()


@pytest.mark.asyncio
async def test_cron_create_rejects_unknown_delivery_target(tmp_path):
    catalog = DeliveryTargetCatalog(tmp_path / "targets.sqlite3")
    manager = SimpleNamespace(create_job=AsyncMock())
    command = CronCommand(
        cron_job_manager=manager,
        delivery_target_catalog=catalog,
    )

    replies = await command.execute(
        InputMessage("m1", "admin", "123", "", True),
        'create 提醒 "0 8 * * *" 早安 --delivery-target 456',
    )

    assert "未找到唯一" in replies[0]["content"]
    manager.create_job.assert_not_awaited()
    catalog.close()
