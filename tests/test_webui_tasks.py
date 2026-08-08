from types import SimpleNamespace

import httpx
import pytest

from core.tasks.models import DeliveryStatus, TaskRecord, TaskStatus
from core.webui.app import create_app


@pytest.mark.asyncio
async def test_task_detail_shows_silent_result_explicitly():
    task = TaskRecord(
        id="silent-task",
        status=TaskStatus.SUCCESS,
        result="检查完成",
        delivery_status=DeliveryStatus.NOT_REQUESTED,
        silent=True,
    )
    task_manager = SimpleNamespace(get_task=lambda task_id: task)
    app = create_app(
        {"agent_engine": SimpleNamespace(_task_manager=task_manager)},
        {},
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/tasks/detail/silent-task")

    assert response.status_code == 200
    assert "静默" in response.text
    assert "未请求投递" in response.text
