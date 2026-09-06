from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.webui.app import create_app


@pytest.mark.asyncio
async def test_status_page_uses_ledger_archive_index():
    app = create_app({}, {})
    app.state.managers["agent_engine"] = SimpleNamespace(
        hindsight=None,
        get_stats=AsyncMock(
            return_value={
                "queue_sizes": {},
                "cost": {
                    "turn_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_hit_rate": 0,
                    "total_cost": 0,
                },
                "active_chats": 0,
                "total_messages": 0,
                "hindsight_health": {"status": "disabled"},
                "learners": {},
            }
        ),
        get_engagement_status=AsyncMock(return_value={}),
    )
    app.state.managers["context_manager"] = SimpleNamespace(
        get_archived_sessions_summary_async=AsyncMock(
            side_effect=AssertionError("legacy archive summary must not be read")
        )
    )
    app.state.managers["archive_index"] = SimpleNamespace(
        chat_ids=AsyncMock(return_value=["should-not-be-used"]),
        chat_summaries_for_webui=AsyncMock(
            return_value=(
                [
                    {"chat_id": "chat-1", "archive_count": 1},
                    {"chat_id": "chat-2", "archive_count": 2},
                    {"chat_id": "prepared-only", "archive_count": 0},
                ],
                2,
            )
        ),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/status")

    assert response.status_code == 200
    assert '>2<' in response.text
