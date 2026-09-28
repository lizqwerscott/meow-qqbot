import httpx
import pytest

from core.engine.token_usage_log import SOURCE_TURN_REPLY, TokenUsageLog
from core.webui.app import create_app


@pytest.mark.asyncio
async def test_usage_page_renders_summary_and_breakdowns():
    log = TokenUsageLog(":memory:")
    log.record(
        chat_id="chat-1",
        model="deepseek-v4-flash",
        provider="deepseek",
        source=SOURCE_TURN_REPLY,
        prompt_tokens=100,
        completion_tokens=20,
        cache_hit_tokens=60,
        cache_miss_tokens=40,
        usage_present=True,
        cost=0.001234,
    )
    app = create_app({}, {})
    app.state.managers["token_usage_log"] = log

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/usage")

    assert response.status_code == 200
    assert "deepseek-v4-flash" in response.text
    assert SOURCE_TURN_REPLY in response.text
    assert "0.001234" in response.text
    log.close()


@pytest.mark.asyncio
async def test_usage_page_without_ledger_reports_unavailable():
    app = create_app({}, {})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/usage")

    assert response.status_code == 200
    assert "用量账本不可用" in response.text
