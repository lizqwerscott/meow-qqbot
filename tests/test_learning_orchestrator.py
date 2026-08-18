from unittest.mock import AsyncMock

import pytest

from core.learners.orchestrator import LearningOrchestrator


@pytest.mark.asyncio
async def test_on_message_deduplicates_idempotency_key(tmp_path):
    orchestrator = LearningOrchestrator(
        {"enabled": True}, data_dir=f"{tmp_path}/learners/"
    )
    orchestrator.jargon.observe = AsyncMock()

    kwargs = {
        "message_text": "hello world",
        "chat_id": "chat",
        "sender_id": "user",
        "message_id": "message",
        "idempotency_key": "admission:chat:message:learner",
    }
    assert await orchestrator.on_message(**kwargs) is True
    assert await orchestrator.on_message(**kwargs) is True
    orchestrator.jargon.observe.assert_awaited_once_with("hello world", "chat")
