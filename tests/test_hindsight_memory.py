from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.engine.hindsight_memory import HindsightMemory


@pytest.mark.asyncio
async def test_add_message_forwards_idempotency_key_as_metadata():
    memory = HindsightMemory()
    client = SimpleNamespace(aretain=AsyncMock())
    memory._client = client

    assert (
        await memory.add_message(
            session_id="chat",
            content="hello",
            sender_id="user",
            idempotency_key="admission:chat:message:hindsight",
        )
        is True
    )

    metadata = client.aretain.call_args.kwargs["metadata"]
    assert metadata["idempotency_key"] == "admission:chat:message:hindsight"


@pytest.mark.asyncio
async def test_add_message_deduplicates_idempotency_key_in_process():
    memory = HindsightMemory()
    client = SimpleNamespace(aretain=AsyncMock())
    memory._client = client

    kwargs = {
        "session_id": "chat",
        "content": "hello",
        "sender_id": "user",
        "idempotency_key": "admission:chat:message:hindsight",
    }
    assert await memory.add_message(**kwargs) is True
    assert await memory.add_message(**kwargs) is True
    client.aretain.assert_awaited_once()
