from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.engine.hindsight_memory import HindsightDocumentRef, HindsightMemory


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


@pytest.mark.asyncio
async def test_add_message_accepts_stable_document_ref_and_legacy_tag():
    memory = HindsightMemory()
    client = SimpleNamespace(aretain=AsyncMock())
    memory._client = client

    document = HindsightDocumentRef(
        document_id="session-123",
        canonical_chat_tag="chat:agent:main:qq:default:group:123",
        legacy_chat_tags=("chat:123",),
    )
    assert (
        await memory.add_message(
            content="hello",
            sender_id="user",
            document=document,
        )
        is True
    )

    kwargs = client.aretain.call_args.kwargs
    assert kwargs["document_id"] == "session-123"
    assert kwargs["tags"] == [
        "user:user",
        "chat:agent:main:qq:default:group:123",
        "chat:123",
    ]


@pytest.mark.asyncio
async def test_search_shared_reads_canonical_and_legacy_aliases_without_duplicates():
    memory = HindsightMemory()

    class Result:
        def __init__(self, text):
            self.type = "experience"
            self.text = text

    class Client:
        def __init__(self):
            self.tags = []

        async def arecall(self, **kwargs):
            self.tags.append(kwargs["tags"][0])
            return SimpleNamespace(results=[Result("same"), Result(kwargs["tags"][0])])

    client = Client()
    memory._client = client
    result = await memory.search_shared(
        "agent:main:qq:default:group:123",
        aliases=["chat:123"],
    )

    assert client.tags == [
        "chat:agent:main:qq:default:group:123",
        "chat:123",
    ]
    assert [episode["summary"] for episode in result["episodes"]] == [
        "same",
        "chat:agent:main:qq:default:group:123",
        "chat:123",
    ]


@pytest.mark.asyncio
async def test_close_uses_async_hindsight_client_close():
    memory = HindsightMemory.__new__(HindsightMemory)
    client = SimpleNamespace(aclose=AsyncMock(), close=Mock())
    memory._client = client

    await memory.close()

    client.aclose.assert_awaited_once_with()
    client.close.assert_not_called()
