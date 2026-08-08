import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest

from core.media.service import MediaService
from core.media.store import MediaStore
from core.message import InputMessage, ResourceMeta


class FakeTranscriber:
    model_name = "small"

    def __init__(self):
        self.preload = AsyncMock()
        self.transcribe = AsyncMock(return_value="你好 hello")


@pytest.mark.asyncio
async def test_image_understanding_disabled_hides_tools(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=AsyncMock(),
        storage_dir=tmp_path,
        image_understanding={"enabled": False},
    )
    assert not service.image_tools_enabled
    assert service.file_tools_enabled


@pytest.mark.asyncio
async def test_inspect_file_authorizes_type_and_truncation(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.md",
        mime_type="text/markdown",
        filename="notes.md",
        data=b"abcdef",
    )
    result = await service.inspect_file(
        chat_id="g1", media_uri=record.media_uri, max_chars=4
    )
    assert result.content == "abcd"
    assert result.truncated
    assert (
        await service.inspect_file(chat_id="g2", media_uri=record.media_uri)
    ).error == ("MEDIA_NOT_AVAILABLE")


@pytest.mark.asyncio
async def test_inspect_file_rejects_unsupported_type(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/archive.zip",
        mime_type="application/zip",
        filename="archive.zip",
        data=b"not-a-zip",
    )
    result = await service.inspect_file(chat_id="g1", media_uri=record.media_uri)
    assert result.error == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.asyncio
async def test_prepare_for_ai_includes_current_text_file_excerpt(tmp_path):
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, file_context_max_chars=4
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=b"abcdef",
    )
    resource = ResourceMeta(
        resource_type="file",
        media_uri=record.media_uri,
        filename="notes.txt",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="看看文件",
        is_group=True,
        resources=[resource],
    )
    context = await service.prepare_for_ai(message)
    text = context.as_text()
    assert "[当前文件]" in text
    assert "abcd" in text
    assert "inspect_file" in text


@pytest.mark.asyncio
async def test_voice_transcription_preloads_caches_and_isolates_sessions(tmp_path):
    transcriber = FakeTranscriber()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    transcriber.preload.assert_awaited_once()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice.ogg",
        mime_type="audio/ogg",
        filename="voice.ogg",
        data=b"voice",
    )
    first = await service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    second = await service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    other_chat = await service.transcribe_voice(
        chat_id="g2", media_uri=record.media_uri
    )
    assert first.transcript == "你好 hello"
    assert not first.cached
    assert second.cached
    transcriber.transcribe.assert_awaited_once()
    assert other_chat.error == "MEDIA_NOT_AVAILABLE"


@pytest.mark.asyncio
async def test_voice_transcription_preserves_voice_type_without_audio_mime(tmp_path):
    transcriber = FakeTranscriber()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice",
        mime_type="application/octet-stream",
        filename="voice",
        data=b"voice",
    )

    result = await service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)

    assert result.transcript == "你好 hello"
    transcriber.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_voice_transcription_single_flight_for_concurrent_requests(tmp_path):
    transcriber = FakeTranscriber()
    started = asyncio.Event()
    release = asyncio.Event()

    async def transcribe(_):
        started.set()
        await release.wait()
        return "你好 hello"

    transcriber.transcribe.side_effect = transcribe
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice.ogg",
        mime_type="audio/ogg",
        filename="voice.ogg",
        data=b"voice",
    )
    first = asyncio.create_task(
        service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    )
    await started.wait()
    second = asyncio.create_task(
        service.transcribe_voice(chat_id="g1", media_uri=record.media_uri)
    )
    release.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.transcript == second_result.transcript == "你好 hello"
    transcriber.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_media_store_migrates_legacy_index_with_resource_type(tmp_path):
    index_path = tmp_path / "index.sqlite3"
    with sqlite3.connect(index_path) as conn:
        conn.execute(
            "CREATE TABLE media_messages ("
            "chat_id TEXT NOT NULL, message_id TEXT NOT NULL, "
            "sender_id TEXT NOT NULL, media_id TEXT NOT NULL, "
            "position INTEGER NOT NULL, created_at REAL NOT NULL, "
            "PRIMARY KEY (chat_id, message_id, media_id)"
            ")"
        )

    store = MediaStore(tmp_path)
    await store.open()
    columns = {
        row["name"]
        for row in store._conn.execute("PRAGMA table_info(media_messages)").fetchall()
    }

    assert "resource_type" in columns


@pytest.mark.asyncio
async def test_prepare_for_ai_includes_current_voice_transcription(tmp_path):
    transcriber = FakeTranscriber()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="voice",
        source_url="https://example.test/voice.ogg",
        mime_type="audio/ogg",
        filename="voice.ogg",
        data=b"voice",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="听听这个",
        is_group=True,
        resources=[ResourceMeta(resource_type="voice", media_uri=record.media_uri)],
    )
    text = (await service.prepare_for_ai(message)).as_text()
    assert "[当前语音]" in text
    assert "你好 hello" in text


@pytest.mark.asyncio
async def test_failed_download_does_not_retain_source_url(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    resource = ResourceMeta(
        resource_type="image",
        resource_id="https://example.test/image.jpg",
        source_url="https://example.test/image.jpg",
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="g1",
        content="",
        is_group=True,
        resources=[resource],
    )
    service._download = AsyncMock(side_effect=ValueError("failed"))
    await service.ingest_message(message)
    assert resource.storage_status == "failed"
    assert resource.source_url == ""
    assert resource.resource_id == ""
