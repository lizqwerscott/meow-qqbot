from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.media.service import MediaService
from core.webui.app import create_app


@pytest.mark.asyncio
async def test_status_page_shows_media_provider_status(tmp_path):
    service = MediaService(http_client=SimpleNamespace(), storage_dir=tmp_path)
    app = create_app({"media_service": service}, {})
    app.state.managers["agent_engine"] = SimpleNamespace(
        hindsight=None,
        get_stats=AsyncMock(
            return_value={
                "queue_sizes": {},
                "active_chats": 0,
                "total_messages": 0,
                "hindsight_health": {"status": "disabled"},
                "cost": {
                    "turn_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_hit_rate": 0,
                    "total_cost": 0,
                },
                "learners": {},
            }
        ),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/status")

    assert response.status_code == 200
    assert "媒体 Provider" in response.text
    assert "未配置可用 Provider" in response.text


@pytest.mark.asyncio
async def test_media_webui_list_renders_usage(tmp_path):
    service = MediaService(http_client=SimpleNamespace(), storage_dir=tmp_path)
    await service.open()
    app = create_app({"media_service": service}, {})
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/media")

    assert response.status_code == 200
    assert "媒体管理" in response.text


@pytest.mark.asyncio
async def test_media_webui_detail_and_preview(tmp_path):
    service = MediaService(http_client=SimpleNamespace(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/cat.png",
        mime_type="image/png",
        filename="cat.png",
        data=b"\x89PNG\r\n\x1a\nimage",
    )
    app = create_app({"media_service": service}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        detail = await client.get(f"/media/{record.media_id}")
        preview = await client.get(f"/media/{record.media_id}/content")
        forbidden = await client.get("/media/../etc/passwd")
    assert detail.status_code == 200
    assert record.media_id in detail.text
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/png")
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert preview.headers["cache-control"] == "private, no-store"
    assert "media://inbound/" in detail.text
    assert forbidden.status_code in (404, 307)


@pytest.mark.asyncio
async def test_media_webui_keeps_missing_record_and_disables_large_preview(tmp_path):
    service = MediaService(
        http_client=SimpleNamespace(),
        storage_dir=tmp_path,
        preview_max_inline_bytes=4,
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/cat.png",
        mime_type="image/png",
        filename="cat.png",
        data=b"\x89PNG\r\n\x1a\nimage",
    )
    Path(record.local_path).unlink()
    item = await service.get_media(record.media_id)
    assert item["storage_status"] == "missing"
    assert await service.get_preview_path(record.media_id) is None


@pytest.mark.asyncio
async def test_media_webui_uses_indexed_path_and_supports_download(tmp_path):
    service = MediaService(http_client=SimpleNamespace(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/original.png",
        mime_type="image/png",
        filename="original.png",
        data=b"\x89PNG\r\n\x1a\nimage",
    )
    renamed = record.local_path.with_name(f"{record.media_id}.renamed")
    record.local_path.rename(renamed)
    service.store._conn.execute(
        "UPDATE media_objects SET local_path=? WHERE media_id=?",
        (str(renamed), record.media_id),
    )
    service.store._conn.commit()
    app = create_app({"media_service": service}, {})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/media/{record.media_id}/content?download=true")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_media_webui_does_not_inline_html_as_text(tmp_path):
    service = MediaService(http_client=SimpleNamespace(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/page.html",
        mime_type="text/html",
        filename="page.html",
        data=b"<script>alert(1)</script>",
    )
    assert await service.get_preview_path(record.media_id) is None


@pytest.mark.asyncio
async def test_media_preview_switch_disables_download(tmp_path):
    service = MediaService(
        http_client=SimpleNamespace(), storage_dir=tmp_path, preview_enabled=False
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="image",
        source_url="https://example.test/cat.png",
        mime_type="image/png",
        filename="cat.png",
        data=b"\x89PNG\r\n\x1a\nimage",
    )
    assert await service.get_download_path(record.media_id) is None
    assert await service.get_text_preview(record.media_id) is None


@pytest.mark.asyncio
async def test_media_webui_requires_authentication(tmp_path):
    service = MediaService(http_client=SimpleNamespace(), storage_dir=tmp_path)
    app = create_app({"media_service": service}, {"token": "secret"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/media/not-registered/content")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@pytest.mark.asyncio
async def test_text_preview_uses_configured_limit(tmp_path):
    service = MediaService(
        http_client=SimpleNamespace(), storage_dir=tmp_path, text_preview_max_chars=4
    )
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/note.txt",
        mime_type="text/plain",
        filename="note.txt",
        data=b"abcdef",
    )
    assert await service.get_text_preview(record.media_id) == "abcd"


@pytest.mark.asyncio
async def test_sessions_webui_repairs_legacy_gap_when_timeline_is_nonempty(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    class ContextManager:
        async def get_all_disk_chat_ids_async(self):
            return ["chat-1"]

        async def get_archived_sessions_summary_async(self):
            return {}

        async def get_chat_history_async(self, chat_id, max_messages=None):
            return [
                {
                    "role": "user",
                    "content": "hello",
                    "message_id": "u1",
                    "timestamp": 1,
                },
                {"role": "assistant", "content": "legacy answer", "timestamp": 2},
            ]

        async def get_session_summary_async(self, chat_id):
            raise AssertionError("timeline summary should be used")

        async def get_archived_files_async(self, chat_id):
            return []

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    await timeline.append_user_message(
        chat_id="chat-1",
        message_id="u1",
        content="hello",
        sender_id="u1",
        timestamp=1,
    )
    app = create_app(
        {"context_manager": ContextManager(), "conversation_timeline": timeline}, {}
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/sessions")
        detail = await client.get("/sessions/chat-1")

    assert response.status_code == 200
    assert ">2</td>" in response.text
    assert detail.status_code == 200
    assert "legacy answer" in detail.text
    assert (await timeline.history("chat-1"))[-1]["content"] == "legacy answer"
    await timeline.close()
