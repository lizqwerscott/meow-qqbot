from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from core.media.service import MediaService
from core.webui.app import create_app


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
