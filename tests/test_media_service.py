from unittest.mock import AsyncMock

import pytest

from core.media.service import MediaService
from core.message import InputMessage, ResourceMeta


@pytest.mark.asyncio
async def test_image_understanding_disabled_hides_tools(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        multimodal=AsyncMock(),
        storage_dir=tmp_path,
        image_understanding={"enabled": False},
    )
    assert not service.tools_enabled


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
