from unittest.mock import AsyncMock

import pytest

from core.media.service import MediaService
from core.tools.deps import ToolDeps
from core.tools.impl.media import create_media_entries


@pytest.mark.asyncio
async def test_media_tools_expose_file_inspection_without_vision(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        image_understanding={"enabled": False},
    )
    entries = create_media_entries(ToolDeps(media_service=service))
    assert [entry.name for entry in entries] == ["inspect_file"]


@pytest.mark.asyncio
async def test_media_tools_expose_voice_transcription(tmp_path):
    transcriber = type("Transcriber", (), {"preload": AsyncMock()})()
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        voice_transcriber=transcriber,
        voice_transcription={"enabled": True},
    )
    entries = create_media_entries(ToolDeps(media_service=service))
    assert "transcribe_voice" in [entry.name for entry in entries]
