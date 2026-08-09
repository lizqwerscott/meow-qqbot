import asyncio
import logging
from types import SimpleNamespace

import pytest

from core.media.capabilities import MediaCapability, MediaCapabilityTimeoutError


class FailingProvider:
    name = "first"
    model_name = "first-model"

    def __init__(self):
        self.calls = 0

    async def execute(self, record, **kwargs):
        self.calls += 1
        raise RuntimeError("unavailable")


class SuccessfulProvider:
    name = "second"
    model_name = "second-model"

    def __init__(self):
        self.calls = 0

    async def execute(self, record, **kwargs):
        self.calls += 1
        return "recognized media"


@pytest.mark.asyncio
async def test_media_capability_falls_back_and_caches_result():
    first = FailingProvider()
    second = SuccessfulProvider()
    capability = MediaCapability(
        name="test",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[first, second],
    )
    record = SimpleNamespace(resource_type="voice", size=10, media_id="media-1")

    initial = await capability.execute(record, cache_key=("hash", "transcript"))
    cached = await capability.execute(record, cache_key=("hash", "transcript"))

    assert initial.content == "recognized media"
    assert initial.provider == "second"
    assert initial.model == "second-model"
    assert not initial.cached
    assert cached.cached
    assert first.calls == second.calls == 1


@pytest.mark.asyncio
async def test_media_capability_logs_underlying_provider_error(caplog):
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[FailingProvider()],
    )
    record = SimpleNamespace(resource_type="voice", size=10, media_id="media-1")

    with caplog.at_level(logging.WARNING, logger="core.media.capabilities"):
        with pytest.raises(RuntimeError, match="voice_transcription providers failed"):
            await capability.execute(record)

    assert "detail=unavailable" in caplog.text


@pytest.mark.asyncio
async def test_media_capability_reports_timeout_after_all_providers_timeout():
    class TimeoutProvider:
        name = "slow"
        model_name = "slow-model"

        async def execute(self, record, **kwargs):
            await asyncio.sleep(2)

    capability = MediaCapability(
        name="test",
        resource_types={"voice"},
        max_bytes=100,
        timeout=0.01,
        concurrency=1,
        providers=[TimeoutProvider()],
    )
    record = SimpleNamespace(resource_type="voice", size=10, media_id="media-1")

    with pytest.raises(MediaCapabilityTimeoutError):
        await capability.execute(record)


@pytest.mark.asyncio
async def test_text_extractor_reads_through_worker_thread(tmp_path, monkeypatch):
    from core.media.service import _TextExtractorProvider

    path = tmp_path / "notes.txt"
    path.write_text("abcdef", encoding="utf-8")
    calls = []

    async def fake_to_thread(function, *args, **kwargs):
        calls.append((function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr("core.media.service.asyncio.to_thread", fake_to_thread)
    result = await _TextExtractorProvider().execute(
        SimpleNamespace(local_path=path), max_chars=4
    )

    assert result == "abcd"
    assert calls
