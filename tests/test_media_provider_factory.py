import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.media.capabilities import (
    MediaCapability,
    MediaCapabilityTimeoutError,
    MediaProviderHealth,
)
from core.media.provider_factory import (
    MediaProviderBuildContext,
    MediaProviderFactory,
    OpenAITranscriptionProvider,
    VisionModelProvider,
)


class FakeRegistry:
    cooldown_manager = None

    def __init__(self, services=None):
        self.services = services or {}

    def get(self, name):
        return self.services.get(name)


class FakeProvider:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.model_name = f"{name}-model"
        self.result = result
        self.error = error
        self.calls = 0

    async def execute(self, record, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class TimeoutAwareProvider(FakeProvider):
    def __init__(self, name, timeout_seconds, result="ok"):
        super().__init__(name, result=result)
        self.timeout_seconds = timeout_seconds


@pytest.fixture
def build_context():
    def _build(media_config, providers_config=None):
        return MediaProviderFactory.build(
            MediaProviderBuildContext(
                model_registry=FakeRegistry(),
                providers_config=providers_config or {},
                media_config=media_config,
                http_client=AsyncMock(),
            )
        )

    return _build


@pytest.mark.asyncio
async def test_factory_explicit_empty_chains_disable_legacy_providers(build_context):
    chains = build_context(
        {
            "image_understanding": {
                "summary": {"providers": []},
                "inspect": {"providers": []},
            },
            "voice_transcription": {"enabled": True, "providers": []},
        }
    )

    assert chains.image_summary == ()
    assert chains.image_inspect == ()
    assert chains.voice_transcription == ()


@pytest.mark.asyncio
async def test_factory_uses_legacy_chain_when_nested_provider_field_is_missing():
    multimodal = SimpleNamespace(model="legacy-vlm")
    chains = MediaProviderFactory.build(
        MediaProviderBuildContext(
            model_registry=FakeRegistry(),
            providers_config={},
            media_config={
                "image_understanding": {
                    "summary": {},
                    "inspect": {},
                }
            },
            http_client=AsyncMock(),
            multimodal=multimodal,
        )
    )

    assert [provider.name for provider in chains.image_summary] == [
        "multimodal",
        "rapidocr",
    ]
    assert [provider.name for provider in chains.image_inspect] == [
        "multimodal",
        "rapidocr",
    ]


@pytest.mark.asyncio
async def test_factory_ignores_malformed_sections_and_provider_config(build_context):
    chains = build_context(
        {
            "providers": "not-a-table",
            "image_understanding": {
                "summary": "not-a-table",
                "inspect": {"providers": []},
            },
            "voice_transcription": {"providers": "not-a-list"},
        }
    )

    assert chains.image_summary == ()
    assert chains.image_inspect == ()
    assert chains.voice_transcription == ()


@pytest.mark.asyncio
async def test_factory_rejects_provider_type_for_wrong_capability(build_context):
    chains = build_context(
        {
            "image_understanding": {"summary": {"providers": ["stt"]}},
            "voice_transcription": {"providers": ["ocr"]},
        },
        {
            "stt": {"type": "openai_transcription", "provider_ref": "api"},
            "ocr": {"type": "rapidocr"},
        },
    )

    assert chains.image_summary == ()
    assert chains.voice_transcription == ()


@pytest.mark.asyncio
async def test_empty_vision_result_falls_through_to_next_provider(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")

    vision_service = SimpleNamespace(
        model="vision-model",
        chat_completion=AsyncMock(return_value=(None, None)),
    )
    vision = VisionModelProvider("vision", vision_service, "vision-model")
    fallback = FakeProvider("fallback", result="fallback description")
    capability = MediaCapability(
        name="image_inspect",
        resource_types={"image"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[vision, fallback],
    )

    result = await capability.execute(
        SimpleNamespace(
            resource_type="image",
            size=5,
            media_id="media-1",
            local_path=image_path,
        )
    )

    assert result.content == "fallback description"
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_provider_health_isolated_by_provider_id():
    health = MediaProviderHealth(cooldown_seconds=60)
    first = FakeProvider("shared-name", error=RuntimeError("failed"))
    second = FakeProvider("shared-name", result="ok")
    first.provider_id = "first"
    second.provider_id = "second"
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[first],
        health=health,
    )
    with pytest.raises(RuntimeError):
        await capability.execute(
            SimpleNamespace(resource_type="voice", size=1, media_id="m1")
        )
    assert health.details()["first"]["last_error_category"] == "failed"
    assert health.available("second")


@pytest.mark.asyncio
async def test_capability_accepts_provider_without_timeout():
    provider = FakeProvider("cloud", result="recognized")
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[provider],
    )

    result = await capability.execute(
        SimpleNamespace(resource_type="voice", size=10, media_id="media-1")
    )
    assert result.content == "recognized"


@pytest.mark.asyncio
async def test_capability_uses_provider_specific_timeout():
    provider = TimeoutAwareProvider("cloud", timeout_seconds=0.01)
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[provider],
    )
    result = await capability.execute(
        SimpleNamespace(resource_type="voice", size=10, media_id="media-1")
    )
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_factory_skips_legacy_ocr_when_dependency_is_unavailable(monkeypatch):
    monkeypatch.setattr("core.media.provider_factory.is_ocr_available", lambda: False)
    chains = MediaProviderFactory.build(
        MediaProviderBuildContext(
            model_registry=FakeRegistry(),
            providers_config={},
            media_config={"image_understanding": {}},
            http_client=AsyncMock(),
        )
    )
    assert chains.image_summary == ()
    assert chains.image_inspect == ()


@pytest.mark.asyncio
async def test_factory_builds_explicit_vision_and_cloud_voice_chain(tmp_path):
    model = SimpleNamespace(model="vision-model")
    registry = FakeRegistry({"provider/model": model})
    transcriber = SimpleNamespace(model_name="small", preload=AsyncMock())
    chains = MediaProviderFactory.build(
        MediaProviderBuildContext(
            model_registry=registry,
            providers_config={
                "api": {"base_url": "https://stt.example/v1", "api_key": "key"}
            },
            media_config={
                "providers": {
                    "vision": {"type": "vision_model", "model_ref": "provider/model"},
                    "cloud": {
                        "type": "openai_transcription",
                        "provider_ref": "api",
                    },
                },
                "image_understanding": {"inspect": {"providers": ["vision"]}},
                "voice_transcription": {"providers": ["cloud"]},
            },
            http_client=AsyncMock(),
            voice_transcriber=transcriber,
        )
    )
    assert chains.image_inspect[0].provider_id == "vision"
    assert chains.voice_transcription[0].provider_id == "cloud"


@pytest.mark.asyncio
async def test_factory_skips_unhashable_provider_type(build_context):
    chains = build_context(
        {
            "image_understanding": {"summary": {"providers": ["bad"]}},
        },
        {"bad": {"type": []}},
    )
    assert chains.image_summary == ()


@pytest.mark.asyncio
async def test_media_capability_falls_back_after_wrapped_dependency_error():
    first = FakeProvider("local", error=RuntimeError("未安装 faster-whisper"))
    second = FakeProvider("cloud", result="recognized")
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[first, second],
    )

    result = await capability.execute(
        SimpleNamespace(resource_type="voice", size=10, media_id="media-1")
    )
    assert result.content == "recognized"
    assert capability.health.snapshot()["local"] == "unavailable"


@pytest.mark.asyncio
async def test_media_capability_total_timeout_does_not_run_later_provider():
    class SlowProvider(FakeProvider):
        async def execute(self, record, **kwargs):
            self.calls += 1
            await asyncio.sleep(0.2)

    import asyncio

    first = SlowProvider("first")
    second = FakeProvider("second", result="late")
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        total_timeout=0.01,
        concurrency=1,
        providers=[first, second],
    )

    with pytest.raises(MediaCapabilityTimeoutError):
        await capability.execute(
            SimpleNamespace(resource_type="voice", size=10, media_id="media-1")
        )
    assert second.calls == 0


@pytest.mark.asyncio
async def test_http_failure_enters_provider_cooldown():
    provider = FakeProvider("cloud", error=httpx.ConnectError("offline"))
    capability = MediaCapability(
        name="voice_transcription",
        resource_types={"voice"},
        max_bytes=100,
        timeout=1,
        concurrency=1,
        providers=[provider],
    )
    record = SimpleNamespace(resource_type="voice", size=10, media_id="media-1")

    with pytest.raises(RuntimeError):
        await capability.execute(record)
    with pytest.raises(RuntimeError):
        await capability.execute(record)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_local_whisper_preload_can_be_disabled():
    transcriber = SimpleNamespace(model_name="small", preload=AsyncMock())
    chains = MediaProviderFactory.build(
        MediaProviderBuildContext(
            model_registry=FakeRegistry(),
            providers_config={},
            media_config={
                "voice_transcription": {
                    "enabled": True,
                    "preload": False,
                }
            },
            http_client=AsyncMock(),
            voice_transcriber=transcriber,
        )
    )

    await chains.voice_transcription[0].preload()
    transcriber.preload.assert_not_awaited()


@pytest.mark.asyncio
async def test_openai_transcription_provider_posts_only_saved_media(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF test")
    request = httpx.Request("POST", "https://stt.example/v1/audio/transcriptions")
    response = httpx.Response(200, json={"text": "你好"}, request=request)
    client = AsyncMock()
    client.post.return_value = response
    provider = OpenAITranscriptionProvider(
        name="cloud_stt",
        http_client=client,
        api_key="secret",
        base_url="https://stt.example/v1",
        model="transcribe-1",
    )

    result = await provider.execute(
        SimpleNamespace(
            local_path=path,
            filename="clip.wav",
            mime_type="audio/wav",
            media_id="media-1",
        )
    )

    assert result == "你好"
    kwargs = client.post.call_args.kwargs
    uploaded = kwargs["files"]["file"]
    assert uploaded[0] == "clip.wav"
    assert uploaded[2] == "audio/wav"
    assert (
        client.post.call_args.args[0] == "https://stt.example/v1/audio/transcriptions"
    )
    assert kwargs["headers"] == {"Authorization": "Bearer secret"}
    assert "media://" not in str(kwargs)
