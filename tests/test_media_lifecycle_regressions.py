import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.bootstrap import ServiceGraph
from core.media.service import MediaService


@pytest.mark.asyncio
async def test_service_graph_retries_transient_gateway_tls_failure(monkeypatch):
    graph = ServiceGraph.__new__(ServiceGraph)
    attempts = 0

    async def get_gateway_url():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            try:
                raise httpx.ReadError("tls read failed")
            except httpx.ReadError as exc:
                raise RuntimeError("Failed to get QQ Bot access token") from exc
        return "wss://gateway.example.test"

    graph.bot_engine = SimpleNamespace(
        api=SimpleNamespace(get_gateway_url=get_gateway_url)
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    assert await graph._get_gateway_url_with_retry() == "wss://gateway.example.test"
    assert attempts == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_service_graph_does_not_retry_non_transport_gateway_failure():
    graph = ServiceGraph.__new__(ServiceGraph)
    get_gateway_url = AsyncMock(side_effect=RuntimeError("invalid credentials"))
    graph.bot_engine = SimpleNamespace(
        api=SimpleNamespace(get_gateway_url=get_gateway_url)
    )

    with pytest.raises(RuntimeError, match="invalid credentials"):
        await graph._get_gateway_url_with_retry()

    get_gateway_url.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_bot_engine_stop_continues_after_nickname_failure(monkeypatch):
    from core.engine.client import BotEngine

    engine = BotEngine.__new__(BotEngine)
    events = []

    async def fail_nickname():
        events.append("nickname")
        raise RuntimeError("nickname failed")

    async def stop_agent():
        events.append("agent")

    async def stop_ws():
        events.append("ws")

    async def close_http():
        events.append("http")

    engine.nickname_manager = SimpleNamespace(
        flush_save=fail_nickname,
        save_auto=lambda: events.append("auto") or asyncio.sleep(0),
    )
    engine.approval_manager = None
    engine.agent_engine = SimpleNamespace(stop=stop_agent)
    engine.ws = SimpleNamespace(async_stop=stop_ws)
    engine._http_client = SimpleNamespace(aclose=close_http)

    await engine.stop()

    assert events == ["nickname", "auto", "agent", "ws", "http"]

    import main as main_module

    events = []

    class BrokenGraph:
        def __init__(self, cfg):
            events.append("build")

        async def build(self):
            return self

        async def start(self):
            events.append("start")
            raise RuntimeError("startup failed")

        async def stop(self):
            events.append("stop")

    monkeypatch.setattr(main_module, "ServiceGraph", BrokenGraph)
    monkeypatch.setattr(main_module, "ConfigLoader", lambda: object())
    monkeypatch.setattr(
        main_module, "setup_logging", lambda: main_module.logging.getLogger("test")
    )

    await main_module.main()

    assert events == ["build", "start", "stop"]


@pytest.mark.asyncio
async def test_service_graph_stop_continues_after_cleanup_failure():
    graph = ServiceGraph.__new__(ServiceGraph)
    events = []

    def cleanup(name, fails=False):
        async def _cleanup():
            events.append(name)
            if fails:
                raise RuntimeError(name)

        return _cleanup

    graph.process_registry = SimpleNamespace(stop=cleanup("process"))
    graph.cron_scheduler = SimpleNamespace(stop=cleanup("cron", fails=True))
    graph.heartbeat_manager = SimpleNamespace(stop=cleanup("heartbeat"))
    graph.bot_engine = SimpleNamespace(stop=cleanup("bot"))
    graph.media_service = SimpleNamespace(close=cleanup("media"))
    graph.tts_service = SimpleNamespace(close=cleanup("tts"))
    graph.model_registry = SimpleNamespace(close=cleanup("models"))
    graph.http_client = SimpleNamespace(aclose=cleanup("http"))
    graph.task_cleanup_task = None
    graph.context_cleanup_task = None
    graph.webui_task = None

    await graph.stop()

    assert events == [
        "process",
        "cron",
        "heartbeat",
        "bot",
        "media",
        "tts",
        "models",
        "http",
    ]
    assert graph.http_client is None
    assert graph.media_service is None
    assert graph.bot_engine is None
    assert graph.model_registry is None


@pytest.mark.asyncio
async def test_malformed_nested_media_config_degrades(tmp_path):
    service = MediaService(
        http_client=AsyncMock(),
        storage_dir=tmp_path,
        image_understanding={"ocr": "bad", "inspect": "bad"},
        voice_transcription=[],
        download_timeout="bad",
        download_concurrency="bad",
    )
    assert service.download_timeout == 15
    assert service.voice_enabled is False
    await service.open()
    await service.close()


@pytest.mark.asyncio
async def test_media_dns_resolution_obeys_download_timeout(tmp_path, monkeypatch):
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, download_timeout=0.01
    )

    async def slow_resolve(*args):
        await asyncio.sleep(1)
        return []

    monkeypatch.setattr(service, "_resolve_addresses", slow_resolve)
    with pytest.raises(ValueError, match="仅允许"):
        await service._validate_url("not-a-url")


@pytest.mark.asyncio
async def test_media_dns_resolution_timeout_is_bounded(tmp_path, monkeypatch):
    service = MediaService(
        http_client=AsyncMock(), storage_dir=tmp_path, download_timeout=0.01
    )

    def slow_getaddrinfo(*args):
        time.sleep(1)
        return []

    monkeypatch.setattr("core.media.service.socket.getaddrinfo", slow_getaddrinfo)
    with pytest.raises(asyncio.TimeoutError):
        await service._resolve_addresses("example.test")
