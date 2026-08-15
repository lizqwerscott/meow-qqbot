import sys
import types

import pytest

from core.media.whisper_transcriber import WhisperTranscriber


def test_whisper_transcriber_defaults_to_multilingual_small_model(tmp_path):
    transcriber = WhisperTranscriber(download_root=tmp_path)
    assert transcriber.model_name == "small"
    assert transcriber.language == ""
    assert transcriber.device == "cpu"


@pytest.mark.asyncio
async def test_whisper_preload_loads_model_once(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    transcriber = WhisperTranscriber(download_root=tmp_path)
    await transcriber.preload()
    await transcriber.preload()
    assert len(calls) == 1
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["download_root"] == str(tmp_path)
    assert calls[0][1]["local_files_only"] is True


@pytest.mark.asyncio
async def test_whisper_preload_downloads_only_when_cache_is_missing(
    tmp_path, monkeypatch
):
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            if kwargs.get("local_files_only"):
                raise LocalEntryNotFoundError("model cache is missing")

    fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    transcriber = WhisperTranscriber(download_root=tmp_path)

    await transcriber.preload()

    assert len(calls) == 2
    assert calls[0][1]["local_files_only"] is True
    assert "local_files_only" not in calls[1][1]


@pytest.mark.asyncio
async def test_whisper_transcribe_joins_segments(tmp_path, monkeypatch):
    class FakeModel:
        def transcribe(self, file_path, **kwargs):
            class Seg:
                text = "在试一下呢"

            return iter([Seg(), type("S", (), {"text": "，猫猫"})()]), None

    transcriber = WhisperTranscriber(download_root=tmp_path)
    monkeypatch.setattr(transcriber, "_model", FakeModel())
    text = await transcriber.transcribe("x.wav")
    assert text == "在试一下呢，猫猫"
