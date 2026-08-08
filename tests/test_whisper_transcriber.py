import pytest

from core.media.whisper_transcriber import WhisperTranscriber


def test_whisper_transcriber_defaults_to_multilingual_small_model(tmp_path):
    transcriber = WhisperTranscriber(download_root=tmp_path)
    assert transcriber.model_name == "small"
    assert transcriber.language == ""


@pytest.mark.asyncio
async def test_whisper_preload_requires_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("core.media.whisper_transcriber.shutil.which", lambda _: None)
    transcriber = WhisperTranscriber(download_root=tmp_path)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        await transcriber.preload()
