import asyncio
import shutil
from pathlib import Path


class WhisperTranscriber:
    """Lazy local adapter for the openai-whisper package."""

    def __init__(
        self,
        *,
        model_name: str = "small",
        language: str = "",
        download_root: str | Path = "data/media/whisper",
    ):
        self.model_name = model_name
        self.language = language.strip()
        self.download_root = Path(download_root)
        self._model = None
        self._load_lock = asyncio.Lock()

    async def transcribe(self, file_path: str) -> str:
        model = await self._get_model()
        options = {"fp16": False}
        if self.language:
            options["language"] = self.language
        result = await asyncio.to_thread(model.transcribe, file_path, **options)
        return str(result.get("text") or "").strip()

    async def preload(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("未找到 ffmpeg，无法启用 Whisper 语音转写")
        await self._get_model()

    async def _get_model(self):
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError("未安装 openai-whisper") from exc
            self.download_root.mkdir(parents=True, exist_ok=True)
            self._model = await asyncio.to_thread(
                whisper.load_model,
                self.model_name,
                download_root=str(self.download_root),
            )
        return self._model
