import asyncio
from pathlib import Path


class WhisperTranscriber:
    """Lazy local adapter for the faster-whisper package (CTranslate2)."""

    def __init__(
        self,
        *,
        model_name: str = "small",
        language: str = "",
        device: str = "cpu",
        compute_type: str | None = None,
        download_root: str | Path = "data/media/whisper",
    ):
        self.model_name = model_name
        self.language = language.strip()
        self.device = device
        self.compute_type = compute_type
        self.download_root = Path(download_root)
        self._model = None
        self._load_lock = asyncio.Lock()

    async def transcribe(self, file_path: str) -> str:
        model = await self._get_model()
        return await asyncio.to_thread(
            self._transcribe_sync, model, file_path, self.language or None
        )

    async def preload(self) -> None:
        await self._get_model()

    async def _get_model(self):
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("未安装 faster-whisper") from exc
            self.download_root.mkdir(parents=True, exist_ok=True)
            self._model = await asyncio.to_thread(
                WhisperModel,
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.download_root),
            )
        return self._model

    @staticmethod
    def _transcribe_sync(model, file_path: str, language: str | None) -> str:
        segments, _info = model.transcribe(
            file_path,
            language=language,
            vad_filter=False,
        )
        # segments is a lazy generator: consume it inside this worker thread so
        # the event loop never blocks on decoding.
        return "".join(seg.text for seg in segments).strip()
