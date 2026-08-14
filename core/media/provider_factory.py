"""Media capability provider construction and compatibility wiring."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from core.media.models import MediaRecord
from core.media.ocr import OcrEngine, OcrProvider, is_ocr_available
from core.media.whisper_transcriber import WhisperTranscriber

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaProviderChains:
    image_summary: tuple[Any, ...] = ()
    image_inspect: tuple[Any, ...] = ()
    voice_transcription: tuple[Any, ...] = ()


@dataclass(frozen=True)
class MediaProviderBuildContext:
    model_registry: Any
    providers_config: dict[str, Any]
    media_config: dict[str, Any]
    http_client: httpx.AsyncClient
    multimodal: Any = None
    ocr_engine: Any = None
    voice_transcriber: Any = None


class VisionModelProvider:
    def __init__(
        self,
        name: str,
        service: Any,
        model_ref: str,
        *,
        timeout_seconds: float | None = None,
    ):
        self.name = name
        self.provider_id = name
        self.service = service
        self.model_name = model_ref
        self.timeout_seconds = timeout_seconds

    async def execute(self, record: MediaRecord, **kwargs: Any) -> str:
        content = await self.service.analyze_image(
            str(record.local_path), prompt=kwargs.get("prompt")
        )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("vision provider returned empty content")
        return content.strip()


class LegacyMultimodalProvider(VisionModelProvider):
    def __init__(self, service: Any):
        super().__init__("multimodal", service, getattr(service, "model", ""))


class LocalWhisperProvider:
    def __init__(
        self,
        transcriber: WhisperTranscriber,
        *,
        name: str = "local_whisper",
        required: bool = False,
        preload_enabled: bool = True,
        timeout_seconds: float | None = None,
    ):
        self.name = name
        self.provider_id = name
        self.transcriber = transcriber
        self.model_name = getattr(transcriber, "model_name", "")
        self.required = required
        self.preload_enabled = preload_enabled
        self.timeout_seconds = timeout_seconds

    async def preload(self) -> None:
        if not self.preload_enabled:
            return
        await self.transcriber.preload()

    async def execute(self, record: MediaRecord, **kwargs: Any) -> str:
        return await self.transcriber.transcribe(str(record.local_path))


class OpenAITranscriptionProvider:
    def __init__(
        self,
        *,
        name: str,
        http_client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        model: str,
        language: str = "",
        timeout_seconds: float | None = None,
    ):
        self.name = name
        self.provider_id = name
        self.http_client = http_client
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.language = language.strip()
        self.timeout_seconds = (
            None if timeout_seconds is None else max(0.1, float(timeout_seconds))
        )

    async def execute(self, record: MediaRecord, **kwargs: Any) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model_name}
        if self.language:
            data["language"] = self.language
        with record.local_path.open("rb") as media_file:
            response = await self.http_client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files={
                    "file": (
                        record.filename or f"{record.media_id}.audio",
                        media_file,
                        record.mime_type,
                    )
                },
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("transcription response has no text")
        return text.strip()


def _optional_timeout_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.1, float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _diagnostic(capability: str, provider_id: str, reason: str) -> None:
    _log.warning(
        "媒体 provider 配置不可用 capability=%s provider=%s reason=%s",
        capability,
        provider_id,
        reason,
    )


def _build_ocr(ctx: MediaProviderBuildContext, provider_id: str, cfg: dict[str, Any]):
    image_cfg = ctx.media_config.get("image_understanding", {}) or {}
    image_cfg = image_cfg if isinstance(image_cfg, Mapping) else {}
    ocr_cfg = image_cfg.get("ocr", {}) or {}
    ocr_cfg = ocr_cfg if isinstance(ocr_cfg, Mapping) else {}
    ocr_cfg = {**ocr_cfg, **cfg}
    if not ocr_cfg.get("enabled", True):
        return None
    if ctx.ocr_engine is None and not is_ocr_available():
        _diagnostic("image", provider_id, "dependency unavailable")
        return None
    engine = ctx.ocr_engine or OcrEngine()
    return OcrProvider(
        engine,
        name=provider_id,
        min_chars=ocr_cfg.get("min_chars", 8),
        max_chars=ocr_cfg.get("max_chars", 2000),
        timeout_seconds=_optional_timeout_seconds(cfg.get("timeout_seconds")),
    )


def _build_local_whisper(
    ctx: MediaProviderBuildContext, provider_id: str, cfg: dict[str, Any]
):
    voice_cfg = ctx.media_config.get("voice_transcription", {}) or {}
    voice_cfg = voice_cfg if isinstance(voice_cfg, Mapping) else {}
    merged = {**voice_cfg, **cfg}
    if not merged.get("enabled", True):
        return None
    transcriber = ctx.voice_transcriber or WhisperTranscriber(
        model_name=merged.get("model", "small"),
        language=merged.get("language", ""),
        device=merged.get("device", "cpu"),
        compute_type=merged.get("compute_type", "int8"),
        download_root=merged.get("download_root", "data/media/whisper"),
    )
    return LocalWhisperProvider(
        transcriber,
        name=provider_id,
        required=bool(merged.get("required", False)),
        preload_enabled=bool(merged.get("preload", True)),
        timeout_seconds=_optional_timeout_seconds(merged.get("timeout_seconds")),
    )


def _build_vision(
    ctx: MediaProviderBuildContext, provider_id: str, cfg: dict[str, Any]
):
    model_ref = str(cfg.get("model_ref", ""))
    service = ctx.model_registry.get(model_ref) if ctx.model_registry else None
    if not model_ref or service is None:
        _diagnostic("image", provider_id, "unknown model_ref")
        return None
    from core.ai.multimodal import MultimodalService

    multimodal = MultimodalService(
        [service],
        model_names=[model_ref],
        cooldown_manager=ctx.model_registry.cooldown_manager,
    )
    return VisionModelProvider(
        provider_id,
        multimodal,
        model_ref,
        timeout_seconds=_optional_timeout_seconds(cfg.get("timeout_seconds")),
    )


def _build_cloud_stt(
    ctx: MediaProviderBuildContext, provider_id: str, cfg: dict[str, Any]
):
    provider_ref = str(cfg.get("provider_ref", ""))
    provider_cfg = ctx.providers_config.get(provider_ref)
    if (
        not provider_ref
        or not isinstance(provider_cfg, Mapping)
        or not isinstance(provider_cfg.get("base_url"), str)
        or not provider_cfg.get("base_url", "").startswith(("http://", "https://"))
        or not isinstance(provider_cfg.get("api_key"), str)
        or not provider_cfg.get("api_key")
    ):
        _diagnostic(
            "voice_transcription", provider_id, "provider has no usable endpoint"
        )
        return None
    base_url = provider_cfg["base_url"]
    api_key = provider_cfg["api_key"]
    timeout_seconds = _optional_timeout_seconds(cfg.get("timeout_seconds"))
    return OpenAITranscriptionProvider(
        name=provider_id,
        http_client=ctx.http_client,
        api_key=str(api_key),
        base_url=str(base_url),
        model=str(cfg.get("model", "gpt-4o-mini-transcribe")),
        language=str(cfg.get("language", "")),
        timeout_seconds=timeout_seconds,
    )


_ALLOWED_PROVIDER_TYPES = {
    "image_summary": frozenset({"rapidocr", "vision_model"}),
    "image_inspect": frozenset({"rapidocr", "vision_model"}),
    "voice_transcription": frozenset({"local_whisper", "openai_transcription"}),
}


_BUILDERS = {
    "rapidocr": _build_ocr,
    "local_whisper": _build_local_whisper,
    "vision_model": _build_vision,
    "openai_transcription": _build_cloud_stt,
}


def _build_chain(
    ctx: MediaProviderBuildContext,
    capability: str,
    provider_ids: list[str],
    provider_configs: dict[str, Any],
) -> tuple[Any, ...]:
    seen: set[str] = set()
    result = []
    for provider_id in provider_ids:
        if provider_id in seen:
            _diagnostic(capability, provider_id, "duplicate provider id")
            continue
        seen.add(provider_id)
        cfg = provider_configs.get(provider_id)
        if not isinstance(cfg, dict):
            _diagnostic(capability, provider_id, "unknown provider id")
            continue
        provider_type = cfg.get("type")
        if not isinstance(provider_type, str):
            _diagnostic(capability, provider_id, "provider type must be a string")
            continue
        builder = _BUILDERS.get(provider_type)
        if builder is None:
            _diagnostic(capability, provider_id, "unknown provider type")
            continue
        if provider_type not in _ALLOWED_PROVIDER_TYPES.get(capability, frozenset()):
            _diagnostic(capability, provider_id, "incompatible provider type")
            continue
        try:
            provider = builder(ctx, provider_id, cfg)
        except Exception:
            _diagnostic(capability, provider_id, "construction failed")
            continue
        if provider is not None:
            result.append(provider)
    return tuple(result)


def _provider_ids_from_config(section: Any, capability: str) -> list[str] | None:
    """Return None for legacy config, [] for invalid/explicitly empty config."""
    if section is None:
        return None
    if not isinstance(section, Mapping):
        _diagnostic(capability, "(section)", "section must be a table")
        return []
    if "providers" not in section:
        _diagnostic(capability, "(section)", "providers field is missing")
        return None
    provider_ids = section["providers"]
    if not isinstance(provider_ids, list) or not all(
        isinstance(provider_id, str) and provider_id.strip()
        for provider_id in provider_ids
    ):
        _diagnostic(capability, "(section)", "providers must be a string list")
        return []
    return provider_ids


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_media_provider_chains(ctx: MediaProviderBuildContext) -> MediaProviderChains:
    media_cfg = _as_mapping(ctx.media_config)
    provider_configs = _as_mapping(media_cfg.get("providers"))
    image_cfg = _as_mapping(media_cfg.get("image_understanding"))
    voice_cfg = _as_mapping(media_cfg.get("voice_transcription"))

    summary_ids = _provider_ids_from_config(image_cfg.get("summary"), "image_summary")
    inspect_ids = _provider_ids_from_config(image_cfg.get("inspect"), "image_inspect")
    voice_ids = _provider_ids_from_config(
        voice_cfg if "providers" in voice_cfg else None, "voice_transcription"
    )

    if summary_ids is None:
        _log.info(
            "媒体 image_summary 使用 legacy 兼容链 provider=%s",
            "multimodal→rapidocr" if ctx.multimodal else "rapidocr",
        )
        summary = tuple(
            provider
            for provider in (
                LegacyMultimodalProvider(ctx.multimodal) if ctx.multimodal else None,
                _build_ocr(ctx, "rapidocr", _as_mapping(image_cfg.get("ocr"))),
            )
            if provider is not None
        )
    else:
        summary = _build_chain(ctx, "image_summary", summary_ids, provider_configs)

    if inspect_ids is None:
        _log.info(
            "媒体 image_inspect 使用 legacy 兼容链 provider=%s",
            "multimodal→rapidocr" if ctx.multimodal else "rapidocr",
        )
        inspect = tuple(
            provider
            for provider in (
                LegacyMultimodalProvider(ctx.multimodal) if ctx.multimodal else None,
                _build_ocr(ctx, "rapidocr", _as_mapping(image_cfg.get("ocr"))),
            )
            if provider is not None
        )
    else:
        inspect = _build_chain(ctx, "image_inspect", inspect_ids, provider_configs)

    if voice_ids is None:
        voice = (
            _build_local_whisper(ctx, "local_whisper", voice_cfg)
            if voice_cfg.get("enabled", False)
            else None
        )
        voice_chain = (voice,) if voice else ()
    else:
        voice_chain = _build_chain(
            ctx, "voice_transcription", voice_ids, provider_configs
        )

    return MediaProviderChains(summary, inspect, voice_chain)


class MediaProviderFactory:
    build = staticmethod(build_media_provider_chains)
