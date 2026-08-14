import asyncio
import ipaddress
import logging
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader
from qqbot_agent_sdk.audio import convert_audio_to_wav, looks_like_silk

from core.media.capabilities import (
    MediaCapability,
    MediaCapabilityTimeoutError,
    safe_error_category,
)
from core.media.models import (
    FileInspection,
    ImageInspection,
    MediaRecord,
    MediaTurnContext,
    PdfInspection,
    VoiceTranscription,
)
from core.media.ocr import OcrEngine, OcrProvider, is_ocr_available
from core.media.provider_factory import LegacyMultimodalProvider, LocalWhisperProvider
from core.media.store import MediaStore
from core.message import InputMessage, ResourceMeta

_log = logging.getLogger(__name__)

_IMAGE_SUMMARY_PROMPT_VERSION = "v2"
_IMAGE_INSPECTION_PROMPT_VERSION = "v2"
_TEXT_EXTRACTION_VERSION = "v1"
_FILE_SUMMARY_VERSION = "v1"


class _TextExtractorProvider:
    name = "builtin_text"
    model_name = ""

    async def execute(self, record: MediaRecord, **kwargs) -> str:
        return await asyncio.to_thread(
            self._read_text, record.local_path, kwargs["max_chars"]
        )

    @staticmethod
    def _read_text(path, max_chars: int) -> str:
        with path.open(encoding="utf-8", errors="replace") as file:
            return file.read(max_chars)


class _FileSummaryProvider:
    name = "file_summary_llm"

    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.model_name = getattr(ai_service, "model", "")

    async def execute(self, record: MediaRecord, **kwargs) -> str:
        text = kwargs["text"]
        summary, _ = await self.ai_service.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是文件摘要助手。请用简洁中文概括文件内容，保留主题、关键事实、"
                        "数字、日期、结论和待办事项。不要臆测，不要复述整段原文。"
                    ),
                },
                {"role": "user", "content": f"请摘要以下文件内容：\n\n{text}"},
            ],
            max_tokens=kwargs.get("max_tokens", 400),
        )
        return summary or ""


class _PdfProvider:
    name = "pypdf"

    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.model_name = getattr(ai_service, "model", "")

    async def execute(self, record: MediaRecord, **kwargs) -> str:
        text, pages = await asyncio.to_thread(self._extract, record.local_path)
        if not text.strip():
            raise ValueError("PDF 没有可提取的文本")
        prompt = kwargs.get("prompt") or "分析这份 PDF 文档。"
        result, _ = await self.ai_service.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是 PDF 文档分析助手。基于提供的文档内容回答用户问题，不要臆测。",
                },
                {"role": "user", "content": f"{prompt}\n\n文档内容：\n{text}"},
            ],
            max_tokens=kwargs.get("max_tokens", 800),
        )
        if not result:
            raise ValueError("PDF 分析返回空内容")
        return f"页数: {pages}\n{result}"

    @staticmethod
    def count_pages(path) -> int:
        return len(PdfReader(str(path)).pages)

    @staticmethod
    def _extract(path) -> tuple[str, int]:
        reader = PdfReader(str(path))
        pages = len(reader.pages)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text[:100_000], pages


def _safe_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError, OverflowError):
        return default
class MediaService:
    """媒体生命周期模块：保存、授权、摘要和按需图片分析。"""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        multimodal=None,
        storage_dir="data/media",
        enabled=True,
        image_understanding=None,
        recent_window_seconds=600,
        recent_max_items=5,
        max_attachments_per_message=5,
        max_image_bytes=10 * 1024 * 1024,
        max_file_bytes=25 * 1024 * 1024,
        download_timeout=15,
        download_concurrency=4,
        max_total_bytes=2 * 1024 * 1024 * 1024,
        preview_enabled=True,
        preview_max_inline_bytes=50 * 1024 * 1024,
        text_preview_max_chars=20_000,
        file_extract_max_chars=20_000,
        file_context_max_chars=4_000,
        file_summary_max_chars=600,
        file_summary_timeout=30,
        file_summary_concurrency=1,
        pdf_max_bytes=50 * 1024 * 1024,
        pdf_max_pages=20,
        pdf_timeout=60,
        pdf_max_tokens=800,
        ai_service=None,
        voice_transcriber=None,
        voice_transcription=None,
        ocr_engine=None,
        provider_chains=None,
    ):
        self.enabled = enabled
        self.http_client = http_client
        self.multimodal = multimodal
        image_config = image_understanding if isinstance(image_understanding, Mapping) else {}
        self.image_enabled = bool(image_config.get("enabled", True))
        self.max_auto_images = _safe_int(image_config.get("max_auto_images", 3), 3)
        self.analysis_timeout = _safe_float(image_config.get("analysis_timeout_seconds", 30), 30)
        self.summary_max_chars = _safe_int(image_config.get("summary_max_chars", 300), 300, 20)
        ocr_cfg = image_config.get("ocr", {})
        ocr_cfg = ocr_cfg if isinstance(ocr_cfg, Mapping) else {}
        self.ocr_enabled = bool(ocr_cfg.get("enabled", True))
        self.ocr_min_chars = max(0, int(ocr_cfg.get("min_chars", 8)))
        self.ocr_max_chars = max(1, int(ocr_cfg.get("max_chars", 2000)))
        self.recent_window_seconds = max(0, int(recent_window_seconds))
        self.recent_max_items = max(0, int(recent_max_items))
        self.max_attachments = max(1, int(max_attachments_per_message))
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.download_timeout = _safe_float(download_timeout, 15)
        self.download_sem = asyncio.Semaphore(_safe_int(download_concurrency, 4, 1))
        self.max_total_bytes = max_total_bytes
        self.preview_enabled = bool(preview_enabled)
        self.preview_max_inline_bytes = max(1, int(preview_max_inline_bytes))
        self.text_preview_max_chars = max(1, int(text_preview_max_chars))
        self.file_extract_max_chars = max(1, int(file_extract_max_chars))
        self.file_context_max_chars = max(1, int(file_context_max_chars))
        self.file_summary_max_chars = max(20, int(file_summary_max_chars))
        self.pdf_max_tokens = max(50, int(pdf_max_tokens))
        self.pdf_max_pages = max(1, int(pdf_max_pages))
        self.pdf_max_bytes = max(1, int(pdf_max_bytes))
        voice_cfg = voice_transcription if isinstance(voice_transcription, Mapping) else {}
        self.voice_transcriber = voice_transcriber
        self.voice_timeout = _safe_float(voice_cfg.get("timeout_seconds", 120), 120)
        self.voice_max_chars = _safe_int(voice_cfg.get("max_chars", 4_000), 4_000, 1)
        self.store = MediaStore(storage_dir)
        self._opened = False
        if provider_chains is None:
            vlm_provider = LegacyMultimodalProvider(multimodal) if multimodal else None
            ocr_provider = None
            if self.ocr_enabled:
                # 显式注入的引擎（测试/替换实现）直接采用；默认引擎要求 rapidocr 可导入
                if ocr_engine is not None or is_ocr_available():
                    ocr_engine = ocr_engine or OcrEngine()
                    ocr_provider = OcrProvider(
                        ocr_engine,
                        min_chars=self.ocr_min_chars,
                        max_chars=self.ocr_max_chars,
                    )
                else:
                    _log.warning("OCR 已启用但 rapidocr 未安装，跳过本地 OCR provider")
            summary_providers = [p for p in (ocr_provider, vlm_provider) if p]
            inspect_providers = [p for p in (vlm_provider, ocr_provider) if p]
            voice_providers = (
                [
                    LocalWhisperProvider(
                        voice_transcriber,
                        preload_enabled=bool(voice_cfg.get("preload", True)),
                    )
                ]
                if voice_transcriber
                else []
            )
        else:
            summary_providers = list(provider_chains.image_summary)
            inspect_providers = list(provider_chains.image_inspect)
            voice_providers = list(provider_chains.voice_transcription)
        self.voice_enabled = bool(
            voice_cfg.get("enabled", bool(voice_providers)) and voice_providers
        )
        # 自动摘要：provider factory 未配置时保留 OCR → legacy VLM 的兼容链
        self.image_capability = MediaCapability(
            name="image_understanding",
            resource_types={"image"},
            max_bytes=self.max_image_bytes,
            timeout=self.analysis_timeout,
            total_timeout=image_config.get("total_timeout_seconds"),
            concurrency=image_config.get("concurrency", 2),
            providers=summary_providers,
        )
        # image 工具：provider factory 未配置时保留 legacy VLM → OCR 的兼容链
        self.image_inspect_capability = MediaCapability(
            name="image_inspection",
            resource_types={"image"},
            max_bytes=self.max_image_bytes,
            timeout=self.analysis_timeout,
            total_timeout=(
                image_config.get("inspect", {}).get(
                    "total_timeout_seconds", image_config.get("total_timeout_seconds")
                )
                if isinstance(image_config.get("inspect", {}), Mapping)
                else image_config.get("total_timeout_seconds")
            ),
            concurrency=image_config.get("concurrency", 2),
            providers=inspect_providers,
        )
        self.file_capability = MediaCapability(
            name="text_extraction",
            resource_types={"file"},
            max_bytes=self.max_file_bytes,
            timeout=5,
            concurrency=1,
            providers=[_TextExtractorProvider()],
            cache_size=10,
        )
        self.file_summary_capability = MediaCapability(
            name="file_summary",
            resource_types={"file"},
            max_bytes=self.max_file_bytes,
            timeout=file_summary_timeout,
            concurrency=file_summary_concurrency,
            providers=[_FileSummaryProvider(ai_service)] if ai_service else [],
            cache_size=100,
        )
        self.pdf_capability = MediaCapability(
            name="pdf_analysis",
            resource_types={"file"},
            max_bytes=self.pdf_max_bytes,
            timeout=pdf_timeout,
            concurrency=1,
            providers=[_PdfProvider(ai_service)] if ai_service else [],
            cache_size=50,
        )
        self.voice_capability = MediaCapability(
            name="voice_transcription",
            resource_types={"voice"},
            max_bytes=self.max_file_bytes,
            timeout=self.voice_timeout,
            total_timeout=voice_cfg.get("total_timeout_seconds"),
            concurrency=voice_cfg.get("concurrency", 1),
            providers=voice_providers,
        )

    def provider_status(self) -> dict[str, list[dict[str, str]]]:
        capabilities = {
            "image_summary": self.image_capability,
            "image_inspect": self.image_inspect_capability,
            "voice_transcription": self.voice_capability,
        }
        result = {}
        for capability_name, capability in capabilities.items():
            health_details = capability.health.details()
            providers = []
            for provider in capability.providers:
                provider_id = getattr(provider, "provider_id", provider.name)
                details = health_details.get(
                    provider_id,
                    {"status": "available", "last_error_category": ""},
                )
                providers.append(
                    {
                        "id": provider_id,
                        "status": details["status"],
                        "last_error_category": details["last_error_category"],
                    }
                )
            result[capability_name] = providers
        return result

    async def open(self):
        if self.enabled and not self._opened:
            await self.store.open()
            await self.voice_capability.preload()
            self._opened = True

    async def close(self):
        await self.store.close()
        self._opened = False

    @property
    def tools_enabled(self) -> bool:
        return self.enabled and (
            self.file_tools_enabled
            or self.image_tools_enabled
            or self.pdf_tools_enabled
        )

    @property
    def image_tools_enabled(self) -> bool:
        return (
            self.enabled
            and self.image_enabled
            and self.image_inspect_capability.enabled
        )

    @property
    def file_tools_enabled(self) -> bool:
        return self.enabled

    @property
    def pdf_tools_enabled(self) -> bool:
        return self.enabled and self.pdf_capability.enabled

    @property
    async def usage(self) -> tuple[int, int]:
        if not self.enabled:
            return 0, 0
        await self.open()
        return await self.store.usage()

    async def cleanup(self, clear_all: bool = False) -> int:
        if not self.enabled:
            return 0
        await self.open()
        if clear_all:
            return await self.store.clear_all()
        return await self.store.cleanup(self.max_total_bytes)

    async def list_media(self, descending: bool = True, limit: int = 200) -> list[dict]:
        if not self.enabled:
            return []
        await self.open()
        return await self.store.list_objects(descending, limit)

    async def get_media(self, media_id: str) -> dict | None:
        if not self.enabled:
            return None
        await self.open()
        return await self.store.get_object(media_id)

    async def get_preview_path(self, media_id: str):
        if not self.enabled or not self.preview_enabled:
            return None
        await self.open()
        content = await self.store.get_content_info(media_id)
        if not content:
            return None
        file_path, item = content
        mime_type = item["mime_type"]
        if not (
            mime_type.startswith(("image/", "audio/", "video/"))
            or mime_type == "text/plain"
        ):
            return None
        if file_path.stat().st_size > self.preview_max_inline_bytes:
            return None
        return file_path, mime_type, item["filename"]

    async def get_download_path(self, media_id: str):
        if not self.enabled or not self.preview_enabled:
            return None
        await self.open()
        content = await self.store.get_content_info(media_id)
        if not content:
            return None
        file_path, item = content
        return file_path, item["mime_type"], item["filename"]

    async def get_text_preview(self, media_id: str, max_chars: int | None = None):
        if not self.enabled or not self.preview_enabled:
            return None
        await self.open()
        content = await self.store.get_content_info(media_id)
        if not content:
            return None
        path, item = content
        max_chars = max_chars or self.text_preview_max_chars
        if (
            item["mime_type"] != "text/plain"
            or path.stat().st_size > self.preview_max_inline_bytes
        ):
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except OSError:
            return None

    async def delete_media(self, media_id: str) -> bool:
        if not self.enabled:
            return False
        await self.open()
        return await self.store.delete_media(media_id)

    async def _prepare_voice_data(
        self, data: bytes, mime: str, resource: ResourceMeta, source_url: str
    ) -> tuple[bytes, str, bytes | None]:
        if resource.resource_type != "voice" or not (
            looks_like_silk(data) or data.startswith(b"\x02#!SILK_V3")
        ):
            return data, mime, None
        wav_path = None
        try:
            wav_path = await convert_audio_to_wav(
                data, resource.filename or source_url, log_tag="MediaService"
            )
            if not wav_path:
                return data, "audio/silk", None
            wav_data = await asyncio.to_thread(Path(wav_path).read_bytes)
            if not wav_data.startswith(b"RIFF"):
                return data, "audio/silk", None
            return wav_data, "audio/wav", data
        except Exception as exc:
            _log.warning(
                "语音转换失败，保留原始 Silk: category=%s", safe_error_category(exc)
            )
            return data, "audio/silk", None
        finally:
            if wav_path is not None:
                try:
                    await asyncio.to_thread(Path(wav_path).unlink, missing_ok=True)
                except OSError:
                    pass

    async def _ingest_resource(
        self,
        resource: ResourceMeta,
        source_url: str,
        chat_id: str,
        message_id: str,
        sender_id: str,
        position: int,
    ) -> MediaRecord:
        data, mime = await self._download(source_url, resource)
        data, mime, original_data = await self._prepare_voice_data(
            data, mime, resource, source_url
        )
        record = await self.store.save(
            chat_id=chat_id,
            message_id=message_id,
            sender_id=sender_id,
            resource_type=resource.resource_type,
            source_url=source_url,
            mime_type=mime or resource.mime_type,
            filename=resource.filename,
            data=data,
            original_data=original_data,
            position=position,
        )
        resource.media_id = record.media_id
        resource.media_uri = record.media_uri
        resource.resource_id = record.media_uri
        resource.hash = record.sha256
        resource.mime_type = record.mime_type
        resource.size = record.size
        resource.storage_status = "ready"
        resource.source_url = ""
        resource.extra["summary"] = record.summary
        return record

    async def ingest_message(self, message: InputMessage) -> None:
        if not self.enabled:
            return
        await self.open()
        resources = message.resources[: self.max_attachments]
        for index, resource in enumerate(resources):
            if resource.resource_type not in {"image", "file", "voice", "video"}:
                continue
            source_url = resource.source_url or resource.resource_id
            if not source_url:
                resource.storage_status = "failed"
                continue
            try:
                record = await self._ingest_resource(
                    resource,
                    source_url,
                    message.chat_id,
                    message.id,
                    message.sender_id,
                    index,
                )
                resource.source_url = ""
            except Exception as exc:
                resource.storage_status = "failed"
                resource.source_url = ""
                resource.resource_id = ""
                _log.warning(
                    "媒体保存失败 [%s]: category=%s",
                    message.id,
                    safe_error_category(exc),
                )
        for resource in message.resources[self.max_attachments :]:
            resource.storage_status = "failed"
            resource.source_url = ""
            resource.resource_id = ""

    async def ingest_replied_resources(self, message: InputMessage) -> None:
        if not self.enabled:
            return
        await self.open()
        for index, resource in enumerate(
            message.replied_resources[: self.max_attachments]
        ):
            if resource.media_uri or not resource.source_url:
                continue
            try:
                record = await self._ingest_resource(
                    resource,
                    resource.source_url,
                    message.chat_id,
                    message.replied_message_id or f"reply:{message.id}",
                    message.replied_author_id or message.sender_id,
                    index,
                )
                resource.source_url = ""
            except Exception as exc:
                resource.storage_status = "failed"
                resource.source_url = ""
                resource.resource_id = ""
                _log.warning(
                    "引用媒体保存失败 [%s]: category=%s",
                    message.id,
                    safe_error_category(exc),
                )

    async def resolve_replied_resources(self, message: InputMessage) -> None:
        if not message.replied_message_id:
            return
        records = await self.store.find_message_media(
            message.chat_id, message.replied_message_id
        )
        for resource, record in zip(message.replied_resources, records):
            resource.media_id = record.media_id
            resource.media_uri = record.media_uri
            resource.resource_id = record.media_uri
            resource.hash = record.sha256
            resource.mime_type = record.mime_type
            resource.size = record.size
            resource.storage_status = "ready"

    async def _download(self, url: str, resource: ResourceMeta) -> tuple[bytes, str]:
        parsed = await self._validate_url(url)
        if resource.resource_type == "image":
            limit = self.max_image_bytes
        elif (
            resource.mime_type == "application/pdf"
            or resource.filename.lower().endswith(".pdf")
        ):
            limit = self.pdf_max_bytes
        else:
            limit = self.max_file_bytes
        async with self.download_sem:
            current = parsed
            for _ in range(4):
                current = await self._validate_url(current)
                async with self.http_client.stream(
                    "GET",
                    current,
                    timeout=self.download_timeout,
                    follow_redirects=False,
                ) as response:
                    self._check_peer_address(response)
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("重定向缺少目标地址")
                        current = str(httpx.URL(current).join(location))
                        continue
                    if response.status_code >= 400:
                        raise ValueError(f"HTTP {response.status_code}")
                    declared = int(response.headers.get("content-length") or 0)
                    if declared > limit:
                        raise ValueError("媒体超过大小限制")
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(chunks) + len(chunk) > limit:
                            raise ValueError("媒体超过大小限制")
                        chunks.extend(chunk)
                    data = bytes(chunks)
                    mime = (
                        (
                            response.headers.get("content-type")
                            or resource.mime_type
                            or ""
                        )
                        .split(";", 1)[0]
                        .lower()
                    )
                    if (
                        resource.resource_type == "image"
                        and mime
                        and not mime.startswith("image/")
                    ):
                        raise ValueError("响应不是图片")
                    if resource.resource_type == "image" and not self._image_signature(
                        data
                    ):
                        raise ValueError("图片文件头校验失败")
                    if (
                        resource.mime_type == "application/pdf"
                        or resource.filename.lower().endswith(".pdf")
                        or mime == "application/pdf"
                    ) and not self._pdf_signature(data):
                        raise ValueError("PDF 文件头校验失败")
                    return data, mime
            else:
                raise ValueError("重定向次数超过限制")

    @staticmethod
    def _image_signature(data: bytes) -> bool:
        return data.startswith(
            (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"RIFF")
        )

    @staticmethod
    def _pdf_signature(data: bytes) -> bool:
        return data.startswith(b"%PDF-")

    async def _validate_url(self, url: str) -> str:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("仅允许 http/https 媒体地址")
        host = parsed.hostname.lower()
        if host == "localhost":
            raise ValueError("禁止访问 localhost")
        addresses = await self._resolve_addresses(host)
        if any(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            for ip in addresses
        ):
            raise ValueError("禁止访问私网媒体地址")
        return url.strip()

    async def _resolve_addresses(self, host: str):
        async def resolve():
            try:
                return [ipaddress.ip_address(host)]
            except ValueError:
                infos = await asyncio.get_running_loop().run_in_executor(
                    None, socket.getaddrinfo, host, None
                )
                return [ipaddress.ip_address(info[4][0]) for info in infos]

        return await asyncio.wait_for(resolve(), timeout=self.download_timeout)

    @staticmethod
    def _check_peer_address(response) -> None:
        stream = getattr(response, "extensions", {}).get("network_stream")
        if stream is None or not hasattr(stream, "get_extra_info"):
            return
        peer = stream.get_extra_info("peername")
        host = peer[0] if isinstance(peer, tuple) and peer else None
        if not host:
            return
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("禁止访问私网媒体地址")

    async def prepare_for_ai(self, message: InputMessage) -> MediaTurnContext:
        if not self.enabled:
            return MediaTurnContext()
        await self.open()
        current = []
        replied = []
        for resource in (*message.resources, *message.replied_resources):
            if len(current) + len(replied) >= self.max_auto_images:
                break
            label = "当前图片" if resource in message.resources else "引用图片"
            block = await self._summarize_resource(
                message.chat_id, resource, label, len(current) + len(replied) + 1
            )
            if block:
                (current if label == "当前图片" else replied).append(block)
        for resource in (*message.resources, *message.replied_resources):
            if resource.resource_type != "file" or not resource.media_uri:
                continue
            label = "当前文件" if resource in message.resources else "引用文件"
            block = await self._summarize_file_context(message.chat_id, resource, label)
            if block:
                (current if label == "当前文件" else replied).append(block)
        for resource in (*message.resources, *message.replied_resources):
            if resource.resource_type != "voice" or not resource.media_uri:
                continue
            label = "当前语音" if resource in message.resources else "引用语音"
            block = await self._transcribe_voice_context(
                message.chat_id, resource, label
            )
            if block:
                (current if label == "当前语音" else replied).append(block)
        used = {
            r.media_uri
            for r in message.resources + message.replied_resources
            if r.media_uri
        }
        recent = [
            r
            for r in await self.store.recent(
                message.chat_id, self.recent_window_seconds, self.recent_max_items
            )
            if r.media_uri not in used
        ]
        recent_text = ""
        if recent:
            lines = ["[近期媒体]"]
            for item in recent:
                age = max(0, int((time.time() - item.created_at) / 60))
                state = (
                    f"摘要: {item.summary}"
                    if item.summary
                    and item.summary_version == _IMAGE_SUMMARY_PROMPT_VERSION
                    else "未分析"
                )
                lines.append(
                    f"- {item.media_uri} | {item.resource_type} | 消息 {item.message_id} | 发送者 {item.sender_id} | {age} 分钟前 | {state}"
                )
            lines.append("[/近期媒体]")
            recent_text = "\n".join(lines)
        return MediaTurnContext(tuple(current), tuple(replied), recent_text)

    async def _summarize_resources(self, message, label):
        blocks = []
        count = 0
        for resource in message.resources:
            if count >= self.max_auto_images:
                break
            block = await self._summarize_resource(
                message.chat_id, resource, label, count + 1
            )
            if block:
                blocks.append(block)
                count += 1
        return blocks

    async def _summarize_resource(self, chat_id, resource, label, number=1):
        if resource.resource_type != "image" or not resource.media_uri:
            return ""
        record = await self.store.authorize(
            chat_id, resource.media_uri, image_only=True
        )
        if not record:
            return f"[{label} {number}]\n引用: {resource.media_uri}\n摘要: 图片不可用；如确有需要，可调用 image。\n[/{label} {number}]"
        summary = (
            record.summary
            if record.summary_version == _IMAGE_SUMMARY_PROMPT_VERSION
            else ""
        )
        if self.image_enabled and self.image_capability.enabled and not summary:
            try:
                result = await self.image_capability.execute(
                    record,
                    cache_key=(
                        record.sha256,
                        "summary",
                        _IMAGE_SUMMARY_PROMPT_VERSION,
                    ),
                    prompt=None,
                )
                summary = result.content
                summary = (summary or "图片未能自动解析")[: self.summary_max_chars]
                await self.store.update_summary(
                    record.media_id,
                    summary,
                    result.model,
                    _IMAGE_SUMMARY_PROMPT_VERSION,
                )
                resource.extra["summary"] = summary
            except Exception:
                summary = "图片未能自动解析；如确有需要，可调用 image。"
        elif not summary:
            summary = "图片未自动分析；如确有需要，可调用 image。"
        resource.extra["summary"] = summary
        return f"[{label} {number}]\n引用: {record.media_uri}\n摘要: {summary}\n[/{label} {number}]"

    async def _extract_file_context(self, chat_id, resource, label):
        inspection = await self.read_file(
            chat_id=chat_id,
            media_uri=resource.media_uri,
            max_chars=self.file_context_max_chars,
        )
        if inspection.error:
            return (
                f"[{label}]\n引用: {resource.media_uri}\n文件: {resource.filename or '未命名'}\n"
                f"提取: {inspection.message}\n[/{label}]"
            )
        suffix = "（已截断，可调用 read_file 查看更多）" if inspection.truncated else ""
        return (
            f"[{label}]\n引用: {inspection.media_uri}\n文件: {resource.filename or '未命名'}\n"
            f"内容{suffix}:\n{inspection.content}\n[/{label}]"
        )

    async def _summarize_file_context(self, chat_id, resource, label):
        record = await self.store.authorize(chat_id, resource.media_uri)
        if not record or not self._is_supported_text_file(record):
            return self._file_summary_fallback(resource, "文件不可用或格式不支持")
        if record.file_summary and record.file_summary_version == _FILE_SUMMARY_VERSION:
            return self._file_summary_block(resource, record.file_summary)
        if not self.file_summary_capability.enabled:
            return self._file_summary_fallback(
                resource, "文件未自动摘要，可调用 read_file"
            )
        try:
            extracted = await self.file_capability.execute(
                record,
                cache_key=(
                    record.sha256,
                    "text",
                    _TEXT_EXTRACTION_VERSION,
                    str(self.file_extract_max_chars),
                ),
                max_chars=self.file_extract_max_chars,
            )
            result = await self.file_summary_capability.execute(
                record,
                cache_key=(record.sha256, "summary", _FILE_SUMMARY_VERSION),
                text=extracted.content,
                max_tokens=self.file_summary_max_chars,
            )
            summary = result.content[: self.file_summary_max_chars]
            await self.store.update_file_summary(
                record.media_id, summary, result.model, _FILE_SUMMARY_VERSION
            )
            return self._file_summary_block(resource, summary)
        except Exception:
            return self._file_summary_fallback(
                resource, "文件未能自动摘要，可调用 read_file"
            )

    @staticmethod
    def _file_summary_block(resource, summary):
        return (
            f"[文件摘要]\n引用: {resource.media_uri}\n文件: {resource.filename or '未命名'}\n"
            f"摘要: {summary}\n[/文件摘要]"
        )

    @staticmethod
    def _file_summary_fallback(resource, message):
        return (
            f"[文件摘要]\n引用: {resource.media_uri}\n文件: {resource.filename or '未命名'}\n"
            f"摘要: {message}\n[/文件摘要]"
        )

    async def read_file(
        self, *, chat_id: str, media_uri: str, max_chars: int | None = None
    ) -> FileInspection:
        if not media_uri.startswith("media://inbound/"):
            return FileInspection(
                media_uri, error="INVALID_MEDIA_URI", message="仅支持受控 media:// 引用"
            )
        if not self.file_tools_enabled:
            return FileInspection(
                media_uri, error="ANALYSIS_FAILED", message="文件提取服务未启用"
            )
        record, auth_error = await self.store.authorize_with_reason(chat_id, media_uri)
        if not record:
            return FileInspection(
                media_uri,
                error=auth_error,
                message=(
                    "文件不属于当前会话"
                    if auth_error == "MEDIA_FORBIDDEN"
                    else "文件已过期或保存失败"
                ),
            )
        if not self._is_supported_text_file(record):
            return FileInspection(
                media_uri,
                error="UNSUPPORTED_MEDIA_TYPE",
                message="仅支持 TXT、Markdown、JSON 和 CSV 文件",
            )
        requested_limit = int(max_chars or self.file_extract_max_chars)
        limit = min(requested_limit, self.file_extract_max_chars)
        try:
            result = await self.file_capability.execute(
                record,
                cache_key=(
                    record.sha256,
                    "text",
                    _TEXT_EXTRACTION_VERSION,
                    str(self.file_extract_max_chars),
                ),
                max_chars=self.file_extract_max_chars + 1,
            )
        except Exception:
            return FileInspection(
                media_uri, error="MEDIA_NOT_AVAILABLE", message="文件不可用"
            )
        text = result.content
        return FileInspection(
            media_uri, content=text[:limit], truncated=len(text) > limit
        )

    async def inspect_file(
        self, *, chat_id: str, media_uri: str, max_chars: int | None = None
    ) -> FileInspection:
        return await self.read_file(
            chat_id=chat_id, media_uri=media_uri, max_chars=max_chars
        )

    async def inspect_pdf(
        self, *, chat_id: str, media_uri: str, prompt: str
    ) -> PdfInspection:
        if not media_uri.startswith("media://inbound/"):
            return PdfInspection(
                media_uri, error="INVALID_MEDIA_URI", message="仅支持受控 media:// 引用"
            )
        if not self.pdf_tools_enabled:
            return PdfInspection(
                media_uri, error="ANALYSIS_FAILED", message="PDF 分析服务未启用"
            )
        record, auth_error = await self.store.authorize_with_reason(chat_id, media_uri)
        if not record:
            return PdfInspection(
                media_uri,
                error=auth_error,
                message=(
                    "PDF 不属于当前会话"
                    if auth_error == "MEDIA_FORBIDDEN"
                    else "PDF 已过期或保存失败"
                ),
            )
        if not self._is_pdf(record):
            return PdfInspection(
                media_uri, error="UNSUPPORTED_MEDIA_TYPE", message="该媒体不是 PDF"
            )
        if record.size > self.pdf_capability.max_bytes:
            return PdfInspection(
                media_uri, error="PDF_TOO_LARGE", message="PDF 超过大小限制"
            )
        try:
            pages = await asyncio.to_thread(_PdfProvider.count_pages, record.local_path)
        except Exception:
            return PdfInspection(
                media_uri, error="ANALYSIS_FAILED", message="PDF 文件不可解析"
            )
        if pages > self.pdf_max_pages:
            return PdfInspection(
                media_uri,
                error="PDF_TOO_MANY_PAGES",
                message=f"PDF 页数超过限制（最多 {self.pdf_max_pages} 页）",
            )
        normalized_prompt = " ".join((prompt or "").split())[:2_000]
        try:
            result = await self.pdf_capability.execute(
                record,
                cache_key=(record.sha256, "pdf", normalized_prompt),
                prompt=normalized_prompt,
                max_tokens=self.pdf_max_tokens,
            )
            pages, _, analysis = result.content.partition("\n")
            return PdfInspection(
                media_uri,
                analysis=analysis or pages,
                pages=int(pages.removeprefix("页数: ").strip() or 0),
                cached=result.cached,
            )
        except MediaCapabilityTimeoutError:
            return PdfInspection(
                media_uri, error="ANALYSIS_TIMEOUT", message="PDF 分析超时"
            )
        except Exception as exc:
            _log.warning("PDF 分析失败: category=%s", safe_error_category(exc))
            return PdfInspection(
                media_uri, error="ANALYSIS_FAILED", message="PDF 分析失败"
            )

    @staticmethod
    def _is_pdf(record: MediaRecord) -> bool:
        return record.mime_type == "application/pdf"

    @staticmethod
    def _is_supported_text_file(record: MediaRecord) -> bool:
        if record.mime_type in {
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
        }:
            return True
        return record.filename.lower().endswith(
            (".txt", ".md", ".markdown", ".json", ".csv")
        )

    async def _transcribe_voice_context(self, chat_id, resource, label):
        result = await self.transcribe_voice(
            chat_id=chat_id, media_uri=resource.media_uri
        )
        if result.error:
            return (
                f"[{label}]\n引用: {resource.media_uri}\n"
                f"转写: {result.message}\n[/{label}]"
            )
        return (
            f"[{label}]\n引用: {result.media_uri}\n转写: {result.transcript}\n"
            f"[/{label}]"
        )

    async def transcribe_voice(
        self, *, chat_id: str, media_uri: str
    ) -> VoiceTranscription:
        if not media_uri.startswith("media://inbound/"):
            return VoiceTranscription(
                media_uri, error="INVALID_MEDIA_URI", message="仅支持受控 media:// 引用"
            )
        if not self.voice_enabled:
            return VoiceTranscription(
                media_uri, error="TRANSCRIPTION_FAILED", message="语音转写服务未启用"
            )
        record, auth_error = await self.store.authorize_with_reason(chat_id, media_uri)
        if not record:
            return VoiceTranscription(
                media_uri,
                error=auth_error,
                message=(
                    "语音不属于当前会话"
                    if auth_error == "MEDIA_FORBIDDEN"
                    else "语音已过期或保存失败"
                ),
            )
        if record.resource_type != "voice" and not record.mime_type.startswith(
            "audio/"
        ):
            return VoiceTranscription(
                media_uri, error="UNSUPPORTED_MEDIA_TYPE", message="该媒体不是语音"
            )
        cached = await self.store.get_transcript(record.media_id)
        if cached:
            return VoiceTranscription(media_uri, transcript=cached[0], cached=True)
        try:
            result = await self.voice_capability.execute(
                record, cache_key=(record.sha256, "transcript")
            )
            transcript = result.content[: self.voice_max_chars]
            await self.store.update_transcript(
                record.media_id, transcript, result.model
            )
            return VoiceTranscription(
                media_uri, transcript=transcript, cached=result.cached
            )
        except MediaCapabilityTimeoutError:
            return VoiceTranscription(
                media_uri, error="TRANSCRIPTION_TIMEOUT", message="语音转写超时"
            )
        except Exception as exc:
            _log.warning("语音转写失败: category=%s", safe_error_category(exc))
            return VoiceTranscription(
                media_uri, error="TRANSCRIPTION_FAILED", message="语音转写失败"
            )

    async def inspect_image(
        self, *, chat_id: str, media_uri: str, question: str
    ) -> ImageInspection:
        if not media_uri.startswith("media://inbound/"):
            return ImageInspection(
                media_uri, error="INVALID_MEDIA_URI", message="仅支持受控 media:// 引用"
            )
        if not question or not question.strip():
            return ImageInspection(
                media_uri, error="ANALYSIS_FAILED", message="请提供具体问题"
            )
        if not self.image_tools_enabled:
            return ImageInspection(
                media_uri, error="ANALYSIS_FAILED", message="图片理解服务未启用"
            )
        record, auth_error = await self.store.authorize_with_reason(chat_id, media_uri)
        if not record:
            return ImageInspection(
                media_uri,
                error=auth_error,
                message=(
                    "图片不属于当前会话"
                    if auth_error == "MEDIA_FORBIDDEN"
                    else "图片已过期或保存失败"
                ),
            )
        if not record.mime_type.startswith("image/"):
            return ImageInspection(
                media_uri, error="UNSUPPORTED_MEDIA_TYPE", message="该媒体不是图片"
            )
        if record.size > self.max_image_bytes:
            return ImageInspection(
                media_uri, error="IMAGE_TOO_LARGE", message="图片超过大小限制"
            )
        normalized_question = " ".join(question.split())[:2000]
        try:
            result = await self.image_inspect_capability.execute(
                record,
                cache_key=(
                    record.sha256,
                    "inspect",
                    normalized_question,
                    _IMAGE_INSPECTION_PROMPT_VERSION,
                ),
                prompt=normalized_question,
            )
            note = ""
            if result.provider == OcrProvider.name:
                note = "视觉模型暂不可用，以下为图片文字识别（OCR）结果，不代表对画面的理解"
            return ImageInspection(
                media_uri, analysis=result.content, cached=result.cached, note=note
            )
        except MediaCapabilityTimeoutError:
            return ImageInspection(
                media_uri, error="ANALYSIS_TIMEOUT", message="图片分析超时"
            )
        except Exception as exc:
            _log.warning("图片分析失败: category=%s", safe_error_category(exc))
            return ImageInspection(
                media_uri, error="ANALYSIS_FAILED", message="图片分析失败"
            )
