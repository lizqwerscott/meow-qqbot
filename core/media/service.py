import asyncio
import ipaddress
import logging
import socket
import time
from collections import OrderedDict
from urllib.parse import urlparse

import httpx

from core.media.models import ImageInspection, MediaRecord, MediaTurnContext
from core.media.store import MediaStore
from core.message import InputMessage, ResourceMeta

_log = logging.getLogger(__name__)


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
    ):
        self.enabled = enabled
        self.http_client = http_client
        self.multimodal = multimodal
        cfg = image_understanding or {}
        self.image_enabled = bool(cfg.get("enabled", True))
        self.max_auto_images = max(0, int(cfg.get("max_auto_images", 3)))
        self.analysis_timeout = max(1, float(cfg.get("analysis_timeout_seconds", 30)))
        self.summary_max_chars = max(20, int(cfg.get("summary_max_chars", 300)))
        self.recent_window_seconds = max(0, int(recent_window_seconds))
        self.recent_max_items = max(0, int(recent_max_items))
        self.max_attachments = max(1, int(max_attachments_per_message))
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.download_timeout = max(1, float(download_timeout))
        self.download_sem = asyncio.Semaphore(max(1, int(download_concurrency)))
        self.vlm_sem = asyncio.Semaphore(max(1, int(cfg.get("concurrency", 2))))
        self.max_total_bytes = max_total_bytes
        self.preview_enabled = bool(preview_enabled)
        self.preview_max_inline_bytes = max(1, int(preview_max_inline_bytes))
        self.text_preview_max_chars = max(1, int(text_preview_max_chars))
        self.store = MediaStore(storage_dir)
        self._opened = False
        self._inspection_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._inspection_cache_max = 200
        self._summary_tasks: dict[str, asyncio.Task] = {}

    async def open(self):
        if self.enabled and not self._opened:
            await self.store.open()
            self._opened = True

    async def close(self):
        await self.store.close()
        self._opened = False

    @property
    def tools_enabled(self) -> bool:
        return self.enabled and self.image_enabled and self.multimodal is not None

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
                data, mime = await self._download(source_url, resource)
                record = await self.store.save(
                    chat_id=message.chat_id,
                    message_id=message.id,
                    sender_id=message.sender_id,
                    resource_type=resource.resource_type,
                    source_url=source_url,
                    mime_type=mime or resource.mime_type,
                    filename=resource.filename,
                    data=data,
                    position=index,
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
            except Exception as exc:
                resource.storage_status = "failed"
                resource.source_url = ""
                resource.resource_id = ""
                _log.warning("媒体保存失败 [%s]: %s", message.id, exc)
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
                data, mime = await self._download(resource.source_url, resource)
                record = await self.store.save(
                    chat_id=message.chat_id,
                    message_id=message.replied_message_id or f"reply:{message.id}",
                    sender_id=message.replied_author_id or message.sender_id,
                    resource_type=resource.resource_type,
                    source_url=resource.source_url,
                    mime_type=mime or resource.mime_type,
                    filename=resource.filename,
                    data=data,
                    position=index,
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
            except Exception as exc:
                resource.storage_status = "failed"
                resource.source_url = ""
                resource.resource_id = ""
                _log.warning("引用媒体保存失败 [%s]: %s", message.id, exc)

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
        limit = (
            self.max_image_bytes
            if resource.resource_type == "image"
            else self.max_file_bytes
        )
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
                    return data, mime
            else:
                raise ValueError("重定向次数超过限制")

    @staticmethod
    def _image_signature(data: bytes) -> bool:
        return data.startswith(
            (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"RIFF")
        )

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
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            infos = await asyncio.get_running_loop().run_in_executor(
                None, socket.getaddrinfo, host, None
            )
            return [ipaddress.ip_address(info[4][0]) for info in infos]

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
                state = f"摘要: {item.summary}" if item.summary else "未分析"
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
            return f"[{label} {number}]\n引用: {resource.media_uri}\n摘要: 图片不可用；如确有需要，可调用 inspect_image。\n[/{label} {number}]"
        summary = record.summary
        if self.image_enabled and self.multimodal and not summary:
            try:
                task = self._summary_tasks.get(record.media_id)
                if task is None:

                    async def _run_summary():
                        async with self.vlm_sem:
                            return await asyncio.wait_for(
                                self.multimodal.analyze_image(str(record.local_path)),
                                timeout=self.analysis_timeout,
                            )

                    task = asyncio.create_task(_run_summary())
                    self._summary_tasks[record.media_id] = task
                try:
                    summary = await task
                finally:
                    if task.done():
                        self._summary_tasks.pop(record.media_id, None)
                summary = (summary or "图片未能自动解析")[: self.summary_max_chars]
                await self.store.update_summary(
                    record.media_id, summary, getattr(self.multimodal, "model", "")
                )
                resource.extra["summary"] = summary
            except Exception:
                summary = "图片未能自动解析；如确有需要，可调用 inspect_image。"
        elif not summary:
            summary = "图片未自动分析；如确有需要，可调用 inspect_image。"
        resource.extra["summary"] = summary
        return f"[{label} {number}]\n引用: {record.media_uri}\n摘要: {summary}\n[/{label} {number}]"

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
        if not self.tools_enabled:
            return ImageInspection(
                media_uri, error="ANALYSIS_FAILED", message="图片理解服务未启用"
            )
        record = await self.store.authorize(chat_id, media_uri)
        if not record:
            return ImageInspection(
                media_uri,
                error="MEDIA_NOT_AVAILABLE",
                message="图片已过期、保存失败或不属于当前会话",
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
        cache_key = (record.sha256, normalized_question)
        if cache_key in self._inspection_cache:
            self._inspection_cache.move_to_end(cache_key)
            return ImageInspection(
                media_uri, analysis=self._inspection_cache[cache_key], cached=True
            )
        try:
            async with self.vlm_sem:
                result = await asyncio.wait_for(
                    self.multimodal.analyze_image(
                        str(record.local_path), prompt=normalized_question
                    ),
                    timeout=self.analysis_timeout,
                )
            analysis = (result or "").strip()
            self._inspection_cache[cache_key] = analysis
            self._inspection_cache.move_to_end(cache_key)
            while len(self._inspection_cache) > self._inspection_cache_max:
                self._inspection_cache.popitem(last=False)
            return ImageInspection(media_uri, analysis=analysis)
        except asyncio.TimeoutError:
            return ImageInspection(
                media_uri, error="ANALYSIS_TIMEOUT", message="图片分析超时"
            )
        except Exception as exc:
            _log.warning("图片分析失败: %s", exc)
            return ImageInspection(
                media_uri, error="ANALYSIS_FAILED", message="图片分析失败"
            )
