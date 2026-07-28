"""
表情管理器 — 自定义表情（faceType=6 + attachments）的下载、缓存、VLM 分析与用户自定义。

流程图：
  is_custom_emoji() 检测 → EmojiManager.get_or_build(attachment)
    → 1. 下载图片
    → 2. 计算 SHA-256 hash
    → 3. 检查 emojis.json 缓存
         ├─ 命中 → 返回 (user 自定义 or auto 的 desc + tags)
         └─ 未命中 → 继续
    → 4. GIF → ImageUtils 转为静态图
    → 5. 保存到 data/emojis/<hash>.<ext>
    → 6. VLM 分析（MultimodalService.analyze_emoji）
    → 7. 写入 JSON 缓存
    → 8. 返回 (display_text, full_description, emotion_tags, emoji_hash)
         display_text  = user_description > auto_summary > auto_description[:20] > "自定义表情"
         full_description = user_description > auto_description > ""
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qqbot_agent_sdk.dto import MessageAttachment

from core.image_utils import ImageUtils

_log = logging.getLogger(__name__)

QQ_FACE_CATEGORIES = {
    "0": "普通表情",
    "1": "动图",
    "2": "小表情",
    "3": "大表情",
    "4": "魔法表情",
    "5": "戳一戳",
    "6": "自定义表情",
    "7": "红包",
    "8": "龙猫",
}


def is_custom_emoji(content: str, attachments: list) -> bool:
    if not attachments:
        return False

    cleaned = re.sub(r"<[^>]+>", "", content).strip()
    if cleaned:
        return False

    face_type_tags = re.findall(r'<faceType=(\d+)[^>]*>', content)
    if not face_type_tags:
        return False

    if "6" not in face_type_tags:
        return False

    return True


class EmojiStorage:
    def __init__(self, json_path: str = "data/emojis/emojis.json"):
        self._path = Path(json_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._data: Dict[str, Any] = self._load()

    def get(self, emoji_hash: str) -> Optional[dict]:
        return self._data.get("emojis", {}).get(emoji_hash)

    async def save(self, record: dict) -> None:
        async with self._lock:
            self._data.setdefault("emojis", {})[record["hash"]] = record
            await self._flush_async()

    async def update(self, emoji_hash: str, **kwargs) -> bool:
        async with self._lock:
            emojis = self._data.setdefault("emojis", {})
            if emoji_hash not in emojis:
                return False
            emojis[emoji_hash].update(kwargs)
            emojis[emoji_hash]["updated_at"] = datetime.now().isoformat()
            await self._flush_async()
            return True

    def list_all(self) -> List[dict]:
        return list(self._data.get("emojis", {}).values())

    async def delete(self, emoji_hash: str) -> bool:
        async with self._lock:
            emojis = self._data.setdefault("emojis", {})
            if emoji_hash not in emojis:
                return False
            del emojis[emoji_hash]
            await self._flush_async()
            return True

    def count(self) -> int:
        return len(self._data.get("emojis", {}))

    def _load(self) -> dict:
        if not self._path.exists():
            _log.info(f"创建新的表情数据文件: {self._path}")
            return {"version": 1, "emojis": {}}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "emojis" not in data:
                _log.warning("表情数据文件格式不正确，重置")
                return {"version": 1, "emojis": {}}
            return data
        except (json.JSONDecodeError, OSError) as e:
            _log.error(f"读取表情数据文件失败: {e}，备份并重置")
            backup = self._path.with_suffix(".json.bak")
            try:
                if self._path.exists():
                    import shutil
                    shutil.copy2(self._path, backup)
                    _log.info(f"已备份损坏文件到: {backup}")
            except Exception as e:
                _log.warning(f"备份损坏文件失败 [{self._path}]: {e}")
            return {"version": 1, "emojis": {}}

    def _flush_sync(self) -> None:
        try:
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
        except OSError as e:
            _log.error(f"写入表情数据文件失败: {e}")

    async def _flush_async(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._flush_sync)


class EmojiManager:
    def __init__(
        self,
        http_client,
        multimodal_service=None,
        emoji_dir: str = "data/emojis/",
        json_path: str = "data/emojis/emojis.json",
    ):
        self._http_client = http_client
        self._multimodal = multimodal_service
        self._emoji_dir = Path(emoji_dir)
        self._emoji_dir.mkdir(parents=True, exist_ok=True)
        self._storage = EmojiStorage(json_path)
        _log.info(
            f"表情管理器已启动，缓存目录: {self._emoji_dir.resolve()}, "
            f"已有 {self._storage.count()} 个表情记录"
        )

    async def get_or_build(
        self, attachment: MessageAttachment
    ) -> Tuple[str, str, List[str], str]:
        image_bytes = await self._download_emoji(attachment.resolved_url)

        emoji_hash = hashlib.sha256(image_bytes).hexdigest()

        cached = self._storage.get(emoji_hash)
        if cached:
            auto_summary = cached.get("auto_summary", "")
            auto_desc = cached.get("auto_description", "")
            user_desc = cached.get("user_description")
            tags = cached.get("user_tags") or cached.get("auto_tags", [])
            full_desc = user_desc or auto_desc or ""
            display = user_desc or auto_summary or (auto_desc[:20] if auto_desc else "自定义表情")
            await self._storage.update(
                emoji_hash, used_count=cached.get("used_count", 0) + 1
            )
            return (display, full_desc, tags, emoji_hash)

        ext = self._infer_ext(attachment.content_type, attachment.filename)
        is_gif = ext == ".gif"

        file_path = self._emoji_dir / f"{emoji_hash}{ext}"
        await asyncio.to_thread(file_path.write_bytes, image_bytes)

        auto_summary, auto_desc, auto_tags = "", "", ""
        temp_paths: list = []
        if self._multimodal:
            try:
                if is_gif:
                    static_bytes = ImageUtils.gif_2_static_image(image_bytes)
                    analysis_path = file_path.with_suffix(".analysis.jpg")
                    await asyncio.to_thread(analysis_path.write_bytes, static_bytes)
                    processed_path = self._maybe_normalize_image(analysis_path)
                    temp_paths = [analysis_path, processed_path]
                else:
                    processed_path = self._maybe_normalize_image(file_path)
                    temp_paths = [processed_path]

                auto_summary, auto_desc, auto_tags = await self._multimodal.analyze_emoji(
                    str(processed_path), is_gif=is_gif
                )

                for p in temp_paths:
                    if p != file_path and p.exists():
                        p.unlink()

                _log.info(
                    f"VLM 表情分析完成 [{emoji_hash[:12]}..]: "
                    f"summary={auto_summary}, desc={auto_desc}, tags={auto_tags}"
                )
            except Exception:
                _log.warning(
                    f"VLM 分析表情失败 [{emoji_hash[:12]}..]",
                    exc_info=True,
                )

        record = {
            "hash": emoji_hash,
            "file_name": f"{emoji_hash}{ext}",
            "url": attachment.resolved_url,
            "auto_summary": auto_summary,
            "auto_description": auto_desc,
            "auto_tags": auto_tags,
            "user_description": None,
            "user_tags": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "used_count": 1,
        }
        await self._storage.save(record)

        display = auto_summary or (auto_desc[:20] if auto_desc else "自定义表情")
        return (display, auto_desc, auto_tags or [], emoji_hash)

    async def set_custom(
        self,
        emoji_hash: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        record = self._storage.get(emoji_hash)
        if record is None:
            _log.warning(f"未找到 emoji: {emoji_hash[:12]}..")
            return False

        updates = {}
        if description is not None:
            updates["user_description"] = description
        if tags is not None:
            clean_tags = [t for t in tags if t.strip()]
            updates["user_tags"] = clean_tags

        await self._storage.update(emoji_hash, **updates)
        _log.info(f"已更新 emoji [{emoji_hash[:12]}..]: desc={description}, tags={tags}")
        return True

    async def reset_to_auto(self, emoji_hash: str) -> bool:
        record = self._storage.get(emoji_hash)
        if record is None:
            return False

        await self._storage.update(
            emoji_hash,
            user_description=None,
            user_tags=None,
        )
        _log.info(f"已重置 emoji [{emoji_hash[:12]}..] 为自动识别结果")
        return True

    async def delete_emoji(self, emoji_hash: str) -> bool:
        record = self._storage.get(emoji_hash)
        if not record:
            return False
        file_name = record.get("file_name")
        if file_name:
            file_path = self._emoji_dir / file_name
            if file_path.exists():
                file_path.unlink()
                _log.info(f"已删除表情文件: {file_name}")
        result = await self._storage.delete(emoji_hash)
        if result:
            _log.info(f"已删除表情记录: {emoji_hash[:12]}..")
        return result

    async def reanalyze_emoji(self, emoji_hash: str) -> bool:
        record = self._storage.get(emoji_hash)
        if not record:
            _log.warning(f"重新分析失败，未找到 emoji: {emoji_hash[:12]}..")
            return False
        if not self._multimodal:
            _log.warning("重新分析失败，多模态服务未配置")
            return False

        file_name = record.get("file_name")
        if not file_name:
            return False
        file_path = self._emoji_dir / file_name
        if not file_path.exists():
            _log.warning(f"重新分析失败，图片文件不存在: {file_path}")
            return False

        is_gif = file_name.lower().endswith(".gif")

        try:
            if is_gif:
                image_bytes = await asyncio.to_thread(file_path.read_bytes)
                static_bytes = ImageUtils.gif_2_static_image(image_bytes)
                analysis_path = file_path.with_suffix(".analysis.jpg")
                await asyncio.to_thread(analysis_path.write_bytes, static_bytes)
                processed_path = self._maybe_normalize_image(analysis_path)
                self._multimodal.invalidate_cache(str(processed_path), "emoji")
                auto_summary, auto_desc, auto_tags = await self._multimodal.analyze_emoji(
                    str(processed_path), is_gif=True
                )
                for p in [analysis_path, processed_path]:
                    if p != file_path and p.exists():
                        p.unlink()
            else:
                processed_path = self._maybe_normalize_image(file_path)
                self._multimodal.invalidate_cache(str(processed_path), "emoji")
                auto_summary, auto_desc, auto_tags = await self._multimodal.analyze_emoji(
                    str(processed_path), is_gif=False
                )

            if not auto_summary and not auto_desc:
                _log.warning(f"重新分析返回空结果 [{emoji_hash[:12]}..]，保留原有数据")
                return False

            await self._storage.update(
                emoji_hash,
                auto_summary=auto_summary,
                auto_description=auto_desc,
                auto_tags=auto_tags,
            )
            _log.info(f"表情重新分析完成 [{emoji_hash[:12]}..]: summary={auto_summary}, desc={auto_desc}, tags={auto_tags}")
            return True
        except Exception:
            _log.warning(
                f"表情重新分析失败 [{emoji_hash[:12]}..]",
                exc_info=True,
            )
            return False

    def count_emojis(self) -> int:
        return self._storage.count()

    async def update_emoji(self, emoji_hash: str, **kwargs) -> bool:
        record = self.get_info(emoji_hash)
        if not record:
            return False
        await self._storage.update(emoji_hash, **kwargs)
        return True

    def get_info(self, emoji_hash: str) -> Optional[dict]:
        return self._storage.get(emoji_hash)

    def list_emojis(
        self, page: int = 1, page_size: int = 20,
        sort_by: str = "has_tags", sort_order: str = "desc",
    ) -> Dict[str, Any]:
        all_emojis = self._storage.list_all()

        sort_fns = {
            "used_count": lambda r: r.get("used_count", 0),
            "has_tags": lambda r: (
                1 if (r.get("user_tags") or r.get("auto_tags")) else 0
            ),
            "has_description": lambda r: (
                1 if (r.get("user_description") or r.get("auto_summary", "") or r.get("auto_description", "")) else 0
            ),
            "has_custom": lambda r: (
                1 if (r.get("user_description") or r.get("user_tags")) else 0
            ),
            "created_at": lambda r: r.get("created_at", ""),
            "hash": lambda r: r.get("hash", ""),
        }
        key_fn = sort_fns.get(sort_by, sort_fns["has_tags"])
        all_emojis.sort(key=key_fn, reverse=(sort_order != "asc"))

        total = len(all_emojis)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = all_emojis[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "emojis": page_items,
        }

    def find_by_hash(self, partial_hash: str) -> Optional[dict]:
        if not partial_hash:
            return None
        records = self._storage.list_all()
        for e in records:
            if e["hash"] == partial_hash:
                return e
        matches = [e for e in records if e["hash"].startswith(partial_hash)]
        if len(matches) == 1:
            return matches[0]
        return None

    def find_emoji(self, query: str) -> Optional[dict]:
        records = self._storage.list_all()
        if not records:
            return None

        keywords = [k.lower().strip() for k in query.split() if k.strip()]
        if not keywords:
            return None

        def _get_desc(r):
            return (r.get("user_description") or r.get("auto_description", "") or "")

        def _get_tags(r):
            return r.get("user_tags") or r.get("auto_tags", []) or []

        best_record = None
        best_score = 0

        for r in records:
            desc = _get_desc(r).lower()
            summary = (r.get("auto_summary", "") or "").lower()
            search_text = f"{desc} {summary}".strip()
            tags = [t.lower().strip('<>') for t in _get_tags(r)]
            score = 0

            for k in keywords:
                if k in tags:
                    score += 10
                elif any(k in t for t in tags):
                    score += 3
                if k in search_text:
                    score += 5

            if score > best_score:
                best_score = score
                best_record = r

        return best_record

    def find_emojis(self, query: str, max_results: int = 5) -> List[dict]:
        records = self._storage.list_all()
        if not records:
            return []

        keywords = [k.lower().strip() for k in query.split() if k.strip()]
        if not keywords:
            return []

        def _score(r):
            desc = (r.get("user_description") or r.get("auto_description", "") or "").lower()
            summary = (r.get("auto_summary", "") or "").lower()
            search_text = f"{desc} {summary}".strip()
            tags = [t.lower().strip('<>') for t in (r.get("user_tags") or r.get("auto_tags", []) or [])]
            score = 0
            matched_keywords = 0

            for k in keywords:
                kw_score = 0
                if k in tags:
                    kw_score += 10
                if k in search_text:
                    kw_score += 5
                for t in tags:
                    if k in t:
                        kw_score += 3
                        break
                if kw_score > 0:
                    matched_keywords += 1
                score += kw_score

            score += matched_keywords * 2
            if score > 0:
                score += min(r.get("used_count", 0) / 10, 3)

            return score

        scored = [(r, _score(r)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [r for r, s in scored if s > 0][:max_results]
        return results

    def get_all_tags(self) -> List[str]:
        tags_set = set()
        for r in self._storage.list_all():
            for t in (r.get("user_tags") or r.get("auto_tags", []) or []):
                clean = t.strip('<>').strip()
                if clean:
                    tags_set.add(clean)
        return sorted(tags_set)

    def get_emoji_catalog_text(self, max_emojis: int = 30) -> str:
        records = self._storage.list_all()
        if not records:
            return ""

        records.sort(key=lambda r: r.get("used_count", 0), reverse=True)
        records = records[:max_emojis]

        lines = ["## 可用表情"]
        lines.append("你可以使用 search_emoji 工具搜索以下表情，然后用 send_emoji 发送。")
        lines.append("")

        for r in records:
            short_hash = r["hash"][:12]
            summary = r.get("auto_summary", "") or ""
            desc = r.get("user_description") or r.get("auto_description", "") or "(无描述)"
            display = f"{summary} - {desc}" if summary else desc
            tags = r.get("user_tags") or r.get("auto_tags", []) or []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"- {display}{tag_str} (hash: {short_hash})")

        return "\n".join(lines)

    async def _download_emoji(self, url: str) -> bytes:
        last_exc = None
        for attempt in range(3):
            try:
                resp = await self._http_client.get(url, timeout=15)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                last_exc = e
                _log.warning(f"下载 emoji 失败 (尝试 {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
        raise RuntimeError(f"下载 emoji 失败 (已重试 3 次): {url}") from last_exc

    @staticmethod
    def _infer_ext(content_type: str, filename: str) -> str:
        ct_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        if content_type in ct_map:
            return ct_map[content_type]

        if filename:
            ext = Path(filename).suffix.lower()
            if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                return ext if ext != ".jpeg" else ".jpg"

        return ".png"

    @staticmethod
    def _maybe_normalize_image(image_path: Path) -> Path:
        try:
            b64 = ImageUtils.image_path_to_base64(str(image_path))
            if b64 is None:
                return image_path

            fmt = "png"
            ext = image_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                fmt = "jpeg"

            normalized_b64, new_fmt, was_modified = (
                ImageUtils.normalize_image_base64_for_model(b64, fmt)
            )
            if not was_modified:
                return image_path

            new_path = image_path.with_stem(image_path.stem + "_norm")
            ImageUtils.base64_to_image(
                normalized_b64, str(new_path)
            )
            _log.debug(f"图片已归一化: {new_path}")
            return new_path
        except Exception as e:
            _log.warning(f"图片归一化失败，使用原图: {e}")
            return image_path
