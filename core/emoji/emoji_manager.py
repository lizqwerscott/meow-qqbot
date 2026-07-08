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
    → 8. 返回 (description, emotion_tags)
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qqbot_agent_sdk.dto import MessageAttachment

from core.utils.utils_image import ImageUtils

_log = logging.getLogger(__name__)

# ============================================================
# 辅助常量 & 函数
# ============================================================

# QQ 内置 faceType 大类映射（内置表情码用不到的，但保留用作 general 参考）
QQ_FACE_CATEGORIES = {
    "0": "普通表情",
    "1": "动图",
    "2": "小表情",
    "3": "大表情",
    "4": "魔法表情",
    "5": "戳一戳",
    "6": "自定义表情",  # 这是我们的关注点
    "7": "红包",
    "8": "龙猫",
}


def is_custom_emoji(content: str, attachments: list) -> bool:
    """
    判断消息是否为 QQ 自定义表情。

    QQ 自定义表情的特征：
    - content 中只包含 <faceType=6,...> 标签，无其他实质文本
    - attachments 中有图片文件

    Args:
        content: 消息文本内容
        attachments: 消息附件列表 (List[MessageAttachment])
    Returns:
        True 如果是自定义表情消息
    """
    if not attachments:
        return False

    # 去掉所有 XML/HTML 标签，看剩下的是否为空
    cleaned = re.sub(r"<[^>]+>", "", content).strip()
    if cleaned:
        # 还有实质文本 → 不是纯表情消息
        return False

    # 检查是否为 faceType=6（自定义表情）
    # content 可能有多种格式，但 faceType=6 是自定义表情的标识
    face_type_tags = re.findall(r'<faceType=(\d+)[^>]*>', content)
    if not face_type_tags:
        return False  # 没有 faceType 标签

    # 只要有一个 faceType=6 就算（用户可能发多个表情）
    if "6" not in face_type_tags:
        return False

    return True


# ============================================================
# EmojiStorage — JSON 文件持久化
# ============================================================


class EmojiStorage:
    """
    Emoji 元数据的 JSON 文件存储。
    线程安全（asyncio.Lock），单协程写。
    """

    def __init__(self, json_path: str = "data/emojis/emojis.json"):
        self._path = Path(json_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._data: Dict[str, Any] = self._load()

    # ── 公开查询接口 ──

    def get(self, emoji_hash: str) -> Optional[dict]:
        """根据 hash 获取 emoji 记录"""
        return self._data.get("emojis", {}).get(emoji_hash)

    def save(self, record: dict) -> None:
        """保存/更新一条 emoji 记录（立即刷盘）"""
        self._data.setdefault("emojis", {})[record["hash"]] = record
        self._flush()

    def update(self, emoji_hash: str, **kwargs) -> bool:
        """部分更新 emoji 记录字段。返回 True 表示成功。"""
        emojis = self._data.setdefault("emojis", {})
        if emoji_hash not in emojis:
            return False
        emojis[emoji_hash].update(kwargs)
        emojis[emoji_hash]["updated_at"] = datetime.now().isoformat()
        self._flush()
        return True

    def list_all(self) -> List[dict]:
        """返回所有 emoji 记录列表"""
        return list(self._data.get("emojis", {}).values())

    def count(self) -> int:
        """返回 emoji 总数"""
        return len(self._data.get("emojis", {}))

    # ── 内部 IO ──

    def _load(self) -> dict:
        """从 JSON 文件加载数据"""
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
            # 备份损坏文件
            backup = self._path.with_suffix(".json.bak")
            try:
                if self._path.exists():
                    import shutil
                    shutil.copy2(self._path, backup)
                    _log.info(f"已备份损坏文件到: {backup}")
            except Exception:
                pass
            return {"version": 1, "emojis": {}}

    def _flush(self) -> None:
        """将数据写回 JSON 文件"""
        try:
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)  # 原子替换
        except OSError as e:
            _log.error(f"写入表情数据文件失败: {e}")


# ============================================================
# EmojiManager — 表情管理器
# ============================================================


class EmojiManager:
    """
    表情管理器：下载、缓存、VLM 分析、用户自定义。

    使用方式：
        manager = EmojiManager(http_client, multimodal_service)
        desc, tags = await manager.get_or_build(attachment)
    """

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

    # ════════════════════════════════════════════════════════════
    # 核心接口
    # ════════════════════════════════════════════════════════════

    async def get_or_build(
        self, attachment: MessageAttachment
    ) -> Tuple[str, List[str]]:
        """
        核心入口：获取表情的描述和情绪标签。

        流程：下载 → 计算 hash → 检查缓存 → GIF 处理 → 保存文件
              → VLM 分析（未命中时） → 缓存结果 → 返回

        Args:
            attachment: 消息附件（需包含 url、content_type、filename）
        Returns:
            (description, emotion_tags)
            如 ("一个微笑的卡通猫头", ["开心", "可爱"])
        """
        # 1. 下载原始图片
        image_bytes = await self._download_emoji(attachment.resolved_url)

        # 2. 计算 SHA-256 hash（内容寻址）
        emoji_hash = hashlib.sha256(image_bytes).hexdigest()

        # 3. 检查缓存
        cached = self._storage.get(emoji_hash)
        if cached:
            desc = cached.get("user_description") or cached.get("auto_description", "") or "[自定义表情]"
            tags = cached.get("user_tags") or cached.get("auto_tags", [])
            self._storage.update(
                emoji_hash, used_count=cached.get("used_count", 0) + 1
            )
            return (desc, tags)

        # 4. GIF 转换 + 文件保存
        ext = self._infer_ext(attachment.content_type, attachment.filename)
        is_gif = ext == ".gif"
        save_bytes = image_bytes

        if is_gif:
            try:
                save_bytes = ImageUtils.gif_2_static_image(image_bytes)
                ext = ".jpg"  # 保存为静态 JPEG
            except Exception as e:
                _log.warning(f"GIF 转静态图失败，使用原始 GIF: {e}")

        file_path = self._emoji_dir / f"{emoji_hash}{ext}"
        file_path.write_bytes(save_bytes)

        # 5. VLM 分析
        auto_desc, auto_tags = "", []
        if self._multimodal:
            try:
                # 对图片做归一化处理后再分析
                processed_path = self._maybe_normalize_image(file_path)
                auto_desc, auto_tags = await self._multimodal.analyze_emoji(
                    str(processed_path)
                )
                _log.info(
                    f"VLM 表情分析完成 [{emoji_hash[:12]}..]: "
                    f"desc={auto_desc}, tags={auto_tags}"
                )
            except Exception as e:
                _log.warning(f"VLM 分析表情失败 [{emoji_hash[:12]}..]: {e}")
                auto_desc = ""
                auto_tags = []
        else:
            # 未配置 VLM，跳过分析
            auto_desc = ""
            auto_tags = []

        # 6. 写入缓存
        record = {
            "hash": emoji_hash,
            "file_name": f"{emoji_hash}{ext}",
            "url": attachment.resolved_url,
            "auto_description": auto_desc,
            "auto_tags": auto_tags,
            "user_description": None,
            "user_tags": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "used_count": 1,
        }
        self._storage.save(record)

        # 返回时应用 fallback：全空时显示通用描述
        final_desc = auto_desc or "[自定义表情]"
        return (final_desc, auto_tags or [])

    # ════════════════════════════════════════════════════════════
    # 用户自定义接口
    # ════════════════════════════════════════════════════════════

    def set_custom(
        self,
        emoji_hash: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        自定义 emoji 的描述和标签。

        Args:
            emoji_hash: 表情 hash
            description: 自定义描述（None 表示不修改）
            tags: 自定义标签列表（None 表示不修改）
        Returns:
            True 如果更新成功
        """
        record = self._storage.get(emoji_hash)
        if record is None:
            _log.warning(f"未找到 emoji: {emoji_hash[:12]}..")
            return False

        updates = {}
        if description is not None:
            updates["user_description"] = description
        if tags is not None:
            # 去重 + 去除空值
            clean_tags = [t for t in tags if t.strip()]
            updates["user_tags"] = clean_tags

        self._storage.update(emoji_hash, **updates)
        _log.info(f"已更新 emoji [{emoji_hash[:12]}..]: desc={description}, tags={tags}")
        return True

    def reset_to_auto(self, emoji_hash: str) -> bool:
        """
        恢复为 VLM 自动识别结果，清除用户自定义。

        Args:
            emoji_hash: 表情 hash
        Returns:
            True 如果恢复成功
        """
        record = self._storage.get(emoji_hash)
        if record is None:
            return False

        self._storage.update(
            emoji_hash,
            user_description=None,
            user_tags=None,
        )
        _log.info(f"已重置 emoji [{emoji_hash[:12]}..] 为自动识别结果")
        return True

    def count_emojis(self) -> int:
        """返回已知表情总数。"""
        return self._storage.count()

    def update_emoji(self, emoji_hash: str, **kwargs) -> bool:
        """更新一条表情记录的字段，返回是否成功。"""
        record = self.get_info(emoji_hash)
        if not record:
            return False
        self._storage.update(emoji_hash, **kwargs)
        return True

    def get_info(self, emoji_hash: str) -> Optional[dict]:
        """
        获取 emoji 的完整信息。

        Args:
            emoji_hash: 表情 hash
        Returns:
            记录 dict，或 None
        """
        return self._storage.get(emoji_hash)

    def list_emojis(
        self, page: int = 1, page_size: int = 20
    ) -> Dict[str, Any]:
        """
        分页列出所有已知 emoji。

        Args:
            page: 页码（从 1 开始）
            page_size: 每页数量
        Returns:
            {"total": N, "page": P, "page_size": S, "emojis": [...]}
        """
        all_emojis = self._storage.list_all()
        # 按使用次数降序排列
        all_emojis.sort(key=lambda r: r.get("used_count", 0), reverse=True)

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

    # ════════════════════════════════════════════════════════════
    # 工具调用支持
    # ════════════════════════════════════════════════════════════

    def find_by_hash(self, partial_hash: str) -> Optional[dict]:
        """根据 hash（支持短前缀）查找 emoji 记录。"""
        records = self._storage.list_all()
        for e in records:
            if e["hash"] == partial_hash:
                return e
        matches = [e for e in records if e["hash"].startswith(partial_hash)]
        if len(matches) == 1:
            return matches[0]
        return None

    def find_emoji(self, query: str) -> Optional[dict]:
        """
        多关键词 OR 匹配，取最高分返回。

        query 按空格分隔为多个关键词，每个关键词独立匹配标签和描述，
        按匹配数 + 匹配精度累计打分，返回最佳的一个。

        Args:
            query: 空格分隔的关键词，如「开心 撒娇 猫娘」
        Returns:
            匹配度最高的单条 emoji 记录，或 None
        """
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
            tags = [t.lower().strip('<>') for t in _get_tags(r)]
            score = 0

            for k in keywords:
                # 标签完全匹配 +10
                if k in tags:
                    score += 10
                # 标签子串匹配 +3
                elif any(k in t for t in tags):
                    score += 3
                # 描述包含 +5
                if k in desc:
                    score += 5

            if score > best_score:
                best_score = score
                best_record = r

        return best_record

    def find_emojis(self, query: str, max_results: int = 5) -> List[dict]:
        """
        多关键词 OR 匹配，返回多条按匹配度排序。

        query 按空格分隔为多个关键词，每个关键词独立匹配标签和描述，
        命中关键词越多、匹配精度越高，得分越高。

        Args:
            query: 空格分隔的关键词，如「开心 撒娇 猫娘」
            max_results: 最多返回数量
        Returns:
            按匹配度降序的 emoji 记录列表
        """
        records = self._storage.list_all()
        if not records:
            return []

        keywords = [k.lower().strip() for k in query.split() if k.strip()]
        if not keywords:
            return []

        def _score(r):
            desc = (r.get("user_description") or r.get("auto_description", "") or "").lower()
            tags = [t.lower().strip('<>') for t in (r.get("user_tags") or r.get("auto_tags", []) or [])]
            score = 0
            matched_keywords = 0

            for k in keywords:
                kw_score = 0
                if k in tags:
                    kw_score += 10
                if k in desc:
                    kw_score += 5
                for t in tags:
                    if k in t:
                        kw_score += 3
                        break
                if kw_score > 0:
                    matched_keywords += 1
                score += kw_score

            # 命中关键词数量 bonus（鼓励覆盖更多关键词）
            score += matched_keywords * 2
            # 使用频率平局决胜
            if score > 0:
                score += min(r.get("used_count", 0) / 10, 3)

            return score

        scored = [(r, _score(r)) for r in records]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [r for r, s in scored if s > 0][:max_results]
        return results

    def get_all_tags(self) -> List[str]:
        """返回所有去重后的标签列表（去尖括号、去重、去空、排序）"""
        tags_set = set()
        for r in self._storage.list_all():
            for t in (r.get("user_tags") or r.get("auto_tags", []) or []):
                clean = t.strip('<>').strip()
                if clean:
                    tags_set.add(clean)
        return sorted(tags_set)

    def get_emoji_catalog_text(self, max_emojis: int = 30) -> str:
        """
        生成 AI 可读的表情目录文本。
        格式：
          - 描述 → [标签1, 标签2] (hash: a1b2c3...)

        Args:
            max_emojis: 最多列出多少个
        Returns:
            目录文本，如果没有表情则返回空字符串
        """
        records = self._storage.list_all()
        if not records:
            return ""

        # 按使用次数降序排列
        records.sort(key=lambda r: r.get("used_count", 0), reverse=True)
        records = records[:max_emojis]

        lines = ["## 可用表情"]
        lines.append("你可以使用 search_emoji 工具搜索以下表情，然后用 send_emoji 发送。")
        lines.append("")

        for r in records:
            short_hash = r["hash"][:12]
            desc = r.get("user_description") or r.get("auto_description", "") or "(无描述)"
            tags = r.get("user_tags") or r.get("auto_tags", []) or []
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"- {desc}{tag_str} (hash: {short_hash})")

        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════
    # 内部辅助
    # ════════════════════════════════════════════════════════════

    async def _download_emoji(self, url: str) -> bytes:
        """
        下载 emoji 图片，带重试。

        Args:
            url: 图片 URL
        Returns:
            图片字节数据
        Raises:
            RuntimeError: 下载失败
        """
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
        """
        根据 content_type 和 filename 推断文件扩展名。

        Returns:
            扩展名，带点，如 ".png", ".jpg", ".gif"
        """
        # 优先用 content_type
        ct_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        if content_type in ct_map:
            return ct_map[content_type]

        # 退化用文件名
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                return ext if ext != ".jpeg" else ".jpg"

        # 默认
        return ".png"

    @staticmethod
    def _maybe_normalize_image(image_path: Path) -> Path:
        """
        对图片做归一化处理，确保 VLM 能正常识别。
        对小图片进行放大填边。

        Returns:
            归一化后的图片路径（可能是新文件）
        """
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

            # 保存归一化后的图片
            new_path = image_path.with_stem(image_path.stem + "_norm")
            import base64 as b64_mod

            ImageUtils.base64_to_image(
                normalized_b64, str(new_path)
            )
            _log.debug(f"图片已归一化: {new_path}")
            return new_path
        except Exception as e:
            _log.warning(f"图片归一化失败，使用原图: {e}")
            return image_path
