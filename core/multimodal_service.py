"""
多模态（视觉）模型服务接口。

封装与视觉大模型（VLM）的交互，提供两个核心方法：
- analyze_image(image_path) → 图片内容描述
- analyze_emoji(image_path) → (内容描述, 情绪标签列表)

复用与主 AI 服务相同的 OpenAI 兼容客户端配置（api_key, base_url, model）。
支持 DeepSeek V4 等兼容 OpenAI Vision API 的模型。
"""

import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)


class MultimodalService:
    """
    多模态（视觉）模型服务。

    使用 OpenAI 兼容的 Vision API 分析图片内容。
    与 AIService 共享相同的 API 密钥和基础 URL，但独立控制参数。

    使用方式：
        service = MultimodalService(api_key="...", base_url="...", model="deepseek-v4-flash")
        description = await service.analyze_image("/path/to/image.jpg")
        desc, tags = await service.analyze_emoji("/path/to/emoji.png")
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "deepseek-v4-flash",
    ):
        """
        初始化多模态服务。

        Args:
            api_key: API 密钥
            base_url: API 基础 URL（OpenAI 兼容格式）
            model: 支持视觉能力的模型名
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 内存缓存：image_md5 → (description, emotion_tags)
        # 防止同一图片在同 session 中重复请求 VLM
        self._cache: Dict[str, Tuple[str, List[str]]] = {}
        self._cache_max = 200

        _log.info(f"多模态服务已启动，模型: {self.model}")

    async def analyze_image(self, image_path: str) -> str:
        """
        分析普通图片，返回主要内容描述。

        Args:
            image_path: 本地图片文件路径
        Returns:
            图片内容描述文本，如 "一只橘猫坐在窗台上晒太阳"
        """
        cache_key = self._get_cache_key(image_path, "image")
        if cache_key in self._cache:
            _log.debug(f"分析图片命中缓存: {image_path}")
            return self._cache[cache_key][0]

        base64_data = self._encode_image(image_path)
        prompt = (
            "请用一句话简要描述这张图片中的主要内容。"
            "只返回描述，不要附加其他文字。"
        )

        result = await self._call_vlm(base64_data, prompt)
        result = (result or "图片").strip()

        self._set_cache(cache_key, result, [])
        return result

    async def analyze_emoji(self, image_path: str) -> Tuple[str, List[str]]:
        """
        分析表情/贴图图片，返回主要内容和情绪标签。

        Args:
            image_path: 本地图片文件路径
        Returns:
            (description, emotion_tags)
            description: "一个微笑的卡通猫头，眼睛弯弯的"
            emotion_tags: ["开心", "可爱", "友好"]
        """
        cache_key = self._get_cache_key(image_path, "emoji")
        if cache_key in self._cache:
            _log.debug(f"分析表情命中缓存: {image_path}")
            cached = self._cache[cache_key]
            return cached[0], cached[1]

        base64_data = self._encode_image(image_path)
        prompt = (
            "请分析这张表情/贴图图片。\n"
            "1. 用一句话描述图片中的主要内容。\n"
            "2. 给出 1-3 个情绪/情感标签。\n\n"
            "格式要求（严格按此格式返回）：\n"
            "描述：<内容描述>\n"
            "情绪：<标签1>、<标签2>、<标签3>"
        )

        result = await self._call_vlm(base64_data, prompt)
        description, emotions = self._parse_emoji_result(result or "")

        self._set_cache(cache_key, description, emotions)
        return description, emotions

    # ── 内部方法 ──

    def _encode_image(self, image_path: str) -> str:
        """
        读取图片文件并转换为 base64 data URI。

        Args:
            image_path: 本地图片路径
        Returns:
            data URI 格式的 base64 字符串，如 "data:image/jpeg;base64,...."
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        # 从文件扩展名推断 MIME 类型
        ext = path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = mime_map.get(ext, "image/jpeg")
        return f"data:{mime};base64,{b64}"

    async def _call_vlm(self, base64_image: str, prompt: str) -> Optional[str]:
        """
        调用视觉模型的核心方法。

        Args:
            base64_image: data URI 格式的 base64 图片
            prompt: 分析提示词
        Returns:
            模型返回文本，或 None
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=300,  # 图片描述不需要太多 token
                temperature=0.3,  # 低温度，描述更精确
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return None
        except Exception as e:
            _log.error(f"调用视觉模型失败: {e}")
            raise

    @staticmethod
    def _parse_emoji_result(result: str) -> Tuple[str, List[str]]:
        """
        解析表情分析的结构化返回。

        期望格式：
            描述：一个微笑的卡通猫头
            情绪：开心、可爱、友好
        """
        description = ""
        emotions: List[str] = []

        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith("描述："):
                description = line[3:].strip()
            elif line.startswith("情绪："):
                raw_tags = line[3:].strip()
                # 支持中文顿号、逗号、空格分隔
                emotions = [
                    t.strip()
                    for t in raw_tags.replace("、", ",")
                    .replace("，", ",")
                    .replace(" ", ",")
                    .split(",")
                    if t.strip()
                ]

        # 如果解析失败，返回原始文本作为描述
        if not description and result:
            description = result.strip()

        return description, emotions

    def _get_cache_key(self, image_path: str, mode: str) -> str:
        """
        生成缓存 key。

        使用文件内容前 8KB 的 MD5 + 分析模式，避免重复请求 VLM。
        """
        try:
            with open(image_path, "rb") as f:
                head = f.read(8192)
            h = hashlib.md5(head).hexdigest()
            return f"{h}:{mode}"
        except Exception:
            # 如果读文件失败，用路径本身
            return f"{image_path}:{mode}"

    def _set_cache(self, key: str, description: str, tags: List[str]) -> None:
        """写入缓存，并控制缓存大小。"""
        if len(self._cache) >= self._cache_max:
            # 简单淘汰：清空一半
            keys_to_remove = list(self._cache.keys())[: self._cache_max // 2]
            for k in keys_to_remove:
                del self._cache[k]
        self._cache[key] = (description, tags)

    def clear_cache(self) -> None:
        """清空内存缓存。"""
        self._cache.clear()
        _log.debug("多模态服务内存缓存已清空")
