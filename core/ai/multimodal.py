"""
多模态（视觉）模型服务接口。

封装与视觉大模型（VLM）的交互，提供两个核心方法：
- analyze_image(image_path) → 图片内容描述
- analyze_emoji(image_path) → (内容描述, 情绪标签列表)

支持多模型 fallback：传入 AIService 列表，按顺序调用直到成功。
"""

import base64
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.ai.service import AIService

_log = logging.getLogger(__name__)


class MultimodalService:
    """
    多模态（视觉）模型服务。

    使用 OpenAI 兼容的 Vision API 分析图片内容。
    支持多 AIService fallback：列表中的模型按顺序调用，失败自动切换。

    使用方式：
        service = MultimodalService([ai_svc_1, ai_svc_2])
        description = await service.analyze_image("/path/to/image.jpg")
        desc, tags = await service.analyze_emoji("/path/to/emoji.png")
    """

    def __init__(self, ai_services: List[AIService]):
        if not ai_services:
            raise ValueError("至少需要一个 AIService")
        self._services = ai_services
        self.model = ai_services[0].model

        self._cache: OrderedDict = OrderedDict()
        self._cache_max = 200

        _log.info(
            f"多模态服务已启动: {len(ai_services)} 个模型, "
            f"主模型: {self.model}"
        )

    async def analyze_image(self, image_path: str) -> str:
        cache_key = self._get_cache_key(image_path, "image")
        if cache_key in self._cache:
            _log.debug(f"分析图片命中缓存: {image_path}")
            self._cache.move_to_end(cache_key)
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

    async def analyze_emoji(self, image_path: str, is_gif: bool = False) -> Tuple[str, List[str]]:
        cache_key = self._get_cache_key(image_path, "emoji")
        if cache_key in self._cache:
            _log.debug(f"分析表情命中缓存: {image_path}")
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            return cached[0], cached[1]

        base64_data = self._encode_image(image_path)

        if is_gif:
            prompt = (
                '这是一张将动图各帧从左到右拼接的图片，展示了一个动态动画过程。'
                '请分析这个动画表现的内容和动作，仅返回以下 JSON 格式'
                '（不要包含其他文字或 markdown 包裹）：\n'
                '{\n'
                '  "description": "一句话描述这个动画表达的内容和动作",\n'
                '  "emotions": ["标签1", "标签2", "标签3"]\n'
                '}\n'
                '其中 emotions 给出 1-3 个情绪/情感标签，反映这个动画传递的情感。'
            )
        else:
            prompt = (
                '请分析这张表情/贴图图片，仅返回以下 JSON 格式'
                '（不要包含其他文字或 markdown 包裹）：\n'
                '{\n'
                '  "description": "一句话描述图片中的主要内容",\n'
                '  "emotions": ["标签1", "标签2", "标签3"]\n'
                '}\n'
                '其中 emotions 给出 1-3 个情绪/情感标签。'
            )

        result = await self._call_vlm(base64_data, prompt)
        description, emotions = self._parse_emoji_result(result or "")

        self._set_cache(cache_key, description, emotions)
        return description, emotions

    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

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
        for idx, svc in enumerate(self._services):
            try:
                response = await svc.client.chat.completions.create(
                    model=svc.model,
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
                    max_tokens=300,
                    temperature=0.3,
                )
                if response.choices and response.choices[0].message.content:
                    _log.debug(
                        f"VLM 调用成功 (模型 [{svc.model}])"
                    )
                    return response.choices[0].message.content
                _log.warning(
                    f"VLM 返回空结果 (模型 [{svc.model}])"
                )
            except Exception as e:
                _log.warning(
                    f"VLM 调用失败 (模型 [{svc.model}], "
                    f"第 {idx + 1}/{len(self._services)} 个): {e}"
                )

        _log.error("所有 VLM 模型均失败")
        return None

    @staticmethod
    def _parse_emoji_result(result: str) -> Tuple[str, List[str]]:
        import json

        text = result.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1]

        try:
            data = json.loads(text)
            desc = data.get("description", "").strip()
            raw_emotions = data.get("emotions", [])
            if isinstance(raw_emotions, list):
                emotions = [
                    e.strip() for e in raw_emotions
                    if isinstance(e, str) and e.strip()
                ]
            else:
                emotions = []
            return desc, emotions
        except (json.JSONDecodeError, TypeError):
            _log.warning(f"解析表情 JSON 失败，原始返回: {result[:200]}")
            return result.strip(), []

    def _get_cache_key(self, image_path: str, mode: str) -> str:
        try:
            with open(image_path, "rb") as f:
                head = f.read(8192)
            h = hashlib.md5(head).hexdigest()
            return f"{h}:{mode}"
        except Exception as e:
            _log.debug(f"计算图片缓存键失败 [{image_path}]: {e}")
            return f"{image_path}:{mode}"

    def _set_cache(self, key: str, description: str, tags: List[str]) -> None:
        if len(self._cache) >= self._cache_max:
            self._cache.popitem(last=False)
        self._cache[key] = (description, tags)

    def clear_cache(self) -> None:
        self._cache.clear()
        _log.debug("多模态服务内存缓存已清空")
