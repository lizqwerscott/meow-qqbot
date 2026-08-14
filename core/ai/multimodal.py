"""
多模态（视觉）模型服务接口。

封装与视觉大模型（VLM）的交互，提供两个核心方法：
- analyze_image(image_path) → 图片内容描述
- analyze_emoji(image_path) → (摘要, 详细描述, 情绪标签列表)

支持多模型 fallback：传入服务列表，按顺序调用直到成功。
"""

import asyncio
import base64
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.ai.cooldown import ModelCooldownManager

_log = logging.getLogger(__name__)


class MultimodalService:
    """
    多模态（视觉）模型服务。

    使用 OpenAI 兼容的 Vision API 分析图片内容。
    支持多 AIService fallback：列表中的模型按顺序调用，失败自动切换。

        使用方式：
            service = MultimodalService([ai_svc_1, ai_svc_2], ["deepseek/vlm", "modelscope/vlm"])
            description = await service.analyze_image("/path/to/image.jpg")
            summary, desc, tags = await service.analyze_emoji("/path/to/emoji.png")
    """

    def __init__(
        self,
        ai_services: List[Any],
        model_names: Optional[List[str]] = None,
        cooldown_manager: Optional[ModelCooldownManager] = None,
    ):
        if not ai_services:
            raise ValueError("至少需要一个 AIService")
        self._services = ai_services
        self._model_names = model_names or [""] * len(ai_services)
        self._cooldown = cooldown_manager
        self.model = ai_services[0].model

        self._cache: OrderedDict = OrderedDict()
        self._cache_max = 200

        _log.info(
            f"多模态服务已启动: {len(ai_services)} 个模型, " f"主模型: {self.model}"
        )

    async def analyze_image(
        self, image_path: str, *, prompt: str | None = None, max_tokens: int = 1024
    ) -> str:
        normalized_prompt = " ".join((prompt or "").split())
        cache_key = await self._get_cache_key(
            image_path, f"image:{normalized_prompt}:v2"
        )
        if cache_key in self._cache:
            _log.debug(f"分析图片命中缓存: {image_path}")
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key][0]

        base64_data = await self._encode_image(image_path)
        prompt = normalized_prompt or (
            "请用一句话简要描述这张图片中的主要内容。只返回描述，不要附加其他文字。"
        )

        result = await self._call_vlm(base64_data, prompt, max_tokens=max_tokens)
        result = (result or "").strip()
        if not result:
            raise RuntimeError("VLM returned no image description")

        self._set_cache(cache_key, result, result, [])
        return result

    async def analyze_emoji(
        self, image_path: str, is_gif: bool = False
    ) -> Tuple[str, str, List[str]]:
        cache_key = await self._get_cache_key(image_path, "emoji")
        if cache_key in self._cache:
            _log.debug(f"分析表情命中缓存: {image_path}")
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            return cached[0], cached[1], cached[2]

        base64_data = await self._encode_image(image_path)

        prompt = self._build_emoji_prompt(is_gif)

        strategies = [
            ("json_object", {"type": "json_object"}, "", 4096),
            ("no_format", None, "", 4096),
            (
                "retry",
                None,
                "\n请务必只返回合法的 JSON 格式，不要添加其他任何文字。",
                4096,
            ),
        ]

        summary = description = ""
        emotions: List[str] = []

        for i, (name, rf, extra, mt) in enumerate(strategies):
            result = await self._call_vlm(
                base64_data,
                prompt + extra,
                max_tokens=mt,
                response_format=rf,
            )
            summary, description, emotions = self._parse_emoji_result(result or "")
            if summary or description:
                break
            _log.warning(f"VLM 表情分析失败 (策略={name}), {i+1}/3 次重试")

        if summary or description:
            self._set_cache(cache_key, summary, description, emotions)
        return summary, description, emotions

    @staticmethod
    def _build_emoji_prompt(is_gif: bool) -> str:
        fmt = (
            "仅返回以下 JSON 格式（不要 markdown 包裹）：\n"
            "{\n"
            '  "summary": "10字以内简短概括",\n'
            '  "description": "尽可能详细的图片描述",\n'
            '  "emotions": ["情感标签1", "情感标签2"]\n'
            "}\n"
        )
        if is_gif:
            return (
                "这是一张将动图各帧从左到右拼接的图片，展示了一个动态动画过程。\n"
                "请详细描述这个动画中的人物、动作、表情、背景和所有视觉细节，越详细越好。\n"
                f"{fmt}"
                "其中 emotions 给出 1-3 个情绪标签，反映这个动画传递的情感。"
            )
        return (
            "请详细描述这张图片中的人物、动物、动作、表情、背景、文字及所有视觉细节，越详细越好。\n"
            f"{fmt}"
            "其中 emotions 给出 1-3 个情绪标签。"
        )

    async def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        image_bytes = await asyncio.to_thread(path.read_bytes)
        b64 = base64.b64encode(image_bytes).decode("ascii")

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

    async def _call_vlm(
        self,
        base64_image: str,
        prompt: str,
        max_tokens: int = 300,
        response_format: Optional[Dict] = None,
    ) -> Optional[str]:
        for idx, svc in enumerate(self._services):
            qualified_name = (
                self._model_names[idx] if idx < len(self._model_names) else ""
            )

            # 冷却检查
            if self._cooldown and await self._cooldown.is_cooled_down(qualified_name):
                _log.info(f"VLM 模型 [{qualified_name}] 处于冷却期，跳过")
                continue

            try:
                content, _ = await svc.chat_completion(
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
                    max_tokens=max_tokens,
                    temperature=0.3,
                    response_format=response_format,
                )
                if content:
                    if self._cooldown and qualified_name:
                        await self._cooldown.record_success(qualified_name)
                    _log.info(f"VLM 调用成功 (模型 [{qualified_name or svc.model}])")
                    return content
                # 空结果不写入全局冷却
                _log.warning(f"VLM 返回空结果 (模型 [{qualified_name or svc.model}])")
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                if self._cooldown and qualified_name:
                    await self._cooldown.record_failure(qualified_name)
                _log.warning(
                    f"VLM 调用失败 (模型 [{qualified_name or svc.model}], "
                    f"第 {idx + 1}/{len(self._services)} 个)",
                    exc_info=True,
                )

        models_tried = ", ".join(
            self._model_names[i] or self._services[i].model
            for i in range(len(self._services))
        )
        _log.error(f"所有 VLM 模型均失败: [{models_tried}]")
        return None

    @staticmethod
    def _parse_emoji_result(result: str) -> Tuple[str, str, List[str]]:
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
            summary = (data.get("summary") or "").strip()
            description = (data.get("description") or "").strip()
            raw_emotions = data.get("emotions", [])
            if isinstance(raw_emotions, list):
                emotions = [
                    e.strip() for e in raw_emotions if isinstance(e, str) and e.strip()
                ]
            else:
                emotions = []
            return summary, description, emotions
        except (json.JSONDecodeError, TypeError):
            _log.warning(
                f"解析表情 JSON 失败，原始返回: {result[:200]}",
                exc_info=True,
            )
            return "", "", []

    async def _get_cache_key(self, image_path: str, mode: str) -> str:
        try:
            head = await asyncio.to_thread(self._read_file_head, image_path)
            h = hashlib.md5(head).hexdigest()
            return f"{h}:{mode}"
        except Exception as e:
            _log.debug(f"计算图片缓存键失败 [{image_path}]: {e}")
            return f"{image_path}:{mode}"

    @staticmethod
    def _read_file_head(path: str, n: int = 8192) -> bytes:
        with open(path, "rb") as f:
            return f.read(n)

    def _set_cache(
        self, key: str, summary: str, description: str, tags: List[str]
    ) -> None:
        if len(self._cache) >= self._cache_max:
            self._cache.popitem(last=False)
        self._cache[key] = (summary, description, tags)

    def invalidate_cache(self, image_path: str, mode: str = "emoji") -> None:
        key = self._get_cache_key(image_path, mode)
        if key in self._cache:
            del self._cache[key]
            _log.debug(f"已清除多模态缓存: {key}")

    def clear_cache(self) -> None:
        self._cache.clear()
        _log.debug("多模态服务内存缓存已清空")
