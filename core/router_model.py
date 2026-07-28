"""RouterModel — 轻量路由模型实现智能分级。

路由模型（如 qwen2.5:7b）接收请求，判断复杂度：
- 简单任务：路由模型直接回复
- 复杂任务：标记 [ESCALATE] 后转发主模型
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

_log = logging.getLogger(__name__)

ROUTER_BASE_PROMPT = """你是一个智能路由助手，负责判断消息复杂度。

简单任务（直接回复，不加任何前缀）：
- 问候/打招呼：你好、在吗、早上好
- 闲聊/情绪回应：哈哈、好的、辛苦了、谢谢、嗯嗯
- 简单问答：今天几号、现在几点
- 模式匹配/复读：同上、+1、确实

复杂任务（回复必须以 [ESCALATE] 开头，后跟重新表述后的请求）：
- 需要搜索记忆或查询信息
- 需要使用工具（搜索表情、查用户、执行命令等）
- 多步推理、分析、总结、代码生成
- 用户明确要求"猫猫"执行某个操作
- 不确定时，标记为复杂任务"""

HEARTBEAT_BASE_PROMPT = """请检查是否有需要关注的事项。

如果没有需要关注的事项，只回复 HEARTBEAT_OK。
如果有需要提醒的，简短说明即可，不要超过 100 字。"""


def _build_route_prompt(character_card: str = "") -> str:
    """构建带角色卡的路由系统提示。"""
    parts = []
    if character_card:
        parts.append(character_card)
    parts.append(ROUTER_BASE_PROMPT)
    return "\n\n".join(parts)


def _build_heartbeat_prompt(character_card: str = "") -> str:
    """构建带角色卡的心跳系统提示。"""
    parts = []
    if character_card:
        parts.append(character_card)
    parts.append(HEARTBEAT_BASE_PROMPT)
    return "\n\n".join(parts)


@dataclass
class RouteDecision:
    action: str  # "direct" | "escalate"
    response: str
    latency_ms: float = 0.0


class RouterModel:
    """轻量路由模型。

    Args:
        config: {
            "api_key": str,
            "base_url": str,
            "model": str,
            "temperature": float (default 0.3),
            "max_tokens": int (default 2000),
            "timeout": int (default 15),
        }
    """

    def __init__(self, config: dict, character_card: str = ""):
        self.model = config.get("model", "qwen2.5:7b")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2000)
        timeout = config.get("timeout", 15)
        self._character_card = character_card

        self._route_prompt = _build_route_prompt(character_card)
        self._heartbeat_prompt = _build_heartbeat_prompt(character_card)

        self._client = AsyncOpenAI(
            api_key=config.get("api_key", "not-needed"),
            base_url=config.get("base_url", "http://localhost:11434/v1"),
            timeout=timeout,
            max_retries=1,
        )
        card_info = " [有角色卡]" if character_card else ""
        _log.info(
            f"RouterModel 已初始化: model={self.model} "
            f"base_url={config.get('base_url', 'http://localhost:11434/v1')}{card_info}"
        )

    async def close(self):
        await self._client.close()

    async def route(
        self,
        content: str,
        chat_id: str = "",
    ) -> RouteDecision:
        """判断消息复杂度并路由。

        Returns:
            RouteDecision:
              action="direct" → 路由模型直接回复
              action="escalate" → 转发主模型，response 是重新表述后的请求
        """
        start = time.monotonic()
        messages = [
            {"role": "system", "content": self._route_prompt},
            {"role": "user", "content": content},
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            latency = (time.monotonic() - start) * 1000
            text = (resp.choices[0].message.content or "").strip()

            escalate_prefix = "[ESCALATE]"
            if text.upper().startswith(escalate_prefix):
                reformulated = text[len(escalate_prefix):].strip()
                if not reformulated:
                    reformulated = content
                _log.info(
                    f"路由 escalate ({latency:.0f}ms): {content[:40]} -> {reformulated[:60]}"
                )
                return RouteDecision("escalate", reformulated, latency)
            else:
                _log.info(
                    f"路由 direct ({latency:.0f}ms): {content[:40]}"
                )
                return RouteDecision("direct", text, latency)

        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            _log.warning(f"路由模型请求失败，fallback 主模型: {e}")
            return RouteDecision("escalate", content, 0.0)

    async def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
    ) -> str:
        """无工具的单轮对话（给 heartbeat 和简单任务用）。

        Returns:
            回复文本，可能包含 HEARTBEAT_OK
        """
        if system_prompt is None:
            system_prompt = self._heartbeat_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            _log.warning(f"RouterModel simple_chat 失败: {e}")
            return ""
