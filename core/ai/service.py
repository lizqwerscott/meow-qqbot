import asyncio
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from httpx import Timeout
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

_log = logging.getLogger(__name__)


class AIService:
    """
    AI 服务类，使用 OpenAI 官方包
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    async def close(self):
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[Dict]]:
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        is_reasoning = self._is_reasoning_model(model_to_use)

        try:
            kwargs: Dict[str, Any] = dict(
                messages=messages,
                model=model_to_use,
                max_tokens=max_tokens_to_use,
            )

            if not (is_reasoning or self.reasoning_effort):
                temperature_to_use = (
                    temperature if temperature is not None else self.temperature
                )
                kwargs["temperature"] = temperature_to_use

            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

            extra_body = self._build_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body

            response = await self.client.chat.completions.create(**kwargs)
            usage = response.usage.model_dump() if response.usage else None
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content, usage
            else:
                return None, usage
        except Exception as e:
            _log.error(f"AI 请求失败: {e}")
            return None, None

    def _is_reasoning_model(self, model: str) -> bool:
        return any(k in model for k in ("o1", "o3", "deepseek", "reasoning"))

    def _build_extra_body(self) -> Optional[Dict[str, Any]]:
        if self.reasoning_effort:
            return {"thinking": {"type": "enabled"}}
        return None

    async def chat_completion_with_tools(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[Any], Optional[Dict]]:
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        is_reasoning = self._is_reasoning_model(model_to_use)

        try:
            # 最终防线：清理孤立的 tool_calls，防止重启恢复后历史不完整导致 API 400
            from core.tools.tool_loop import ensure_messages_consistent
            ensure_messages_consistent(messages)

            kwargs: Dict[str, Any] = dict(
                messages=messages,
                model=model_to_use,
                max_tokens=max_tokens_to_use,
            )

            if is_reasoning or self.reasoning_effort:
                pass
            else:
                temperature_to_use = (
                    temperature if temperature is not None else self.temperature
                )
                kwargs["temperature"] = temperature_to_use

            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

            extra_body = self._build_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body

            if tools:
                kwargs["tools"] = tools

            response = await self.client.chat.completions.create(**kwargs)
            usage = response.usage.model_dump() if response.usage else None
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message, usage
            return None, usage
        except Exception as e:
            _log.error(f"AI 请求失败（带工具）: {e}")
            return None, None
