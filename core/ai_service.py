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
        max_tokens: int = 1000,
    ):
        """
        初始化 AI 服务

        Args:
            api_key: OpenAI API 密钥，如果为 None 则从环境变量读取
            base_url: API 基础 URL，支持 OpenAI 兼容接口
            model: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    async def close(self):
        """关闭客户端"""
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
    ) -> str | None:
        """
        发送聊天补全请求

        Args:
            messages: 消息列表
            model: 模型名称，如果为 None 则使用默认模型
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Returns:
            AI 生成的文本内容
        """
        model_to_use = model or self.model
        temperature_to_use = (
            temperature if temperature is not None else self.temperature
        )
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=model_to_use,
                temperature=temperature_to_use,
                max_tokens=max_tokens_to_use,
            )
            # 提取文本内容
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            else:
                return None
        except Exception as e:
            _log.error(f"AI 请求失败: {e}")
            return None
