import asyncio
import os
from typing import Any, Dict, List, Optional

from botpy import logging
from httpx import Timeout
from openai import AsyncOpenAI

_log = logging.get_logger()


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

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        格式化消息为 OpenAI 格式

        Args:
            messages: 消息列表，每个消息是包含 role 和 content 的字典

        Returns:
            格式化后的消息列表
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted.append({"role": msg["role"], "content": msg["content"]})
            else:
                # 尝试转换为字典格式
                try:
                    if hasattr(msg, "to_dict"):
                        msg_dict = msg.to_dict()
                        if "role" in msg_dict and "content" in msg_dict:
                            formatted.append(
                                {
                                    "role": msg_dict["role"],
                                    "content": msg_dict["content"],
                                }
                            )
                except Exception:
                    _log.warning(f"无法格式化消息: {msg}")
        return formatted

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        发送聊天补全请求

        Args:
            messages: 消息列表
            model: 模型名称，如果为 None 则使用默认模型
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            stream: 是否使用流式响应
            **kwargs: 其他 OpenAI API 参数

        Returns:
            AI 生成的文本内容
        """
        model_to_use = model or self.model
        temperature_to_use = (
            temperature if temperature is not None else self.temperature
        )
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens

        # 格式化消息
        formatted_messages = self._format_messages(messages)
        if not formatted_messages:
            raise ValueError("No valid messages provided")

        try:
            if stream:
                return await self._stream_completion(
                    messages=formatted_messages,
                    model=model_to_use,
                    temperature=temperature_to_use,
                    max_tokens=max_tokens_to_use,
                    **kwargs,
                )
            else:
                return await self._normal_completion(
                    messages=formatted_messages,
                    model=model_to_use,
                    temperature=temperature_to_use,
                    max_tokens=max_tokens_to_use,
                    **kwargs,
                )
        except Exception as e:
            _log.error(f"AI 请求失败: {e}")
            raise

    async def generate_with_context(
        self,
        chat_id: str,
        user_message: str,
        context_manager: Any,
        system_prompt: Optional[str] = None,
        max_context_messages: int = 8,
        **kwargs,
    ) -> str:
        """
        使用上下文管理器生成响应

        Args:
            chat_id: 聊天 ID
            user_message: 用户消息
            context_manager: 上下文管理器实例
            system_prompt: 系统提示
            max_context_messages: 最大上下文消息数
            **kwargs: 其他参数传递给 chat_completion

        Returns:
            AI 生成的文本
        """
        # 获取历史上下文
        history_messages = []
        if hasattr(context_manager, "get_history_as_dicts"):
            history_dicts = context_manager.get_history_as_dicts(
                chat_id, max_context_messages
            )
            history_messages = history_dicts
        elif hasattr(context_manager, "get_history"):
            history = context_manager.get_history(chat_id, max_context_messages)
            if history and hasattr(history[0], "to_dict"):
                history_messages = [msg.to_dict() for msg in history]
            elif history and isinstance(history[0], dict):
                history_messages = history

        # 生成响应
        return await self.generate_response(
            prompt=user_message,
            context_messages=history_messages,
            system_prompt=system_prompt,
            **kwargs,
        )
