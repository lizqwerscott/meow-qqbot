"""ModelScopeService — ModelScope 免费推理服务。

OpenAI 兼容 API，通过 httpx response hook 捕获限流头。
额度耗尽时自动返回 None，由上层 fallback 链处理。
"""

import logging
from typing import Any, Dict, Iterable, List, Optional

import httpx
from httpx import Timeout
from openai.types.chat import ChatCompletionMessageParam

from core.ai.service import AIService
from core.ai.rate_limit import ModelScopeRateLimit

_log = logging.getLogger(__name__)


class ModelScopeService(AIService):
    """ModelScope 模型服务。

    继承 AIService，所有 chat_completion / chat_completion_with_tools 行为一致，
    额外特性：
    - 使用带 response hook 的自定义 httpx 客户端，自动解析限流头
    - 额度耗尽时返回 (None, None)，触发上层 fallback
    - 暴露 rate_limit / can_call / quota_info 供外部决策
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
        self._rate_limit = ModelScopeRateLimit()

        http_client = httpx.AsyncClient(
            timeout=Timeout(timeout),
            event_hooks={"response": [self._on_response]},
        )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            http_client=http_client,
        )

    def _on_response(self, response: httpx.Response):
        if response.is_success:
            self._rate_limit = ModelScopeRateLimit.from_headers(response.headers)

    @property
    def rate_limit(self) -> ModelScopeRateLimit:
        return self._rate_limit

    @property
    def can_call(self) -> bool:
        return self._rate_limit.can_call

    @property
    def quota_info(self) -> dict:
        rl = self._rate_limit
        return {
            "user_limit": rl.user_limit,
            "user_remaining": rl.user_remaining,
            "model_limit": rl.model_limit,
            "model_remaining": rl.model_remaining,
            "exhausted": rl.exhausted,
        }

    async def chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[Dict]]:
        if not self.can_call:
            _log.warning(
                f"ModelScope [{model or self.model}] 额度耗尽，跳过调用: "
                f"user_remaining={self._rate_limit.user_remaining}, "
                f"model_remaining={self._rate_limit.model_remaining}"
            )
            return None, None
        return await super().chat_completion(
            messages=messages, model=model, temperature=temperature, max_tokens=max_tokens,
        )

    async def chat_completion_with_tools(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[Any], Optional[Dict]]:
        if not self.can_call:
            _log.warning(
                f"ModelScope [{model or self.model}] 额度耗尽，跳过调用"
            )
            return None, None
        return await super().chat_completion_with_tools(
            messages=messages, tools=tools, model=model,
            temperature=temperature, max_tokens=max_tokens,
        )
