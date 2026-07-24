"""OllamaService — 本地 Ollama 模型服务。

使用 ollama-python 库直接调用本地或远程 Ollama 服务。
支持工具调用、多模态视觉、自定义 host 和 api_key。
"""

import logging
from typing import Any, Dict, Iterable, List, Optional

from ollama import AsyncClient as OllamaAsyncClient
from openai.types.chat import ChatCompletionMessageParam

_log = logging.getLogger(__name__)


class OllamaService:
    """Ollama 模型服务。

    不继承 AIService，直接使用 ollama-python 的 AsyncClient。
    与 AIService 保持相同的方法签名，通过 duck-typing 兼容。
    """

    def __init__(
        self,
        api_key: str = "",
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 120,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or ""
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        auth = None
        if self.api_key:
            auth = (self.api_key, "")

        self._client = OllamaAsyncClient(host=self.host, auth=auth)

    async def close(self):
        pass

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
        try:
            raw = list(messages)
            self._normalize_vision_messages(raw)
            response = await self._client.chat(
                model=model_to_use,
                messages=raw,
                options={
                    "temperature": temperature if temperature is not None else self.temperature,
                    "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
                },
            )
            content = response.get("message", {}).get("content")
            usage = self._build_usage(response)
            return content, usage
        except Exception as e:
            _log.error(f"Ollama 请求失败: {e}")
            return None, None

    @staticmethod
    def _normalize_vision_messages(messages: List[Dict]) -> None:
        """就地转换 OpenAI vision 格式为 ollama 格式。

        OpenAI: content=[{"type":"image_url","image_url":{"url":"data:..."}}, {"type":"text","text":"..."}]
        Ollama:  content="text", images=["base64"]
        """
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            text_parts: List[str] = []
            images: List[str] = []
            has_image = False
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type")
                    if t == "image_url":
                        has_image = True
                        url = part.get("image_url", {}).get("url", "")
                        if "," in url:
                            images.append(url.split(",", 1)[1])
                        else:
                            images.append(url)
                    elif t == "text":
                        text_parts.append(part.get("text", ""))
            if has_image:
                msg["content"] = " ".join(text_parts) if text_parts else ""
                msg["images"] = images

    async def chat_completion_with_tools(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[Any], Optional[Dict]]:
        model_to_use = model or self.model
        try:
            from core.tools.tool_loop import ensure_messages_consistent
            ensure_messages_consistent(messages)

            kwargs: Dict[str, Any] = dict(
                model=model_to_use,
                messages=list(messages),
                options={
                    "temperature": temperature if temperature is not None else self.temperature,
                    "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
                },
            )

            if tools:
                kwargs["tools"] = tools

            response = await self._client.chat(**kwargs)
            message = response.get("message", {})
            content = message.get("content")
            raw_tool_calls = message.get("tool_calls")

            usage = self._build_usage(response)

            result = _ChatMessage(content=content, raw_tool_calls=raw_tool_calls)
            return result, usage
        except Exception as e:
            _log.error(f"Ollama 请求失败（带工具）: {e}")
            return None, None

    def _build_usage(self, response: dict) -> Optional[Dict[str, int]]:
        prompt_tokens = response.get("prompt_eval_count")
        eval_tokens = response.get("eval_count")
        if prompt_tokens is not None:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": eval_tokens or 0,
                "total_tokens": prompt_tokens + (eval_tokens or 0),
            }
        return None


class _ChatMessage:
    """轻量兼容层，模拟 OpenAI ChatCompletionMessage 的 content / tool_calls 接口。

    ToolLoop 等消费方通过 .content 和 .tool_calls 访问结果。
    """

    def __init__(self, content: Optional[str] = None, raw_tool_calls: Optional[List[Dict]] = None):
        self.content = content
        self._raw_tool_calls = raw_tool_calls or []
        self._cached: Optional[List[Any]] = None

    @property
    def tool_calls(self) -> Optional[List[Any]]:
        if not self._raw_tool_calls:
            return None
        if self._cached is not None:
            return self._cached

        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        result = []
        for i, tc in enumerate(self._raw_tool_calls):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, dict):
                import json
                args = json.dumps(args, ensure_ascii=False)
            result.append(
                ChatCompletionMessageToolCall(
                    id=tc.get("id", f"call_{i}"),
                    type="function",
                    function=Function(name=fn.get("name", ""), arguments=args),
                )
            )
        self._cached = result
        return self._cached

    def model_dump(self) -> dict:
        return {
            "content": self.content,
            "tool_calls": [tc.model_dump() for tc in self.tool_calls] if self.tool_calls else None,
        }
