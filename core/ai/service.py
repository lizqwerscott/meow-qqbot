import asyncio
import logging
import os
from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from core.ai.protocol import (
    AssistantMessage,
    AssistantToolCall,
    StreamAbortedError,
    StreamBuffer,
    StreamCallbacks,
    ensure_messages_consistent,
    log_llm_error,
)

_log = logging.getLogger(__name__)


async def _consume_chunk_stream(
    stream: Any,
    callbacks: StreamCallbacks | None,
    buffer: StreamBuffer,
) -> dict[str, Any] | None:
    """迭代 chat-completions SSE 流，把增量聚合进缓冲；返回 usage（若有）。

    主路径与 stream_options 降级重试路径共用，保证两处聚合语义一致。
    """
    async for chunk in stream:
        if not chunk.choices and chunk.usage:
            return chunk.usage.model_dump()
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue
        if getattr(delta, "content", None):
            buffer.text_parts.append(delta.content)
            if callbacks and callbacks.on_text:
                await callbacks.on_text("".join(buffer.text_parts))
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            buffer.reasoning_parts.append(reasoning)
            if callbacks and callbacks.on_reasoning:
                await callbacks.on_reasoning("".join(buffer.reasoning_parts))
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index or 0
            while len(buffer.tool_calls) <= idx:
                buffer.tool_calls.append(
                    AssistantToolCall(id="", name="", arguments="")
                )
            target = buffer.tool_calls[idx]
            if tc.id:
                target.id = tc.id
            if tc.function:
                if tc.function.name:
                    target.name = tc.function.name
                if tc.function.arguments:
                    target.arguments += tc.function.arguments
    return None


class AIService:
    """
    AI 服务类，使用 OpenAI 官方包。

    两级返回粒度是设计意图（契约）：
    - `chat_completion`（无工具）返回 `str` 文本，供学习者/归档/简单对话消费；
    - `chat_completion_with_tools` 返回 `AssistantMessage`（含 tool_calls /
      reasoning_content / raw），供工具循环消费。
    新增调用方按所需粒度选方法；需要完整消息的无工具场景，再引入消息级方法。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-3.5-turbo",
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
        http_client: Any | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
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
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        is_reasoning = self._is_reasoning_model(model_to_use)

        try:
            kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model_to_use,
                "max_tokens": max_tokens_to_use,
            }

            if not (is_reasoning or self.reasoning_effort):
                temperature_to_use = (
                    temperature if temperature is not None else self.temperature
                )
                kwargs["temperature"] = temperature_to_use

            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort

            if response_format:
                kwargs["response_format"] = response_format

            extra_body = self._build_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body

            response = await self.client.chat.completions.create(**kwargs)
            usage = response.usage.model_dump() if response.usage else None
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content, usage
            else:
                return None, usage
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_llm_error(e, model_to_use)
            return None, None

    def _is_reasoning_model(self, model: str) -> bool:
        return any(k in model for k in ("o1", "o3", "deepseek", "reasoning"))

    async def chat_completion_stream(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: StreamCallbacks | None = None,
    ) -> tuple[AssistantMessage | None, dict[str, Any] | None]:
        """流式版 chat_completion_with_tools（chat-completions SSE）。

        内部聚合增量片段，返回的 AssistantMessage 与非流式完全一致；
        callbacks 收到的是**累计文本**（含 reasoning），调用方按需转发。
        """
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        is_reasoning = self._is_reasoning_model(model_to_use)

        # 聚合缓冲放 try 外：create() 早期抛异常时 except 分支也要能引用（断流抛错）
        buffer = StreamBuffer()
        usage: dict[str, Any] | None = None

        try:
            msgs = list(messages)
            ensure_messages_consistent(msgs)

            kwargs: dict[str, Any] = {
                "messages": msgs,
                "model": model_to_use,
                "max_tokens": max_tokens_to_use,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

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

            if tools:
                kwargs["tools"] = tools

            stream = await self.client.chat.completions.create(**kwargs)
            try:
                usage = await _consume_chunk_stream(stream, callbacks, buffer)
            finally:
                close_fn = getattr(stream, "close", None)
                if close_fn is not None:
                    await close_fn()

            return buffer.assemble(usage)
        except asyncio.CancelledError:
            raise
        except StreamAbortedError:
            raise
        except Exception as e:
            err_str = str(e)
            # 部分网关（如 ollama /v1）不认 stream_options，重试一次不带它
            low = err_str.lower()
            if (
                "stream_options" in low
                or "include_usage" in low
                or "unrecognized" in low
            ):
                kwargs.pop("stream_options", None)
                # 重试前清空聚合缓冲：重试是一次全新的生成，旧增量必须丢弃
                # （防御：网关在流中途才拒绝该参数时，缓冲里已有半截内容）。
                buffer.reset()
                usage = None
                # 通知调用方同样归零转发偏移：首尝试的增量可能已流过 on_text，
                # ToolLoop 的 st.sent 还停在旧文本上，新文本会从错误偏移切片。
                if callbacks and callbacks.on_reset:
                    await callbacks.on_reset()
                try:
                    stream = await self.client.chat.completions.create(**kwargs)
                    try:
                        usage = await _consume_chunk_stream(stream, callbacks, buffer)
                    finally:
                        close_fn = getattr(stream, "close", None)
                        if close_fn is not None:
                            await close_fn()
                except asyncio.CancelledError:
                    raise
                except Exception as e2:
                    # 降级重试也失败：与主路径一致抛错，由上层决定回退/终止
                    _log.error("流式降级重试也失败 [%s]: %s", model_to_use, e2)
                    raise StreamAbortedError(f"流式降级重试失败: {e2}") from e2
                return buffer.assemble(usage)
            log_llm_error(e, model_to_use, tag="（流式）")
            # 聚合到一半断流：不把半截结果当完整返回——上层会误判成功、跳过
            # fallback、投递截断回复。抛 StreamAbortedError，由 ToolLoop 依
            # 转发状态决定回退（零转发）或终止（已转发部分文本，避免双回复）。
            raise StreamAbortedError(f"流式响应中断: {e}") from e

    def _build_extra_body(self) -> dict[str, Any] | None:
        if self.reasoning_effort:
            return {"thinking": {"type": "enabled"}}
        return None

    async def chat_completion_with_tools(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[AssistantMessage | None, dict[str, Any] | None]:
        """带工具调用的一次性对话，返回统一协议对象 AssistantMessage。

        Returns:
            (AssistantMessage | None, usage_dict | None)
        """
        model_to_use = model or self.model
        max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
        is_reasoning = self._is_reasoning_model(model_to_use)

        try:
            # 最终防线：清理孤立的 tool_calls，防止重启恢复后历史不完整导致 API 400
            msgs = list(messages)
            ensure_messages_consistent(msgs)

            kwargs: dict[str, Any] = {
                "messages": msgs,
                "model": model_to_use,
                "max_tokens": max_tokens_to_use,
            }

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
                msg = response.choices[0].message
                return (
                    AssistantMessage(
                        content=msg.content,
                        tool_calls=(
                            [
                                AssistantToolCall(
                                    id=tc.id,
                                    name=tc.function.name,
                                    arguments=tc.function.arguments,
                                )
                                for tc in msg.tool_calls
                            ]
                            if getattr(msg, "tool_calls", None)
                            else None
                        ),
                        reasoning_content=getattr(msg, "reasoning_content", None),
                        raw=msg,
                    ),
                    usage,
                )
            return None, usage
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_llm_error(e, model_to_use, tag="（带工具）")
            return None, None
