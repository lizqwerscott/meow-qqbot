"""DeepSeekResponsesService — DeepSeek Responses API 服务（OpenAI SDK responses 资源）。

DeepSeek 新增的 Responses API（base_url 即 https://api.deepseek.com，无需 /v1 前缀）：
- 非流式/流式均走 `client.responses.create(...)`，与 chat-completions 是两个端点。
- 目前仅支持 deepseek-v4-flash 模型。
- 输入为 input items（message / function_call / function_call_output / reasoning），
  与 chat-completions 的 messages 数组不通用，需要转换（_messages_to_input）。
- 流式事件以 response.completed / response.incomplete / response.failed 收尾，
  无 `data: [DONE]`；usage 在收尾事件携带的完整 response 对象里。
- 不支持 stream_options（忽略即可）；不支持的参数静默忽略。

对外契约与 AIService 一致（满足 LLMService 协议），可注册为 provider type
`deepseek_responses` 进入 ModelRegistry fallback 链。
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
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

DEFAULT_BASE_URL = "https://api.deepseek.com"


def _messages_to_input(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """chat-completions messages → (instructions, Responses input items)。

    转换规则（对齐 DeepSeek 兼容性明细）：
    - 第一条 system → instructions（"作为第一条 system 消息"）；后续 system → message item。
    - assistant 的 reasoning_content → 明文 reasoning item（服务端归并到相邻 assistant）。
    - assistant 无 content 时只发 function_call items（服务端自动归并），
      避免空 content 的 message item 触发 400。
    - tool → function_call_output（output 必须是字符串）。
    """
    instructions: str | None = None
    items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            if instructions is None:
                instructions = content
            else:
                items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
        elif role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        elif role == "assistant":
            reasoning = msg.get("reasoning_content")
            if reasoning:
                items.append(
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": reasoning}],
                    }
                )
            if content:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id"),
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id"),
                    "output": content,
                }
            )

    return instructions, items


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """chat-completions 风格 tools → Responses API 风格（function 扁平化）。"""
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        converted.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return converted or None


def _convert_response_format(
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """response_format → Responses `text.format`（DeepSeek 完整支持 format）。"""
    if not response_format:
        return None
    if response_format.get("type") == "json_object":
        return {"type": "json_object"}
    if response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema") or {}
        return {
            "type": "json_schema",
            "name": schema.get("name", "schema"),
            "schema": schema.get("schema", {}),
        }
    return None


def _normalize_usage(usage: Any) -> dict[str, Any] | None:
    """Responses usage → cost_tracker 认识的 chat-completions 键。

    Responses: input_tokens / output_tokens / input_tokens_details.cached_tokens。
    CostTracker 读取: prompt_tokens / completion_tokens / prompt_cache_hit_tokens /
    prompt_cache_miss_tokens。
    """
    if usage is None:
        return None
    u = usage if isinstance(usage, dict) else usage.model_dump()
    input_tokens = u.get("input_tokens", 0) or 0
    output_tokens = u.get("output_tokens", 0) or 0
    in_details = u.get("input_tokens_details") or {}
    cached = in_details.get("cached_tokens", 0) or 0
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "prompt_cache_hit_tokens": cached,
        "prompt_cache_miss_tokens": max(input_tokens - cached, 0),
    }


def _parse_output(response: Any) -> AssistantMessage | None:
    """非流式响应 output items → AssistantMessage。"""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[AssistantToolCall] = []
    for item in getattr(response, "output", None) or []:
        t = getattr(item, "type", None)
        if t == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text" and getattr(
                    part, "text", None
                ):
                    content_parts.append(part.text)
        elif t == "function_call":
            tool_calls.append(
                AssistantToolCall(
                    id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                    name=getattr(item, "name", ""),
                    arguments=getattr(item, "arguments", ""),
                )
            )
        elif t == "reasoning":
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if text:
                    reasoning_parts.append(text)

    content = "".join(content_parts) or None
    reasoning_content = "".join(reasoning_parts) or None
    if not (content or reasoning_content or tool_calls):
        return None
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls or None,
        reasoning_content=reasoning_content,
    )


class DeepSeekResponsesService:
    """DeepSeek Responses API 服务，满足 LLMService 协议。

    非流式与流式共用同一套消息/工具转换与输出解析，差异仅在 stream 参数。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "deepseek-v4-flash",
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
        http_client: Any | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or DEFAULT_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
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

    # ── 请求组装 ──

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        instructions, items = _messages_to_input(messages)
        kwargs: dict[str, Any] = {"model": model, "stream": stream}
        if items:
            kwargs["input"] = items
        if instructions is not None:
            kwargs["instructions"] = instructions

        max_output = max_tokens if max_tokens is not None else self.max_tokens
        kwargs["max_output_tokens"] = max_output

        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            # DeepSeek 文档：temperature 思考模式下不生效（不报错）；非思考模式总是传。
            kwargs["temperature"] = (
                temperature if temperature is not None else self.temperature
            )

        converted_tools = _convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        text_format = _convert_response_format(response_format)
        if text_format:
            kwargs["text"] = {"format": text_format}

        return kwargs

    # ── 协议方法 ──

    async def chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        model_to_use = model or self.model
        try:
            msgs = list(messages)
            ensure_messages_consistent(msgs)
            kwargs = self._build_kwargs(
                msgs, None, model_to_use, temperature, max_tokens, response_format
            )
            response = await self.client.responses.create(**kwargs)
            usage = _normalize_usage(getattr(response, "usage", None))
            msg = _parse_output(response)
            return (msg.content if msg else None), usage
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_llm_error(e, model_to_use, service="DeepSeek")
            return None, None

    async def chat_completion_with_tools(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[AssistantMessage | None, dict[str, Any] | None]:
        model_to_use = model or self.model
        try:
            msgs = list(messages)
            ensure_messages_consistent(msgs)
            kwargs = self._build_kwargs(
                msgs, tools, model_to_use, temperature, max_tokens
            )
            response = await self.client.responses.create(**kwargs)
            usage = _normalize_usage(getattr(response, "usage", None))
            return _parse_output(response), usage
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_llm_error(e, model_to_use, service="DeepSeek")
            return None, None

    async def chat_completion_stream(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: StreamCallbacks | None = None,
    ) -> tuple[AssistantMessage | None, dict[str, Any] | None]:
        """流式：SSE 事件聚合，返回的 AssistantMessage 与非流式一致。

        事件映射：
        - response.output_text.delta → 累计文本（回调）
        - response.reasoning_text.delta → 累计思维链（回调）
        - response.function_call_arguments.delta/done → 按 item_id 聚合工具调用
        - response.completed / response.incomplete → 取 usage
        - response.failed → 错误日志
        """
        model_to_use = model or self.model

        # 聚合缓冲放 try 外：create() 早期抛异常时 except 分支也要能引用（断流抛错）
        buffer = StreamBuffer()
        # item_id -> [name, arguments_buf]（保持出现顺序）
        tool_calls: OrderedDict[str, list[str]] = OrderedDict()
        # item_id/call_id -> function name。
        # DeepSeek 实测：function_call_arguments.done 事件的 name 为 None，
        # name 只在 output_item.added/done 的 item 上；双 key 兼容两种情况。
        call_names: dict[str, str] = {}
        # item_id/call_id -> 真实 call_id（对齐非流式 _parse_output 的 call_id 优先；
        # 若服务端 item_id ≠ call_id，回注 function_call_output 需用 call_id 关联）
        call_ids: dict[str, str] = {}
        usage: dict[str, Any] | None = None

        try:
            msgs = list(messages)
            ensure_messages_consistent(msgs)
            kwargs = self._build_kwargs(
                msgs, tools, model_to_use, temperature, max_tokens, stream=True
            )

            stream = await self.client.responses.create(**kwargs)
            try:
                async for event in stream:
                    et = getattr(event, "type", "")
                    if et in (
                        "response.output_item.added",
                        "response.output_item.done",
                    ):
                        item = getattr(event, "item", None)
                        itype = getattr(item, "type", None) if item else None
                        if itype == "function_call":
                            name = getattr(item, "name", None)
                            call_id = getattr(item, "call_id", None) or getattr(
                                item, "id", None
                            )
                            if name:
                                call_names[getattr(item, "id", None)] = name
                                call_names[call_id] = name
                            if call_id:
                                call_ids[getattr(item, "id", None)] = call_id
                                call_ids[call_id] = call_id
                        elif itype == "message" and et == "response.output_item.added":
                            # 模型修订输出（草稿→终稿）：新 message item 开始，
                            # 之前 item 的文本必须丢弃——否则草稿+终稿在缓冲里
                            # 拼接成翻倍内容（线上事故：两条部分重复的消息）。
                            buffer.text_parts.clear()
                        elif (
                            itype == "reasoning" and et == "response.output_item.added"
                        ):
                            # 修订后的新 reasoning item：旧思维链丢弃（仅日志用途）
                            buffer.reasoning_parts.clear()
                    elif et == "response.output_text.delta":
                        buffer.text_parts.append(event.delta)
                        if callbacks and callbacks.on_text:
                            await callbacks.on_text("".join(buffer.text_parts))
                    elif et == "response.reasoning_text.delta":
                        buffer.reasoning_parts.append(event.delta)
                        if callbacks and callbacks.on_reasoning:
                            await callbacks.on_reasoning(
                                "".join(buffer.reasoning_parts)
                            )
                    elif et == "response.function_call_arguments.delta":
                        buf = tool_calls.setdefault(event.item_id, ["", ""])
                        buf[1] += event.delta
                    elif et == "response.function_call_arguments.done":
                        buf = tool_calls.setdefault(event.item_id, ["", ""])
                        name = getattr(event, "name", None) or call_names.get(
                            event.item_id
                        )
                        if name:
                            buf[0] = name
                        if getattr(event, "arguments", None) and not buf[1]:
                            buf[1] = event.arguments
                    elif et in (
                        "response.completed",
                        "response.incomplete",
                    ):
                        resp = getattr(event, "response", None)
                        if resp is not None:
                            usage = _normalize_usage(getattr(resp, "usage", None))
                    elif et == "response.failed":
                        resp = getattr(event, "response", None)
                        err = getattr(resp, "error", None) if resp else None
                        _log.error("Responses API 流式失败 [%s]: %s", model_to_use, err)
                        # API 侧失败：不把半截聚合当完整结果返回，抛错交上层决策
                        raise StreamAbortedError(f"Responses API 流式失败: {err}")
            finally:
                close_fn = getattr(stream, "close", None)
                if close_fn is not None:
                    await close_fn()

            buffer.tool_calls = [
                AssistantToolCall(
                    id=call_ids.get(item_id, item_id),
                    name=buf[0],
                    arguments=buf[1],
                )
                for item_id, buf in tool_calls.items()
                if buf[0]
            ]
            return buffer.assemble(usage)
        except asyncio.CancelledError:
            raise
        except StreamAbortedError:
            raise
        except Exception as e:
            log_llm_error(e, model_to_use, service="DeepSeek", tag="（流式）")
            # 聚合到一半断流：不把半截结果当完整返回（上层会误判成功、跳过
            # fallback、投递截断回复）。抛 StreamAbortedError，由 ToolLoop 依
            # 转发状态决定回退（零转发）或终止（已转发部分文本，避免双回复）。
            raise StreamAbortedError(f"流式响应中断: {e}") from e
