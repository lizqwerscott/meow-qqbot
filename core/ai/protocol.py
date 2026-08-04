"""AI 协议抽象层 — 核心循环只依赖本模块，不感知底层 LLM 协议。

对应 Pi 架构文档的「事件流抽象」：无论底层是 OpenAI SSE / Anthropic SSE / 其他协议，
统一翻译成项目自己的消息对象（AssistantMessage / AssistantToolCall），
ToolLoop / FallbackRunner / ModelRegistry 只消费本模块的类型。

设计要点:
- raw 字段保留原始 message 对象，翻译不丢信息（如 reasoning_content 等扩展字段）。
- tool_calls_data 属性把「回注 messages 所需的 wire dict 组装」收进协议对象，
  ToolLoop 不再手写 dict。
- ensure_messages_consistent 本在 tool_loop.py，因 service.py 需要它而形成
  函数内 import 的循环依赖隐患，随协议层一并下沉到此处。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam

_log = logging.getLogger(__name__)


@dataclass
class AssistantToolCall:
    """一条工具调用（与底层协议的 wire 格式解耦）。

    arguments 保持 JSON 字符串形式，与 openai wire 格式一致。
    """

    id: str
    name: str
    arguments: str

    def to_wire(self) -> dict[str, Any]:
        """转回 API 请求所需的 dict 格式（用于回注 messages 的 assistant 消息）。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class AssistantMessage:
    """模型返回的统一消息。ToolLoop / FallbackRunner / ModelRegistry 只消费本类型。"""

    content: str | None = None
    tool_calls: list[AssistantToolCall] | None = None
    reasoning_content: str | None = None

    # 保留原始 message 对象（如 openai ChatCompletionMessage），
    # 供需要原始字段的调用方使用（诊断、日志、未来扩展字段探测）。
    raw: Any = field(default=None, repr=False, compare=False)

    @property
    def tool_calls_data(self) -> list[dict[str, Any]] | None:
        """tool_calls 的 wire dict 形式（= 原 ToolLoop 手写的 tool_calls_data）。"""
        if not self.tool_calls:
            return None
        return [tc.to_wire() for tc in self.tool_calls]

    def to_wire(self) -> dict[str, Any]:
        """转回 API 请求所需的整条 assistant 消息 dict（用于回注 messages）。

        与旧 ToolLoop 手写组装逐字段一致：reasoning_content 有条件设置，
        tool_calls 仅在有调用时携带（空数组会触发部分 API 400）。
        """
        wire: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.reasoning_content:
            wire["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            wire["tool_calls"] = self.tool_calls_data
        return wire


class StreamAbortedError(Exception):
    """流式响应中途失败（create 成功后迭代中断 / API 侧 failed 事件）。

    服务层不得把半截聚合结果当「正常完成」返回——调用方（ToolLoop）无法区分
    截断与完整，会把截断回复标记为成功并跳过 fallback。正确约定：流式失败
    必须抛本异常，由调用方依据自身的转发状态决定回退（零转发）还是终止
    （已实时转发部分文本，回退会双回复）。
    """


@dataclass
class StreamCallbacks:
    """流式回调：on_text / on_reasoning 收到的是**累计文本**（非增量）。

    传累计文本而非原始 delta，是为了让调用方（如 ToolLoop 分片发送）
    无需自己拼接，直接按累计长度切片即可。
    """

    on_text: Callable[[str], Awaitable[None]] | None = None
    on_reasoning: Callable[[str], Awaitable[None]] | None = None


class LLMService(Protocol):
    """LLM 服务契约（结构化子类型：AIService / ModelScopeService 自动满足）。

    ModelRegistry / FallbackRunner / ProviderFactory 只依赖本协议，
    新增 provider 必须实现这些方法 + close，否则静态检查报错。

    流式方法 chat_completion_stream 返回的 AssistantMessage 与非流式
    chat_completion_with_tools 完全一致（内部聚合 delta），调用方可按
    callbacks 是否提供决定要不要实时转发文本。

    流式中途失败（断流/API 侧 failed）时抛 StreamAbortedError——返回值
    只表示「正常完成」，半截聚合不得冒充完整结果。
    """

    model: str
    base_url: str | None

    async def chat_completion(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]: ...

    async def chat_completion_with_tools(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[AssistantMessage | None, dict[str, Any] | None]: ...

    async def chat_completion_stream(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        callbacks: StreamCallbacks | None = None,
    ) -> tuple[AssistantMessage | None, dict[str, Any] | None]: ...

    async def close(self) -> None: ...


def ensure_messages_consistent(messages: list[dict[str, Any]]) -> None:
    """清理 messages 中孤立的 tool_calls 并修复 tool 响应顺序。

    场景：某轮 AI 返回了 tool_calls，但后续处理异常导致
    对应的 tool 响应消息未追加完整。下次请求时 API 会 400。
    修复：删除最后一个没有对应 tool 响应的 assistant 消息。

    此外，因竞态条件可能在 assistant(tool_calls) 和 tool 响应之间
    插入了用户消息，导致 API 400。修复：将 tool 响应紧跟在对应的
    tool_calls 之后，被插入的消息移到 tool 响应后面。
    """
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tc_ids = {
                tc["id"]
                for tc in msg["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            }
            if not tc_ids:
                _log.warning(
                    f"移除无 ID 的 tool_calls 消息: count={len(msg['tool_calls'])}"
                )
                i += 1
                continue
            result.append(msg)
            i += 1
            tools = []
            interleaved = []
            while i < len(messages) and tc_ids:
                m = messages[i]
                if m.get("role") == "tool" and m.get("tool_call_id") in tc_ids:
                    tools.append(m)
                    tc_ids.discard(m["tool_call_id"])
                    i += 1
                elif not tc_ids:
                    break
                else:
                    interleaved.append(m)
                    i += 1
            if tc_ids:
                _log.warning(
                    f"tool_calls 缺少响应: missing_ids={tc_ids}, "
                    f"移除第 {len(result) - 1} 条 assistant"
                )
                result.pop()
                result.extend(interleaved)
            else:
                result.extend(tools)
                result.extend(interleaved)
        else:
            if msg.get("role") == "tool":
                _log.warning(
                    f"移除孤立的 tool 消息: tool_call_id={msg.get('tool_call_id')}"
                )
            else:
                result.append(messages[i])
            i += 1
    messages[:] = result
