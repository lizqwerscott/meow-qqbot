"""ToolLoop — AI 工具调用编排循环

执行 AI 工具调用循环：
- AI → tool_calls → 执行 → 结果回注 → 重复
- 每轮文本即时发送并记录上下文
- Queue Steering：工具循环期间 drain 新消息并注入
"""

import asyncio
import itertools
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from core.message import InputMessage, MessageType

from core.tools.executor import ToolContext

_log = logging.getLogger(__name__)


def ensure_messages_consistent(messages: List[dict]) -> None:
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
                tc["id"] for tc in msg["tool_calls"]
                if isinstance(tc, dict) and tc.get("id")
            }
            if not tc_ids:
                _log.warning(f"移除无 ID 的 tool_calls 消息: count={len(msg['tool_calls'])}")
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
            result.append(messages[i])
            i += 1
    messages[:] = result


class ToolLoop:
    """AI 工具调用编排循环。

    职责：
    1. 调用 AI 获取响应
    2. 解析 tool_calls 并分派给 ToolExecutor
    3. 结果回注到 messages 并继续下一轮
    4. Queue Steering：新消息注入
    """

    def __init__(
        self,
        ai_service: Any,
        tool_executor: Any,
        *,
        cost_tracker: Any = None,
        context_manager: Any = None,
        session_manager: Any = None,
        prompt_builder: Any = None,
        hindsight_memory: Any = None,
        max_rounds: int = -1,
        model_registry: Any = None,
    ):
        self.ai_service = ai_service
        self.tool_executor = tool_executor
        self.cost_tracker = cost_tracker
        self.context_manager = context_manager
        self.session_manager = session_manager
        self.prompt_builder = prompt_builder
        self.hindsight = hindsight_memory
        self._max_tool_rounds = max_rounds
        self._model_registry = model_registry

    async def run(
        self,
        messages: List[dict],
        tools: Optional[List[dict]],
        chat_id: str,
        is_group: bool,
        reply_to: str,
        reply_callback: Callable,
        sender_id: str = "",
        get_user_nickname: Optional[Callable[[str], str]] = None,
        delivery_channel: str = "",
        reply_to_message_id: str = "",
        model_chain: Optional[List[str]] = None,
    ) -> bool:
        """执行工具调用循环。

        Args:
            delivery_channel: 后台任务时传入真实聊天 ID，供 send_emoji 等工具使用
            model_chain: 模型链（如 ["cheap", "primary"]），启用 fallback。

        Returns:
            sent_emoji: 是否在循环中发送了表情。
        """
        sent_emoji = False
        current_model_name: Optional[str] = None

        if self._max_tool_rounds == -1:
            _rounds: Any = itertools.count()
        else:
            _rounds = range(self._max_tool_rounds)

        # ── 预解析模型链：循环前一次性找出可用模型 ──
        resolved_model_name: Optional[str] = None
        resolved_service: Any = None
        if model_chain and self._model_registry:
            resolved = await self._model_registry.resolve_model_chain(model_chain)
            if resolved:
                resolved_model_name, resolved_service = resolved
                _log.info(f"工具循环预解析模型: [{resolved_model_name}]")
            else:
                _log.warning(f"模型链全部冷却/无效: {model_chain}")
                await reply_callback(chat_id, "所有模型均不可用，请稍后重试", reply_to, is_group)
                return False

        for round_idx in _rounds:
            # ── 防御：清理 messages 中孤立的 tool_calls ──
            ensure_messages_consistent(messages)

            # ── AI 调用（使用预解析模型，失败则重新解析） ──
            message = None
            usage = None
            failed_models: set = set()
            while True:
                svc = resolved_service or self.ai_service
                current_model_name = resolved_model_name or None

                try:
                    message, usage = await svc.chat_completion_with_tools(
                        messages=messages, tools=tools,
                    )
                except Exception as e:
                    _log.warning(f"模型 [{current_model_name}] 调用异常: {e}")
                    message, usage = None, None

                if message is not None:
                    if resolved_model_name and self._model_registry:
                        self._model_registry.cooldown_manager.record_success(
                            resolved_model_name
                        )
                    break

                # ── 失败 → 累积 + 记录冷却 ──
                if resolved_model_name:
                    failed_models.add(resolved_model_name)
                    if self._model_registry:
                        self._model_registry.cooldown_manager.record_failure(
                            resolved_model_name
                        )

                if not model_chain or not self._model_registry:
                    break

                remaining = [m for m in model_chain if m not in failed_models]

                if not remaining:
                    # 链中所有模型都已尝试并失败
                    await reply_callback(
                        chat_id, "所有模型均不可用", reply_to, is_group,
                    )
                    return False

                _log.warning(
                    f"模型 [{resolved_model_name}] 失败，从剩余链重新解析: {remaining}"
                )
                new_resolved = await self._model_registry.resolve_model_chain(remaining)
                if new_resolved:
                    resolved_model_name, resolved_service = new_resolved
                    continue

                # 剩余链全在冷却：走完整 fallback 链做最后一次兜底
                message, usage, fb_name = await self._model_registry.chat_with_fallback(
                    remaining, messages, tools,
                )
                if fb_name:
                    resolved_model_name = fb_name
                    resolved_service = self._model_registry.get(fb_name)
                current_model_name = fb_name
                break

            model_for_cost = current_model_name or self.ai_service.model
            if usage and self.cost_tracker:
                self.cost_tracker.record_turn(chat_id, model_for_cost, usage)

            if message is None:
                await reply_callback(chat_id, "AI 服务异常", reply_to, is_group)
                break

            response_text = message.content or ""
            tool_calls = message.tool_calls or []

            reasoning = getattr(message, "reasoning_content", None) or None
            if reasoning:
                _log.info(
                    f"[工具循环 第{round_idx + 1}轮 思考过程]\n{reasoning}"
                )

            _log.info(
                f"[工具循环 第{round_idx + 1}轮] "
                f"text={response_text[:50]!r}... "
                f"tool_calls={[tc.function.name for tc in tool_calls]}"
            )

            tool_calls_data = None
            if tool_calls:
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            if response_text:
                await reply_callback(
                    chat_id=chat_id,
                    content=response_text,
                    message_id=reply_to,
                    is_group=is_group,
                )

            if response_text or tool_calls:
                await self.context_manager.add_assistant_message_async(
                    chat_id,
                    response_text or "",
                    reply_to,
                    tool_calls=tool_calls_data,
                    reasoning_content=reasoning,
                )

            if not tool_calls:
                break

            assistant_msg: dict = {"role": "assistant", "content": response_text or None}
            reasoning_content = getattr(message, "reasoning_content", None)
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            assistant_msg["tool_calls"] = tool_calls_data
            messages.append(assistant_msg)

            ctx = ToolContext(
                chat_id=chat_id,
                is_group=is_group,
                reply_to=reply_to,
                sender_id=sender_id,
                reply_callback=reply_callback,
                delivery_channel=delivery_channel,
                reply_to_message_id=reply_to_message_id,
            )

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    _log.warning(f"工具参数解析失败: {tc.function.arguments}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "参数解析失败"}),
                    })
                    continue

                try:
                    result = await self.tool_executor.execute(tc.function.name, args, ctx)
                    content = result.content
                    if result.sent_emoji:
                        sent_emoji = True
                except Exception as e:
                    _log.error(f"工具 [{tc.function.name}] 执行异常: {e}")
                    content = json.dumps({"error": f"执行异常: {e}"}, ensure_ascii=False)
                    # 先记录 tool 响应再传播，避免历史中留下孤立 tool_calls
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    })
                    await self.context_manager.add_tool_result_async(
                        chat_id, tc.function.name, content, tc.id,
                    )
                    if isinstance(e, asyncio.CancelledError):
                        raise
                    continue

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })
                await self.context_manager.add_tool_result_async(
                    chat_id, tc.function.name, content, tc.id,
                )

                # 在 tool 响应之后写入表情标记，避免插在 assistant(tc) 和 tool 之间
                if result.sent_emoji:
                    await self.context_manager.add_assistant_message_async(
                        chat_id, "[助手发送了一个表情]", reply_to,
                    )

            if get_user_nickname:
                steer_msgs = await self._drain_steering_messages(
                    chat_id=chat_id,
                    current_sender_id=sender_id,
                    messages=messages,
                    get_user_nickname=get_user_nickname,
                )
                messages.extend(steer_msgs)

        return sent_emoji

    async def _drain_steering_messages(
        self,
        chat_id: str,
        current_sender_id: str,
        messages: List[dict],
        get_user_nickname: Callable[[str], str],
    ) -> List[dict]:
        """从会话队列中 drain 新消息，注入到当前工具循环。

        同一用户 → 不注入记忆（减少冗余调用）
        不同用户 → 走一次 build_memory_context
        """
        if not self.session_manager:
            return []

        queue = await self.session_manager.get_queue(chat_id)
        drained: List[InputMessage] = []
        while not queue.empty():
            try:
                drained.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        steered: List[dict] = []
        for msg in drained:
            nick = get_user_nickname(msg.sender_id) or msg.sender_id
            content = f"[来自 {nick} 的新消息]: {msg.content}"
            user_msg: dict = {"role": "user", "content": content}
            steered.append(user_msg)

            await self.context_manager.add_user_message_async(
                chat_id, content, msg.id,
                sender_id=msg.sender_id, name=nick,
            )

            if self.hindsight and msg.msg_type != MessageType.CARD:
                await self.hindsight.add_message(
                    session_id=chat_id,
                    sender_id=msg.sender_id,
                    sender_name=nick,
                    content=content,
                    context=self.hindsight.msg_type_to_context(msg.msg_type),
                    timestamp=msg.timestamp,
                    resources=msg.resources,
                )

            if msg.sender_id != current_sender_id and self.hindsight and self.prompt_builder:
                memory_text = await self.prompt_builder.build_memory_context(
                    sender_id=msg.sender_id,
                    input_message=msg,
                )
                if memory_text:
                    steered.append({
                        "role": "system",
                        "content": memory_text,
                    })

            queue.task_done()

        return steered
