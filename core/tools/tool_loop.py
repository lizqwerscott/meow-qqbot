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
    """清理 messages 中孤立的 tool_calls。

    场景：某轮 AI 返回了 tool_calls，但后续处理异常导致
    对应的 tool 响应消息未追加完整。下次请求时 API 会 400。
    修复：删除最后一个没有对应 tool 响应的 assistant 消息。
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue

        # 检查此 assistant 之后是否有足够的 tool 响应
        expected_ids = {
            tc["id"] for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        }
        if not expected_ids:
            # tool_calls 没有 ID，可能格式异常，移除
            _log.warning(f"移除无 ID 的 tool_calls 消息: count={len(tool_calls)}")
            messages.pop(i)
            continue

        # 统计之后 tool 消息中出现的 tool_call_id
        found_ids: set = set()
        for j in range(i + 1, len(messages)):
            if messages[j].get("role") == "tool":
                tid = messages[j].get("tool_call_id")
                if tid:
                    found_ids.add(tid)

        if not expected_ids.issubset(found_ids):
            missing = expected_ids - found_ids
            _log.warning(f"tool_calls 缺少响应: missing_ids={missing}, 移除第 {i} 条 assistant")
            messages.pop(i)


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

        for round_idx in _rounds:
            # ── 防御：清理 messages 中孤立的 tool_calls ──
            ensure_messages_consistent(messages)

            if model_chain and self._model_registry:
                message, usage, current_model_name = await self._model_registry.chat_with_fallback(
                    model_chain, messages, tools,
                )
            else:
                current_model_name = None
                message, usage = await self.ai_service.chat_completion_with_tools(
                    messages=messages,
                    tools=tools,
                )

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
                except BaseException as e:
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
