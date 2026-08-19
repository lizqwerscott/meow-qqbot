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
from typing import Any, Awaitable, Callable, List, Optional

from core.ai.fallback_runner import FallbackRunner
from core.ai.protocol import ensure_messages_consistent
from core.managers.session_manager import PendingInbound
from core.tools._types import ToolContext
from core.tools.impl import execute as execute_tool
from core.tools.stream_delivery import StreamDelivery, is_silent_reply_text

_log = logging.getLogger(__name__)


class ToolLoop:
    """AI 工具调用编排循环。

    职责：
    1. 调用 AI 获取响应
    2. 解析 tool_calls 并分派给 ToolExecutor
    3. 结果回注到 messages 并继续下一轮
    4. Queue Steering：新消息注入
    """

    def __init__(self, ctx, *, prompt_builder=None, session_manager=None):
        self.ai_service = ctx.ai.ai_service
        self._perm = ctx.mgmt.permission_manager
        self.cost_tracker = ctx.mgmt.cost_tracker
        self.context_manager = ctx.mgmt.context_manager
        self.session_manager = session_manager
        self.prompt_builder = prompt_builder
        self.hindsight = ctx.memory.hindsight_memory
        self._max_tool_rounds = ctx.ai.max_tool_rounds
        self._model_registry = ctx.ai.model_registry
        # 流式回复（block 模式，对齐 openclaw qqbot 插件）：文本累积到
        # stream_block_chars 或距上次发送空闲 stream_block_idle_ms 才发一块，
        # 不做逐句连发，避免 QQ 群聊刷屏。
        self._stream_reply = bool(getattr(ctx.ai, "stream_reply", False))
        self._stream_block_chars = max(
            int(getattr(ctx.ai, "stream_block_chars", 800) or 800), 64
        )
        self._stream_block_idle = max(
            float(getattr(ctx.ai, "stream_block_idle_ms", 1000) or 1000) / 1000.0,
            0.05,
        )

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
        binding_manager=None,
        tier: Optional[str] = None,
        stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        delivery_state_callback: Optional[Callable[[], Awaitable[None]]] = None,
        tool_reply_callback: Optional[Callable] = None,
        tool_reply_names: Optional[set[str] | frozenset[str]] = None,
        reply_state_callback: Optional[Callable[[bool], Awaitable[None]]] = None,
        steering_enabled: bool = False,
        steering_admission_callback: Optional[
            Callable[[PendingInbound], Awaitable[Any]]
        ] = None,
        inbound_message_ids: Optional[List[str]] = None,
    ) -> tuple[bool, bool]:
        """执行工具调用循环。

        Args:
            delivery_channel: 后台任务时传入真实聊天 ID，供 send_emoji 等工具使用
            model_chain: 模型链（如 ["cheap", "primary"]），启用 fallback。
            binding_manager: SessionBindingManager（启用 session 绑定优化）
            tier: 当前消息的 RuleRouter 分类档位（用于 session 绑定键）
            stream_callback: 流式文本转发回调（async (text_chunk) -> None）。
                传入后（且 [ai].stream_reply 开启）AI 文本增量会在静默探测期后
                按自然边界分片实时转发；未传入则只聚合不转发，行为与非流式一致。

        Returns:
            (sent_emoji, text_committed)
            - sent_emoji: 是否在循环中发送了表情
            - text_committed: 文本是否已投递/承诺投递（含发送失败但已承诺的部分，防重复）
        """
        sent_emoji = False
        text_committed = False
        message_delivered = False
        current_model_name: Optional[str] = None
        suppress_reply = False
        inbound_message_ids = list(inbound_message_ids or [])

        if self._max_tool_rounds == -1:
            _rounds: Any = itertools.count()
        else:
            _rounds = range(self._max_tool_rounds)

        # ── 预解析模型链：FallbackRunner 统一编排（支持 session 绑定） ──
        runner = None
        if model_chain and self._model_registry:
            runner = FallbackRunner(self._model_registry, model_chain)
            ok = await runner.try_acquire_with_binding(binding_manager, chat_id, tier)
            if not ok:
                _log.warning(f"模型链全部冷却/无效: {model_chain}")
                try:
                    await reply_callback(
                        chat_id, "所有模型均不可用，请稍后重试", reply_to, is_group
                    )
                except Exception as cb_err:
                    _log.warning("回复 callback 失败 [%s]: %s", chat_id[:12], cb_err)
                return False, True

        for round_idx in _rounds:
            # ── 防御：清理 messages 中孤立的 tool_calls ──
            ensure_messages_consistent(messages)

            # ── AI 调用（FallbackRunner 统一回退编排） ──
            message = None
            usage = None
            if runner:
                runner.reset_failures()

            while True:
                svc = runner.service() if runner else self.ai_service
                if svc is None:
                    _log.error("FallbackRunner: service() 返回 None，回退默认服务")
                    svc = self.ai_service
                current_model_name = runner.current if runner else None

                _log.debug(
                    "provider payload [%s..] round=%d inbound_message_ids=%s",
                    chat_id[:12],
                    round_idx + 1,
                    inbound_message_ids,
                )

                delivery = StreamDelivery(
                    chat_id=chat_id,
                    stream_callback=stream_callback,
                    message_delivered=lambda: message_delivered,
                    block_chars=self._stream_block_chars,
                    idle_seconds=self._stream_block_idle,
                )

                was_exception = False
                try:
                    # 协议已声明 chat_completion_stream，直接调用（不防御式探测）
                    if self._stream_reply:
                        cb = None
                        request_messages = list(messages)
                        if stream_callback is not None:
                            cb = delivery.callbacks
                        message, usage = await svc.chat_completion_stream(
                            messages=request_messages,
                            tools=tools,
                            callbacks=cb,
                        )
                    else:
                        message, usage = await svc.chat_completion_with_tools(
                            messages=list(messages),
                            tools=tools,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    was_exception = True
                    _log.warning(f"模型 [{current_model_name}] 调用异常: {e}")
                    message, usage = None, None
                    # 无条件取消当前 generation 的空闲定时器：即使尚未转发，timer 也引用本轮状态，
                    # 不取消会在 fallback 到下一模型后触发幽灵文本（双回复）
                    delivery.complete()
                    if delivery.forwarded:
                        # 已实时转发过部分文本：无法干净回退（会双回复），记冷却后终止
                        if runner:
                            await runner.mark_failure(record_cooldown=True)
                        _log.error(
                            "流式转发中途失败，已发送部分文本，不再回退: "
                            "model=%s chat=%s",
                            current_model_name,
                            chat_id[:12],
                        )
                        # 断流收尾：把已累积未发出的尾巴按块补发，不丢回复结尾。
                        # 已实时转发过部分文本无法干净回退（会双回复），
                        # 这里尽力把剩余文本送达；失败也不再重试。
                        await delivery.abort()
                        break

                if message is not None:
                    delivery.complete()
                    if runner:
                        await runner.mark_success(
                            mgr=binding_manager,
                            chat_id=chat_id,
                            tier=tier,
                        )
                    break

                # ── 失败 → 回退 ──
                if runner:
                    await runner.mark_failure(record_cooldown=was_exception)
                    if await runner.acquire():
                        if binding_manager and tier:
                            await binding_manager.bind(chat_id, tier, runner.current)
                        _log.warning(f"模型链剩余模型: {runner.remaining}，继续尝试")
                        continue

                    # 剩余链全在冷却/失败：走兜底
                    fallback_result = await runner.last_resort(messages, tools)
                    if fallback_result.ok:
                        message = fallback_result.message
                        usage = fallback_result.usage
                        current_model_name = fallback_result.model_name
                        if binding_manager and tier:
                            await binding_manager.bind(chat_id, tier, runner.current)
                        await runner.mark_success(
                            mgr=binding_manager,
                            chat_id=chat_id,
                            tier=tier,
                        )
                    else:
                        try:
                            await reply_callback(
                                chat_id,
                                "所有模型均不可用",
                                reply_to,
                                is_group,
                            )
                        except Exception as cb_err:
                            _log.warning(
                                "回复 callback 失败 [%s]: %s", chat_id[:12], cb_err
                            )
                        return sent_emoji, True
                    break
                else:
                    # 无模型链（使用默认 ai_service），尝试一次
                    break

            model_for_cost = current_model_name or self.ai_service.model
            if usage and self.cost_tracker:
                self.cost_tracker.record_turn(chat_id, model_for_cost, usage)

            # 流式已转发部分文本但调用异常：终止整个循环（部分文本已送达）
            if delivery.forwarded and message is None:
                delivery.complete()
                return sent_emoji, True

            if message is None:
                try:
                    await reply_callback(chat_id, "AI 服务异常", reply_to, is_group)
                except Exception as cb_err:
                    _log.warning("回复 callback 失败 [%s]: %s", chat_id[:12], cb_err)
                text_committed = True
                break

            response_text = message.content or ""
            tool_calls = message.tool_calls or []

            reasoning = message.reasoning_content
            if reasoning:
                _log.info(f"[工具循环 第{round_idx + 1}轮 思考过程]\n{reasoning}")

            _log.info(
                f"[工具循环 第{round_idx + 1}轮] "
                f"text={response_text[:50]!r}... "
                f"tool_calls={[tc.name for tc in tool_calls]}"
            )

            tool_calls_data = message.tool_calls_data

            # ── 预检：本轮是否有 heartbeat_respond(notify=false) ──
            for tc in tool_calls:
                if tc.name == "heartbeat_respond":
                    try:
                        tc_args = json.loads(tc.arguments)
                        if not tc_args.get("notify", True):
                            suppress_reply = True
                    except json.JSONDecodeError:
                        pass

            if response_text:
                if message_delivered:
                    _log.info(
                        f"[工具循环 第{round_idx + 1}轮] send_message 已投递，跳过后续文本发送"
                    )
                elif is_silent_reply_text(response_text) or suppress_reply:
                    if delivery.forwarded:
                        # 探测期内不应发生；若发生（超长静默追加），已转发部分无法撤回
                        _log.warning(
                            f"[工具循环 第{round_idx + 1}轮] 静默回复但流式已转发部分文本，"
                            "无法撤回"
                        )
                    else:
                        _log.info(f"[工具循环 第{round_idx + 1}轮] 静默回复，跳过发送")
                else:

                    async def _reply_remaining(content: str) -> None:
                        await reply_callback(
                            chat_id=chat_id,
                            content=content,
                            message_id=reply_to,
                            is_group=is_group,
                        )

                    await delivery.finish(response_text, _reply_remaining)
                    text_committed = text_committed or delivery.text_committed

            if reply_state_callback:
                await reply_state_callback(
                    not response_text
                    or is_silent_reply_text(response_text)
                    or suppress_reply
                    or message_delivered
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

            messages.append(message.to_wire())

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
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    _log.warning(
                        "工具参数解析失败: %s",
                        tc.arguments,
                    )
                    content = json.dumps({"error": "参数解析失败"})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": content,
                        }
                    )
                    try:
                        await self.context_manager.add_tool_result_async(
                            chat_id,
                            tc.name,
                            content,
                            tc.id,
                        )
                    except Exception as persist_err:
                        _log.warning(
                            "持久化参数解析失败结果 [%s]: %s",
                            tc.name,
                            persist_err,
                        )
                    continue

                try:
                    if tool_reply_callback and tc.name in (tool_reply_names or ()):
                        tool_ctx = ToolContext(
                            chat_id=ctx.chat_id,
                            is_group=ctx.is_group,
                            reply_to=ctx.reply_to,
                            sender_id=ctx.sender_id,
                            reply_callback=tool_reply_callback,
                            delivery_channel=ctx.delivery_channel,
                            reply_to_message_id=ctx.reply_to_message_id,
                        )
                    else:
                        tool_ctx = ctx
                    result = await execute_tool(tc.name, args, tool_ctx, self._perm)
                    content = result.content
                    if result.sent_emoji:
                        sent_emoji = True
                    if result.sent_text:
                        text_committed = True
                        message_delivered = True
                        if delivery_state_callback:
                            await delivery_state_callback()
                    if result.no_reply:
                        suppress_reply = True
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    _log.error(
                        "工具 [%s] 执行异常: %s",
                        tc.name,
                        e,
                        exc_info=True,
                    )
                    content = json.dumps(
                        {"error": f"执行异常: {e}"}, ensure_ascii=False
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": content,
                        }
                    )
                    try:
                        await self.context_manager.add_tool_result_async(
                            chat_id,
                            tc.name,
                            content,
                            tc.id,
                        )
                    except Exception as persist_err:
                        _log.warning(
                            "持久化工具结果失败 [%s]: %s",
                            tc.name,
                            persist_err,
                        )
                    continue

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )
                await self.context_manager.add_tool_result_async(
                    chat_id,
                    tc.name,
                    content,
                    tc.id,
                )

                # 在 tool 响应之后写入表情标记，避免插在 assistant(tc) 和 tool 之间
                if result.sent_emoji:
                    await self.context_manager.add_assistant_message_async(
                        chat_id,
                        "[助手发送了一个表情]",
                        reply_to,
                    )

            if steering_enabled and steering_admission_callback:
                steer_msgs = await self._drain_steering_messages(
                    chat_id=chat_id,
                    admission_callback=steering_admission_callback,
                    inbound_message_ids=inbound_message_ids,
                )
                if steer_msgs:
                    suppress_reply = False
                    message_delivered = False
                    messages.extend(steer_msgs)

        return sent_emoji, text_committed

    async def _drain_steering_messages(
        self,
        *,
        chat_id: str,
        admission_callback: Callable[[PendingInbound], Awaitable[Any]],
        inbound_message_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """Admit the leased pending batch after a complete tool batch only."""
        if not self.session_manager:
            return []

        lease = await self.session_manager.claim_pending_for_steer(chat_id)
        if lease is None:
            return []

        steered: List[dict] = []
        try:
            for pending in lease.items:

                async def _admit_and_commit(item: PendingInbound) -> Any:
                    admitted = await admission_callback(item)
                    await self.session_manager.commit(lease, item)
                    return admitted

                admission_task = asyncio.create_task(_admit_and_commit(pending))
                try:
                    admitted = await asyncio.shield(admission_task)
                except asyncio.CancelledError:
                    await asyncio.shield(admission_task)
                    raise
                if inbound_message_ids is not None:
                    inbound_message_ids.append(pending.message.id)
                if admitted is not None:
                    steered.append(admitted.prompt_message)
                    steered.extend(getattr(admitted, "additional_prompt_messages", ()))
        except asyncio.CancelledError:
            await self.session_manager.requeue_front(lease)
            raise
        except Exception:
            requeued = await self.session_manager.requeue_front(lease)
            _log.warning(
                "steering 准入失败，已恢复 %d 条消息 [%s..]",
                requeued,
                chat_id[:12],
                exc_info=True,
            )
        return steered
