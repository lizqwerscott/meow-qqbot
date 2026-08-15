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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from core.ai.fallback_runner import FallbackRunner
from core.ai.protocol import StreamCallbacks, ensure_messages_consistent
from core.managers.session_manager import PendingInbound
from core.markdown_split import (
    MARKDOWN_SAFE_CHUNK_BYTE_LIMIT,
    markdown_safe_cut,
    pending_starts_incomplete,
    trailing_structure,
    utf8_prefix,
    utf8len,
)
from core.tools._types import ToolContext
from core.tools.impl import execute as execute_tool

_log = logging.getLogger(__name__)

_SILENT_TOKENS = frozenset({"NO_REPLY", "HEARTBEAT_OK"})
_ACK_MAX_CHARS = 100
# 静默探测期：最长静默回复 = 最长 token (HEARTBEAT_OK=12) + _ACK_MAX_CHARS 追加字符。
# 累计文本（strip 后）不超过探测期时绝不转发，保证 NO_REPLY/HEARTBEAT_OK 不会被流式漏出。
_STREAM_PROBE_CHARS = _ACK_MAX_CHARS + 12


@dataclass
class _StreamBlockState:
    """block 流式转发状态（工具循环每轮新建，杜绝跨轮泄漏）。

    sent: 已承诺/已转发的字符数（相对最新累计文本的索引）
    forwarded: 是否转发过任何块
    text: 最新累计文本（on_text 回调更新）
    last_flush: 上次转发时刻 (monotonic)
    timer: 空闲 flush 任务（流结束/异常时须 cancel，防幽灵文本）
    """

    sent: int = 0
    forwarded: bool = False
    text: str = ""
    last_flush: float | None = None
    timer: asyncio.Task | None = field(default=None, repr=False, compare=False)
    # 是否有块正在发送（锁内 await 中）：发送期间抑制空闲定时器调度，
    # 避免多个 flush 任务排队等锁、锁释放后连发（破坏空闲分块节奏）。
    sending: bool = False
    # 串行化所有块的发送。空闲 flush 是后台任务：timer 在任务开始时已置 None，
    # 上一块 await 发送（QQ API 慢 / 重试退避 1s/2s/4s）期间 on_text 会调度新的
    # 空闲定时器 → 两个发送同时在途，后发的块可能先到 → 气泡乱序（线上事故：
    # 收尾句被插到列表中间）。锁保证同一轮内严格按生成顺序投递。
    flush_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )

    def cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


def _is_silent_reply_text(text: str) -> bool:
    stripped = text.strip().strip("`").strip()
    if not stripped:
        return True
    for token in _SILENT_TOKENS:
        if stripped == token:
            return True
        if stripped.startswith(token):
            remaining = stripped[len(token) :].lstrip("`").strip("：:，, \t")
            if not remaining or len(remaining) < _ACK_MAX_CHARS:
                return True
    return False


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

                # ── 流式转发状态（每轮新建，杜绝跨轮泄漏） ──
                st = _StreamBlockState()

                async def _flush_stream_block(
                    allow_partial: bool = False, flush_incomplete: bool = False
                ) -> None:
                    """发送当前累积的 pending 文本（block 投递）。

                    allow_partial=True（达块大小）：用 markdown 安全切点切块，
                    不切断表格/代码围栏；allow_partial=False（空闲 flush）：
                    pending 以未完成结构开头时跳过（等流继续，收尾补发兜底）。

                    flush_incomplete=True（断流收尾补发）：流已死不再增长，
                    无视结构完整性直接发出剩余文本（否则 cut==0 跳过会死循环）。

                    先承诺后发送：取消竞态下（await 中注入 CancelledError）消息可能已
                    送达，sent 先递增可防止收尾补发把同一段再发一遍。

                    注意：stream_callback 在持有 flush_lock 时被 await——回调不得
                    同步调用本函数（asyncio.Lock 不可重入，会死锁）。
                    """
                    # 串行化：同一轮内只允许一个块在途。空闲 flush 是后台任务，
                    # 不持锁会与后续 flush 并发 → 后发先至、气泡乱序（收尾句
                    # 插到列表中间的线上事故）。on_text 的块 flush / 空闲 flush /
                    # 断流收尾补发都走这一把锁。
                    async with st.flush_lock:
                        nonlocal text_committed
                        pending = st.text[st.sent :]
                        if not pending or message_delivered:
                            # message_delivered: send_message 等工具已投递过消息，
                            # 流式文本与工具投递重复，跳过（收尾补发同样会被拦截）
                            return
                        if not allow_partial and pending_starts_incomplete(
                            pending, st.text[: st.sent]
                        ):
                            return
                        # 统一找安全切点：达块大小按 limit 切；空闲 flush/首块按当前全文
                        # 扫描（末尾是半截表头/半截行时切回结构前，宁可少发）
                        limit = (
                            self._stream_block_chars if allow_partial else len(pending)
                        )
                        cut = markdown_safe_cut(
                            pending,
                            limit,
                            initial=trailing_structure(st.text[: st.sent]),
                        )
                        if cut == 0 and not flush_incomplete:
                            # 整个 pending 都在列表项/表格/围栏内（无安全切点）：
                            # 未达单条字节上限 → 跳过本次发送，等安全切点出现
                            # （项后空行/新标记/围栏闭合）或收尾补发兜底——避免
                            # 项内/中线切块（QQ 上列表项断成半截的根因之一）。
                            # 已达字节上限 → 退化为行尾硬切（宁超块大小也不
                            # 无限持有）；无换行可用（单行超限）时按 UTF-8 安全
                            # 字节边界切——超限整行整段发出会被 SDK 截断（内容
                            # 丢失），中线切块是上限约束下的唯一出路。
                            if utf8len(pending) < MARKDOWN_SAFE_CHUNK_BYTE_LIMIT:
                                return
                            cut = pending.rfind("\n") + 1
                            if cut <= 0:
                                cut = utf8_prefix(
                                    pending, MARKDOWN_SAFE_CHUNK_BYTE_LIMIT
                                )
                        if 0 < cut < len(pending):
                            pending = pending[:cut]
                        st.forwarded = True
                        # 先承诺后发送：await 之前就推进 sent。
                        # 重复气泡根因（18:59 线上事故）：旧实现发送后才推进，而
                        # _idle_flush_task 在 flush 开始前已把 timer 置 None——发送
                        # 在途期间 on_text 会再次调度空闲定时器，新 flush 以旧偏移
                        # 二次捕获同一段文本（逐字节相同的重复气泡）；on_reset 在
                        # 发送期间触发同样会把偏移算错（0 + len(pending)）。
                        # 发送前推进后：任何并发 on_text / on_reset / 收尾补发看到的
                        # 偏移都已是「本块已投递」，同一段文本不可能被再次捕获。
                        # 发送失败/取消不回退：已承诺的文本无法撤回（与旧语义一致）。
                        st.sent += len(pending)
                        text_committed = True
                        st.sending = True
                        try:
                            await stream_callback(pending)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            # 发送失败：与 reply_callback 失败同等对待（记录、不重发），
                            # sent 已推进，后续 flush / 收尾补发不会重发同段。
                            _log.warning("流式块发送失败 [%s]: %s", chat_id[:12], e)
                        finally:
                            st.sending = False
                        # 空闲间隔从「发送完成」起算：排队 flush 不按发送开始时间
                        # 计算延迟（避免锁释放后连发）
                        st.last_flush = time.monotonic()

                async def _idle_flush_task(delay: float) -> None:
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        return
                    st.timer = None
                    try:
                        await _flush_stream_block()
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        _log.warning("空闲 flush 失败 [%s]: %s", chat_id[:12], e)

                def _schedule_idle_flush() -> None:
                    """安排空闲定时器：距上次转发 ≥ idle 时强制发一块（首块立即发）。

                    发送在途（st.sending）时不调度——等待中的 flush 会在锁释放后
                    连发，破坏空闲分块节奏；等本次发送完成后的增量再起定时器。
                    """
                    if st.timer is not None or st.sending:
                        return
                    now = time.monotonic()
                    last = st.last_flush or (now - self._stream_block_idle)
                    delay = max(0.0, self._stream_block_idle - (now - last))
                    st.timer = asyncio.create_task(_idle_flush_task(delay))

                async def _on_stream_text(text_so_far: str) -> None:
                    """block 聚合：达块大小立即发，否则空闲超时触发；静默探测期内不发。"""
                    st.text = text_so_far
                    pending = text_so_far[st.sent :]
                    if not pending:
                        return
                    # 静默探测期（对 strip 后长度判定，防前导空白绕过）：绝不转发
                    if len(text_so_far.strip()) <= _STREAM_PROBE_CHARS:
                        return
                    if len(pending) >= self._stream_block_chars:
                        await _flush_stream_block(allow_partial=True)
                    else:
                        _schedule_idle_flush()

                async def _on_stream_reset() -> None:
                    """服务内部降级重试（全新生成）：转发状态归零。

                    首尝试的增量可能已流过 on_text，st.sent 停在旧文本偏移上；
                    不归零的话新文本会从错误偏移切片（开头被跳过或整条被吞）。
                    注意：不重置 st.forwarded——旧尝试已转发的块无法撤回，
                    后续若再断流仍需按「已转发」终止而非回退（防双回复）。
                    """
                    st.cancel_timer()
                    st.sent = 0
                    st.text = ""

                was_exception = False
                try:
                    # 协议已声明 chat_completion_stream，直接调用（不防御式探测）
                    if self._stream_reply:
                        cb = None
                        request_messages = list(messages)
                        if stream_callback is not None:
                            cb = StreamCallbacks(
                                on_text=_on_stream_text,
                                on_reset=_on_stream_reset,
                            )
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
                    # 无条件取消空闲定时器：即使尚未转发，timer 也引用本轮的 st，
                    # 不取消会在 fallback 到下一模型后触发幽灵文本（双回复）
                    st.cancel_timer()
                    if st.forwarded:
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
                        try:
                            while st.sent < len(st.text) and not message_delivered:
                                await _flush_stream_block(
                                    allow_partial=True, flush_incomplete=True
                                )
                        except asyncio.CancelledError:
                            raise
                        except Exception as cb_err:
                            _log.warning(
                                "断流收尾补发失败 [%s]: %s", chat_id[:12], cb_err
                            )
                        break

                if message is not None:
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
            if st.forwarded and message is None:
                st.cancel_timer()
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

            # 流已结束：取消挂起的空闲定时器，剩余文本由下方收尾逻辑一次性补发
            st.cancel_timer()

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
                elif _is_silent_reply_text(response_text) or suppress_reply:
                    if st.forwarded:
                        # 探测期内不应发生；若发生（超长静默追加），已转发部分无法撤回
                        _log.warning(
                            f"[工具循环 第{round_idx + 1}轮] 静默回复但流式已转发部分文本，"
                            "无法撤回"
                        )
                    else:
                        _log.info(f"[工具循环 第{round_idx + 1}轮] 静默回复，跳过发送")
                else:
                    # 收尾补发也走同一把锁：此刻可能有空闲 flush 仍在途
                    # （await stream_callback 未返回），不串行化会再次乱序。
                    # （同 _flush_stream_block：锁内 await reply_callback，
                    # 回调不得同步重入本函数——锁不可重入会死锁。）
                    async with st.flush_lock:
                        # 流式已转发前缀 → 只发剩余部分，避免重复
                        remaining = response_text[st.sent :]
                        if remaining:
                            try:
                                await reply_callback(
                                    chat_id=chat_id,
                                    content=remaining,
                                    message_id=reply_to,
                                    is_group=is_group,
                                )
                            except Exception as cb_err:
                                _log.warning(
                                    "回复 callback 失败 [%s]: %s",
                                    chat_id[:12],
                                    cb_err,
                                )
                        # 同步 sent：即使补发走了 reply_callback，也标记已投递，
                        # 防止残留定时器 flush 再次发送同段文本
                        st.sent = len(response_text)
                        text_committed = True

            if reply_state_callback:
                await reply_state_callback(
                    not response_text
                    or _is_silent_reply_text(response_text)
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
