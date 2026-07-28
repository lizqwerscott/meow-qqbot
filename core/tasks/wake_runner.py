"""WakeRunner — wake 执行引擎。

职责：
1. 预检（含唤醒分类 + 10 步 skip chain）
2. 构建 AI messages（消费系统事件，构造 event prompt）
3. 调用 AgentEngine.run_wake_turn()
4. per-source delivery strategy dispatch
5. system event snapshot 消费

handler 签名: async (PendingWake) → WakeRunResult
"""

import logging
import time
from typing import Any, Callable, Optional

from .wake_coalescer import (
    PendingWake, WakeRunResult,
    SOURCE_INTERVAL, SOURCE_MANUAL,
    SOURCE_EXEC, SOURCE_CRON, SOURCE_TASK,
    WakeTurnResult,
)
from .preflight import PreflightContext, PreflightResult, run_preflight
from .system_event_prompt import build_system_events_prompt
from .delivery_strategy import DeliveryStrategy

_log = logging.getLogger(__name__)


class WakeRunner:
    """Wake 执行引擎。作为 handler 注册到 WakeCoalescer。"""

    def __init__(
        self,
        agent_engine: Any,
        system_events: Any,
        cooldown: Any,
        delivery_strategies: Optional[dict[str, DeliveryStrategy]] = None,
        active_hours: tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
        session_active_check: Optional[Callable[[str], bool]] = None,
        has_cron_check: Optional[Callable[[], bool]] = None,
        # C2: 新增跳过检查
        main_lane_busy_check: Optional[Callable[[], bool]] = None,
        agent_busy_check: Optional[Callable[[], bool]] = None,
        skip_when_busy: bool = False,
        session_lane_busy_check: Optional[Callable[[str], bool]] = None,
        # C5: session isolation
        isolated_session_key_fn: Optional[Callable[[str], str]] = None,
        # #3: pending delivery deferral
        delivery_pending_check: Optional[Callable[[], bool]] = None,
    ):
        self._agent = agent_engine
        self._events = system_events
        self._cooldown = cooldown
        self._delivery = delivery_strategies or {}
        self._active_hours = active_hours
        self._session_active = session_active_check or (lambda _: False)
        self._has_cron = has_cron_check or (lambda: False)
        self._main_lane_busy = main_lane_busy_check or (lambda: False)
        self._agent_busy = agent_busy_check or (lambda: False)
        self._skip_when_busy = skip_when_busy
        self._session_lane_busy = session_lane_busy_check or (lambda _: False)
        self._resolve_isolated_key = isolated_session_key_fn
        self._delivery_pending = delivery_pending_check or (lambda: False)

    async def __call__(self, pw: PendingWake) -> WakeRunResult:
        """handler 接口（传给 set_wake_handler）。"""
        started = time.time()
        event_key = pw.session_key or "heartbeat:events"

        # ── 1. Preflight ──
        preflight_events = bool(self._events and self._events.has_events(event_key))

        ctx = PreflightContext(
            source=pw.source,
            intent=pw.intent,
            session_key=pw.session_key,
            cooldown=self._cooldown,
            active_hours=self._active_hours,
            has_system_events=preflight_events,
            has_extra_prompt=bool(pw.extra_prompt),
            is_session_active=self._session_active(pw.session_key),
            has_cron_jobs=self._has_cron(),
            source_is_interval=(pw.source == SOURCE_INTERVAL),
            source_is_manual=(pw.source == SOURCE_MANUAL),
            source_is_exec=(pw.source == SOURCE_EXEC),
            source_is_cron=(pw.source == SOURCE_CRON),
            source_is_background=(pw.source == SOURCE_TASK),
            is_main_lane_busy=self._main_lane_busy(),
            is_agent_busy=self._agent_busy(),
            skip_when_busy=self._skip_when_busy,
            is_session_lane_busy=self._session_lane_busy(pw.session_key),
            is_delivery_pending=self._delivery_pending(),
        )
        pf_result = run_preflight(ctx)
        if pf_result.skip_reason:
            _log.info(
                "[WakeRunner] wake 跳过: source=%s intent=%s reason=%s",
                pw.source, pw.intent, pf_result.skip_reason,
            )
            return WakeRunResult(
                status="skipped",
                skip_reason=pf_result.skip_reason,
                duration_ms=(time.time() - started) * 1000,
            )
        _log.info("[WakeRunner] 预检通过: source=%s intent=%s → run_wake_turn()", pw.source, pw.intent)

        # ── 2. 系统事件注入 ──

        if pw.source in (SOURCE_INTERVAL, SOURCE_MANUAL):
            chat_id = pw.session_key
            if self._resolve_isolated_key:
                chat_id = self._resolve_isolated_key(pw.session_key)
            messages, tools = await self._agent.prompt_builder.build_heartbeat_messages(
                prompt=pw.extra_prompt,
                system_prompt_mode="minimal" if pw.source == "interval" else "normal",
                session_mode="isolated",
                admin_chat_id=self._agent._admin_id[0] if self._agent._admin_id else "",
                chat_id=chat_id,
                system_event_key=event_key,
            )
        else:
            event_text = build_system_events_prompt(self._events, event_key) or ""
            base_prompt = pw.extra_prompt
            if event_text:
                base_prompt = f"{base_prompt}\n\n{event_text}" if base_prompt else event_text
            is_group = (self._agent.context_manager.get_chat_type(pw.session_key)
                        if self._agent.context_manager else False) or False
            from core.message import InputMessage
            msg = InputMessage(
                id=f"wake_{pw.session_key}_{int(time.time())}",
                sender_id="system",
                chat_id=pw.session_key,
                content=base_prompt or "[系统事件]",
                is_group=is_group,
                is_at_mention=False,
            )
            messages, tools = await self._agent.prompt_builder.build(
                chat_id=pw.session_key,
                is_group=is_group,
                user_nickname="系统",
                sender_id="system",
                input_message=msg,
                cost_tracker=self._agent.cost_tracker,
            )

        # ── 3. 执行 AI turn ──
        turn_ok = False
        try:
            if self._cooldown:
                self._cooldown.record_run_start()
            result = await self._agent.run_wake_turn(
                source=pw.source,
                intent=pw.intent,
                reason=pw.reason,
                session_key=pw.session_key,
                delivery_target=pw.delivery_target,
                messages=messages,
                tools=tools,
            )
            turn_ok = True
        except Exception as e:
            _log.error("run_wake_turn 异常 [%s]: %s", pw.source, e)
            result = WakeTurnResult(error=str(e))

        # ── 5. 消费系统事件快照（仅在 AI 成功执行后） ──
        if self._events and turn_ok:
            try:
                self._events.consume_snapshot(event_key)
            except Exception:
                pass
        elif self._events and not turn_ok:
            _log.debug("AI turn 失败，保留系统事件快照供重试 [%s]", event_key[:20])

        # ── 6. Delivery ──
        if turn_ok:
            strategy = self._delivery.get(pw.source)
            if strategy:
                try:
                    await strategy.deliver(result, delivery_target=pw.delivery_target)
                except Exception as e:
                    _log.error("[WakeRunner] delivery 异常: source=%s strategy=%s err=%s", pw.source, type(strategy).__name__, e)

        duration = (time.time() - started) * 1000
        status = "ran" if turn_ok else "failed"
        _log.info(
            "[WakeRunner] wake 完成: source=%s status=%s duration=%.0fms",
            pw.source, status, duration,
        )
        return WakeRunResult(
            status=status,
            skip_reason=str(result.error) if not turn_ok and result.error else "",
            duration_ms=duration,
            result=result,
        )
