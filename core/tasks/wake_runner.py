"""WakeRunner — wake 执行引擎。

职责：
1. 预检（含唤醒分类 + 10 步 skip chain）
2. 构建 AI messages（消费系统事件，构造 event prompt）
3. 调用 AgentEngine.run_wake_turn()
4. per-source delivery strategy dispatch
5. system event snapshot 消费

handler 签名: async (PendingWake) → WakeRunResult
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from core.engine.system_events import SystemEventBusy

from .delivery_strategy import DeliveryStrategy
from .preflight import PreflightContext, run_preflight
from .system_event_prompt import build_system_events_prompt
from .wake_coalescer import (
    SOURCE_CRON,
    SOURCE_EXEC,
    SOURCE_INTERVAL,
    SOURCE_MANUAL,
    SOURCE_TASK,
    PendingWake,
    WakeRunResult,
)

_log = logging.getLogger(__name__)


class WakeRunner:
    """Wake 执行引擎。作为 handler 注册到 WakeCoalescer。"""

    def __init__(
        self,
        agent_engine: Any,
        system_events: Any,
        cooldown: Any,
        delivery_strategies: Optional[dict[str, DeliveryStrategy]] = None,
        active_hours: tuple[Optional[str], Optional[str], Optional[str]] = (
            None,
            None,
            None,
        ),
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
        self._active_wake_tasks: dict[str, set[asyncio.Task]] = {}

    @staticmethod
    def _wake_key(pw: PendingWake) -> str:
        return pw.session_key or "heartbeat:events"

    def _is_session_active(self, session_key: str) -> bool:
        current_task = asyncio.current_task()
        active_tasks = self._active_wake_tasks.get(session_key)
        if active_tasks and any(task is not current_task for task in active_tasks):
            return True
        return self._session_active(session_key)

    def _release_event_lease(self, event_key: str) -> None:
        if not self._events:
            return
        try:
            release = getattr(self._events, "release_snapshot_for_session", None)
            if release:
                release(event_key)
        except Exception:
            _log.exception("释放系统事件 lease 失败 [%s]", event_key[:20])

    async def __call__(self, pw: PendingWake) -> WakeRunResult:
        task = asyncio.current_task()
        if task is None:
            return await self._run(pw)
        key = self._wake_key(pw)
        active_tasks = self._active_wake_tasks.setdefault(key, set())
        active_tasks.add(task)
        try:
            return await self._run(pw)
        finally:
            active_tasks.discard(task)
            if not active_tasks:
                self._active_wake_tasks.pop(key, None)

    async def _run(self, pw: PendingWake) -> WakeRunResult:
        """handler 接口（传给 set_wake_handler）。"""
        started = time.time()
        event_key = self._wake_key(pw)

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
            is_session_active=self._is_session_active(pw.session_key),
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
                pw.source,
                pw.intent,
                pf_result.skip_reason,
            )
            return WakeRunResult(
                status="skipped",
                skip_reason=pf_result.skip_reason,
                duration_ms=(time.time() - started) * 1000,
            )
        _log.info(
            "[WakeRunner] 预检通过: source=%s intent=%s → run_wake_turn()",
            pw.source,
            pw.intent,
        )

        if pw.source == SOURCE_TASK and not pw.work_plan_id:
            _log.warning(
                "[WakeRunner] 拒绝无 WorkPlan scope 的 task wake: session=%s",
                pw.session_key[:40],
            )
            return WakeRunResult(
                status="skipped",
                skip_reason="missing-work-plan-scope",
                duration_ms=(time.time() - started) * 1000,
            )

        # ── 2. 系统事件注入 ──
        work_plan_leases = []

        async def _release_work_plan_leases() -> None:
            if work_plan_leases:
                await self._agent.work_plan_service.settle_wake_inbox(
                    work_plan_leases, success=False
                )

        if pw.source == SOURCE_TASK:
            claim_wake_inbox = getattr(
                getattr(self._agent, "work_plan_service", None),
                "claim_wake_inbox",
                None,
            )
            if claim_wake_inbox is None:
                raise RuntimeError("work-plan inbox service is unavailable")
            work_plan_leases = await claim_wake_inbox(
                pw.session_key, work_plan_id=pw.work_plan_id
            )
            if not work_plan_leases:
                return WakeRunResult(
                    status="skipped",
                    skip_reason="no-pending-work-plan-events",
                    duration_ms=(time.time() - started) * 1000,
                )
            pw.extra_prompt = "\n\n".join(
                [pw.extra_prompt, *(lease.prompt for lease in work_plan_leases)]
            ).strip()

        planner_ids = {lease.owner_id for lease in work_plan_leases}
        planner_sender_id = (
            next(iter(planner_ids)) if len(planner_ids) == 1 else "system"
        )
        planner_lease_id = (
            work_plan_leases[0].lease_id if len(work_plan_leases) == 1 else ""
        )
        planner_plan_id = (
            work_plan_leases[0].plan_id if len(work_plan_leases) == 1 else ""
        )
        consumer_evidence_recorded = False
        consumer_evidence_callback = None
        if len(work_plan_leases) == 1:
            wake_lease = work_plan_leases[0]

            async def consumer_evidence_callback(action: str) -> None:
                nonlocal consumer_evidence_recorded
                await self._agent.work_plan_service.record_consumer_evidence(
                    wake_lease, action
                )
                consumer_evidence_recorded = True

        try:
            if pw.source in (SOURCE_INTERVAL, SOURCE_MANUAL):
                chat_id = pw.session_key
                if self._resolve_isolated_key:
                    chat_id = self._resolve_isolated_key(pw.session_key)
                messages, tools = (
                    await self._agent.prompt_builder.build_heartbeat_messages(
                        prompt=pw.extra_prompt,
                        system_prompt_mode=(
                            "minimal" if pw.source == "interval" else "normal"
                        ),
                        session_mode="isolated",
                        admin_chat_id=(
                            self._agent._admin_id[0] if self._agent._admin_id else ""
                        ),
                        chat_id=chat_id,
                        system_event_key=event_key,
                    )
                )
            elif pw.source == SOURCE_CRON:
                messages, tools = (
                    await self._agent.prompt_builder.build_system_event_messages(
                        prompt=pw.extra_prompt or "[系统事件]",
                        system_event_key=event_key,
                    )
                )
            elif pw.source == SOURCE_TASK:
                messages, tools = (
                    await self._agent.prompt_builder.build_system_event_messages(
                        prompt=pw.extra_prompt or "[系统事件]",
                        system_event_key=event_key,
                        work_plan_consumer=True,
                    )
                )
            elif pw.source == SOURCE_EXEC:
                messages, tools = (
                    await self._agent.prompt_builder.build_system_event_messages(
                        prompt=pw.extra_prompt or "[系统事件]",
                        system_event_key=event_key,
                    )
                )
            else:
                event_text = build_system_events_prompt(self._events, event_key) or ""
                base_prompt = pw.extra_prompt
                if event_text:
                    base_prompt = (
                        f"{base_prompt}\n\n{event_text}" if base_prompt else event_text
                    )
                is_group = (
                    self._agent.context_manager.get_chat_type(pw.session_key)
                    if self._agent.context_manager
                    else False
                ) or False
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
                    turn_id=msg.id,
                    timeline_snapshot=await self._agent._get_prompt_timeline_snapshot(
                        pw.session_key, current_turn_id=msg.id
                    ),
                    protocol_snapshot=(),
                )
        except SystemEventBusy:
            await _release_work_plan_leases()
            return WakeRunResult(
                status="skipped",
                skip_reason="requests-in-flight",
                duration_ms=(time.time() - started) * 1000,
            )
        except asyncio.CancelledError:
            self._release_event_lease(event_key)
            await _release_work_plan_leases()
            raise
        except Exception as e:
            self._release_event_lease(event_key)
            await _release_work_plan_leases()
            _log.error("wake prompt 构建异常 [%s]: %s", pw.source, e)
            raise

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
                planner_sender_id=planner_sender_id,
                planner_lease_id=planner_lease_id,
                planner_plan_id=planner_plan_id,
                consumer_evidence_callback=consumer_evidence_callback,
                work_plan_consumer=(pw.source == SOURCE_TASK),
                messages=messages,
                tools=tools,
            )
            turn_ok = not bool(getattr(result, "error", ""))
        except asyncio.CancelledError:
            self._release_event_lease(event_key)
            await _release_work_plan_leases()
            raise
        except Exception as e:
            _log.error("run_wake_turn 异常 [%s]: %s", pw.source, e)
            self._release_event_lease(event_key)
            await _release_work_plan_leases()
            raise

        if not turn_ok:
            try:
                self._release_event_lease(event_key)
            except Exception:
                pass
            await _release_work_plan_leases()
            _log.debug("AI turn 失败，保留系统事件快照供重试 [%s]", event_key[:20])
            raise RuntimeError(result.error or "wake turn failed")

        # ── 5. Delivery ──
        strategy_key = pw.source
        if pw.source == SOURCE_CRON and pw.session_key == "heartbeat:events":
            strategy_key = "cron-heartbeat"
        elif pw.source == SOURCE_EXEC:
            strategy_key = "cron-heartbeat"
        elif pw.source == SOURCE_TASK:
            strategy_key = "work-plan"
        strategy = self._delivery.get(strategy_key)
        if pw.source == SOURCE_TASK and strategy is None:
            await _release_work_plan_leases()
            raise RuntimeError("work-plan delivery strategy is unavailable")
        if strategy:
            try:
                await strategy.deliver(result, delivery_target=pw.delivery_target)
            except asyncio.CancelledError:
                self._release_event_lease(event_key)
                await _release_work_plan_leases()
                raise
            except Exception as e:
                self._release_event_lease(event_key)
                await _release_work_plan_leases()
                _log.error(
                    "[WakeRunner] delivery 异常: source=%s strategy=%s err=%s",
                    pw.source,
                    type(strategy).__name__,
                    e,
                )
                raise

        # ── 6. 消费系统事件快照（AI 和 delivery 均成功后） ──
        if self._events:
            try:
                self._events.consume_snapshot(event_key)
            except Exception:
                pass
        if work_plan_leases:
            if not consumer_evidence_recorded:
                await _release_work_plan_leases()
                raise RuntimeError("WorkPlan consumer produced no acknowledgement")
            await self._agent.work_plan_service.settle_wake_inbox(
                work_plan_leases, success=True
            )
        duration = (time.time() - started) * 1000
        status = "ran" if turn_ok else "failed"
        _log.info(
            "[WakeRunner] wake 完成: source=%s status=%s duration=%.0fms",
            pw.source,
            status,
            duration,
        )
        return WakeRunResult(
            status=status,
            skip_reason=str(result.error) if not turn_ok and result.error else "",
            duration_ms=duration,
            result=result,
        )
