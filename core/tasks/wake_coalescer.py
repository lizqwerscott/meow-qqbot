"""WakeCoalescer — module-level singleton 合并调度层。

与 OpenClaw heartbeat-wake.ts 对应，纯合并职责：
1. 接收请求 → 优先级合并 → coalesce 延迟 → 调用 handler
2. handler 返回 retryable skip → 自动重试
3. manual/manual → 同步路径 execute_immediate() 立即返回结果

不持有 system_events / agent_engine / cooldown / active_hours。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

Callback = Callable[[], None]

_log = logging.getLogger(__name__)

# ── 常量 ──

SOURCE_INTERVAL = "interval"
SOURCE_MANUAL = "manual"
SOURCE_EXEC = "exec-event"
SOURCE_CRON = "cron"
SOURCE_TASK = "background-task"
SOURCE_SYSTEM = "system"
SOURCE_RETRY = "retry"

WakeSource = str

INTENT_MANUAL = "manual"  # priority 0 — 最高
INTENT_IMMEDIATE = "immediate"  # priority 1
INTENT_EVENT = "event"  # priority 2
INTENT_SCHEDULED = "scheduled"  # priority 3 — 最低

WakeIntent = str

_INTENT_PRIORITY = {
    INTENT_MANUAL: 0,
    INTENT_IMMEDIATE: 1,
    INTENT_EVENT: 2,
    INTENT_SCHEDULED: 3,
}

RETRYABLE_SKIP_REASONS = frozenset(
    {
        "requests-in-flight",
        "cron-in-progress",
        "lanes-busy",
    }
)

# ── 数据类型 ──


@dataclass
class PendingWake:
    source: WakeSource = SOURCE_SYSTEM
    intent: WakeIntent = INTENT_SCHEDULED
    reason: str = ""
    agent_id: str = ""
    session_key: str = ""
    delivery_target: str = ""
    extra_prompt: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class WakeRunResult:
    status: str = "ran"  # "ran" | "skipped" | "failed"
    skip_reason: str = ""
    duration_ms: float = 0
    result: Any = None


@dataclass
class WakeTurnResult:
    """run_wake_turn 的返回容器（AI turn 完成后的结果）。"""

    should_notify: bool = False
    notification_text: str = ""
    deliver_to_user: str = ""  # 非空时投递到该 chat_id，否则 DM 管理员
    captured_replies: list = field(default_factory=list)
    error: str = ""
    turn_id: str = ""


WakeHandler = Callable[[PendingWake], "Awaitable[WakeRunResult]"]

# ── Module-level state ──

_handler: Optional[WakeHandler] = None
_handler_generation: int = 0
_pending: dict[str, PendingWake] = {}
_retry_count: dict[str, int] = {}
_timer: Optional[asyncio.TimerHandle] = None
_timer_due_at: float = 0
_running: bool = False
_loop: Optional[asyncio.AbstractEventLoop] = None

DEFAULT_COALESCE_MS = 250
DEFAULT_RETRY_MS = 1000
MAX_RETRY_COUNT = 10
RETRY_EXHAUSTED_MS = 30_000

# ── 内部 ──


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        try:
            _loop = asyncio.get_running_loop()
        except RuntimeError:
            _loop = asyncio.get_event_loop()
    return _loop


def _target_key(pw: PendingWake) -> str:
    return f"{pw.agent_id or ''}::{pw.session_key or ''}"


def _merge_pending(key: str, pw: PendingWake) -> None:
    existing = _pending.get(key)
    if not existing:
        _log.debug(
            "[Coalescer] 新 pending: key=%s source=%s intent=%s",
            key[:24],
            pw.source,
            pw.intent,
        )
        _pending[key] = pw
        return
    new_prio = _INTENT_PRIORITY.get(pw.intent, 99)
    old_prio = _INTENT_PRIORITY.get(existing.intent, 99)
    if existing.source == SOURCE_EXEC or pw.source == SOURCE_EXEC:
        prompts = [
            prompt for prompt in (existing.extra_prompt, pw.extra_prompt) if prompt
        ]
        if prompts:
            pw.extra_prompt = "\n\n".join(dict.fromkeys(prompts))
    if new_prio < old_prio or (
        new_prio == old_prio and pw.timestamp >= existing.timestamp
    ):
        _log.debug(
            "[Coalescer] 合并覆盖: key=%s old=%s(prio=%d) new=%s(prio=%d)",
            key[:24],
            existing.intent,
            old_prio,
            pw.intent,
            new_prio,
        )
        _pending[key] = pw


async def _drain_pending() -> None:
    global _running
    gen = _handler_generation
    _running = True
    batch: list[PendingWake] = []
    current_index = 0
    try:
        batch = list(_pending.values())
        _pending.clear()
        _log.info("[Coalescer] _drain_pending: batch=%d", len(batch))
        for idx, pw in enumerate(batch):
            current_index = idx
            if _handler_generation != gen:
                # handler 已替换，剩余未处理项重新入队
                remaining = batch[idx:]
                for rpw in remaining:
                    _merge_pending(_target_key(rpw), rpw)
                _log.warning(
                    "[Coalescer] handler 已替换，重新入队 %d 项 pending",
                    len(remaining),
                )
                break
            if not _handler:
                continue
            try:
                result = await _handler(pw)
            except Exception as e:
                _log.error("[Coalescer] handler 异常 [%s]: %s", pw.session_key[:12], e)
                # 重试入队
                key = _target_key(pw)
                retries = _retry_count.get(key, 0) + 1
                if retries > MAX_RETRY_COUNT:
                    _log.warning(
                        "[Coalescer] handler 重试超限: key=%s retries=%d → 延长重试间隔",
                        key[:24],
                        retries - 1,
                    )
                    _retry_count[key] = MAX_RETRY_COUNT
                    pw.timestamp = time.time()
                    _merge_pending(key, pw)
                    _schedule(RETRY_EXHAUSTED_MS)
                else:
                    _retry_count[key] = retries
                    pw.timestamp = time.time()
                    _merge_pending(key, pw)
                    _log.info(
                        "[Coalescer] handler 异常重试: key=%s retry=%d/%d → 1000ms 后",
                        key[:24],
                        retries,
                        MAX_RETRY_COUNT,
                    )
                    _schedule(DEFAULT_RETRY_MS)
                continue
            if (
                result.status == "skipped"
                and result.skip_reason in RETRYABLE_SKIP_REASONS
            ):
                key = _target_key(pw)
                retries = _retry_count.get(key, 0) + 1
                if retries > MAX_RETRY_COUNT:
                    _log.warning(
                        "[Coalescer] 重试超限: key=%s source=%s intent=%s retries=%d → 延长重试间隔",
                        key[:24],
                        pw.source,
                        pw.intent,
                        retries - 1,
                    )
                    _retry_count[key] = MAX_RETRY_COUNT
                    pw.timestamp = time.time()
                    _merge_pending(key, pw)
                    _schedule(RETRY_EXHAUSTED_MS)
                    continue
                _retry_count[key] = retries
                pw.timestamp = time.time()  # 更新重试时间戳，避免被新请求覆盖
                _merge_pending(key, pw)
                _log.info(
                    "[Coalescer] retryable skip: key=%s reason=%s retry=%d/%d → 1000ms 后重试",
                    key[:24],
                    result.skip_reason,
                    retries,
                    MAX_RETRY_COUNT,
                )
                _schedule(DEFAULT_RETRY_MS)
            else:
                key = _target_key(pw)
                _retry_count.pop(key, None)
    except asyncio.CancelledError:
        for pending in batch[current_index:]:
            _merge_pending(_target_key(pending), pending)
        raise
    finally:
        _running = False
        if _pending:
            _log.debug(
                "[Coalescer] _drain_pending: 仍有 pending %d 项，继续调度",
                len(_pending),
            )
            _schedule(DEFAULT_COALESCE_MS)
        else:
            _log.info("[Coalescer] _drain_pending 完成: 无剩余 pending")


def _schedule(coalesce_ms: int) -> None:
    global _timer, _timer_due_at
    now = time.time()
    due_at = now + coalesce_ms / 1000
    if _timer and _timer_due_at <= due_at:
        _log.debug(
            "[Coalescer] 跳过调度: 已有更早的 timer (due_at=%.3f)", _timer_due_at
        )
        return
    if _timer:
        _log.debug("[Coalescer] 取消旧 timer, 重设为 %.0fms 后", coalesce_ms)
        _timer.cancel()
    loop = _ensure_loop()
    _timer_due_at = due_at

    handle: asyncio.TimerHandle

    def fire() -> None:
        global _timer, _timer_due_at
        if _timer is not handle:
            return
        _timer = None
        _timer_due_at = 0
        asyncio.ensure_future(_drain_pending(), loop=loop)

    handle = loop.call_later(coalesce_ms / 1000, fire)
    _timer = handle
    _log.debug("[Coalescer] 调度 _drain_pending: %dms 后", coalesce_ms)


# ── 公开 API ──


def request_wake(
    source: WakeSource = SOURCE_SYSTEM,
    intent: WakeIntent = INTENT_EVENT,
    *,
    reason: str = "",
    agent_id: str = "",
    session_key: str = "",
    delivery_target: str = "",
    extra_prompt: str = "",
    coalesce_ms: int = DEFAULT_COALESCE_MS,
) -> None:
    """标准入口 — fire and forget。所有来源的合并调度。"""
    if not session_key and not agent_id:
        return
    pw = PendingWake(
        source=source,
        intent=intent,
        reason=reason,
        agent_id=agent_id,
        session_key=session_key,
        delivery_target=delivery_target or session_key,
        extra_prompt=extra_prompt,
    )
    key = _target_key(pw)
    _log.info(
        "[Coalescer] 收到 wake: source=%s intent=%s reason=%s session=%s has_prompt=%s",
        source,
        intent,
        reason,
        session_key[:20],
        bool(extra_prompt),
    )
    _merge_pending(key, pw)
    _schedule(coalesce_ms)


async def execute_immediate(
    source: WakeSource,
    intent: WakeIntent,
    *,
    reason: str = "",
    session_key: str = "",
    delivery_target: str = "",
    extra_prompt: str = "",
) -> WakeRunResult:
    """同步路径 — 跳过合并，立即由 handler 执行并返回结果。

    用于 manual/manual 场景（猫猫心跳 命令需要返回值）。
    """
    pw = PendingWake(
        source=source,
        intent=intent,
        reason=reason,
        session_key=session_key,
        delivery_target=delivery_target or session_key,
        extra_prompt=extra_prompt,
    )
    if _handler:
        return await _handler(pw)
    return WakeRunResult(status="failed", skip_reason="no handler")


def set_wake_handler(handler: Optional[WakeHandler]) -> Callback:
    """注册 handler。返回 disposer 函数，调用可安全清除本次注册。"""
    global _handler, _handler_generation, _timer, _timer_due_at, _running
    _handler_generation += 1
    generation = _handler_generation
    _handler = handler
    _retry_count.clear()
    if _timer:
        _timer.cancel()
        _timer = None
        _timer_due_at = 0
    _running = False
    if handler and _pending:
        _schedule(DEFAULT_COALESCE_MS)

    def dispose() -> None:
        global _handler, _handler_generation
        if _handler_generation != generation:
            return  # stale disposer — 新 handler 已注册
        _handler_generation += 1
        _handler = None

    return dispose


def clear_pending() -> None:
    global _timer, _timer_due_at
    count = len(_pending)
    _pending.clear()
    _retry_count.clear()
    if _timer:
        _timer.cancel()
        _timer = None
        _timer_due_at = 0
    if count:
        _log.info("[Coalescer] clear_pending: 清除了 %d 项 pending", count)


def get_status() -> dict:
    return {
        "pending": list(_pending.keys()),
        "pending_count": len(_pending),
        "running": _running,
        "has_handler": _handler is not None,
        "retry_count": dict(_retry_count),
    }
