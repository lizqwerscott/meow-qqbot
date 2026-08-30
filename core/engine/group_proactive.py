"""Scheduled, opt-in proactive participation for configured group chats."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
from collections import Counter
from datetime import datetime
from typing import Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.engine.engagement_config import EngagementConfig
from core.engine.group_engagement import GroupEngagementDecision, GroupEngagementManager
from core.engine.proactive_state import ProactiveStateStore

_log = logging.getLogger(__name__)


class GroupProactiveScheduler:
    """Scan an explicit group allowlist and submit bounded proactive turns.

    This module owns timer lifecycle and scheduling policy only.  Reservation,
    cooldown, and budget decisions remain in ``GroupEngagementManager``; the
    callback owns prompt construction, model execution, and delivery.
    """

    def __init__(
        self,
        config: EngagementConfig,
        engagement: GroupEngagementManager,
        run_turn: Callable[[GroupEngagementDecision], Awaitable[object]],
        *,
        is_busy: Callable[[str], bool | Awaitable[bool]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        state_store: ProactiveStateStore | None = None,
    ):
        self.config = config
        self.engagement = engagement
        self._run_turn = run_turn
        self._is_busy = is_busy or (lambda _chat_id: False)
        self._clock = clock
        self._wall_clock = wall_clock
        self._task: asyncio.Task | None = None
        self._next_due: dict[str, float] = {}
        self._metrics: Counter[str] = Counter()
        self._state_store = state_store or getattr(engagement, "_state_store", None)
        self._loaded_due_chats: set[str] = set()
        self._metrics_hydrated = False
        self._zone = self._load_zone(config.group_proactive_timezone)

    @staticmethod
    def _load_zone(name: str):
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            _log.warning("proactive timezone 无效，回退 UTC: %s", name)
            return ZoneInfo("UTC")

    def _in_active_hours(self) -> bool:
        start = self._parse_minutes(self.config.group_proactive_active_hours_start)
        end = self._parse_minutes(self.config.group_proactive_active_hours_end)
        now = datetime.fromtimestamp(self._wall_clock(), self._zone)
        current = now.hour * 60 + now.minute
        if start == end:
            return True
        if start < end:
            return start <= current < end
        return current >= start or current < end

    @staticmethod
    def _parse_minutes(value: str) -> int:
        hour, minute = (int(part) for part in value.split(":"))
        return hour * 60 + minute

    @staticmethod
    def _jitter_seconds(chat_id: str, maximum: int) -> int:
        if maximum <= 0:
            return 0
        digest = hashlib.sha256(chat_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (maximum + 1)

    async def _call_busy(self, chat_id: str) -> bool:
        result = self._is_busy(chat_id)
        return bool(await result) if inspect.isawaitable(result) else bool(result)

    async def _metric(self, name: str) -> None:
        if self._state_store is not None and not self._metrics_hydrated:
            self._metrics.update(
                await self._state_store.metric_totals("scheduler")
            )
            self._metrics_hydrated = True
        self._metrics[name] += 1
        if self._state_store is not None:
            await self._state_store.increment_metric("scheduler", name)

    async def _restore_due(self, chat_id: str, now: float) -> None:
        if self._state_store is None or chat_id in self._loaded_due_chats:
            return
        persisted = await self._state_store.get(chat_id)
        wall_remaining = persisted.next_due_at - self._wall_clock()
        if wall_remaining > 0:
            self._next_due[chat_id] = now + wall_remaining
        self._loaded_due_chats.add(chat_id)

    async def tick_once(self) -> None:
        """Run one deterministic scan of the configured proactive allowlist."""
        if self.config.group_proactive_mode == "off":
            await self._metric("disabled")
            return
        if not self._in_active_hours():
            await self._metric("inactive_hours")
            return

        now = self._clock()
        for chat_id in self.config.group_proactive_active_chats:
            await self._restore_due(chat_id, now)
            due = self._next_due.get(chat_id)
            if due is not None and now < due:
                await self._metric("not_due")
                continue
            self._next_due[chat_id] = now + self.config.group_proactive_interval_seconds + self._jitter_seconds(
                chat_id, self.config.group_proactive_jitter_seconds
            )
            if self._state_store is not None:
                await self._state_store.set_next_due(
                    chat_id,
                    self._wall_clock()
                    + self.config.group_proactive_interval_seconds
                    + self._jitter_seconds(
                        chat_id, self.config.group_proactive_jitter_seconds
                    ),
                )
            if await self._call_busy(chat_id):
                await self._metric("session_busy")
                continue

            decision = await self.engagement.reserve_proactive(chat_id)
            await self.engagement.observe_proactive(decision)
            if decision.shadow:
                await self._metric("shadow")
                await self.engagement.complete(decision, delivered=False, silent=True)
                continue
            if not decision.allowed:
                await self._metric(f"skip:{decision.reason}")
                continue

            if not await self.engagement.start(decision):
                await self.engagement.complete(decision, delivered=False, silent=True)
                await self._metric("skip:reservation_expired")
                continue
            await self._metric("reserved")
            try:
                result = await self._run_turn(decision)
                delivered = bool(
                    getattr(result, "delivered", False)
                    or getattr(result, "text_committed", False)
                    or getattr(result, "tool_text_delivered", False)
                    or getattr(result, "sent_emoji", False)
                )
                silent = bool(
                    getattr(result, "silent", False)
                    or getattr(result, "final_reply_silent", False)
                    or not delivered
                )
                await self.engagement.complete(
                    decision, delivered=delivered, silent=silent
                )
                await self._metric("delivered" if delivered else "silent")
            except asyncio.CancelledError:
                await self.engagement.complete(decision, delivered=False, silent=True)
                raise
            except Exception:
                await self.engagement.complete(decision, delivered=False, silent=True)
                await self._metric("failed")
                _log.exception("proactive turn failed chat=%s", chat_id[:12])

    async def _run(self) -> None:
        try:
            while True:
                await self.tick_once()
                await asyncio.sleep(min(60.0, float(self.config.group_proactive_interval_seconds)))
        except asyncio.CancelledError:
            raise

    async def start(self) -> None:
        """Start the idempotent scheduler loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the scheduler loop without leaving a timer task behind."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def snapshot_metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def next_due(self) -> dict[str, float]:
        return dict(self._next_due)

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()
