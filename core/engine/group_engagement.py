"""Local gate and budget state for optional group ambient participation."""

import time
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional, Sequence

from core.engine.ambient_delivery import (
    AmbientDeliveryDecision,
    decide_ambient_delivery,
)
from core.engine.engagement_config import EngagementConfig
from core.engine.proactive_state import ProactiveStateStore


class EngagementTrigger(StrEnum):
    REACTIVE = "reactive"
    PROACTIVE = "proactive"


class EngagementPhase(StrEnum):
    IDLE = "idle"
    RESERVED = "reserved"
    THINKING = "thinking"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class GroupEngagementDecision:
    chat_id: str
    generation: int
    allowed: bool
    shadow: bool
    reason: str
    reply_anchor_id: str = ""
    expires_at: float = 0.0
    trigger: EngagementTrigger = EngagementTrigger.REACTIVE


@dataclass
class _SessionState:
    phase: EngagementPhase = EngagementPhase.IDLE
    generation: int = 0
    window_started: float = 0.0
    turns_in_window: int = 0
    cooldown_until: float = 0.0
    proactive_window_started: float = 0.0
    proactive_turns_in_window: int = 0
    proactive_cooldown_until: float = 0.0
    reservation: Optional[GroupEngagementDecision] = None
    provider_started: bool = False


class GroupEngagementManager:
    """Manage ambient participation without interpreting message semantics via LLM."""

    def __init__(
        self,
        config: EngagementConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_sessions: int = 1024,
        state_store: ProactiveStateStore | None = None,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self._clock = clock
        self._max_sessions = max(1, max_sessions)
        self._sessions: dict[str, _SessionState] = {}
        self._metrics: Counter[str] = Counter()
        self._state_store = state_store
        self._wall_clock = wall_clock
        self._loaded_proactive_chats: set[str] = set()
        self._metrics_hydrated = False

    async def _ensure_proactive_state(self, chat_id: str) -> _SessionState:
        now = self._clock()
        state = self._state(chat_id, now)
        if self._state_store is None or chat_id in self._loaded_proactive_chats:
            return state
        persisted = await self._state_store.get(chat_id)
        wall_now = self._wall_clock()
        if persisted.proactive_window_started_at > 0:
            elapsed = wall_now - persisted.proactive_window_started_at
            if elapsed >= self.config.group_proactive_window_seconds:
                state.proactive_window_started = now
                state.proactive_turns_in_window = 0
            else:
                state.proactive_window_started = now - max(0.0, elapsed)
                state.proactive_turns_in_window = persisted.proactive_turns_in_window
        else:
            state.proactive_window_started = now
            state.proactive_turns_in_window = 0
        cooldown_remaining = max(
            0.0, persisted.proactive_cooldown_until - wall_now
        )
        state.proactive_cooldown_until = (
            now + cooldown_remaining if cooldown_remaining > 0 else 0.0
        )
        self._loaded_proactive_chats.add(chat_id)
        return state

    async def _persist_proactive_state(self, chat_id: str, state: _SessionState) -> None:
        if self._state_store is None:
            return
        now = self._clock()
        wall_now = self._wall_clock()
        window_elapsed = max(0.0, now - state.proactive_window_started)
        cooldown_remaining = max(0.0, state.proactive_cooldown_until - now)
        await self._state_store.save(
            chat_id,
            proactive_window_started_at=wall_now - window_elapsed,
            proactive_turns_in_window=state.proactive_turns_in_window,
            proactive_cooldown_until=wall_now + cooldown_remaining,
        )

    def _state(self, chat_id: str, now: float) -> _SessionState:
        state = self._sessions.get(chat_id)
        if state is None:
            if len(self._sessions) >= self._max_sessions:
                self._sessions.pop(next(iter(self._sessions)))
            state = _SessionState(window_started=now)
            self._sessions[chat_id] = state
        if now - state.window_started >= self.config.group_ambient_window_seconds:
            state.window_started = now
            state.turns_in_window = 0
        if (
            state.proactive_window_started == 0.0
            or now - state.proactive_window_started
            >= self.config.group_proactive_window_seconds
        ):
            state.proactive_window_started = now
            state.proactive_turns_in_window = 0
        return state

    @staticmethod
    def _message(item):
        return getattr(item, "message", item)

    @classmethod
    def _content(cls, item) -> str:
        return str(getattr(cls._message(item), "content", "") or "")

    @classmethod
    def _message_id(cls, item) -> str:
        return str(getattr(cls._message(item), "id", "") or "")

    @classmethod
    def _has_media(cls, item) -> bool:
        message = cls._message(item)
        return bool(getattr(message, "resources", ()))

    @classmethod
    def _anchor(cls, batch: Sequence) -> str:
        text_items = [item for item in batch if cls._content(item).strip()]
        questions = [
            item
            for item in text_items
            if "?" in cls._content(item) or "？" in cls._content(item)
        ]
        return cls._message_id((questions or text_items or list(batch))[-1])

    def _candidate_reason(self, batch: Sequence) -> str:
        if not batch:
            return "empty_batch"
        if len(batch) >= self.config.group_ambient_min_messages:
            return "batch_threshold"
        if self.config.group_ambient_allow_single_question and any(
            "?" in self._content(item) or "？" in self._content(item) for item in batch
        ):
            return "single_question"
        if self.config.group_ambient_allow_single_media and len(batch) == 1:
            if self._has_media(batch[0]):
                return "single_media"
        return "below_threshold"

    async def evaluate(
        self, chat_id: str, *, batch: Sequence
    ) -> GroupEngagementDecision:
        now = self._clock()
        state = self._state(chat_id, now)
        state.generation += 1
        generation = state.generation
        anchor = self._anchor(batch)
        oldest = min(
            (float(getattr(item, "enqueued_at", now)) for item in batch),
            default=now,
        )
        expires_at = oldest + self.config.group_ambient_max_age_seconds

        if now >= expires_at and batch:
            return GroupEngagementDecision(
                chat_id, generation, False, False, "stale_batch", anchor, expires_at
            )
        if self.config.group_ambient_mode == "off":
            return GroupEngagementDecision(
                chat_id, generation, False, False, "disabled", anchor, expires_at
            )
        if self.config.group_ambient_mode == "shadow":
            reason = self._candidate_reason(batch)
            return GroupEngagementDecision(
                chat_id,
                generation,
                False,
                reason != "below_threshold",
                reason,
                anchor,
                expires_at,
            )
        if chat_id not in self.config.group_ambient_active_chats:
            return GroupEngagementDecision(
                chat_id,
                generation,
                False,
                False,
                "active_allowlist",
                anchor,
                expires_at,
            )
        if state.reservation is not None:
            return GroupEngagementDecision(
                chat_id,
                generation,
                False,
                False,
                "already_reserved",
                anchor,
                expires_at,
            )
        if now < state.cooldown_until:
            return GroupEngagementDecision(
                chat_id, generation, False, False, "cooldown", anchor, expires_at
            )
        reason = self._candidate_reason(batch)
        if reason == "below_threshold":
            return GroupEngagementDecision(
                chat_id, generation, False, False, reason, anchor, expires_at
            )
        if state.turns_in_window >= self.config.group_ambient_max_turns_per_window:
            return GroupEngagementDecision(
                chat_id,
                generation,
                False,
                False,
                "budget_exhausted",
                anchor,
                expires_at,
            )

        decision = GroupEngagementDecision(
            chat_id, generation, True, False, reason, anchor, expires_at
        )
        state.reservation = decision
        state.phase = EngagementPhase.RESERVED
        state.provider_started = False
        return decision

    async def reserve_proactive(self, chat_id: str) -> GroupEngagementDecision:
        """Reserve one proactive opportunity using an independent budget."""
        now = self._clock()
        state = await self._ensure_proactive_state(chat_id)
        await self._persist_proactive_state(chat_id, state)
        state.generation += 1
        generation = state.generation
        expires_at = now + self.config.group_proactive_reservation_seconds

        def denied(reason: str, *, shadow: bool = False):
            return GroupEngagementDecision(
                chat_id,
                generation,
                False,
                shadow,
                reason,
                "",
                expires_at,
                EngagementTrigger.PROACTIVE,
            )

        if self.config.group_proactive_mode == "off":
            return denied("proactive_disabled")
        if chat_id not in self.config.group_proactive_active_chats:
            return denied("proactive_allowlist")
        if state.reservation is not None:
            return denied("session_busy")
        if now < state.cooldown_until or now < state.proactive_cooldown_until:
            return denied("cooldown")
        if state.proactive_turns_in_window >= self.config.group_proactive_max_turns_per_window:
            return denied("proactive_budget_exhausted")

        shadow = self.config.group_proactive_mode == "shadow"
        decision = GroupEngagementDecision(
            chat_id,
            generation,
            not shadow,
            shadow,
            "proactive_candidate",
            "",
            expires_at,
            EngagementTrigger.PROACTIVE,
        )
        state.reservation = decision
        state.phase = EngagementPhase.RESERVED
        state.provider_started = False
        return decision
    async def start(self, decision: GroupEngagementDecision) -> bool:
        """Consume one budget turn exactly when the provider request starts."""
        state = self._sessions.get(decision.chat_id)
        now = self._clock()
        if state is None or state.reservation != decision or now >= decision.expires_at:
            return False
        state.provider_started = True
        if decision.trigger is EngagementTrigger.PROACTIVE:
            state.proactive_turns_in_window += 1
        else:
            state.turns_in_window += 1
        state.phase = EngagementPhase.THINKING
        if decision.trigger is EngagementTrigger.PROACTIVE:
            await self._persist_proactive_state(decision.chat_id, state)
        return True

    async def complete(
        self,
        decision: GroupEngagementDecision,
        *,
        delivered: bool,
        silent: bool,
    ) -> bool:
        state = self._sessions.get(decision.chat_id)
        if state is None or state.reservation != decision:
            return False
        state.reservation = None
        state.provider_started = False
        state.phase = EngagementPhase.COOLDOWN
        if decision.trigger is EngagementTrigger.PROACTIVE:
            state.proactive_cooldown_until = self._clock() + (
                self.config.group_proactive_cooldown_seconds
                if delivered and not silent
                else self.config.group_proactive_quiet_cooldown_seconds
            )
        else:
            state.cooldown_until = self._clock() + (
                self.config.group_ambient_cooldown_seconds
                if delivered and not silent
                else self.config.group_ambient_quiet_cooldown_seconds
            )
        if decision.trigger is EngagementTrigger.PROACTIVE:
            await self._persist_proactive_state(decision.chat_id, state)
        return True

    def decide_delivery(
        self,
        content: str | None,
        *,
        tool_delivered: bool = False,
        reply_anchor_id: str = "",
    ) -> AmbientDeliveryDecision:
        return decide_ambient_delivery(
            content,
            delivery_mode=self.config.group_ambient_delivery_mode,
            tool_delivered=tool_delivered,
            reply_anchor_id=reply_anchor_id,
        )

    def observe(self, decision: GroupEngagementDecision) -> None:
        """Record aggregate gate outcomes without retaining message content."""
        self._metrics[f"reason:{decision.reason}"] += 1
        if decision.allowed:
            self._metrics[
                "active_reserved"
                if decision.trigger is EngagementTrigger.REACTIVE
                else "proactive_reserved"
            ] += 1
        if decision.shadow:
            if decision.trigger is EngagementTrigger.PROACTIVE:
                self._metrics["proactive_shadow_candidates"] += 1
            else:
                self._metrics["shadow_candidates"] += 1

    async def observe_proactive(self, decision: GroupEngagementDecision) -> None:
        """Observe a proactive gate and persist its aggregate counters."""
        if decision.trigger is not EngagementTrigger.PROACTIVE:
            self.observe(decision)
            return
        if self._state_store is not None and not self._metrics_hydrated:
            self._metrics.update(
                await self._state_store.metric_totals("engagement")
            )
            self._metrics_hydrated = True
        self.observe(decision)
        if self._state_store is None:
            return
        await self._state_store.increment_metric(
            "engagement", f"reason:{decision.reason}"
        )
        if decision.allowed:
            await self._state_store.increment_metric(
                "engagement", "proactive_reserved"
            )
        if decision.shadow:
            await self._state_store.increment_metric(
                "engagement", "proactive_shadow_candidates"
            )

    def snapshot_metrics(self) -> dict[str, int]:
        return dict(self._metrics)


    def phase(self, chat_id: str) -> EngagementPhase:
        state = self._sessions.get(chat_id)
        return state.phase if state else EngagementPhase.IDLE

    def is_active_chat(self, chat_id: str) -> bool:
        return (
            self.config.group_ambient_mode == "active"
            and chat_id in self.config.group_ambient_active_chats
        )
