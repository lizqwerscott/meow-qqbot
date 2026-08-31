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


@dataclass
class _SessionState:
    phase: EngagementPhase = EngagementPhase.IDLE
    generation: int = 0
    window_started: float = 0.0
    turns_in_window: int = 0
    cooldown_until: float = 0.0
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
    ):
        self.config = config
        self._clock = clock
        self._max_sessions = max(1, max_sessions)
        self._sessions: dict[str, _SessionState] = {}
        self._metrics: Counter[str] = Counter()

    def reconfigure(self, config: EngagementConfig) -> None:
        """Install a complete engagement snapshot for future decisions."""
        self.config = config

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

    async def start(self, decision: GroupEngagementDecision) -> bool:
        """Consume one budget turn exactly when the provider request starts."""
        state = self._sessions.get(decision.chat_id)
        now = self._clock()
        if state is None or state.reservation != decision or now >= decision.expires_at:
            return False
        if state.provider_started:
            return True
        state.provider_started = True
        state.turns_in_window += 1
        state.phase = EngagementPhase.THINKING
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
        state.cooldown_until = self._clock() + (
            self.config.group_ambient_cooldown_seconds
            if delivered and not silent
            else self.config.group_ambient_quiet_cooldown_seconds
        )
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
            self._metrics["active_reserved"] += 1
        if decision.shadow:
            self._metrics["shadow_candidates"] += 1

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
