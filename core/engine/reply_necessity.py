"""Deterministic reply admission for group ambient turns.

The gate is deliberately independent from mode routing: it only decides whether
an already-admitted ambient batch deserves one Chat planner opportunity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Sequence


class ReplyAdmission(StrEnum):
    SKIP = "skip"
    ADMIT = "admit"
    DEFER = "defer"


@dataclass(frozen=True)
class ReplyNecessityInput:
    source: str
    chat_id: str
    batch: Sequence[object] = ()
    pending_count: int = 0
    active_chat: bool = False
    mode: str = "off"
    cooldown_until: float = 0.0
    window_budget_remaining: int = 1
    now: float = 0.0
    bot_presence_ratio: float = 0.0
    frequency_factor: float = 1.0
    explicit_wake: bool = False


@dataclass(frozen=True)
class ReplyNecessityDecision:
    admission: ReplyAdmission
    score: float | None
    threshold: float
    reason: str
    score_breakdown: tuple[tuple[str, float], ...] = ()

    @property
    def admitted(self) -> bool:
        return self.admission is ReplyAdmission.ADMIT


class ReplyNecessityGate:
    """Fail-closed, explainable two-stage ambient admission gate."""

    def __init__(
        self,
        *,
        threshold: float = 80,
        bot_presence_penalty: float = 20,
        frequency_factor: float = 1.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0 <= threshold <= 100:
            raise ValueError("threshold must be between 0 and 100")
        self.threshold = float(threshold)
        self.bot_presence_penalty = max(0.0, float(bot_presence_penalty))
        self.frequency_factor = max(0.0, min(1.0, float(frequency_factor)))
        self._clock = clock

    @staticmethod
    def _message(item: object) -> object:
        return getattr(item, "message", item)

    @classmethod
    def _text(cls, item: object) -> str:
        return str(getattr(cls._message(item), "content", "") or "").strip()

    @classmethod
    def _question(cls, text: str) -> bool:
        return "?" in text or "？" in text or text.endswith(("吗", "呢", "嘛"))

    @classmethod
    def _direct_request(cls, text: str) -> bool:
        return any(
            word in text.lower()
            for word in ("帮我", "请", "能不能", "please", "could you")
        )

    @classmethod
    def _opinion_request(cls, text: str) -> bool:
        return any(word in text for word in ("觉得", "怎么看", "建议", "意见", "推荐"))

    def evaluate(self, request: ReplyNecessityInput) -> ReplyNecessityDecision:
        now = request.now or self._clock()
        if request.source not in {
            "ambient",
            "group_ambient",
        }:
            return ReplyNecessityDecision(
                ReplyAdmission.SKIP, None, self.threshold, "not_ambient"
            )
        if request.explicit_wake:
            return ReplyNecessityDecision(
                ReplyAdmission.ADMIT, 100, self.threshold, "explicit_wake"
            )
        if request.mode != "active" or not request.active_chat:
            return ReplyNecessityDecision(
                ReplyAdmission.SKIP, None, self.threshold, "inactive_chat"
            )
        if request.pending_count <= 0 or not request.batch:
            return ReplyNecessityDecision(
                ReplyAdmission.SKIP, None, self.threshold, "no_pending"
            )
        if now >= request.cooldown_until and request.window_budget_remaining <= 0:
            return ReplyNecessityDecision(
                ReplyAdmission.DEFER, None, self.threshold, "budget_exhausted"
            )
        if now < request.cooldown_until:
            return ReplyNecessityDecision(
                ReplyAdmission.DEFER, None, self.threshold, "cooldown"
            )

        texts = [self._text(item) for item in request.batch]
        text = " ".join(texts)
        breakdown: list[tuple[str, float]] = []
        if self._question(text):
            breakdown.append(("question", 15))
        if self._direct_request(text):
            breakdown.append(("direct_request", 20))
        if self._opinion_request(text):
            breakdown.append(("opinion_request", 20))
        length = len(text)
        if length >= 80:
            breakdown.append(("text_length", 10))
        elif length >= 20:
            breakdown.append(("text_length", 5))
        if request.pending_count > 1:
            breakdown.append(
                ("pending_pressure", min(20, 5 * (request.pending_count - 1)))
            )
        if length <= 8 and not self._question(text):
            breakdown.append(("short_reaction", -25))
        if request.bot_presence_ratio > 0:
            breakdown.append(
                (
                    "recent_bot_presence",
                    -self.bot_presence_penalty * min(1.0, request.bot_presence_ratio),
                )
            )
        raw = max(0.0, min(100.0, sum(value for _, value in breakdown)))
        factor = max(0.0, min(1.0, request.frequency_factor * self.frequency_factor))
        score = raw * factor
        breakdown.append(("frequency_factor", factor))
        admission = (
            ReplyAdmission.ADMIT if score >= self.threshold else ReplyAdmission.SKIP
        )
        return ReplyNecessityDecision(
            admission,
            score,
            self.threshold,
            "threshold" if admission is ReplyAdmission.ADMIT else "below_threshold",
            tuple(breakdown),
        )
