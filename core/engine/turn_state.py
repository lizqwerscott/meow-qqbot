"""Immutable scheduler-owned lifecycle state for one conversation turn."""

from dataclasses import dataclass, replace
from enum import StrEnum

from core.managers.session_manager import InboundIntent


class TurnPhase(StrEnum):
    ACTIVE = "active"
    AWAITING_APPROVAL = "awaiting_approval"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TurnStateError(RuntimeError):
    """Raised when a stale or illegal scheduler lifecycle transition is requested."""


_ALLOWED_TRANSITIONS = {
    TurnPhase.ACTIVE: frozenset(
        {TurnPhase.AWAITING_APPROVAL, TurnPhase.FINALIZING, TurnPhase.CANCELLED}
    ),
    TurnPhase.AWAITING_APPROVAL: frozenset(
        {TurnPhase.ACTIVE, TurnPhase.FINALIZING, TurnPhase.CANCELLED}
    ),
    TurnPhase.FINALIZING: frozenset({TurnPhase.COMPLETED, TurnPhase.CANCELLED}),
    TurnPhase.COMPLETED: frozenset(),
    TurnPhase.CANCELLED: frozenset(),
}


@dataclass(frozen=True)
class TurnState:
    """One frozen turn identity plus a compare-and-swap lifecycle revision."""

    turn_id: str
    chat_id: str
    intent: InboundIntent
    principal_id: str
    queue_revision: int
    principal_role: str = ""
    phase: TurnPhase = TurnPhase.ACTIVE
    revision: int = 0
    cancellation_generation: int = 0
    approval_plan_id: str = ""
    task_anchor_message_id: str = ""
    task_correlation_id: str = ""

    def transition(
        self,
        phase: TurnPhase,
        *,
        expected_revision: int,
        approval_plan_id: str | None = None,
    ) -> "TurnState":
        if expected_revision != self.revision:
            raise TurnStateError(
                f"stale turn revision: expected={expected_revision} actual={self.revision}"
            )
        if phase not in _ALLOWED_TRANSITIONS[self.phase]:
            raise TurnStateError(f"illegal turn transition: {self.phase} -> {phase}")
        if phase is TurnPhase.AWAITING_APPROVAL and not approval_plan_id:
            raise TurnStateError("awaiting approval requires an approval plan id")
        if approval_plan_id is not None and phase is not TurnPhase.AWAITING_APPROVAL:
            raise TurnStateError(
                "approval plan id is only valid while awaiting approval"
            )
        if approval_plan_id is not None and self.phase is not TurnPhase.ACTIVE:
            raise TurnStateError(
                "approval plan can only be replaced from an active turn"
            )
        return replace(
            self,
            phase=phase,
            revision=self.revision + 1,
            cancellation_generation=(
                self.cancellation_generation + (phase is TurnPhase.CANCELLED)
            ),
            approval_plan_id=(
                approval_plan_id
                if approval_plan_id is not None
                else self.approval_plan_id
            ),
        )
