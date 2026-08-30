"""Durable audit state for scheduler-owned conversation turns.

This store deliberately keeps no prompt, provider response, or tool checkpoint. A
restart may report an interrupted turn but must never replay a side-effecting loop.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from core.engine.turn_state import TurnPhase, TurnState


@dataclass(frozen=True)
class PersistedTurnState:
    turn_id: str
    chat_id: str
    intent: str
    principal_id: str
    principal_role: str
    queue_revision: int
    phase: str
    revision: int
    cancellation_generation: int
    approval_plan_id: str
    task_anchor_message_id: str
    updated_at: float
    task_correlation_id: str = ""
    wait_count: int = 0
    wait_reason: str = ""
    wait_deadline: float | None = None
    interrupted_by_restart: bool = False
    delivery_ids: tuple[str, ...] = ()
    last_delivery_status: str = ""

    @classmethod
    def from_turn_state(cls, state: TurnState) -> "PersistedTurnState":
        return cls(
            turn_id=state.turn_id,
            chat_id=state.chat_id,
            intent=state.intent.value,
            principal_id=state.principal_id,
            principal_role=state.principal_role,
            queue_revision=state.queue_revision,
            phase=state.phase.value,
            revision=state.revision,
            cancellation_generation=state.cancellation_generation,
            approval_plan_id=state.approval_plan_id,
            task_anchor_message_id=state.task_anchor_message_id,
            updated_at=time.time(),
            task_correlation_id=state.task_correlation_id,
            wait_count=state.wait_count,
            wait_reason=state.wait_reason,
            wait_deadline=state.wait_deadline,
        )


class TaskStateStore:
    """Small atomic JSON store for turn lifecycle recovery and audit."""

    def __init__(self, data_dir: str = "data/tasks"):
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._path = path / "turn_states.json"
        self._lock = asyncio.Lock()
        self._states: dict[str, PersistedTurnState] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                self._states = {}
                return

            known_fields = {field.name for field in fields(PersistedTurnState)}
            loaded: dict[str, PersistedTurnState] = {}
            for raw_state in raw:
                if not isinstance(raw_state, dict) or not raw_state.get("turn_id"):
                    continue
                phase = str(raw_state.get("phase", ""))
                if phase not in {item.value for item in TurnPhase}:
                    continue
                delivery_ids = raw_state.get("delivery_ids", ())
                if not isinstance(delivery_ids, (list, tuple)):
                    delivery_ids = ()
                payload = {
                    key: value
                    for key, value in raw_state.items()
                    if key in known_fields
                }
                payload.update(
                    {
                        "principal_role": str(payload.get("principal_role", "")),
                        "task_correlation_id": str(
                            payload.get("task_correlation_id", "")
                        ),
                        "wait_count": int(payload.get("wait_count", 0)),
                        "wait_reason": str(payload.get("wait_reason", "")),
                        "wait_deadline": payload.get("wait_deadline"),
                        "delivery_ids": tuple(
                            str(delivery_id)
                            for delivery_id in delivery_ids
                            if delivery_id
                        ),
                        "phase": phase,
                    }
                )
                try:
                    state = PersistedTurnState(**payload)
                except (TypeError, ValueError):
                    continue
                loaded[state.turn_id] = state
            self._states = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            self._states = {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(state) for state in self._states.values()], indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    async def put(self, state: TurnState) -> None:
        async with self._lock:
            self._states[state.turn_id] = PersistedTurnState.from_turn_state(state)
            await asyncio.to_thread(self._save)

    async def mark_interrupted_on_restart(self) -> list[PersistedTurnState]:
        """Terminally record active work without replaying side-effecting loops.

        A persisted WAITING turn has no in-flight provider or tool call.  Keep it
        as a recovery hint until a matching later inbound message safely starts a
        new turn; all other nonterminal phases are cancelled immediately.
        """
        async with self._lock:
            interrupted = []
            for turn_id, state in tuple(self._states.items()):
                if state.phase in {
                    TurnPhase.COMPLETED.value,
                    TurnPhase.CANCELLED.value,
                    TurnPhase.WAITING.value,
                }:
                    continue
                updated = PersistedTurnState(
                    **{
                        **asdict(state),
                        "phase": TurnPhase.CANCELLED.value,
                        "revision": state.revision + 1,
                        "cancellation_generation": state.cancellation_generation + 1,
                        "updated_at": time.time(),
                        "interrupted_by_restart": True,
                    }
                )
                self._states[turn_id] = updated
                interrupted.append(updated)
            if interrupted:
                await asyncio.to_thread(self._save)
            return interrupted

    async def claim_waiting_recoveries(
        self,
        *,
        chat_id: str,
        principal_id: str,
        intent: str,
    ) -> list[PersistedTurnState]:
        """Consume restart-surviving waits that a new authorized message resumes.

        This deliberately returns audit metadata only. Callers must create a new
        scheduler turn for the new inbound message rather than replay the old
        turn, its provider request, or any prior tool authorization.
        """
        async with self._lock:
            claimed: list[PersistedTurnState] = []
            for turn_id, state in tuple(self._states.items()):
                if (
                    state.phase != TurnPhase.WAITING.value
                    or state.chat_id != chat_id
                    or state.principal_id != principal_id
                    or state.intent != intent
                ):
                    continue
                updated = PersistedTurnState(
                    **{
                        **asdict(state),
                        "phase": TurnPhase.CANCELLED.value,
                        "revision": state.revision + 1,
                        "cancellation_generation": state.cancellation_generation + 1,
                        "updated_at": time.time(),
                        "interrupted_by_restart": True,
                    }
                )
                self._states[turn_id] = updated
                claimed.append(updated)
            if claimed:
                await asyncio.to_thread(self._save)
            return claimed

    async def expire_waiting_turns(
        self, *, now: float | None = None
    ) -> list[PersistedTurnState]:
        """Terminate expired restart-surviving waits without replaying side effects."""
        current_time = time.time() if now is None else now
        async with self._lock:
            expired: list[PersistedTurnState] = []
            for turn_id, state in tuple(self._states.items()):
                if (
                    state.phase != TurnPhase.WAITING.value
                    or state.wait_deadline is None
                    or state.wait_deadline > current_time
                ):
                    continue
                updated = PersistedTurnState(
                    **{
                        **asdict(state),
                        "phase": TurnPhase.CANCELLED.value,
                        "revision": state.revision + 1,
                        "cancellation_generation": state.cancellation_generation + 1,
                        "updated_at": current_time,
                    }
                )
                self._states[turn_id] = updated
                expired.append(updated)
            if expired:
                await asyncio.to_thread(self._save)
            return expired

    async def record_delivery(
        self, turn_id: str, delivery_id: str, status: str
    ) -> None:
        """Attach ledger evidence to an existing turn without copying its content."""
        async with self._lock:
            state = self._states.get(turn_id)
            if state is None:
                return
            delivery_ids = tuple(dict.fromkeys((*state.delivery_ids, delivery_id)))
            self._states[turn_id] = PersistedTurnState(
                **{
                    **asdict(state),
                    "delivery_ids": delivery_ids,
                    "last_delivery_status": status,
                    "updated_at": time.time(),
                }
            )
            await asyncio.to_thread(self._save)

    def get(self, turn_id: str) -> Optional[PersistedTurnState]:
        return self._states.get(turn_id)
