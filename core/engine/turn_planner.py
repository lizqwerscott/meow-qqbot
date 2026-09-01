"""Turn-level planner primitives shared by Chat and Agent integrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Awaitable, Callable

from core.engine.planner_control import (
    PlannerAction,
    PlannerControl,
    PlannerControlError,
    parse_planner_control,
)


class PlannerResultKind(StrEnum):
    REPLIED = "replied"
    NO_REPLY = "no_reply"
    WAITING = "waiting"
    HANDED_OFF = "handed_off"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PlannerRequest:
    turn_id: str
    mode: str
    source: str
    messages: list[dict]
    tools: list[dict]
    wait_seconds: int = 0
    max_waits: int = 1


@dataclass(frozen=True)
class PlannerResult:
    kind: PlannerResultKind
    turn_id: str
    reason: str = ""
    task_summary: str = ""
    wait_seconds: int = 0
    work_plan_id: str | None = None


class TurnPlanner:
    """Small orchestration boundary; model/tool execution remains delegated."""

    def __init__(
        self,
        *,
        provider: Callable[..., Awaitable[Any]] | None = None,
        handoff: (
            Callable[[PlannerRequest, PlannerControl], Awaitable[str | None]] | None
        ) = None,
    ):
        self.provider = provider
        self.handoff = handoff
        self._wait_counts: dict[str, int] = {}

    def reset(self, turn_id: str) -> None:
        self._wait_counts.pop(turn_id, None)

    async def consume_control(
        self, request: PlannerRequest, control: PlannerControl
    ) -> PlannerResult:
        """Interpret one control action emitted by the existing ToolLoop.

        Provider calls and normal tool execution stay in ToolLoop.  This is the
        sole control-action boundary used by both direct and future planner
        provider integrations.
        """
        if control.action is PlannerAction.NO_REPLY:
            self.reset(request.turn_id)
            return PlannerResult(
                PlannerResultKind.NO_REPLY, request.turn_id, control.reason
            )
        if control.action is PlannerAction.WAIT:
            count = self._wait_counts.get(request.turn_id, 0) + 1
            self._wait_counts[request.turn_id] = count
            if count > max(1, request.max_waits):
                self.reset(request.turn_id)
                return PlannerResult(
                    PlannerResultKind.COMPLETED, request.turn_id, "wait_limit"
                )
            return PlannerResult(
                PlannerResultKind.WAITING,
                request.turn_id,
                control.reason,
                wait_seconds=control.wait_seconds or request.wait_seconds,
            )
        if request.source not in {"private", "explicit", "user"}:
            return PlannerResult(
                PlannerResultKind.FAILED,
                request.turn_id,
                "request_agent_not_allowed",
            )
        return PlannerResult(
            PlannerResultKind.HANDED_OFF,
            request.turn_id,
            control.reason,
            control.task_summary,
        )

    async def run(self, request: PlannerRequest) -> PlannerResult:
        if self.provider is None:
            return PlannerResult(
                PlannerResultKind.FAILED,
                request.turn_id,
                "planner provider unavailable",
            )
        try:
            response = await self.provider(
                messages=request.messages, tools=request.tools, mode=request.mode
            )
            control = self._control(response, request)
            if control is not None:
                result = await self.consume_control(request, control)
                if result.kind is not PlannerResultKind.HANDED_OFF:
                    return result
                if self.handoff is None:
                    return PlannerResult(
                        PlannerResultKind.FAILED, request.turn_id, "handoff_unavailable"
                    )
                plan_id = await self.handoff(request, control)
                if not plan_id:
                    return PlannerResult(
                        PlannerResultKind.FAILED, request.turn_id, "handoff_failed"
                    )
                self.reset(request.turn_id)
                return replace(result, work_plan_id=plan_id)
            text = str(
                getattr(response, "content", None)
                or (response.get("content", "") if isinstance(response, dict) else "")
            )
            self.reset(request.turn_id)
            return PlannerResult(
                PlannerResultKind.REPLIED if text else PlannerResultKind.COMPLETED,
                request.turn_id,
            )
        except PlannerControlError as exc:
            self.reset(request.turn_id)
            return PlannerResult(
                PlannerResultKind.FAILED,
                request.turn_id,
                f"invalid_planner_control:{exc}",
            )
        except Exception as exc:
            return PlannerResult(PlannerResultKind.FAILED, request.turn_id, str(exc))

    @staticmethod
    def _control(response: Any, request: PlannerRequest) -> PlannerControl | None:
        if isinstance(response, PlannerControl):
            return response
        calls = getattr(response, "tool_calls", None)
        content = getattr(response, "content", None)
        if calls is None and isinstance(response, dict):
            calls = response.get("tool_calls")
            content = response.get("content")
        control_calls = [
            call
            for call in calls or ()
            if (
                getattr(call, "name", None)
                or (call.get("name") if isinstance(call, dict) else "")
            )
            == "planner_control"
        ]
        if not control_calls:
            return None
        call = control_calls[0]
        arguments = getattr(call, "arguments", None) or call.get("arguments", "")
        allowed = {PlannerAction.WAIT, PlannerAction.NO_REPLY}
        if request.source in {"private", "explicit", "user"}:
            allowed.add(PlannerAction.REQUEST_AGENT)
        control = parse_planner_control(arguments, allowed_actions=allowed)
        visible_text = str(content or "").strip()
        if (
            len(control_calls) != 1
            or len(calls or ()) != 1
            or (
                visible_text
                and not (
                    visible_text == "NO_REPLY"
                    and control.action is PlannerAction.NO_REPLY
                )
            )
        ):
            raise PlannerControlError(
                "planner_control must be the only call without visible text"
            )
        return control
