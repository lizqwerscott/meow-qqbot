"""Strict internal control protocol for a Chat planner turn.

``planner_control`` is intentionally not registered in the general tool
registry.  It is a provider-visible control message that the future
``TurnPlanner`` consumes before normal tool execution, so it cannot obtain
ordinary tool delivery or side effects by accident.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from core.ai.protocol import AssistantMessage


class PlannerControlError(ValueError):
    """Raised for a fail-closed planner-control protocol violation."""


class PlannerAction(StrEnum):
    WAIT = "wait"
    NO_REPLY = "no_reply"
    REQUEST_AGENT = "request_agent"


DEFAULT_PLANNER_ACTIONS = frozenset(PlannerAction)
MAX_WAIT_SECONDS = 300
MAX_REASON_CHARS = 240
MAX_TASK_SUMMARY_CHARS = 600


@dataclass(frozen=True)
class PlannerControl:
    """A parsed control action with only the fields valid for that action."""

    action: PlannerAction
    reason: str = ""
    wait_seconds: int | None = None
    task_summary: str = ""


def planner_control_tool(
    allowed_actions: Iterable[PlannerAction | str] = DEFAULT_PLANNER_ACTIONS,
) -> dict[str, Any]:
    """Return the strict OpenAI function schema for the allowed action subset."""
    actions = _normalize_actions(allowed_actions)
    variants = []
    for action in actions:
        if action is PlannerAction.WAIT:
            variants.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"const": action.value},
                        "wait_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_WAIT_SECONDS,
                        },
                        "reason": {
                            "type": ["string", "null"],
                            "maxLength": MAX_REASON_CHARS,
                        },
                    },
                    "required": ["action", "wait_seconds", "reason"],
                }
            )
        elif action is PlannerAction.NO_REPLY:
            variants.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"const": action.value},
                        "reason": {
                            "type": ["string", "null"],
                            "maxLength": MAX_REASON_CHARS,
                        },
                    },
                    "required": ["action", "reason"],
                }
            )
        else:
            variants.append(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"const": action.value},
                        "task_summary": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_TASK_SUMMARY_CHARS,
                        },
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REASON_CHARS,
                        },
                    },
                    "required": ["action", "task_summary", "reason"],
                }
            )
    return {
        "type": "function",
        "function": {
            "name": "planner_control",
            "description": "Internal turn control. Never combine with visible text or another tool call.",
            "strict": True,
            "parameters": {"oneOf": variants},
        },
    }


def parse_planner_control(
    arguments: str | Mapping[str, Any],
    *,
    allowed_actions: Iterable[PlannerAction | str] = DEFAULT_PLANNER_ACTIONS,
) -> PlannerControl:
    """Parse one action exactly; unknown fields and duplicate JSON keys fail closed."""
    actions = _normalize_actions(allowed_actions)
    payload = _parse_arguments(arguments)
    action_value = payload.get("action")
    try:
        action = PlannerAction(action_value)
    except (TypeError, ValueError) as exc:
        raise PlannerControlError("planner_control action is invalid") from exc
    if action not in actions:
        raise PlannerControlError(
            "planner_control action is not allowed for this profile"
        )

    if action is PlannerAction.WAIT:
        _require_fields(
            payload, {"action", "wait_seconds", "reason"}, {"action", "wait_seconds"}
        )
        wait_seconds = payload["wait_seconds"]
        if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, int):
            raise PlannerControlError("wait_seconds must be an integer")
        if not 1 <= wait_seconds <= MAX_WAIT_SECONDS:
            raise PlannerControlError("wait_seconds is outside the allowed range")
        return PlannerControl(
            action=action,
            wait_seconds=wait_seconds,
            reason=_optional_reason(payload),
        )

    if action is PlannerAction.NO_REPLY:
        _require_fields(payload, {"action", "reason"}, {"action"})
        return PlannerControl(action=action, reason=_optional_reason(payload))

    _require_fields(
        payload,
        {"action", "task_summary", "reason"},
        {"action", "task_summary", "reason"},
    )
    task_summary = _required_text(payload, "task_summary", MAX_TASK_SUMMARY_CHARS)
    reason = _required_text(payload, "reason", MAX_REASON_CHARS)
    return PlannerControl(action=action, task_summary=task_summary, reason=reason)


def classify_planner_control_response(
    message: AssistantMessage,
    *,
    allowed_actions: Iterable[PlannerAction | str] = DEFAULT_PLANNER_ACTIONS,
) -> PlannerControl | None:
    """Extract one control call or reject an ambiguous assistant response.

    ``None`` means the response is not a planner-control response. A regular
    tool call remains for the normal tool loop. Any response that combines a
    planner control call with text or another tool call is a terminal protocol
    error, never a recoverable tool result.
    """
    calls = tuple(message.tool_calls or ())
    control_calls = tuple(call for call in calls if call.name == "planner_control")
    if not control_calls:
        return None
    if len(control_calls) != 1 or len(calls) != 1:
        raise PlannerControlError("planner_control must be the only tool call")
    if (message.content or "").strip():
        raise PlannerControlError(
            "planner_control cannot include visible assistant text"
        )
    return parse_planner_control(
        control_calls[0].arguments, allowed_actions=allowed_actions
    )


def _normalize_actions(
    actions: Iterable[PlannerAction | str],
) -> tuple[PlannerAction, ...]:
    resolved: set[PlannerAction] = set()
    for action in actions:
        try:
            resolved.add(PlannerAction(action))
        except ValueError as exc:
            raise PlannerControlError(f"unknown planner action: {action}") from exc
    if not resolved:
        raise PlannerControlError(
            "planner_control requires at least one allowed action"
        )
    return tuple(action for action in PlannerAction if action in resolved)


def _parse_arguments(arguments: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments, object_pairs_hook=_reject_duplicate_keys)
        except (TypeError, json.JSONDecodeError, PlannerControlError) as exc:
            raise PlannerControlError(
                "planner_control arguments must be valid JSON"
            ) from exc
    elif isinstance(arguments, Mapping):
        payload = dict(arguments)
    else:
        raise PlannerControlError("planner_control arguments must be an object")
    if not isinstance(payload, dict):
        raise PlannerControlError("planner_control arguments must be an object")
    return payload


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise PlannerControlError(f"duplicate planner_control field: {key}")
        payload[key] = value
    return payload


def _require_fields(
    payload: Mapping[str, Any], allowed: set[str], required: set[str]
) -> None:
    unknown = set(payload) - allowed
    missing = required - set(payload)
    if unknown:
        raise PlannerControlError("planner_control contains unknown fields")
    if missing:
        raise PlannerControlError("planner_control is missing required fields")


def _optional_reason(payload: Mapping[str, Any]) -> str:
    if "reason" not in payload or payload["reason"] is None:
        return ""
    return _required_text(payload, "reason", MAX_REASON_CHARS, allow_empty=True)


def _required_text(
    payload: Mapping[str, Any], field: str, max_chars: int, *, allow_empty: bool = False
) -> str:
    value = payload[field]
    if not isinstance(value, str):
        raise PlannerControlError(f"{field} must be a string")
    if len(value) > max_chars:
        raise PlannerControlError(f"{field} exceeds the allowed length")
    if not allow_empty and not value.strip():
        raise PlannerControlError(f"{field} cannot be blank")
    return value
