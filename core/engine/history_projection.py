"""Shared migration helpers for the visible timeline projection."""

import inspect
from typing import Any, List, Sequence


async def read_legacy_history_bounded(
    context_manager: Any, chat_id: str, *, max_messages: int = 100
) -> list[dict[str, Any]]:
    """Read legacy history only through an explicit bounded compatibility path."""
    limit = max(0, int(max_messages))
    reader = getattr(context_manager, "get_chat_history_async", None)
    if not callable(reader) or limit == 0:
        return []
    try:
        signature = inspect.signature(reader)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        result = await reader(chat_id, max_messages=limit)
    else:
        parameters = tuple(signature.parameters.values())
        max_parameter = next(
            (parameter for parameter in parameters if parameter.name == "max_messages"),
            None,
        )
        if max_parameter is not None:
            if max_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                result = await reader(chat_id, limit)
            elif max_parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                result = await reader(chat_id, limit)
            else:
                result = await reader(chat_id, max_messages=limit)
        elif any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        ):
            result = await reader(chat_id, max_messages=limit)
        else:
            bounded_parameter_names = {
                "limit",
                "max_count",
                "max_events",
                "max_messages",
            }
            bounded_parameter = next(
                (
                    parameter
                    for parameter in parameters[1:]
                    if parameter.name in bounded_parameter_names
                    and parameter.kind
                    in {
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                ),
                None,
            )
            if bounded_parameter is None:
                result = await reader(chat_id)
            elif bounded_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                result = await reader(chat_id, limit)
            else:
                result = await reader(chat_id, **{bounded_parameter.name: limit})
    return list(result or ())[-limit:]


def visible_legacy_history(messages: Sequence[dict[str, Any]]) -> List[dict[str, Any]]:
    """Project legacy storage into user-visible conversation messages only."""
    visible: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if role == "assistant" and message.get("tool_calls"):
            continue
        content = message.get("raw_content", message.get("content", ""))
        if not content:
            continue
        copied = dict(message)
        copied["content"] = content
        visible.append(copied)
    return visible
