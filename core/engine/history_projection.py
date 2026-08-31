"""Shared migration helpers for the visible timeline projection."""

import time
from typing import Any, Iterable, List, Sequence

from core.managers.chat_message import ChatMessage


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


def merge_timeline_visible_events(
    messages: List[Any],
    events: Iterable[Any],
    skip_event_ids: set[str] | None = None,
) -> List[Any]:
    """Materialize timeline-only visible events into legacy ChatMessage history."""
    events = tuple(events)
    event_by_key = {
        (event.role, event.message_id): event
        for event in events
        if event.role in {"user", "assistant"}
        and event.message_id
        and getattr(event, "event_id", None)
    }
    existing = {
        (message.role, message.message_id)
        for message in messages
        if message.message_id
        and not (message.role == "assistant" and message.tool_calls)
    }
    additions: list[ChatMessage] = []
    enriched: list[Any] = []
    for message in messages:
        event = event_by_key.get((message.role, message.message_id))
        if event is not None and not getattr(message, "event_id", None):
            message = ChatMessage(
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                message_id=message.message_id,
                sender_id=message.sender_id,
                name=message.name,
                tool_call_id=message.tool_call_id,
                tool_name=message.tool_name,
                tool_calls=message.tool_calls,
                reasoning_content=message.reasoning_content,
                event_id=event.event_id,
            )
        enriched.append(message)
    for event in events:
        if event.role not in {"user", "assistant"} or not event.content:
            continue
        if getattr(event, "event_id", None) in (skip_event_ids or set()):
            continue
        message_id = event.message_id or event.event_id
        if not message_id:
            continue
        key = (event.role, message_id)
        if key in existing:
            continue
        additions.append(
            ChatMessage(
                role=event.role,
                content=event.content,
                timestamp=event.timestamp or time.time(),
                message_id=message_id,
                sender_id=getattr(event, "sender_id", "") or None,
                name="系统" if event.role == "assistant" else None,
                event_id=event.event_id,
            )
        )
        existing.add(key)
    if not additions:
        return enriched

    original_positions = {id(message): index for index, message in enumerate(enriched)}
    merged = [*enriched, *additions]
    merged.sort(
        key=lambda message: (
            message.timestamp,
            original_positions.get(id(message), len(enriched)),
        )
    )
    return merged
