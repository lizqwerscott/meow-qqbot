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
    messages: List[Any], events: Iterable[Any]
) -> List[Any]:
    """Materialize timeline-only visible events into legacy ChatMessage history."""
    existing = {
        (message.role, message.message_id)
        for message in messages
        if message.message_id
        and not (message.role == "assistant" and message.tool_calls)
    }
    additions: list[ChatMessage] = []
    for event in events:
        if event.role not in {"user", "assistant"} or not event.content:
            continue
        message_id = event.message_id or event.event_id
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
            )
        )
        existing.add(key)
    if not additions:
        return messages

    original_positions = {id(message): index for index, message in enumerate(messages)}
    merged = [*messages, *additions]
    merged.sort(
        key=lambda message: (
            message.timestamp,
            original_positions.get(id(message), len(messages)),
        )
    )
    return merged
