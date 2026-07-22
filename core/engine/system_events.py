"""Lightweight in-memory queue for human-readable system events.

每个 session (chat_id) 一个独立队列。纯内存、不持久化。
事件在下一轮 prompt 构建前被 drain，格式化为 System: [HH:MM:SS] 文本 行注入上下文。
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)

# 匹配行首的 "System:" 前缀，用于防注入（多行模式）
_SYSTEM_PREFIX_RE = re.compile(r"^\s*System:\s*", re.IGNORECASE | re.MULTILINE)

MAX_EVENTS = 20


@dataclass
class SystemEvent:
    text: str
    ts: float
    context_key: Optional[str] = None


@dataclass
class _SessionQueue:
    queue: list = field(default_factory=list)
    _seen: set = field(default_factory=set)


class SystemEventQueue:
    def __init__(self):
        self._queues: dict[str, _SessionQueue] = {}
        self._snapshots: dict[str, set[tuple]] = {}

    # ── 公开 API ──

    def enqueue(
        self,
        session_key: str,
        text: str,
        context_key: Optional[str] = None,
        *,
        replace: bool = False,
    ) -> bool:
        """推入系统事件到指定 session 的队列。

        Args:
            session_key: 会话 ID（meow-qqbot 中即 chat_id）
            text: 事件文本
            context_key: 用于去重和 replace 的键，如 "cron:<id>"
            replace: 为 True 且 context_key 匹配时，覆盖队列中同 key 的旧事件

        Returns:
            True 表示事件真正入队，False 表示被去重或文本为空
        """
        if not session_key or not session_key.strip():
            _log.warning("system events require a non-empty session_key")
            return False

        cleaned = self._sanitize(text)
        if not cleaned:
            return False

        sq = self._get_or_create(session_key)

        if replace and context_key:
            return self._replace_in_queue(sq, cleaned, context_key)

        # 去重：同 session 内 (text, context_key) 相同则跳过
        dedup_key = (cleaned, context_key)
        if dedup_key in sq._seen:
            return False

        event = SystemEvent(text=cleaned, ts=time.time(), context_key=context_key)
        sq.queue.append(event)
        sq._seen.add(dedup_key)

        if len(sq.queue) > MAX_EVENTS:
            removed = sq.queue.pop(0)
            self._remove_from_seen(sq, removed)

        _log.debug("system event enqueued [%s..]: %s", session_key[:12], cleaned[:60])
        return True

    def peek_and_snapshot(self, session_key: str) -> list[SystemEvent]:
        """Peek 事件并记录快照。后续 consume_snapshot 只移除快照内的事件，
        不会误删 peek 后新增的事件。"""
        events = self.peek(session_key)
        self._snapshots[session_key] = {(e.text, e.context_key) for e in events}
        return events

    def consume_snapshot(self, session_key: str) -> None:
        """只消费 peek_and_snapshot 时快照内的事件，保留快照外的事件。"""
        keys = self._snapshots.pop(session_key, set())
        if not keys:
            return
        sq = self._queues.get(session_key)
        if not sq:
            return
        sq.queue = [e for e in sq.queue if (e.text, e.context_key) not in keys]
        sq._seen = {(e.text, e.context_key) for e in sq.queue}
        if not sq.queue:
            self._queues.pop(session_key, None)
        _log.debug("consumed snapshot %d events [%s..]", len(keys), session_key[:12])

    def drain(self, session_key: str) -> list[SystemEvent]:
        """取出并清空该 session 的所有排队事件。"""
        sq = self._queues.get(session_key)
        if not sq or not sq.queue:
            return []

        out = list(sq.queue)
        sq.queue.clear()
        sq._seen.clear()
        self._queues.pop(session_key, None)
        _log.debug("drained %d system events [%s..]", len(out), session_key[:12])
        return out

    def peek(self, session_key: str) -> list[SystemEvent]:
        """只读查看该 session 的排队事件（不清空）。"""
        sq = self._queues.get(session_key)
        if not sq:
            return []
        return list(sq.queue)

    def has_events(self, session_key: str) -> bool:
        sq = self._queues.get(session_key)
        return bool(sq and sq.queue)

    def clear(self, session_key: str) -> None:
        self._queues.pop(session_key, None)

    def clear_all(self) -> None:
        self._queues.clear()

    # ── 内部方法 ──

    @staticmethod
    def _sanitize(text: str) -> str:
        cleaned = _SYSTEM_PREFIX_RE.sub("", text).strip()
        return cleaned

    def _get_or_create(self, session_key: str) -> _SessionQueue:
        if session_key not in self._queues:
            self._queues[session_key] = _SessionQueue()
        return self._queues[session_key]

    def _replace_in_queue(self, sq: _SessionQueue, text: str, context_key: str) -> bool:
        """替换同 context_key 的事件。没有匹配到则追加。"""
        for i, event in enumerate(sq.queue):
            if event.context_key == context_key:
                if event.text == text:
                    return False
                remove_key = (event.text, event.context_key)
                sq._seen.discard(remove_key)
                sq.queue[i].text = text
                sq.queue[i].ts = time.time()
                sq._seen.add((text, context_key))
                return True

        add_key = (text, context_key)
        if add_key in sq._seen:
            return False

        event = SystemEvent(text=text, ts=time.time(), context_key=context_key)
        sq.queue.append(event)
        sq._seen.add(add_key)

        if len(sq.queue) > MAX_EVENTS:
            removed = sq.queue.pop(0)
            self._remove_from_seen(sq, removed)
        return True

    @staticmethod
    def _remove_from_seen(sq: _SessionQueue, event: SystemEvent) -> None:
        key = (event.text, event.context_key)
        sq._seen.discard(key)
