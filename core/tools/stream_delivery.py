"""流式回复的分块、顺序与 generation 投递。"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from core.ai.protocol import StreamCallbacks, StreamReset, StreamSnapshot
from core.markdown_split import (
    MARKDOWN_SAFE_CHUNK_BYTE_LIMIT,
    markdown_safe_cut,
    pending_starts_incomplete,
    trailing_structure,
    utf8_prefix,
    utf8len,
)

_log = logging.getLogger(__name__)

_SILENT_TOKENS = frozenset({"NO_REPLY", "HEARTBEAT_OK"})
_ACK_MAX_CHARS = 100
STREAM_PROBE_CHARS = _ACK_MAX_CHARS + 12


def is_silent_reply_text(text: str) -> bool:
    stripped = text.strip().strip("`").strip()
    if not stripped:
        return True
    for token in _SILENT_TOKENS:
        if stripped == token:
            return True
        if stripped.startswith(token):
            remaining = stripped[len(token) :].lstrip("`").strip("：:，, \t")
            if not remaining or len(remaining) < _ACK_MAX_CHARS:
                return True
    return False


@dataclass
class _GenerationDelivery:
    generation: int = 0
    text: str = ""
    sent: int = 0
    last_flush: float | None = None
    timer: asyncio.Task | None = field(default=None, repr=False, compare=False)
    sending: bool = False
    flush_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )

    def cancel_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


class StreamDelivery:
    """隐藏流式文本投递状态的深模块。

    `stream_callback` 是流式块投递 adapter，`reply_callback` 由 `finish` 接收，
    用于发送模型结束后剩余的完整文本。两者都只接收一个文本参数。
    """

    def __init__(
        self,
        *,
        chat_id: str,
        stream_callback: Callable[[str], Awaitable[None]] | None,
        message_delivered: Callable[[], bool],
        block_chars: int,
        idle_seconds: float,
    ) -> None:
        self._chat_id = chat_id
        self._stream_callback = stream_callback
        self._message_delivered = message_delivered
        self._block_chars = block_chars
        self._idle_seconds = idle_seconds
        self._current = _GenerationDelivery()
        self._forwarded = False
        self._text_committed = False

    @property
    def callbacks(self) -> StreamCallbacks:
        return StreamCallbacks(on_snapshot=self.on_snapshot, on_reset=self.on_reset)

    @property
    def forwarded(self) -> bool:
        return self._forwarded

    @property
    def text_committed(self) -> bool:
        return self._text_committed

    def complete(self) -> None:
        """结束当前 generation 的后台投递活动。"""
        self._current.cancel_timer()

    async def on_snapshot(self, snapshot: StreamSnapshot) -> None:
        generation = self._current
        if snapshot.generation != generation.generation:
            _log.debug(
                "忽略过期流快照 [%s]: got=%s current=%s",
                self._chat_id[:12],
                snapshot.generation,
                generation.generation,
            )
            return
        generation.text = snapshot.text
        pending = snapshot.text[generation.sent :]
        if not pending or self._stream_callback is None:
            return
        if len(snapshot.text.strip()) <= STREAM_PROBE_CHARS:
            return
        if len(pending) >= self._block_chars:
            await self._flush(generation, allow_partial=True)
        else:
            self._schedule_idle_flush(generation)

    async def on_reset(self, reset: StreamReset) -> None:
        self._current.cancel_timer()
        self._current = _GenerationDelivery(generation=reset.generation)

    async def abort(self) -> None:
        self.complete()
        try:
            while not self._message_delivered() and self._current.sent < len(
                self._current.text
            ):
                await self._flush(
                    self._current,
                    allow_partial=True,
                    flush_incomplete=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log.warning("断流收尾补发失败 [%s]: %s", self._chat_id[:12], error)

    async def finish(
        self,
        response_text: str,
        reply_callback: Callable[[str], Awaitable[None]],
    ) -> None:
        self.complete()
        if self._message_delivered():
            return
        generation = self._current
        generation.text = response_text
        async with generation.flush_lock:
            remaining = response_text[generation.sent :]
            if remaining:
                try:
                    await reply_callback(remaining)
                except Exception as error:
                    _log.warning(
                        "回复 callback 失败 [%s]: %s", self._chat_id[:12], error
                    )
            generation.sent = len(response_text)
            self._text_committed = True

    async def _flush(
        self,
        generation: _GenerationDelivery,
        *,
        allow_partial: bool = False,
        flush_incomplete: bool = False,
    ) -> None:
        async with generation.flush_lock:
            if generation is not self._current:
                return
            pending = generation.text[generation.sent :]
            if not pending or self._message_delivered():
                return
            if not allow_partial and pending_starts_incomplete(
                pending, generation.text[: generation.sent]
            ):
                return
            limit = self._block_chars if allow_partial else len(pending)
            cut = markdown_safe_cut(
                pending,
                limit,
                initial=trailing_structure(generation.text[: generation.sent]),
            )
            if cut == 0 and not flush_incomplete:
                if utf8len(pending) < MARKDOWN_SAFE_CHUNK_BYTE_LIMIT:
                    return
                cut = pending.rfind("\n") + 1
                if cut <= 0:
                    cut = utf8_prefix(pending, MARKDOWN_SAFE_CHUNK_BYTE_LIMIT)
            if 0 < cut < len(pending):
                pending = pending[:cut]
            self._forwarded = True
            generation.sent += len(pending)
            self._text_committed = True
            generation.sending = True
            try:
                if self._stream_callback is not None:
                    await self._stream_callback(pending)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _log.warning("流式块发送失败 [%s]: %s", self._chat_id[:12], error)
            finally:
                generation.sending = False
            generation.last_flush = time.monotonic()

    async def _idle_flush_task(
        self, generation: _GenerationDelivery, delay: float
    ) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if generation.timer is asyncio.current_task():
            generation.timer = None
        try:
            await self._flush(generation)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log.warning("空闲 flush 失败 [%s]: %s", self._chat_id[:12], error)

    def _schedule_idle_flush(self, generation: _GenerationDelivery) -> None:
        if (
            generation is not self._current
            or generation.timer is not None
            or generation.sending
        ):
            return
        now = time.monotonic()
        last = generation.last_flush or (now - self._idle_seconds)
        delay = max(0.0, self._idle_seconds - (now - last))
        generation.timer = asyncio.create_task(self._idle_flush_task(generation, delay))
