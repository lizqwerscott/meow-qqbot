"""ArchiveManager — 会话归档 + 自动摘要 + 上下文回放

消息驱动触发：每 dispatch() 检查日期边界，跨天则：
1. 重命名 JSONL → .archived.<timestamp>（保留全部原始消息）
2. 提取最后 N 条 user/assistant 消息 → 写入 .md 摘要
3. 最近 M 条消息回放到新 session
4. 首次 build() 时注入归档摘要（仅一次，后续不再重复）
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.managers.chat_message import (
    ChatMessage,
    group_user_messages,
    strip_content_prefix,
)

_log = logging.getLogger(__name__)

_DEFAULT_SUMMARY_COUNT = 15
_DEFAULT_REPLAY_COUNT = 6
_DEFAULT_ARCHIVE_HOUR = 4
_DEFAULT_SUMMARY_DAYS = 2
_DEFAULT_RETENTION_DAYS = 30

# ── 工具函数 ──


def _format_archive_timestamp(t: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(t or time.time())
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def _daily_reset_at(hour: int, t: Optional[float] = None) -> float:
    ts = t or time.time()
    dt = datetime.fromtimestamp(ts)
    try:
        today_reset = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    except ValueError:
        today_reset = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            hours=hour
        )
    if today_reset.timestamp() > ts:
        today_reset -= timedelta(days=1)
    return today_reset.timestamp()


def _date_str(t: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(t or time.time())
    return dt.strftime("%Y-%m-%d")


def _get_memory_dir(memory_root: str, chat_id: str) -> Path:
    return Path(memory_root) / chat_id


# ── ArchiveResult ──


class ArchiveResult:
    """归档操作的返回信息。"""

    def __init__(
        self,
        chat_id: str,
        reason: str,
        archive_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        replay_count: int = 0,
    ):
        self.chat_id = chat_id
        self.reason = reason
        self.archive_path = archive_path
        self.summary_path = summary_path
        self.replay_count = replay_count


# ── ArchiveManager ──


class ArchiveManager:
    """会话归档管理器。"""

    def __init__(
        self,
        context_manager: Any,
        memory_dir: str = "data/archives/memory/",
        archive_hour: int = _DEFAULT_ARCHIVE_HOUR,
        replay_count: int = _DEFAULT_REPLAY_COUNT,
        summary_count: int = _DEFAULT_SUMMARY_COUNT,
        summary_days: int = _DEFAULT_SUMMARY_DAYS,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        merge_window_seconds: int = 15,
    ):
        self._cm = context_manager
        self._memory_dir = memory_dir
        self._archive_hour = archive_hour
        self._replay_count = replay_count
        self._summary_count = summary_count
        self._summary_days = summary_days
        self._retention_days = retention_days
        self.merge_window_seconds = merge_window_seconds

        self._pending_injection: Set[str] = set()

    @property
    def _store(self):
        return self._cm.store

    # ── 公开方法 ──

    async def archive_if_stale(
        self, chat_id: str, is_group: bool
    ) -> Optional[ArchiveResult]:
        async def _do(ctx):
            if ctx.is_empty():
                return None

            now = time.time()
            today_reset = _daily_reset_at(self._archive_hour, now)

            if ctx.last_activity >= today_reset:
                return None

            return await self._do_archive(ctx, chat_id, is_group, "daily")

        return await self._cm._with_context_locked(chat_id, _do)

    def load_recent_summaries(self, chat_id: str) -> Optional[str]:
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        if not mem_dir.is_dir():
            return None

        now = time.time()
        parts: List[str] = []
        for day_offset in range(self._summary_days):
            date = _date_str(now - day_offset * 86400)
            day_file = mem_dir / f"{date}.md"
            if day_file.is_file():
                try:
                    text = day_file.read_text(encoding="utf-8").strip()
                    if text:
                        parts.append(f"{date}:\n{text}")
                except Exception as e:
                    _log.warning(
                        "读取归档摘要失败 [%s..] %s: %s",
                        chat_id[:12],
                        day_file.name,
                        e,
                    )

        return "\n\n---\n\n".join(parts) if parts else None

    async def load_recent_summaries_async(self, chat_id: str) -> Optional[str]:
        return await asyncio.to_thread(self.load_recent_summaries, chat_id)

    def consume_summary(self, chat_id: str) -> Optional[str]:
        if chat_id not in self._pending_injection:
            return None
        self._pending_injection.discard(chat_id)
        return self.load_recent_summaries(chat_id)

    async def consume_summary_async(self, chat_id: str) -> Optional[str]:
        if chat_id not in self._pending_injection:
            return None
        self._pending_injection.discard(chat_id)
        return await self.load_recent_summaries_async(chat_id)

    async def get_session_status_async(self, chat_id: str) -> Dict[str, Any]:
        history = await self._cm.get_chat_history_async(chat_id)
        return {
            "message_count": len(history),
            "last_activity": (
                history[-1].get("timestamp", time.time()) if history else None
            ),
            "archive_count": len(
                await asyncio.to_thread(
                    lambda: list(
                        (_get_memory_dir(self._memory_dir, chat_id)).glob("*.md")
                    )
                )
            ),
        }

    async def archive_manual(self, chat_id: str, is_group: bool) -> ArchiveResult:
        async def _do(ctx):
            return await self._do_archive(ctx, chat_id, is_group, "manual")

        return await self._cm._with_context_locked(chat_id, _do)

    def cleanup_old_archives(self) -> int:
        retention_seconds = self._retention_days * 86400
        removed = self._store.cleanup_stale_archives(retention_seconds)

        # 清理过期的 .md 摘要
        mem_root = Path(self._memory_dir)
        cutoff = time.time() - retention_seconds
        if mem_root.is_dir():
            for chat_dir in mem_root.iterdir():
                if not chat_dir.is_dir():
                    continue
                for f in chat_dir.iterdir():
                    if f.suffix == ".md":
                        try:
                            mtime = f.stat().st_mtime
                            if mtime < cutoff:
                                f.unlink()
                                removed += 1
                        except Exception as e:
                            _log.warning("清理摘要文件失败 %s: %s", f.name, e)

        if removed:
            _log.info("归档清理完成: 移除了 %d 个文件", removed)
        return removed

    async def cleanup_old_archives_async(self) -> int:
        return await asyncio.to_thread(self.cleanup_old_archives)

    async def list_archives_async(self, chat_id: str) -> List[dict]:
        return await asyncio.to_thread(self._list_memory_files, chat_id)

    def _list_memory_files(self, chat_id: str) -> List[dict]:
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        if not mem_dir.is_dir():
            return []
        return [
            {"path": str(path), "size": path.stat().st_size}
            for path in mem_dir.glob("*.md")
        ]

    # ── 内部方法（需在 per-chat 锁内调用） ──

    async def _do_archive(
        self, ctx: Any, chat_id: str, is_group: bool, reason: str
    ) -> ArchiveResult:
        store = self._store
        now = time.time()
        ts = _format_archive_timestamp(now)
        date = _date_str(now)

        # 1. 确保数据落盘（后台线程）
        await asyncio.to_thread(
            store.flush,
            chat_id,
            [m.to_dict() for m in ctx.get_history()],
        )

        # 2. 收集消息
        all_msgs = ctx.get_history()

        # 3. 提取回放消息
        replay_msgs = self._extract_replay_messages(all_msgs, self._replay_count)

        # 4. 生成摘要文本
        summary_text = self._format_summary_text(
            all_msgs,
            self._summary_count,
            is_group,
            chat_id,
            date,
        )

        # 5. 归档（后台线程）
        archive_path = await asyncio.to_thread(store.archive, chat_id, ts)

        # 6. 写入摘要 .md（后台线程）
        summary_path: Optional[str] = None
        if summary_text:
            summary_path = await self._write_memory_file(chat_id, date, summary_text)

        # 7. 清空 + 回放新历史
        ctx.set_messages(replay_msgs)
        ctx.last_activity = time.time()

        # 8. 写入新数据（后台线程）
        await asyncio.to_thread(
            store.flush,
            chat_id,
            [m.to_dict() for m in replay_msgs],
        )

        _log.info(
            "归档完成 [%s..]: reason=%s replay=%d summary=%s",
            chat_id[:12],
            reason,
            len(replay_msgs),
            summary_path or "无",
        )

        result = ArchiveResult(
            chat_id=chat_id,
            reason=reason,
            archive_path=archive_path,
            summary_path=summary_path,
            replay_count=len(replay_msgs),
        )

        if summary_text and reason != "manual":
            self._pending_injection.add(chat_id)

        return result

    # ── 消息提取 ──

    def _extract_replay_messages(self, messages: List[Any], count: int) -> List[Any]:
        result: List[ChatMessage] = []
        for msg in reversed(messages):
            if msg.role == "tool":
                continue
            if msg.role == "assistant" and msg.tool_calls:
                continue
            if (
                msg.role == "assistant"
                and msg.content
                and "[助手发送了一个表情]" in msg.content
            ):
                continue
            if msg.sender_id == "system":
                continue

            content = msg.content or ""
            if msg.role == "user":
                content = strip_content_prefix(content)

            cleaned = ChatMessage(
                role=msg.role,
                content=content,
                timestamp=msg.timestamp,
                message_id=msg.message_id,
                sender_id=msg.sender_id,
                name=msg.name,
                tool_call_id=msg.tool_call_id,
                tool_name=msg.tool_name,
                tool_calls=msg.tool_calls,
                reasoning_content=msg.reasoning_content,
            )
            result.append(cleaned)

            if len(result) >= count:
                break

        result.reverse()
        return result

    def _format_summary_text(
        self,
        messages: List[Any],
        count: int,
        is_group: bool,
        chat_id: str,
        date: str,
    ) -> Optional[str]:
        # 1. 正向收集有效消息
        selected: List[ChatMessage] = []
        for msg in messages:
            if msg.role == "tool":
                continue
            if msg.role == "assistant" and msg.tool_calls:
                continue
            if (
                msg.role == "assistant"
                and msg.content
                and "[助手发送了一个表情]" in msg.content
            ):
                continue
            if msg.sender_id == "system":
                continue

            if msg.role == "user":
                raw = strip_content_prefix(msg.content or "")
                if not raw:
                    continue
            else:
                raw = msg.content or ""
                if not raw:
                    continue

            selected.append(msg)

        # 取最后 count 条
        selected = selected[-count:]
        if not selected:
            return None

        # 2. 分组合并
        groups = group_user_messages(selected)

        lines: List[str] = []
        for group in groups:
            _build_summary_group(lines, group, self.merge_window_seconds)

        chat_type = "群聊" if is_group else "私聊"
        short_id = chat_id[:16] + "…" if len(chat_id) > 16 else chat_id

        parts = [
            f"# Session: {date}",
            "",
            f"- **Chat**: {short_id}",
            f"- **Type**: {chat_type}",
            f"- **Messages**: {len(lines)}",
            "",
            "## 对话记录",
            "",
        ]
        parts.extend(lines)
        return "\n".join(parts)

    async def _write_memory_file(
        self, chat_id: str, date: str, text: str
    ) -> Optional[str]:
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        try:
            mem_dir.mkdir(parents=True, exist_ok=True)
            file_path = mem_dir / f"{date}.md"
            await asyncio.to_thread(file_path.write_text, text, encoding="utf-8")
            _log.info(
                "归档摘要已写入 [%s..] %s (%d 字符)",
                chat_id[:12],
                file_path.name,
                len(text),
            )
            return str(file_path)
        except Exception as e:
            _log.warning(
                "写入归档摘要失败 [%s..]: %s",
                chat_id[:12],
                e,
            )
            return None


def _build_summary_group(
    lines: List[str], group: List[ChatMessage], window_seconds: int
) -> None:
    """将合并分组格式化为一行或多行，追加到 lines。"""
    first = group[0]

    if first.role != "user":
        content = first.content or ""
        lines.append(f"猫猫: {content}")
        return

    display = first.name or first.sender_id or "未知"
    content_parts: List[str] = []
    prev_ts = first.timestamp

    for msg in group:
        raw = strip_content_prefix(msg.content or "").strip()
        if not raw:
            continue

        if content_parts:
            gap = msg.timestamp - prev_ts
            if gap > window_seconds:
                ts_marker = time.strftime("[%H:%M:%S]", time.localtime(msg.timestamp))
                content_parts.append(ts_marker)

        content_parts.append(raw)
        prev_ts = msg.timestamp

    if content_parts:
        joined = "\n".join(content_parts)
        lines.append(f"{display}: {joined}")
