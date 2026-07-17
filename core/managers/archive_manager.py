"""ArchiveManager — 会话归档 + 自动摘要 + 上下文回放

消息驱动触发：每 dispatch() 检查日期边界，跨天则：
1. 重命名 JSONL → .archived.<timestamp>（保留全部原始消息）
2. 提取最后 N 条 user/assistant 消息 → 写入 .md 摘要
3. 最近 M 条消息回放到新 session
4. 首次 build() 时注入归档摘要（仅一次，后续不再重复）
"""

import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Set

_log = logging.getLogger(__name__)

# ── 常量 ──

_RE_PREFIX = re.compile(
    r'^\[.*? 在 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:\s*'
)

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
    """计算最近的归档时间点的时间戳。

    返回已经过去的、离当前最近的 `hour:00`。如果今天的 hour:00 还没到，
    则回退到昨天的 hour:00，防止 0:00~hour 之间每条消息都误触发归档。
    """
    dt = datetime.fromtimestamp(t or time.time())
    today_reset = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    if today_reset.timestamp() > (t or time.time()):
        today_reset -= timedelta(days=1)
    return today_reset.timestamp()


def _strip_content_prefix(content: str) -> str:
    """去除 content 开头重复的 [name 在 timestamp]: 前缀。"""
    while _RE_PREFIX.match(content):
        content = _RE_PREFIX.sub('', content)
    return content


def _date_str(t: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(t or time.time())
    return dt.strftime("%Y-%m-%d")


def _get_memory_dir(memory_root: str, chat_id: str) -> Path:
    return Path(memory_root) / chat_id


def _get_archive_path(jsonl_path: Path, ts: str) -> Path:
    return jsonl_path.parent / f"{jsonl_path.name}.archived.{ts}"


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
        cache_dir: str,
        memory_dir: str = "data/archives/memory/",
        archive_hour: int = _DEFAULT_ARCHIVE_HOUR,
        replay_count: int = _DEFAULT_REPLAY_COUNT,
        summary_count: int = _DEFAULT_SUMMARY_COUNT,
        summary_days: int = _DEFAULT_SUMMARY_DAYS,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ):
        self._cm = context_manager
        self._cache_dir = cache_dir
        self._memory_dir = memory_dir
        self._archive_hour = archive_hour
        self._replay_count = replay_count
        self._summary_count = summary_count
        self._summary_days = summary_days
        self._retention_days = retention_days

        # 待注入摘要的 chat_id 集合（归档后首次 build 消耗后移除）
        self._pending_injection: Set[str] = set()

    # ── 公开方法 ──

    async def archive_if_stale(
        self, chat_id: str, is_group: bool
    ) -> Optional[ArchiveResult]:
        """检查日期边界，跨天则归档（在 per-chat 锁内完成检查和归档）。

        由 dispatch() 在 add_user_message 前调用。
        Returns:
            ArchiveResult 或 None（未跨天或无历史时）。
        """
        async def _do():
            ctx = self._cm.get_context(chat_id)

            if ctx.is_empty():
                return None

            now = time.time()
            today_reset = _daily_reset_at(self._archive_hour, now)

            # last_activity >= today_reset → 未跨天
            if ctx.last_activity >= today_reset:
                return None

            return self._do_archive(ctx, chat_id, is_group, "daily")

        return await self._cm.with_chat_lock(chat_id, _do)

    def load_recent_summaries(self, chat_id: str) -> Optional[str]:
        """读取最近 N 天的 memory/*.md 文件，返回拼接后的文本。

        N 由 summary_days 配置控制，默认 2（含今天）。
        """
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
                        chat_id[:12], day_file.name, e,
                    )

        return "\n\n---\n\n".join(parts) if parts else None

    def consume_summary(self, chat_id: str) -> Optional[str]:
        """获取并消耗待注入的摘要。首次 build() 注入后不再重复。"""
        if chat_id not in self._pending_injection:
            return None
        self._pending_injection.discard(chat_id)
        return self.load_recent_summaries(chat_id)

    async def archive_manual(self, chat_id: str, is_group: bool) -> ArchiveResult:
        """手动触发归档（猫猫 /归档 执行）。在 per-chat 锁内执行。"""
        async def _do():
            ctx = self._cm.get_context(chat_id)
            return self._do_archive(ctx, chat_id, is_group, "manual")
        return await self._cm.with_chat_lock(chat_id, _do)

    def cleanup_old_archives(self) -> int:
        """清理超过保留天数的归档文件和摘要目录。"""
        now = time.time()
        cutoff = now - self._retention_days * 86400
        removed = 0

        # 清理 .archived.* 文件
        if Path(self._cache_dir).is_dir():
            for f in Path(self._cache_dir).iterdir():
                if ".archived." in f.name:
                    try:
                        mtime = f.stat().st_mtime
                        if mtime < cutoff:
                            f.unlink()
                            removed += 1
                    except Exception as e:
                        _log.warning("清理归档文件失败 %s: %s", f.name, e)

        # 清理过期的 .md 摘要
        mem_root = Path(self._memory_dir)
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

    # ── 内部方法（需在 per-chat 锁内调用） ──

    def _do_archive(
        self, ctx: Any, chat_id: str, is_group: bool, reason: str
    ) -> ArchiveResult:
        """执行实际的归档操作（调用方已持有 per-chat 锁）。"""
        # 1. 确保数据落盘
        ctx.save()

        # 2. 准备路径
        path = ctx._get_cache_path()
        now = time.time()
        ts = _format_archive_timestamp(now)
        archive_path: Optional[Path] = None

        # 3. 收集消息
        all_msgs = list(ctx.history)

        # 4. 提取回放消息
        replay_msgs = self._extract_replay_messages(
            all_msgs, self._replay_count
        )

        # 5. 生成摘要文本
        date = _date_str(now)
        summary_text = self._format_summary_text(
            all_msgs, self._summary_count, is_group, chat_id, date
        )

        # 6. 重命名 JSONL
        if path and path.exists():
            archive_path = _get_archive_path(path, ts)
            try:
                path.rename(archive_path)
                _log.info(
                    "归档会话 [%s..] %s → %s",
                    chat_id[:12], path.name, archive_path.name,
                )
            except Exception as e:
                _log.warning(
                    "重命名 JSONL 失败 [%s..]: %s", chat_id[:12], e,
                )
                archive_path = None

        # 7. 写入摘要 .md
        summary_path: Optional[str] = None
        if summary_text:
            summary_path = self._write_memory_file(
                chat_id, date, summary_text
            )

        # 8. 清空上下文
        ctx.history.clear()
        ctx._flushed_count = 0

        # 9. 回放消息到新的 session
        for msg in replay_msgs:
            ctx.history.append(msg)

        # 10. 更新 last_activity 为当前时间，防止后续重复归档
        ctx.last_activity = time.time()

        # 11. 写入新 JSONL（含回放消息）
        ctx.save()

        _log.info(
            "归档完成 [%s..]: reason=%s replay=%d summary=%s",
            chat_id[:12], reason, len(replay_msgs),
            summary_path or "无",
        )

        result = ArchiveResult(
            chat_id=chat_id,
            reason=reason,
            archive_path=str(archive_path) if archive_path else None,
            summary_path=summary_path,
            replay_count=len(replay_msgs),
        )

        # 标记待注入（仅当有摘要且非手动归档时才自动注入）
        if summary_text and reason != "manual":
            self._pending_injection.add(chat_id)

        return result

    # ── 消息提取 ──

    def _extract_replay_messages(
        self, messages: List[Any], count: int
    ) -> List[Any]:
        """从消息列表中提取最近 N 条有效消息用于回放。

        过滤规则：
        - 跳过 role=tool
        - 跳过 assistant(tool_calls)
        - 跳过 assistant(content="[助手发送了一个表情]")
        - 跳过 sender_id=system
        - 清理 reasoning_content
        - user 消息清理 content 前缀
        """
        from core.managers.context_manager import ChatMessage

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

            # 清理
            content = msg.content or ""
            if msg.role == "user":
                content = _strip_content_prefix(content)

            cleaned = ChatMessage(
                role=msg.role,
                content=content,
                timestamp=msg.timestamp,
                message_id=msg.message_id,
                sender_id=msg.sender_id,
                name=msg.name,
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
        """从消息列表中提取最近 N 条有效消息，格式化为 .md 文件内容。"""
        lines: List[str] = []
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

            # 取显示名
            if msg.role == "user":
                display = msg.name or msg.sender_id or "未知"
                content = _strip_content_prefix(msg.content or "")
            else:
                display = "猫猫"
                content = msg.content or ""

            if not content:
                continue

            lines.append(f"{display}: {content}")
            if len(lines) >= count:
                break

        if not lines:
            return None

        lines.reverse()

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

    def _write_memory_file(
        self, chat_id: str, date: str, text: str
    ) -> Optional[str]:
        """写入摘要 .md 文件。"""
        mem_dir = _get_memory_dir(self._memory_dir, chat_id)
        try:
            mem_dir.mkdir(parents=True, exist_ok=True)
            file_path = mem_dir / f"{date}.md"
            file_path.write_text(text, encoding="utf-8")
            _log.info(
                "归档摘要已写入 [%s..] %s (%d 字符)",
                chat_id[:12], file_path.name, len(text),
            )
            return str(file_path)
        except Exception as e:
            _log.warning(
                "写入归档摘要失败 [%s..]: %s", chat_id[:12], e,
            )
            return None
