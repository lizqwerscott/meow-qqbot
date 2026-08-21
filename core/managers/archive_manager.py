"""ArchiveManager — 会话归档 + 自动摘要 + 上下文回放

消息驱动触发：每条消息前检查消息流中是否存在"今天之前"的消息
（按消息时间戳判断跨天，不依赖 last_activity），跨天则归档一次（同一天
内只归档一次，状态持久化跨重启）：
1. 仅将本次尚未归档的旧消息写入 .archived.<timestamp>
2. 旧消息中取最后 N 条 user/assistant 消息 → 写入 .md 摘要
3. 保留：今天的消息全部保留；按连续时间段的切点决定是否携带昨天尾段
4. 首次 build() 时注入归档摘要（仅一次，后续不再重复）
"""

import asyncio
import inspect
import json
import logging
import threading
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
_DEFAULT_ARCHIVE_HOUR = 4
_DEFAULT_SUMMARY_DAYS = 2
_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_REPLAY_GAP_SECONDS = 600

# ── 工具函数 ──


def _format_archive_timestamp(t: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(t or time.time())
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def _date_str(t: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(t or time.time())
    return dt.strftime("%Y-%m-%d")


def _previous_date_str(t: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(t or time.time())
    return (dt.date() - timedelta(days=1)).isoformat()


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
        replay_count: Optional[int] = None,
        replay_gap_seconds: int = _DEFAULT_REPLAY_GAP_SECONDS,
        summary_count: int = _DEFAULT_SUMMARY_COUNT,
        summary_days: int = _DEFAULT_SUMMARY_DAYS,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        merge_window_seconds: int = 15,
    ):
        self._cm = context_manager
        self._memory_dir = memory_dir
        self._archive_hour = archive_hour
        # replay_count 是旧配置的兼容参数；回放改为按完整时间段切分。
        self._legacy_replay_count = replay_count
        self._replay_gap_seconds = max(0, replay_gap_seconds)
        self._summary_count = summary_count
        self._summary_days = summary_days
        self._retention_days = retention_days
        self.merge_window_seconds = merge_window_seconds

        self._pending_injection: Set[str] = set()
        # chat_id → 上次自动归档的日期（防止同一天重复归档；持久化跨重启）
        self._last_daily_archive: Dict[str, str] = {}
        # chat_id → 当前 active history 中已写入旧 archive 的回放前缀指纹。
        # 下一次归档跳过此段，避免回放消息进入第二份 archive。
        self._replayed_prefix_keys: Dict[str, List[str]] = {}
        # 已知键集合用于区分新版空前缀与旧版仅记录归档日期的 state。
        self._replayed_prefix_known: Set[str] = set()
        self._daily_state_lock = threading.Lock()
        self._daily_state_path = Path(memory_dir).parent / "daily_archive_state.json"
        self._load_daily_state()

    @property
    def _store(self):
        return self._cm.store

    @property
    def replay_gap_seconds(self) -> int:
        """昨天原始消息回放的连续会话间隔阈值（秒）。"""
        return self._replay_gap_seconds

    @property
    def summary_count(self) -> int:
        """摘要取最近 N 条有效消息。"""
        return self._summary_count

    # ── 同日归档状态持久化 ──

    def _load_daily_state(self) -> None:
        """恢复同日守卫和已归档回放前缀，兼容旧版 {chat_id: date} 格式。"""
        try:
            if not self._daily_state_path.is_file():
                return
            data = json.loads(self._daily_state_path.read_text(encoding="utf-8"))
            for chat_id, value in data.items():
                if isinstance(value, dict):
                    archived_on = value.get("archived_on")
                    if archived_on:
                        self._last_daily_archive[chat_id] = str(archived_on)
                    replayed_keys = value.get("replayed_prefix_keys", [])
                    if isinstance(replayed_keys, list) and all(
                        isinstance(key, str) for key in replayed_keys
                    ):
                        self._replayed_prefix_keys[chat_id] = replayed_keys
                        self._replayed_prefix_known.add(chat_id)
                elif isinstance(value, str):
                    self._last_daily_archive[chat_id] = value
        except Exception as e:
            _log.warning("加载归档状态失败 %s: %s", self._daily_state_path, e)

    def _save_daily_state(self) -> None:
        try:
            self._daily_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._daily_state_lock:
                state = {
                    chat_id: {
                        "archived_on": self._last_daily_archive.get(chat_id),
                        "replayed_prefix_keys": self._replayed_prefix_keys.get(
                            chat_id, []
                        ),
                    }
                    for chat_id in set(self._last_daily_archive)
                    | set(self._replayed_prefix_keys)
                }
                self._daily_state_path.write_text(
                    json.dumps(state, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as e:
            _log.warning("保存归档状态失败 %s: %s", self._daily_state_path, e)

    @staticmethod
    def _message_key(message: Any) -> str:
        """生成跨重启稳定的消息指纹，用于识别已归档的回放前缀。"""
        return json.dumps(
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp,
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "name": message.name,
                "tool_call_id": message.tool_call_id,
                "tool_name": message.tool_name,
                "tool_calls": message.tool_calls,
                "reasoning_content": message.reasoning_content,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _replayed_prefix_length(self, chat_id: str, messages: List[Any]) -> int:
        keys = self._replayed_prefix_keys.get(chat_id, [])
        if not keys and chat_id not in self._replayed_prefix_known:
            # v1 state 只保存了归档日。active history 中早于该日的连续前缀
            # 必然是上一轮留下的回放，首次读取时将其升级为 v2 指纹。
            last_archived_on = self._last_daily_archive.get(chat_id)
            if last_archived_on:
                legacy_prefix = []
                for message, unit_start in zip(
                    messages, self._unit_start_timestamps(messages)
                ):
                    if _date_str(unit_start) >= last_archived_on:
                        break
                    legacy_prefix.append(message)
                if legacy_prefix:
                    keys = [self._message_key(message) for message in legacy_prefix]
                    self._replayed_prefix_keys[chat_id] = keys
                    _log.info(
                        "已迁移归档回放前缀 [%s..]: %d 条",
                        chat_id[:12],
                        len(keys),
                    )
            self._replayed_prefix_known.add(chat_id)

        if not keys:
            return 0

        # ChatContext 是有界 deque，后续消息可能将回放前缀的最早记录挤出。
        # active history 的开头仍与已知前缀的一个后缀相同，就必须继续跳过
        # 这部分，不能将剩余回放消息视为新的待归档历史。
        max_overlap = min(len(messages), len(keys))
        for length in range(max_overlap, 0, -1):
            actual_keys = [self._message_key(message) for message in messages[:length]]
            if actual_keys == keys[-length:]:
                if length != len(keys):
                    _log.info(
                        "回放前缀已截断 [%s..]: 保留 %d/%d 条已归档记录",
                        chat_id[:12],
                        length,
                        len(keys),
                    )
                return length

        _log.warning("回放前缀不匹配 [%s..]，按未归档消息处理", chat_id[:12])
        return 0

    # ── 公开方法 ──

    async def archive_if_stale(
        self, chat_id: str, is_group: bool
    ) -> Optional[ArchiveResult]:
        async def _do(ctx):
            if ctx.is_empty():
                return None

            now = time.time()
            today = _date_str(now)

            history = ctx.get_history()
            if not self._crossed_day(history, today):
                return None

            # active history 可能以已归档的回放前缀开头。只有该前缀之后仍有
            # 今天之前的新消息时，才需要生成新的 archive 文件。
            replayed_prefix_length = self._replayed_prefix_length(chat_id, history)
            unit_start_times = self._unit_start_timestamps(history)
            has_unarchived_old_history = any(
                _date_str(unit_start) < today
                for unit_start in unit_start_times[replayed_prefix_length:]
            )
            if not has_unarchived_old_history:
                return None

            # 同一天通常只会进入一次；但迟到消息可能在此前归档之后才写入
            # history，必须允许它作为新的旧单元在当天补归档。
            if self._last_daily_archive.get(chat_id) == today:
                _log.info("检测到迟到旧消息，追加同日归档 [%s..]", chat_id[:12])

            result = await self._do_archive(ctx, chat_id, is_group, "daily")
            if result is not None:
                self._last_daily_archive[chat_id] = today
                self._save_daily_state()
            return result

        return await self._cm._with_context_locked(chat_id, _do)

    @staticmethod
    def _tool_transaction_start_indices(messages: List[Any]) -> Dict[int, int]:
        """返回 tool result 下标到发起该调用的 assistant 下标的映射。

        此映射只用于分区和回放切点判断；ChatMessage.timestamp 始终保持原值。
        """
        call_owners: Dict[str, int] = {}
        for index, message in enumerate(messages):
            if message.role != "assistant" or not message.tool_calls:
                continue
            for call in message.tool_calls:
                if isinstance(call, dict) and call.get("id"):
                    call_owners[call["id"]] = index

        return {
            index: call_owners[message.tool_call_id]
            for index, message in enumerate(messages)
            if message.role == "tool" and message.tool_call_id in call_owners
        }

    @classmethod
    def _unit_start_timestamps(cls, messages: List[Any]) -> List[float]:
        """计算每条记录所属单元的开始时间，不修改记录本身的 timestamp。"""
        transaction_starts = cls._tool_transaction_start_indices(messages)
        return [
            messages[transaction_starts.get(index, index)].timestamp
            for index in range(len(messages))
        ]

    @classmethod
    def _crossed_day(cls, messages: List[Any], today: str) -> bool:
        """是否有按所属单元开始时间归属到今天之前的消息。"""
        return any(
            _date_str(unit_start) < today
            for unit_start in cls._unit_start_timestamps(messages)
        )

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

    async def archive_manual(
        self, chat_id: str, is_group: Optional[bool] = None
    ) -> ArchiveResult:
        if is_group is None:
            is_group = self._cm.get_chat_type(chat_id)
            if is_group is None:
                _log.warning("未记录聊天类型 [%s..]，按私聊归档", chat_id[:12])
                is_group = False

        async def _do(ctx):
            return await self._do_archive(ctx, chat_id, is_group, "manual")

        result = await self._cm._with_context_locked(chat_id, _do)
        if result is not None:
            # 手动归档同样可能保留已归档的回放前缀；重启后必须能识别它。
            self._save_daily_state()
        return result

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

        # 清理过期的同日归档状态条目
        cutoff_date = _date_str(time.time() - retention_seconds)
        stale_chats = [
            cid for cid, d in self._last_daily_archive.items() if d < cutoff_date
        ]
        if stale_chats:
            for cid in stale_chats:
                self._last_daily_archive.pop(cid, None)
                self._replayed_prefix_keys.pop(cid, None)
                self._replayed_prefix_known.discard(cid)
            self._save_daily_state()

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

        # ChatContext 可能还有旧 history 的异步保存任务。必须先等待，避免
        # 该任务在归档完成后将已归档消息重新追加到 active JSONL。
        wait_for_save = getattr(ctx, "wait_for_save_async", None)
        if callable(wait_for_save):
            pending_save = wait_for_save()
            if inspect.isawaitable(pending_save):
                await pending_save

        # 1. 确保数据落盘（后台线程）
        await asyncio.to_thread(
            store.flush,
            chat_id,
            [m.to_storage_dict() for m in ctx.get_history()],
        )

        # 2. 收集消息。active history 的前缀可能是上一轮已进入 archive 的
        # 回放消息；它们仅用于上下文，不应再次写入新 archive。
        all_msgs = ctx.get_history()
        unit_start_times = self._unit_start_timestamps(all_msgs)
        replayed_prefix_length = self._replayed_prefix_length(chat_id, all_msgs)
        unarchived_indices = list(range(replayed_prefix_length, len(all_msgs)))
        old_indices = [
            index
            for index in unarchived_indices
            if _date_str(unit_start_times[index]) < date
        ]
        old_history_indices = [
            index
            for index, unit_start in enumerate(unit_start_times)
            if _date_str(unit_start) == _previous_date_str(now)
        ]
        today_indices = [
            index
            for index in unarchived_indices
            if _date_str(unit_start_times[index]) >= date
        ]
        old_msgs = [all_msgs[index] for index in old_indices]

        # 3. 今天及以后的单元全部保留。只允许回放昨天最后一个连续
        # 时间段；更早的历史仍会归档，但不能作为原始上下文进入今天。
        following_unit_start = (
            unit_start_times[today_indices[0]] if today_indices else None
        )
        replay_indices = self._select_replay_indices(
            all_msgs,
            old_history_indices,
            self._replay_gap_seconds,
            unit_start_times,
            following_unit_start,
        )

        base_keep_indices = set(today_indices) | set(replay_indices)
        # tool 调用与其全部结果是一个协议事务。跨日时，只要事务中任一消息
        # 必须保留，就将 assistant tool_calls 和每个对应 tool result 一起保留。
        keep_indices = self._close_tool_transactions(all_msgs, set(base_keep_indices))
        # 仅因工具事务闭包而保留的旧消息尚不完整归档，留到整个事务都属于
        # 旧历史时再写入 archive；普通回放消息仍会在本次 archive 中保留副本。
        transaction_carried_indices = keep_indices - base_keep_indices
        archive_indices = [
            index for index in old_indices if index not in transaction_carried_indices
        ]
        archived_msgs = [all_msgs[index] for index in archive_indices]
        sorted_keep_indices = sorted(keep_indices)
        keep_msgs = [
            self._copy_for_history(all_msgs[index]) for index in sorted_keep_indices
        ]

        # 4. 生成摘要：只总结被归档的部分（今天之前），
        #    今天的消息永远不会被卷进摘要。
        summary_text = self._format_summary_text(
            archived_msgs,
            self._summary_count,
            is_group,
            chat_id,
            date,
        )

        # 5. 仅写入本次尚未归档的旧消息；回放前缀不会再次进入 archive。
        archive_path = await asyncio.to_thread(
            store.archive_messages,
            chat_id,
            ts,
            [message.to_storage_dict() for message in archived_msgs],
        )

        # 6. 写入摘要 .md（后台线程）
        summary_path: Optional[str] = None
        if summary_text:
            summary_path = await self._write_memory_file(chat_id, date, summary_text)

        # 7. 保留新历史（今天全部 + 昨天最后一个连续时间段），并持久化其中
        # 已归档的回放前缀。该前缀在下一天只参与上下文，不会第二次写入 archive。
        ctx.set_messages(keep_msgs)
        archived_keep_indices = set(range(replayed_prefix_length)) | set(
            archive_indices
        )
        replayed_prefix_keys = []
        for index in sorted_keep_indices:
            if (
                index not in archived_keep_indices
                or _date_str(unit_start_times[index]) >= date
            ):
                break
            replayed_prefix_keys.append(self._message_key(all_msgs[index]))
        self._replayed_prefix_keys[chat_id] = replayed_prefix_keys
        self._replayed_prefix_known.add(chat_id)
        ctx.last_activity = time.time()

        # 8. 写入新数据（后台线程）。没有任何保留消息时必须删除 active
        # JSONL；JSONLContextStore.flush([]) 为兼容旧行为会直接返回。
        if keep_msgs:
            await asyncio.to_thread(
                store.flush,
                chat_id,
                [m.to_storage_dict() for m in keep_msgs],
            )
        else:
            await asyncio.to_thread(store.delete, chat_id)

        _log.info(
            "归档完成 [%s..]: reason=%s keep=%d (replay=%d+今天%d) summary=%s",
            chat_id[:12],
            reason,
            len(keep_msgs),
            len(replay_indices),
            len(today_indices),
            summary_path or "无",
        )

        result = ArchiveResult(
            chat_id=chat_id,
            reason=reason,
            archive_path=archive_path,
            summary_path=summary_path,
            replay_count=len(keep_msgs),
        )

        if summary_text and reason != "manual":
            self._pending_injection.add(chat_id)

        return result

    # ── 消息提取 ──

    def _is_replayable(self, msg: Any) -> bool:
        """消息是否参与回放/摘要（过滤 tool、工具调用、表情、system、空内容）。

        回放与摘要共用同一套谓词，避免过滤逻辑漂移。
        """
        if msg.role == "tool":
            return False
        if msg.role == "assistant" and msg.tool_calls:
            return False
        if (
            msg.role == "assistant"
            and msg.content
            and "[助手发送了一个表情]" in msg.content
        ):
            return False
        if msg.sender_id == "system":
            return False
        content = msg.content or ""
        if msg.role == "user":
            content = strip_content_prefix(content)
        return bool(content.strip())

    def _select_replay_indices(
        self,
        messages: List[Any],
        candidate_indices: List[int],
        gap_seconds: int,
        unit_start_times: Optional[List[float]] = None,
        following_unit_start: Optional[float] = None,
    ) -> List[int]:
        """选择最后一个完整连续时间段中的普通消息及完整工具事务。

        ``gap_seconds`` 是两个相邻消息所属单元的最大间隔。超过该间隔才允许
        在两段对话之间切开；不再按固定条数截断尾部。工具调用与结果即使
        本身不参与摘要，也必须作为同一个协议单元一起回放。
        """
        if gap_seconds <= 0 or not candidate_indices:
            return []
        unit_start_times = unit_start_times or self._unit_start_timestamps(messages)
        if (
            following_unit_start is not None
            and following_unit_start - unit_start_times[candidate_indices[-1]]
            > gap_seconds
        ):
            return []

        segment_start = len(candidate_indices) - 1
        for position in range(len(candidate_indices) - 1, 0, -1):
            previous = candidate_indices[position - 1]
            current = candidate_indices[position]
            if unit_start_times[current] - unit_start_times[previous] > gap_seconds:
                break
            segment_start = position - 1

        segment_indices = candidate_indices[segment_start:]
        if not any(self._is_replayable(messages[index]) for index in segment_indices):
            return []

        selected = {
            index for index in segment_indices if self._is_replayable(messages[index])
        }
        # 将段内工具事务作为原子单元加入；闭包会补齐跨段边界的配对记录。
        selected.update(
            index
            for index in segment_indices
            if messages[index].role == "tool"
            or (messages[index].role == "assistant" and messages[index].tool_calls)
        )
        return sorted(self._close_tool_transactions(messages, selected))

    def _close_tool_transactions(
        self, messages: List[Any], keep_indices: Set[int]
    ) -> Set[int]:
        """扩展保留集，使 tool 调用与对应结果不可被跨日切分。"""
        call_owners: Dict[str, int] = {}
        call_results: Dict[str, List[int]] = {}
        assistant_call_ids: Dict[int, Set[str]] = {}

        for index, message in enumerate(messages):
            if message.role == "assistant" and message.tool_calls:
                call_ids = {
                    call.get("id")
                    for call in message.tool_calls
                    if isinstance(call, dict) and call.get("id")
                }
                if call_ids:
                    assistant_call_ids[index] = call_ids
                    for call_id in call_ids:
                        call_owners[call_id] = index
            elif message.role == "tool" and message.tool_call_id:
                call_results.setdefault(message.tool_call_id, []).append(index)

        pending = list(keep_indices)
        while pending:
            index = pending.pop()
            message = messages[index]

            if message.role == "tool" and message.tool_call_id:
                owner = call_owners.get(message.tool_call_id)
                if owner is not None and owner not in keep_indices:
                    keep_indices.add(owner)
                    pending.append(owner)

            for call_id in assistant_call_ids.get(index, set()):
                for result_index in call_results.get(call_id, []):
                    if result_index not in keep_indices:
                        keep_indices.add(result_index)
                        pending.append(result_index)

        return keep_indices

    @staticmethod
    def _copy_for_history(msg: Any) -> ChatMessage:
        """复制保留消息，恢复历史时去除旧 JSONL 的 user 内容前缀。"""
        content = msg.content or ""
        if msg.role == "user":
            content = strip_content_prefix(content)
        return ChatMessage(
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

    def _extract_replay_messages(
        self, messages: List[Any], gap_seconds: Optional[int] = None
    ) -> List[Any]:
        indices = self._select_replay_indices(
            messages,
            list(range(len(messages))),
            self._replay_gap_seconds if gap_seconds is None else gap_seconds,
        )
        return [self._copy_for_history(messages[index]) for index in indices]

    def _format_summary_text(
        self,
        messages: List[Any],
        count: int,
        is_group: bool,
        chat_id: str,
        date: str,
    ) -> Optional[str]:
        # 1. 正向收集有效消息（与回放共用同一套过滤谓词，避免漂移）
        selected: List[ChatMessage] = []
        for msg in messages:
            if not self._is_replayable(msg):
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
