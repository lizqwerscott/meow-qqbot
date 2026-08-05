import asyncio
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Set

from core.managers.chat_message import (
    ChatMessage,
    _estimate_tokens,
    group_user_messages,
    strip_content_prefix,
)
from core.managers.context_store import ContextStore

_log = logging.getLogger(__name__)


class ChatContext:

    def __init__(
        self,
        chat_id: str,
        store: ContextStore,
        max_history: int = 10000,
        merge_window_seconds: int = 15,
    ):
        self.chat_id = chat_id
        self.store = store
        self.max_history = max_history
        self.merge_window_seconds = merge_window_seconds
        self.history = deque(maxlen=max_history)
        self.last_activity = time.time()
        self.lock = asyncio.Lock()
        self._save_task: Optional[asyncio.Task] = None
        self._save_pending = False

    # ── 消息添加 ──

    def add_message(
        self,
        role: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            message_id=message_id,
            sender_id=sender_id,
            name=name,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        self.history.append(message)
        self.last_activity = time.time()
        self._try_schedule_save()

    def add_user_message(
        self,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.add_message("user", content, message_id, sender_id=sender_id, name=name)

    def add_assistant_message(
        self,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        self.add_message(
            "assistant",
            content,
            message_id,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def add_tool_result(
        self,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        self.add_message(
            "tool", content, tool_call_id=tool_call_id, tool_name=tool_name
        )

    # ── 消息读取 ──

    def get_history(self, max_messages: Optional[int] = None) -> List[ChatMessage]:
        if max_messages is None:
            return list(self.history)
        return list(self.history)[-max_messages:]

    def get_history_as_dicts(self, max_messages: Optional[int] = None) -> List[Dict]:
        messages = self.get_history(max_messages)
        return [msg.to_dict() for msg in messages]

    def get_history_as_dicts_merged(
        self, max_messages: Optional[int] = None
    ) -> List[Dict]:
        """返回合并后的消息 dict 列表（仅合并连续同用户消息）。

        原始 ChatContext.history 不变，所有现有系统不受影响。
        """
        messages = self.get_history(max_messages)
        groups = group_user_messages(messages)
        result: List[Dict] = []
        for group in groups:
            if len(group) == 1 or group[0].role != "user":
                result.append(group[0].to_dict())
            else:
                merged = _build_merged_dict(group, self.merge_window_seconds)
                if merged is not None:
                    result.append(merged)
        return result

    def get_conversation_context(self, max_messages: Optional[int] = None) -> str:
        messages = self.get_history(max_messages)
        context_lines = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            time_str = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            context_lines.append(f"[{time_str}] {role_label}: {msg.content}")
        return "\n".join(context_lines)

    def get_last_message(self) -> Optional[ChatMessage]:
        if self.history:
            return self.history[-1]
        return None

    def remove_last_message_if(self, role: str, message_id: str) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        if last.role == role and last.message_id == message_id:
            self.history.pop()
            self.last_activity = time.time()
            return True
        return False

    # ── 状态查询 ──

    def estimate_tokens_for_history(self) -> int:
        total = 0
        for msg in self.history:
            total += _estimate_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += _estimate_tokens(tc.get("function", {}).get("name"))
                    total += _estimate_tokens(tc.get("function", {}).get("arguments"))
            if msg.reasoning_content:
                total += _estimate_tokens(msg.reasoning_content)
        return total

    def get_history_count(self) -> int:
        return len(self.history)

    def is_empty(self) -> bool:
        return len(self.history) == 0

    def get_inactivity_time(self) -> float:
        return time.time() - self.last_activity

    # ── 历史管理 ──

    def clear_history(self) -> None:
        self.history.clear()
        self.store.delete(self.chat_id)

    def set_messages(self, messages: List[ChatMessage]) -> None:
        self.history = deque(messages, maxlen=self.max_history)
        self.last_activity = time.time()
        self._try_schedule_save()

    def restore_from_store(self) -> bool:
        data = self.store.load(self.chat_id)
        if data is None:
            return False
        self._restore_from_data(data)
        return True

    async def restore_from_store_async(self) -> bool:
        data = await self.store.load_async(self.chat_id)
        if data is None:
            return False
        self._restore_from_data(data)
        return True

    def _restore_from_data(self, data: List[dict]) -> None:
        for item in data:
            try:
                self.history.append(ChatMessage.from_dict(item))
            except Exception as e:
                _log.warning("跳过损坏的历史条目 [%s..]: %s", self.chat_id[:12], e)

        removed = self.remove_orphaned_tool_calls()
        if removed:
            _log.info(
                "恢复后清理了 %d 条孤立 tool_calls/tool 消息 [%s..]",
                removed,
                self.chat_id[:12],
            )

        if self._is_expired():
            _log.info(
                "会话缓存已过期 [%s..]，交由 ArchiveManager 处理",
                self.chat_id[:12],
            )
            self.last_activity = (
                self.history[-1].timestamp if self.history else time.time()
            )
        else:
            self.last_activity = time.time()
            _log.info(
                "从缓存恢复会话 [%s..] (%d 条)",
                self.chat_id[:12],
                len(self.history),
            )

    def _is_expired(self, max_age: float = 86400) -> bool:
        if not self.history:
            return True
        latest = self.history[-1].timestamp
        return (time.time() - latest) > max_age

    # ── 异步持久化调度 ──

    def _try_schedule_save(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.store.flush(
                self.chat_id,
                [message.to_dict() for message in self.history],
            )
            return
        if self._save_task and not self._save_task.done():
            self._save_pending = True
            return
        self._save_pending = False
        messages = [m.to_dict() for m in self.history]
        self._save_task = asyncio.ensure_future(
            asyncio.to_thread(self.store.flush, self.chat_id, messages)
        )
        self._save_task.add_done_callback(self._on_save_done)

    def _on_save_done(self, task: asyncio.Task) -> None:
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error:
            _log.error(
                "持久化到存储失败 [%s..]: %s",
                self.chat_id[:12],
                error,
            )
        if self._save_pending:
            self._try_schedule_save()

    # ── 异步消息添加（带锁） ──

    async def add_message_async(
        self,
        role: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        async with self.lock:
            self.add_message(
                role,
                content,
                message_id,
                sender_id=sender_id,
                name=name,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )

    async def add_assistant_message_async(
        self,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        async with self.lock:
            self.add_assistant_message(
                content,
                message_id,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )

    async def add_tool_result_async(
        self,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        async with self.lock:
            self.add_tool_result(tool_name, content, tool_call_id)

    # ── 修剪 (Pruning) ──

    @staticmethod
    def _find_protected_boundary(
        messages: List[ChatMessage], keep_last_assistants: int
    ) -> int:
        assistant_count = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                assistant_count += 1
                if assistant_count >= keep_last_assistants:
                    return i
        return 0

    def get_pruned_history(
        self,
        max_messages: int = 12,
        max_tool_results: int = 5,
        keep_last_assistants: int = 3,
        soft_trim: int = 20000,
        hard_clear: int = 180000,
    ) -> List[Dict]:
        all_msgs = list(self.history)
        if not all_msgs:
            return []

        boundary = self._find_protected_boundary(all_msgs, keep_last_assistants)

        result = []
        tool_count = 0

        for i, msg in enumerate(all_msgs):
            d = msg.to_dict()

            if msg.role != "tool":
                result.append(d)
                continue

            tool_count += 1
            is_protected = i >= boundary
            is_overflow = tool_count > max_tool_results

            if is_protected and not is_overflow:
                result.append(d)
            else:
                content = msg.content
                if len(content) > hard_clear:
                    d["content"] = (
                        "[工具 " + (msg.tool_name or "未知") + " 的调用结果已裁剪]"
                    )
                elif len(content) > soft_trim:
                    d["content"] = (
                        content[:1500] + "\n\n…[中间内容已裁剪]…\n\n" + content[-1500:]
                    )
                result.append(d)

        result = result[-max_messages:]

        valid_ids: Set[str] = set()
        cleaned: List[Dict] = []
        for d in result:
            if d.get("role") == "assistant" and d.get("tool_calls"):
                for tc in d["tool_calls"]:
                    valid_ids.add(tc["id"])
                cleaned.append(d)
            elif d.get("role") == "tool":
                if d.get("tool_call_id") in valid_ids:
                    cleaned.append(d)
            else:
                cleaned.append(d)

        responded_ids: Set[str] = set()
        for d in cleaned:
            if d.get("role") == "tool" and d.get("tool_call_id"):
                responded_ids.add(d["tool_call_id"])

        for d in cleaned:
            if d.get("role") == "assistant" and d.get("tool_calls"):
                call_ids = {tc["id"] for tc in d["tool_calls"]}
                if not call_ids.issubset(responded_ids):
                    del d["tool_calls"]
                    if not d.get("content"):
                        d["content"] = None

        return cleaned

    # ── 孤儿消息清理 ──

    def remove_orphaned_tool_calls(self) -> int:
        removed = 0
        result = []
        history_list = list(self.history)
        expected_ids: set = set()
        i = 0
        while i < len(history_list):
            msg = history_list[i]
            if msg.role == "assistant" and msg.tool_calls:
                tc_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                if tc_ids:
                    found_ids = set()
                    for j in range(i + 1, len(history_list)):
                        m = history_list[j]
                        if m.role == "tool" and m.tool_call_id:
                            found_ids.add(m.tool_call_id)
                    if not tc_ids.issubset(found_ids):
                        missing = tc_ids - found_ids
                        _log.warning(
                            "移除孤立 tool_calls [%s..]: missing_ids=%s",
                            self.chat_id[:12],
                            missing,
                        )
                        i += 1
                        removed += 1
                        continue
                    expected_ids.update(tc_ids)
            elif msg.role == "tool" and msg.tool_call_id:
                if msg.tool_call_id in expected_ids:
                    expected_ids.discard(msg.tool_call_id)
                else:
                    _log.warning(
                        "移除孤立 tool 消息 [%s..]: tool_call_id=%s",
                        self.chat_id[:12],
                        msg.tool_call_id,
                    )
                    i += 1
                    removed += 1
                    continue
            result.append(msg)
            i += 1

        if removed > 0:
            self.history = deque(result, maxlen=self.max_history)

        return removed


def _build_merged_dict(group: List[ChatMessage], window_seconds: int) -> Optional[Dict]:
    """将一组连续同发送人的 user 消息合并为单个 dict。

    group 由 group_user_messages() 产出，保证所有 msg.role == "user"
    且 sender_id 相同。

    合并规则：
    - 用组内第一个有效内容的消息生成前缀
    - 间隔 ≤ window_seconds：\n 直接拼接
    - 间隔 > window_seconds：插入 [HH:MM:SS] 标记后拼接
    - 内容为空的消息跳过
    """
    first = _first_non_empty(group)
    if first is None:
        return None

    d = first.to_dict()
    merged = [d["content"]]
    prev_ts = first.timestamp

    for msg in group:
        if msg is first:
            continue
        raw = strip_content_prefix(msg.content).strip()
        if not raw:
            continue

        gap = msg.timestamp - prev_ts
        if gap > window_seconds:
            ts_marker = time.strftime("[%H:%M:%S]", time.localtime(msg.timestamp))
            merged.append(ts_marker)
        merged.append(raw)
        prev_ts = msg.timestamp

    d["content"] = "\n".join(merged)
    return d


def _first_non_empty(group: List[ChatMessage]) -> Optional[ChatMessage]:
    """找到组内第一个有非空内容的消息。"""
    for msg in group:
        if msg.content.strip():
            return msg
    return None
