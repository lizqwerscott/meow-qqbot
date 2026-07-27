import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from core.managers.chat_message import ChatMessage, _estimate_tokens, group_user_messages, strip_content_prefix
from core.managers.context_store import ContextStore

_log = logging.getLogger(__name__)


class ChatContext:

    def __init__(
        self,
        chat_id: str,
        store: ContextStore,
        max_history: int = 10000,
        compact_threshold_tokens: int = 950000,
        keep_recent_tokens: int = 50000,
        merge_window_seconds: int = 15,
    ):
        self.chat_id = chat_id
        self.store = store
        self.max_history = max_history
        self.compact_threshold_tokens = compact_threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.merge_window_seconds = merge_window_seconds
        self.history = deque(maxlen=max_history)
        self.last_activity = time.time()
        self.lock = asyncio.Lock()
        self._save_task: Optional[asyncio.Task] = None

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

    def get_history_as_dicts(
        self, max_messages: Optional[int] = None
    ) -> List[Dict]:
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

    def get_conversation_context(
        self, max_messages: Optional[int] = None
    ) -> str:
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

    # ── 状态查询 ──

    def estimate_tokens_for_history(self) -> int:
        total = 0
        for msg in self.history:
            total += _estimate_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += _estimate_tokens(
                        tc.get("function", {}).get("name")
                    )
                    total += _estimate_tokens(
                        tc.get("function", {}).get("arguments")
                    )
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
        self.history.clear()
        for msg in messages:
            self.history.append(msg)

    def restore_from_store(self) -> bool:
        data = self.store.load(self.chat_id)
        if data is None:
            return False
        for item in data:
            self.history.append(ChatMessage.from_dict(item))

        if self._is_expired():
            _log.info(
                "会话缓存已过期 [%s..]，交由 ArchiveManager 处理",
                self.chat_id[:12],
            )
            self.last_activity = self.history[-1].timestamp
            return False
        self.last_activity = time.time()
        _log.info(
            "从缓存恢复会话 [%s..] (%d 条)",
            self.chat_id[:12], len(self.history),
        )
        return True

    def _is_expired(self, max_age: float = 86400) -> bool:
        if not self.history:
            return True
        latest = self.history[-1].timestamp
        return (time.time() - latest) > max_age

    # ── 异步持久化调度 ──

    def _try_schedule_save(self) -> None:
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(self._schedule_save())
        except RuntimeError:
            pass

    async def _schedule_save(self) -> None:
        if self._save_task and not self._save_task.done():
            return
        messages = [m.to_dict() for m in self.history]
        self._save_task = asyncio.create_task(
            asyncio.to_thread(self.store.flush, self.chat_id, messages)
        )

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
                content, message_id, tool_calls=tool_calls,
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
                        "[工具 " + (msg.tool_name or "未知")
                        + " 的调用结果已裁剪]"
                    )
                elif len(content) > soft_trim:
                    d["content"] = (
                        content[:1500]
                        + "\n\n…[中间内容已裁剪]…\n\n"
                        + content[-1500:]
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
                tc_ids = {
                    tc.get("id") for tc in msg.tool_calls if tc.get("id")
                }
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
                            self.chat_id[:12], missing,
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
                        self.chat_id[:12], msg.tool_call_id,
                    )
                    i += 1
                    removed += 1
                    continue
            result.append(msg)
            i += 1

        if removed > 0:
            self.history = deque(result, maxlen=self.max_history)

        return removed

    # ── Token 辅助方法 ──

    def _split_by_token_budget(
        self, messages: List[ChatMessage], recent_budget: int
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        recent: List[ChatMessage] = []
        total = 0
        recent_tool_ids: set = set()

        for msg in reversed(messages):
            tokens = _estimate_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tokens += _estimate_tokens(
                        tc.get("function", {}).get("name")
                    )
                    tokens += _estimate_tokens(
                        tc.get("function", {}).get("arguments")
                    )
            if msg.reasoning_content:
                tokens += _estimate_tokens(msg.reasoning_content)

            if total + tokens > recent_budget and recent:
                if (
                    msg.role == "tool"
                    and msg.tool_call_id
                    and msg.tool_call_id in recent_tool_ids
                ):
                    recent.insert(0, msg)
                    total += tokens
                    continue

                if msg.tool_calls:
                    tc_ids = {
                        tc.get("id") for tc in msg.tool_calls if tc.get("id")
                    }
                    if tc_ids & recent_tool_ids:
                        recent.insert(0, msg)
                        total += tokens
                        for tc in msg.tool_calls:
                            tid = tc.get("id")
                            if tid:
                                recent_tool_ids.discard(tid)
                        continue

                break

            recent.insert(0, msg)
            total += tokens
            if msg.role == "tool" and msg.tool_call_id:
                recent_tool_ids.add(msg.tool_call_id)

        old = (
            messages[: -len(recent)]
            if recent
            else messages[:-1] if len(messages) > 1 else []
        )
        return old, recent

    # ── 压缩 (Compaction) ──

    def _format_for_summary(self, messages: List[ChatMessage]) -> str:
        lines = []
        for m in messages:
            time_str = time.strftime(
                "%m-%d %H:%M", time.localtime(m.timestamp)
            )
            if m.role == "user":
                display_name = m.name or m.sender_id or "用户"
                lines.append(f"[{time_str}] {display_name}: {m.content}")
            elif m.role == "assistant":
                if m.tool_calls:
                    tools = ", ".join(
                        tc["function"]["name"] for tc in m.tool_calls
                    )
                    lines.append(
                        f"[{time_str}] 助手(调用工具: {tools}): {m.content}"
                    )
                else:
                    lines.append(f"[{time_str}] 助手: {m.content}")
            elif m.role == "tool":
                tname = m.tool_name or "工具"
                content_preview = m.content[:100].replace("\n", " ")
                lines.append(
                    f"[{time_str}] {tname} 返回: {content_preview}..."
                )
        return "\n".join(lines)

    async def compact_history_if_needed(
        self, ai_service: Any, force: bool = False
    ) -> tuple[bool, Optional[Dict]]:
        estimated = self.estimate_tokens_for_history()
        if not force and estimated < self.compact_threshold_tokens:
            return False, None

        all_msgs = list(self.history)
        old_msgs, recent_msgs = self._split_by_token_budget(
            all_msgs, self.keep_recent_tokens
        )

        if not old_msgs:
            return False, None

        text = self._format_for_summary(old_msgs)
        _log.info(
            "正在压缩 [%s..] %d 条消息 → 摘要 "
            "(估算 %d tokens > %d)",
            self.chat_id[:12], len(old_msgs), estimated,
            self.compact_threshold_tokens,
        )

        try:
            summary, usage = await ai_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个对话摘要助手。请将以下对话内容压缩为一段"
                            "简洁的摘要，保留重要的事实、决定、用户偏好、约定等"
                            "关键信息。不要添加原文没有的内容。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"请总结以下对话：\n\n{text}",
                    },
                ],
                max_tokens=500,
            )
        except Exception as e:
            _log.warning(f"压缩失败: {e!r}")
            return False, None

        if not summary:
            _log.warning("压缩返回空结果，跳过")
            return False, usage

        summary = summary.strip()
        _log.info(
            "压缩完成: %d 条 → 摘要 (%d 字符)",
            len(old_msgs), len(summary),
        )

        timestamp = old_msgs[0].timestamp
        new_history = deque(maxlen=self.max_history)
        new_history.append(
            ChatMessage(
                role="assistant",
                content=f"【历史对话摘要】\n{summary}",
                timestamp=timestamp,
                name="系统",
            )
        )
        for m in recent_msgs:
            new_history.append(m)
        self.history = new_history
        return True, usage


def _build_merged_dict(
    group: List[ChatMessage], window_seconds: int
) -> Optional[Dict]:
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
