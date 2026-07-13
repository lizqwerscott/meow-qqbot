import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

_log = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """聊天消息记录"""

    role: str  # "user" | "assistant" | "tool"
    content: str
    timestamp: float
    message_id: Optional[str] = None
    sender_id: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    reasoning_content: Optional[str] = None

    def to_dict(self) -> Dict:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id,
                "content": self.content,
            }

        time_str = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
        )

        content = self.content
        if self.role == "user":
            display_name = self.name or self.sender_id or "未知"
            content = f"[{display_name} 在 {time_str}]: {self.content}"

        d: Dict = {
            "role": self.role,
            "content": content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
        }
        if self.role == "user" and self.name is not None:
            d["name"] = self.name
        if self.role == "assistant":
            if self.tool_calls:
                d["tool_calls"] = self.tool_calls
                if not self.content:
                    d["content"] = None
            if self.reasoning_content:
                d["reasoning_content"] = self.reasoning_content
        return d


class ChatContext:
    """
    单个聊天的上下文管理器
    每个 chat_id 对应一个实例
    """

    def __init__(
        self,
        chat_id: str,
        max_history: int = 30,
        compact_threshold: int = 25,
        keep_recent: int = 8,
    ):
        self.chat_id = chat_id
        self.max_history = max_history
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent
        self.history = deque(maxlen=max_history)
        self.last_activity = time.time()
        self.lock = asyncio.Lock()

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
            "assistant", content, message_id,
            tool_calls=tool_calls, reasoning_content=reasoning_content,
        )

    def add_tool_result(
        self,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        self.add_message("tool", content, tool_call_id=tool_call_id, tool_name=tool_name)

    def get_history(self, max_messages: Optional[int] = None) -> List[ChatMessage]:
        if max_messages is None:
            return list(self.history)
        return list(self.history)[-max_messages:]

    def get_history_as_dicts(self, max_messages: Optional[int] = None) -> List[Dict]:
        messages = self.get_history(max_messages)
        return [msg.to_dict() for msg in messages]

    def get_conversation_context(self, max_messages: Optional[int] = None) -> str:
        messages = self.get_history(max_messages)
        context_lines = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            time_str = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            context_lines.append(f"[{time_str}] {role_label}: {msg.content}")
        return "\n".join(context_lines)

    def clear_history(self) -> None:
        self.history.clear()

    def get_history_count(self) -> int:
        return len(self.history)

    def is_empty(self) -> bool:
        return len(self.history) == 0

    def get_last_message(self) -> Optional[ChatMessage]:
        if self.history:
            return self.history[-1]
        return None

    def get_inactivity_time(self) -> float:
        return time.time() - self.last_activity

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
                role, content, message_id,
                sender_id=sender_id, name=name,
                tool_call_id=tool_call_id, tool_name=tool_name,
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
                content, message_id,
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

    # ── 修剪 (Pruning) — 纯内存，不修改原始数据 ──

    @staticmethod
    def _find_protected_boundary(
        messages: List[ChatMessage], keep_last_assistants: int
    ) -> int:
        """找到保护区的起始索引（该索引之后的消息不做内容裁剪）"""
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
        soft_trim: int = 3000,
        hard_clear: int = 10000,
    ) -> List[Dict]:
        """获取经过裁剪的历史记录（纯内存，不修改原始 ChatMessage）"""
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
                        f"[工具 {msg.tool_name or '未知'} 的调用结果已裁剪]"
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

    # ── 压缩 (Compaction) — 调用 AI 总结旧对话 ──

    def _format_for_summary(self, messages: List[ChatMessage]) -> str:
        """将消息列表格式化为纯文本供 AI 总结"""
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
        """如果历史超过阈值，用 AI 将旧消息压缩为摘要。
        返回 (是否执行了压缩, usage dict 或 None)。
        """
        if not force and len(self.history) < self.compact_threshold:
            return False, None

        all_msgs = list(self.history)
        old_msgs = all_msgs[: -self.keep_recent]
        recent_msgs = all_msgs[-self.keep_recent :]

        if not old_msgs:
            return False, None

        text = self._format_for_summary(old_msgs)
        _log.info(
            f"正在压缩 [{self.chat_id[:12]}..] "
            f"{len(old_msgs)} 条消息 → 摘要"
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

        summary.strip()
        _log.info(f"压缩完成: {len(old_msgs)} 条 → 摘要 ({len(summary)} 字符)")

        timestamp = old_msgs[0].timestamp
        new_history = deque(maxlen=self.max_history)
        new_history.append(ChatMessage(
            role="assistant",
            content=f"【历史对话摘要】\n{summary}",
            timestamp=timestamp,
            name="系统",
        ))
        for m in recent_msgs:
            new_history.append(m)
        self.history = new_history
        return True, usage


class ChatContextManager:
    """
    聊天上下文管理器
    管理所有 chat_id 的上下文
    """

    def __init__(
        self,
        max_history_per_chat: int = 30,
        cleanup_interval: int = 3600,
        compact_threshold: int = 25,
        keep_recent: int = 8,
        max_conversation_messages: int = 12,
        max_tool_results: int = 5,
        keep_last_assistants: int = 3,
        soft_trim: int = 3000,
        hard_clear: int = 10000,
    ):
        self.max_history_per_chat = max_history_per_chat
        self.cleanup_interval = cleanup_interval
        self.compact_threshold = compact_threshold
        self.keep_recent = keep_recent
        self.max_conversation_messages = max_conversation_messages
        self.max_tool_results = max_tool_results
        self.keep_last_assistants = keep_last_assistants
        self.soft_trim = soft_trim
        self.hard_clear = hard_clear
        self.contexts: Dict[str, ChatContext] = {}
        self._ctx_lock = asyncio.Lock()                     # 保护 self.contexts 字典
        self._chat_locks: Dict[str, asyncio.Lock] = {}      # 每 chat 操作锁

    async def _get_chat_lock(self, chat_id: str) -> asyncio.Lock:
        async with self._ctx_lock:
            if chat_id not in self._chat_locks:
                self._chat_locks[chat_id] = asyncio.Lock()
            return self._chat_locks[chat_id]

    def get_context(self, chat_id: str) -> ChatContext:
        if chat_id not in self.contexts:
            self.contexts[chat_id] = ChatContext(
                chat_id,
                max_history=self.max_history_per_chat,
                compact_threshold=self.compact_threshold,
                keep_recent=self.keep_recent,
            )
        return self.contexts[chat_id]

    async def get_context_async(self, chat_id: str) -> ChatContext:
        async with self._ctx_lock:
            return self.get_context(chat_id)

    def add_user_message(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        context = self.get_context(chat_id)
        context.add_user_message(content, message_id, sender_id=sender_id, name=name)

    def add_assistant_message(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        context = self.get_context(chat_id)
        context.add_assistant_message(
            content, message_id,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def add_tool_result(
        self,
        chat_id: str,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        context = self.get_context(chat_id)
        context.add_tool_result(tool_name, content, tool_call_id)

    async def add_user_message_async(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.add_user_message(
                chat_id, content, message_id, sender_id=sender_id, name=name
            )

    async def add_assistant_message_async(
        self,
        chat_id: str,
        content: str,
        message_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        reasoning_content: Optional[str] = None,
    ) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.add_assistant_message(
                chat_id, content, message_id,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )

    async def add_tool_result_async(
        self,
        chat_id: str,
        tool_name: str,
        content: str,
        tool_call_id: str,
    ) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.add_tool_result(chat_id, tool_name, content, tool_call_id)

    def get_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        if chat_id not in self.contexts:
            return []
        return self.contexts[chat_id].get_history_as_dicts(max_messages)

    def get_chat_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        return self.get_history(chat_id, max_messages)

    async def get_chat_history_async(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            return self.get_chat_history(chat_id, max_messages)

    async def get_pruned_history_async(
        self,
        chat_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict]:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            if chat_id not in self.contexts:
                return []
            ctx = self.contexts[chat_id]
            return ctx.get_pruned_history(
                max_messages=max_messages or self.max_conversation_messages,
                max_tool_results=self.max_tool_results,
                keep_last_assistants=self.keep_last_assistants,
                soft_trim=self.soft_trim,
                hard_clear=self.hard_clear,
            )

    def get_conversation_context(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> str:
        if chat_id not in self.contexts:
            return ""
        return self.contexts[chat_id].get_conversation_context(max_messages)

    def clear_history(self, chat_id: str) -> None:
        if chat_id in self.contexts:
            self.contexts[chat_id].clear_history()

    def clear_chat_history(self, chat_id: str) -> None:
        self.clear_history(chat_id)

    async def clear_chat_history_async(self, chat_id: str) -> None:
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            self.clear_chat_history(chat_id)

    def remove_context(self, chat_id: str) -> None:
        if chat_id in self.contexts:
            del self.contexts[chat_id]
        self._chat_locks.pop(chat_id, None)

    def cleanup_inactive_contexts(self, max_inactivity: int = 7200) -> List[str]:
        removed = []
        current_time = time.time()
        for chat_id, context in list(self.contexts.items()):
            if context.get_inactivity_time() > max_inactivity:
                removed.append(chat_id)
                del self.contexts[chat_id]
                self._chat_locks.pop(chat_id, None)
        return removed

    async def cleanup_inactive_contexts_async(
        self, max_inactivity: int = 7200
    ) -> List[str]:
        async with self._ctx_lock:
            removed = self.cleanup_inactive_contexts(max_inactivity)
            for cid in removed:
                self._chat_locks.pop(cid, None)
            return removed

    def get_all_chat_ids(self) -> List[str]:
        return list(self.contexts.keys())

    def get_all_chats(self) -> Dict[str, ChatContext]:
        return self.contexts.copy()

    def get_context_count(self) -> int:
        return len(self.contexts)

    def get_total_messages_count(self) -> int:
        total = 0
        for context in self.contexts.values():
            total += len(context.history)
        return total
