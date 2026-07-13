import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from deepseek_tokenizer import ds_token

_log = logging.getLogger(__name__)


def _estimate_tokens(text: Optional[str]) -> int:
    """用 deepseek-tokenizer 精确估算 token 数。"""
    if not text:
        return 0
    return len(ds_token.encode(text))


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

    @staticmethod
    def from_dict(data: dict) -> "ChatMessage":
        return ChatMessage(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
            message_id=data.get("message_id"),
            sender_id=data.get("sender_id"),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            tool_calls=data.get("tool_calls"),
            reasoning_content=data.get("reasoning_content"),
        )


class ChatContext:
    """
    单个聊天的上下文管理器
    每个 chat_id 对应一个实例
    """

    def __init__(
        self,
        chat_id: str,
        max_history: int = 10000,
        compact_threshold_tokens: int = 950000,
        keep_recent_tokens: int = 50000,
        cache_dir: Optional[str] = None,
    ):
        self.chat_id = chat_id
        self.max_history = max_history
        self.compact_threshold_tokens = compact_threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.history = deque(maxlen=max_history)
        self.last_activity = time.time()
        self.lock = asyncio.Lock()
        self._cache_dir: Optional[str] = cache_dir
        self._save_task: Optional[asyncio.Task] = None

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

    def estimate_tokens_for_history(self) -> int:
        """估算当前历史的总 token 数"""
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

    def remove_orphaned_tool_calls(self) -> int:
        """移除历史中孤立的 assistant(tool_calls) 消息（没有对应 tool 响应的）。

        当工具执行被 CancelledError 中断时，可能出现 assistant 已写入但
        tool 响应未写入的情况。此方法清理这类孤立消息，防止后续 API 400 错误。

        Returns:
            移除的消息数量。
        """
        removed = 0
        result = []
        history_list = list(self.history)
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
                            f"移除孤立 tool_calls [{self.chat_id[:12]}..]: "
                            f"missing_ids={missing}"
                        )
                        i += 1
                        removed += 1
                        continue
            result.append(msg)
            i += 1

        if removed > 0:
            self.history = deque(result, maxlen=self.max_history)

        return removed

    # ── 本地缓存持久化 ──

    def _get_cache_path(self) -> Optional[Path]:
        if not self._cache_dir:
            return None
        return Path(self._cache_dir) / f"{self.chat_id}.json"

    def _is_expired(self, max_age: float = 86400) -> bool:
        if not self.history:
            return True
        latest = self.history[-1].timestamp
        return (time.time() - latest) > max_age

    def _serialize(self) -> list:
        return [msg.to_dict() for msg in self.history]

    def _deserialize(self, data: list) -> None:
        for item in data:
            self.history.append(ChatMessage.from_dict(item))

    def save(self) -> None:
        path = self._get_cache_path()
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self._serialize()
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            _log.warning("保存会话缓存失败 [%s..]: %s", self.chat_id[:12], e)

    def load(self) -> bool:
        """从本地缓存加载历史。如果已过期则删除缓存文件并返回 False。"""
        path = self._get_cache_path()
        if not path or not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data:
                return False
            self._deserialize(data)
            if self._is_expired():
                _log.info("会话缓存已过期，遗忘 [%s..]", self.chat_id[:12])
                self.history.clear()
                path.unlink(missing_ok=True)
                return False
            self.last_activity = time.time()
            _log.info("从本地缓存恢复会话 [%s..] (%d 条)", self.chat_id[:12], len(self.history))
            return True
        except Exception as e:
            _log.warning("加载会话缓存失败 [%s..]: %s", self.chat_id[:12], e)
            return False

    async def _schedule_save(self) -> None:
        if self._save_task and not self._save_task.done():
            return
        self._save_task = asyncio.create_task(asyncio.to_thread(self.save))

    def _try_schedule_save(self) -> None:
        if not self._cache_dir:
            return
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(self._schedule_save())
        except RuntimeError:
            pass

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
        soft_trim: int = 20000,
        hard_clear: int = 180000,
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

    def _split_by_token_budget(
        self, messages: List[ChatMessage], recent_budget: int
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        """按 token 预算将消息分为 old（待压缩）和 recent（保留原样）。

        返回 (old_msgs, recent_msgs)，其中 recent 的总 token 不超过 budget。
        注意：保证 assistant(tool_calls) 和其对应的 tool(responses) 不被拆散。
        """
        recent: List[ChatMessage] = []
        total = 0
        recent_tool_ids: set = set()  # 记录 recent 中出现的 tool_call_id

        for msg in reversed(messages):
            tokens = _estimate_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tokens += _estimate_tokens(tc.get("function", {}).get("name"))
                    tokens += _estimate_tokens(tc.get("function", {}).get("arguments"))
            if msg.reasoning_content:
                tokens += _estimate_tokens(msg.reasoning_content)

            if total + tokens > recent_budget and recent:
                # ── 决定 break 前，检查是否拆散了 tool 配对 ──
                # 如果当前 msg 是 tool 响应，对应的 assistant(tc) 可能在 recent 中？
                # 如果是 assistant(tc)，对应的 tool 响应在 recent 中但未被包含？
                # 两种都要确保配对完整性。

                # case 1: 当前 msg 是 tool 响应且其 call_id 在 recent 的 assistant 中
                if msg.role == "tool" and msg.tool_call_id and msg.tool_call_id in recent_tool_ids:
                    # tool 响应不能没有前面的 assistant → 加入 recent
                    recent.insert(0, msg)
                    total += tokens
                    continue

                # case 2: 当前 msg 是 assistant(tc)，检查其 tool_calls 是否已有响应在 recent 中
                if msg.tool_calls:
                    tc_ids = {tc.get("id") for tc in msg.tool_calls if tc.get("id")}
                    if tc_ids & recent_tool_ids:
                        # 对应响应已加入 recent，assistant 也必须加入
                        recent.insert(0, msg)
                        total += tokens
                        for tc in msg.tool_calls:
                            tid = tc.get("id")
                            if tid:
                                recent_tool_ids.discard(tid)
                        continue

                # 真的超预算了，break
                break

            recent.insert(0, msg)
            total += tokens
            # 记录 tool_call_id
            if msg.role == "tool" and msg.tool_call_id:
                recent_tool_ids.add(msg.tool_call_id)

        old = messages[: -len(recent)] if recent else messages[:-1] if len(messages) > 1 else []
        return old, recent

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
        """如果历史超过 token 阈值，用 AI 将旧消息压缩为摘要。
        返回 (是否执行了压缩, usage dict 或 None)。
        触发条件：估算 token 数 > compact_threshold_tokens
        """
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
            f"正在压缩 [{self.chat_id[:12]}..] "
            f"{len(old_msgs)} 条消息 → 摘要 "
            f"(估算 {estimated} tokens > {self.compact_threshold_tokens})"
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
        max_history_per_chat: int = 10000,
        cleanup_interval: int = 3600,
        compact_threshold_tokens: int = 950000,
        keep_recent_tokens: int = 50000,
        max_tool_results: int = 5,
        keep_last_assistants: int = 3,
        soft_trim: int = 20000,
        hard_clear: int = 180000,
        cache_dir: Optional[str] = None,
    ):
        self.max_history_per_chat = max_history_per_chat
        self.cleanup_interval = cleanup_interval
        self.compact_threshold_tokens = compact_threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.max_tool_results = max_tool_results
        self.keep_last_assistants = keep_last_assistants
        self.soft_trim = soft_trim
        self.hard_clear = hard_clear
        self.cache_dir = cache_dir
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
            ctx = ChatContext(
                chat_id,
                max_history=self.max_history_per_chat,
                compact_threshold_tokens=self.compact_threshold_tokens,
                keep_recent_tokens=self.keep_recent_tokens,
                cache_dir=self.cache_dir,
            )
            ctx.load()
            self.contexts[chat_id] = ctx
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
                max_messages=max_messages or len(ctx.history),
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

    async def compact_history_if_needed(
        self, chat_id: str, ai_service, force: bool = False
    ) -> tuple[bool, Optional[Dict], "ChatContext"]:
        """用 per-chat 锁保护压缩过程，与 add_user_message_async 共用同一锁。"""
        lock = await self._get_chat_lock(chat_id)
        async with lock:
            context = self.get_context(chat_id)
            compacted, usage = await context.compact_history_if_needed(
                ai_service, force=force
            )
        return compacted, usage, context

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

    def get_all_disk_chat_ids(self) -> List[str]:
        disk_ids: set[str] = set()
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            if cache_path.is_dir():
                for f in cache_path.glob("*.json"):
                    disk_ids.add(f.stem)
        memory_ids = set(self.contexts.keys())
        return sorted(disk_ids | memory_ids)

    def get_all_chats(self) -> Dict[str, ChatContext]:
        return self.contexts.copy()

    def get_context_count(self) -> int:
        return len(self.contexts)

    def get_total_messages_count(self) -> int:
        total = 0
        for context in self.contexts.values():
            total += len(context.history)
        return total
