import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ChatMessage:
    """聊天消息记录"""

    role: str  # "user" 或 "assistant"
    content: str
    timestamp: float
    message_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


class ChatContext:
    """
    单个聊天的上下文管理器
    每个 chat_id 对应一个实例
    """

    def __init__(self, chat_id: str, max_history: int = 8):
        """
        初始化聊天上下文

        Args:
            chat_id: 聊天ID（用户ID或群聊ID）
            max_history: 最大历史记录条数，默认为8条
        """
        self.chat_id = chat_id
        self.max_history = max_history
        self.history = deque(maxlen=max_history)  # 使用deque自动限制大小
        self.last_activity = time.time()
        self.lock = asyncio.Lock()  # 异步锁，确保线程安全

    def add_message(
        self, role: str, content: str, message_id: Optional[str] = None
    ) -> None:
        """
        添加消息到历史记录

        Args:
            role: 角色，"user" 或 "assistant"
            content: 消息内容
            message_id: 消息ID（可选）
        """
        message = ChatMessage(
            role=role, content=content, timestamp=time.time(), message_id=message_id
        )
        self.history.append(message)
        self.last_activity = time.time()

    def add_user_message(self, content: str, message_id: Optional[str] = None) -> None:
        """添加用户消息"""
        self.add_message("user", content, message_id)

    def add_assistant_message(
        self, content: str, message_id: Optional[str] = None
    ) -> None:
        """添加助手消息"""
        self.add_message("assistant", content, message_id)

    def get_history(self, max_messages: Optional[int] = None) -> List[ChatMessage]:
        """
        获取历史记录

        Args:
            max_messages: 最大返回消息数，None表示返回全部

        Returns:
            历史消息列表
        """
        if max_messages is None:
            return list(self.history)
        return list(self.history)[-max_messages:]

    def get_history_as_dicts(self, max_messages: Optional[int] = None) -> List[Dict]:
        """
        获取历史记录（字典格式）

        Args:
            max_messages: 最大返回消息数

        Returns:
            历史消息字典列表
        """
        messages = self.get_history(max_messages)
        return [msg.to_dict() for msg in messages]

    def get_conversation_context(self, max_messages: Optional[int] = None) -> str:
        """
        获取对话上下文文本

        Args:
            max_messages: 最大消息数

        Returns:
            格式化的对话上下文
        """
        messages = self.get_history(max_messages)
        context_lines = []

        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            time_str = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
            context_lines.append(f"[{time_str}] {role_label}: {msg.content}")

        return "\n".join(context_lines)

    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()

    def get_history_count(self) -> int:
        """获取历史记录数量"""
        return len(self.history)

    def is_empty(self) -> bool:
        """是否为空"""
        return len(self.history) == 0

    def get_last_message(self) -> Optional[ChatMessage]:
        """获取最后一条消息"""
        if self.history:
            return self.history[-1]
        return None

    def get_inactivity_time(self) -> float:
        """获取不活跃时间（秒）"""
        return time.time() - self.last_activity

    async def add_message_async(
        self, role: str, content: str, message_id: Optional[str] = None
    ) -> None:
        """
        异步添加消息（线程安全）
        """
        async with self.lock:
            self.add_message(role, content, message_id)


class ChatContextManager:
    """
    聊天上下文管理器
    管理所有 chat_id 的上下文
    """

    def __init__(self, max_history_per_chat: int = 8, cleanup_interval: int = 3600):
        """
        初始化上下文管理器

        Args:
            max_history_per_chat: 每个聊天最大历史记录数，默认为8条
            cleanup_interval: 清理不活跃聊天的间隔（秒），默认为1小时
        """
        self.max_history_per_chat = max_history_per_chat
        self.cleanup_interval = cleanup_interval
        self.contexts: Dict[str, ChatContext] = {}
        self.lock = asyncio.Lock()

    def get_context(self, chat_id: str) -> ChatContext:
        """
        获取或创建聊天上下文

        Args:
            chat_id: 聊天ID

        Returns:
            ChatContext 实例
        """
        if chat_id not in self.contexts:
            self.contexts[chat_id] = ChatContext(chat_id, self.max_history_per_chat)
        return self.contexts[chat_id]

    async def get_context_async(self, chat_id: str) -> ChatContext:
        """
        异步获取或创建聊天上下文（线程安全）
        """
        async with self.lock:
            return self.get_context(chat_id)

    def add_user_message(
        self, chat_id: str, content: str, message_id: Optional[str] = None
    ) -> None:
        """
        添加用户消息到指定聊天

        Args:
            chat_id: 聊天ID
            content: 消息内容
            message_id: 消息ID（可选）
        """
        context = self.get_context(chat_id)
        context.add_user_message(content, message_id)

    def add_assistant_message(
        self, chat_id: str, content: str, message_id: Optional[str] = None
    ) -> None:
        """
        添加助手消息到指定聊天

        Args:
            chat_id: 聊天ID
            content: 消息内容
            message_id: 消息ID（可选）
        """
        context = self.get_context(chat_id)
        context.add_assistant_message(content, message_id)

    async def add_user_message_async(
        self, chat_id: str, content: str, message_id: Optional[str] = None
    ) -> None:
        """
        异步添加用户消息（线程安全）
        """
        async with self.lock:
            self.add_user_message(chat_id, content, message_id)

    async def add_assistant_message_async(
        self, chat_id: str, content: str, message_id: Optional[str] = None
    ) -> None:
        """
        异步添加助手消息（线程安全）
        """
        async with self.lock:
            self.add_assistant_message(chat_id, content, message_id)

    def get_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        """
        获取聊天历史记录

        Args:
            chat_id: 聊天ID
            max_messages: 最大消息数

        Returns:
            历史消息字典列表
        """
        if chat_id not in self.contexts:
            return []
        return self.contexts[chat_id].get_history_as_dicts(max_messages)

    def get_chat_history(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        """
        获取聊天历史记录（兼容性别名）

        Args:
            chat_id: 聊天ID
            max_messages: 最大消息数

        Returns:
            历史消息字典列表
        """
        return self.get_history(chat_id, max_messages)

    async def get_chat_history_async(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> List[Dict]:
        """
        异步获取聊天历史记录（线程安全）
        """
        async with self.lock:
            return self.get_chat_history(chat_id, max_messages)

    def get_conversation_context(
        self, chat_id: str, max_messages: Optional[int] = None
    ) -> str:
        """
        获取对话上下文文本

        Args:
            chat_id: 聊天ID
            max_messages: 最大消息数

        Returns:
            格式化的对话上下文
        """
        if chat_id not in self.contexts:
            return ""
        return self.contexts[chat_id].get_conversation_context(max_messages)

    def clear_history(self, chat_id: str) -> None:
        """
        清空指定聊天的历史记录

        Args:
            chat_id: 聊天ID
        """
        if chat_id in self.contexts:
            self.contexts[chat_id].clear_history()

    def clear_chat_history(self, chat_id: str) -> None:
        """
        清空指定聊天的历史记录（兼容性别名）

        Args:
            chat_id: 聊天ID
        """
        self.clear_history(chat_id)

    async def clear_chat_history_async(self, chat_id: str) -> None:
        """
        异步清空聊天历史记录（线程安全）
        """
        async with self.lock:
            self.clear_chat_history(chat_id)

    def remove_context(self, chat_id: str) -> None:
        """
        移除聊天上下文

        Args:
            chat_id: 聊天ID
        """
        if chat_id in self.contexts:
            del self.contexts[chat_id]

    def cleanup_inactive_contexts(self, max_inactivity: int = 7200) -> List[str]:
        """
        清理不活跃的聊天上下文

        Args:
            max_inactivity: 最大不活跃时间（秒），默认为2小时

        Returns:
            被清理的聊天ID列表
        """
        removed = []
        current_time = time.time()

        for chat_id, context in list(self.contexts.items()):
            if context.get_inactivity_time() > max_inactivity:
                removed.append(chat_id)
                del self.contexts[chat_id]

        return removed

    async def cleanup_inactive_contexts_async(
        self, max_inactivity: int = 7200
    ) -> List[str]:
        """
        异步清理不活跃的聊天上下文（线程安全）
        """
        async with self.lock:
            return self.cleanup_inactive_contexts(max_inactivity)

    def get_all_chat_ids(self) -> List[str]:
        """
        获取所有聊天ID

        Returns:
            聊天ID列表
        """
        return list(self.contexts.keys())

    def get_all_chats(self) -> Dict[str, ChatContext]:
        """
        获取所有聊天上下文

        Returns:
            聊天ID到上下文的映射
        """
        return self.contexts.copy()

    def get_context_count(self) -> int:
        """
        获取上下文数量

        Returns:
            上下文数量
        """
        return len(self.contexts)

    def get_total_messages_count(self) -> int:
        """
        获取所有上下文中的总消息数

        Returns:
            总消息数
        """
        total = 0
        for context in self.contexts.values():
            total += len(context.history)
        return total
