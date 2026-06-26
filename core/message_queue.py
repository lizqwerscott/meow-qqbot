import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InputMessage:
    """输入消息数据结构"""

    id: str
    sender_id: str
    chat_id: str
    content: str
    is_group: bool
    is_at_mention: bool = False
    mentioned_ids: List[str] = field(default_factory=list)
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ProcessedMessage:
    """处理完毕的消息数据结构"""

    id: str
    chat_id: str
    content: str
    original_message_id: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MessageQueue:
    """
    消息队列管理器
    包含两个队列：
    1. input_queue: 保存输入的消息
    2. processed_queue: 保存处理完毕的消息
    """

    def __init__(self, max_size: int = 40):
        """
        初始化消息队列

        Args:
            max_size: 每个队列的最大容量，默认为40
        """
        self.max_size = max_size
        self.input_queue = asyncio.Queue(maxsize=max_size)
        self.processed_queue = asyncio.Queue(maxsize=max_size)
        self._input_messages = {}  # 用于快速查找输入消息
        self._processed_messages = {}  # 用于快速查找处理完毕的消息

    async def put_input_message(self, message: InputMessage) -> bool:
        """
        添加输入消息到队列

        Args:
            message: 输入消息对象

        Returns:
            bool: 是否成功添加
        """
        try:
            await self.input_queue.put(message)
            self._input_messages[message.id] = message
            return True
        except asyncio.QueueFull:
            # 队列已满，尝试移除最旧的消息
            try:
                # 获取但不移除最旧的消息
                old_message = await self.input_queue.get()
                # 从缓存中移除
                if old_message.id in self._input_messages:
                    del self._input_messages[old_message.id]
                # 添加新消息
                await self.input_queue.put(message)
                self._input_messages[message.id] = message
                return True
            except Exception:
                return False

    async def get_input_message(
        self, timeout: Optional[float] = None
    ) -> Optional[InputMessage]:
        """
        从输入队列获取消息

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            Optional[InputMessage]: 输入消息对象，超时返回None
        """
        try:
            if timeout is None:
                message = await self.input_queue.get()
            else:
                message = await asyncio.wait_for(self.input_queue.get(), timeout)

            # 从缓存中移除
            if message.id in self._input_messages:
                del self._input_messages[message.id]

            return message
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None

    async def put_processed_message(self, message: ProcessedMessage) -> bool:
        """
        添加处理完毕的消息到队列

        Args:
            message: 处理完毕的消息对象

        Returns:
            bool: 是否成功添加
        """
        try:
            await self.processed_queue.put(message)
            self._processed_messages[message.id] = message
            return True
        except asyncio.QueueFull:
            # 队列已满，尝试移除最旧的消息
            try:
                # 获取但不移除最旧的消息
                old_message = await self.processed_queue.get()
                # 从缓存中移除
                if old_message.id in self._processed_messages:
                    del self._processed_messages[old_message.id]
                # 添加新消息
                await self.processed_queue.put(message)
                self._processed_messages[message.id] = message
                return True
            except Exception:
                return False

    async def get_processed_message(
        self, timeout: Optional[float] = None
    ) -> Optional[ProcessedMessage]:
        """
        从处理完毕队列获取消息

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            Optional[ProcessedMessage]: 处理完毕的消息对象，超时返回None
        """
        try:
            if timeout is None:
                message = await self.processed_queue.get()
            else:
                message = await asyncio.wait_for(self.processed_queue.get(), timeout)

            # 从缓存中移除
            if message.id in self._processed_messages:
                del self._processed_messages[message.id]

            return message
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None

    def get_input_message_by_id(self, message_id: str) -> Optional[InputMessage]:
        """
        根据ID获取输入消息（不从队列中移除）

        Args:
            message_id: 消息ID

        Returns:
            Optional[InputMessage]: 输入消息对象，不存在返回None
        """
        return self._input_messages.get(message_id)

    def get_processed_message_by_id(
        self, message_id: str
    ) -> Optional[ProcessedMessage]:
        """
        根据ID获取处理完毕的消息（不从队列中移除）

        Args:
            message_id: 消息ID

        Returns:
            Optional[ProcessedMessage]: 处理完毕的消息对象，不存在返回None
        """
        return self._processed_messages.get(message_id)

    def remove_input_message(self, message_id: str) -> bool:
        """
        根据ID移除输入消息

        Args:
            message_id: 消息ID

        Returns:
            bool: 是否成功移除
        """
        if message_id in self._input_messages:
            del self._input_messages[message_id]
            return True
        return False

    def remove_processed_message(self, message_id: str) -> bool:
        """
        根据ID移除处理完毕的消息

        Args:
            message_id: 消息ID

        Returns:
            bool: 是否成功移除
        """
        if message_id in self._processed_messages:
            del self._processed_messages[message_id]
            return True
        return False

    def input_queue_size(self) -> int:
        """
        获取输入队列当前大小

        Returns:
            int: 队列大小
        """
        return self.input_queue.qsize()

    def processed_queue_size(self) -> int:
        """
        获取处理完毕队列当前大小

        Returns:
            int: 队列大小
        """
        return self.processed_queue.qsize()

    def input_queue_empty(self) -> bool:
        """
        检查输入队列是否为空

        Returns:
            bool: 是否为空
        """
        return self.input_queue.empty()

    def processed_queue_empty(self) -> bool:
        """
        检查处理完毕队列是否为空

        Returns:
            bool: 是否为空
        """
        return self.processed_queue.empty()

    def input_queue_full(self) -> bool:
        """
        检查输入队列是否已满

        Returns:
            bool: 是否已满
        """
        return self.input_queue.full()

    def processed_queue_full(self) -> bool:
        """
        检查处理完毕队列是否已满

        Returns:
            bool: 是否已满
        """
        return self.processed_queue.full()

    def get_all_input_messages(self) -> List[InputMessage]:
        """
        获取所有输入消息（不从队列中移除）

        Returns:
            List[InputMessage]: 输入消息列表
        """
        return list(self._input_messages.values())

    def get_all_processed_messages(self) -> List[ProcessedMessage]:
        """
        获取所有处理完毕的消息（不从队列中移除）

        Returns:
            List[ProcessedMessage]: 处理完毕的消息列表
        """
        return list(self._processed_messages.values())

    def clear_input_queue(self):
        """清空输入队列"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._input_messages.clear()

    def clear_processed_queue(self):
        """清空处理完毕队列"""
        while not self.processed_queue.empty():
            try:
                self.processed_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._processed_messages.clear()

    def clear_all(self):
        """清空所有队列"""
        self.clear_input_queue()
        self.clear_processed_queue()
