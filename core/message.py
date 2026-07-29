import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional


class MessageType(StrEnum):
    TEXT = "text"
    CARD = "card"
    EMOJI = "emoji"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    FILE = "file"


@dataclass
class ResourceMeta:
    """消息资源元数据（表情、图片、语音、视频、文件 统一表示）。"""

    resource_type: str             # "emoji" | "image" | "voice" | "video" | "file"
    resource_id: str               # 本地 hash / 远程 URL / 唯一标识
    hash: str = ""                 # SHA‑256（去重 / 缓存用）
    mime_type: str = ""            # content-type
    width: int = 0
    height: int = 0
    size: int = 0                  # 文件字节数
    duration: float = 0            # 语音/视频 秒数
    filename: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)  # 兜底扩展


@dataclass
class InputMessage:
    """输入消息数据结构"""

    id: str
    sender_id: str
    chat_id: str
    content: str
    is_group: bool
    is_at_mention: bool = False
    bot_id: str = ""
    mentioned_ids: List[str] = field(default_factory=list)
    replied_content: str = ""
    replied_author: str = ""
    msg_type: MessageType = MessageType.TEXT
    timestamp: Optional[float] = None
    model_chain: Optional[List[str]] = None
    tier: Optional[str] = None
    resources: List[ResourceMeta] = field(default_factory=list)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()