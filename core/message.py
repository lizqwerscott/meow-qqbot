import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from core.session_identity import DeliveryTarget, build_chat_session_key


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

    resource_type: str  # "emoji" | "image" | "voice" | "video" | "file"
    resource_id: str = ""  # 上游原始标识，兼容字段
    media_id: str = ""
    media_uri: str = ""
    source_url: str = ""
    storage_status: str = ""
    hash: str = ""  # SHA‑256（去重 / 缓存用）
    mime_type: str = ""  # content-type
    width: int = 0
    height: int = 0
    size: int = 0  # 文件字节数
    duration: float = 0  # 语音/视频 秒数
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
    delivery_target: Optional[DeliveryTarget] = None
    session_key: str = ""
    is_at_mention: bool = False
    bot_id: str = ""
    mentioned_ids: List[str] = field(default_factory=list)
    replied_content: str = ""
    replied_author: str = ""
    replied_author_id: str = ""
    replied_message_id: str = ""
    task_correlation_id: str = ""
    msg_type: MessageType = MessageType.TEXT
    timestamp: Optional[float] = None
    model_chain: Optional[List[str]] = None
    tier: Optional[str] = None
    resources: List[ResourceMeta] = field(default_factory=list)
    replied_resources: List[ResourceMeta] = field(default_factory=list)
    session_mode: Literal["chat", "agent"] | None = None
    reasoning_effort: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        internal_key = self.chat_id.startswith(
            (
                "task:",
                "cron:",
                "heartbeat:",
                "work-plan:",
                "workplan:",
                "agent:",
                "system:",
                "exec:",
                "subagent:",
            )
        )
        if self.delivery_target is None and not self.session_key and internal_key:
            self.session_key = self.chat_id
        if self.delivery_target is None and not self.session_key:
            self.delivery_target = DeliveryTarget(
                channel="qq",
                account_id="default",
                chat_type="group" if self.is_group else "direct",
                target_id=self.chat_id,
            )
        if not self.session_key and self.delivery_target is not None:
            self.session_key = build_chat_session_key(self.delivery_target)
