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
    replied_content: str = ""
    replied_author: str = ""
    timestamp: float = None
    model_chain: Optional[List[str]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
