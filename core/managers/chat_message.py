import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deepseek_tokenizer import ds_token

_log = logging.getLogger(__name__)

# 匹配 to_dict() 中 user 消息添加的 [发言人 在 YYYY-MM-DD HH:MM:SS]: 前缀
_RE_PREFIX = re.compile(
    r'^\[.*? 在 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:\s*'
)


def strip_content_prefix(content: str) -> str:
    """移除消息内容中由 to_dict() 添加的 [NAME 在 TIME]: 前缀。
    
    仅在从旧 JSONL 数据恢复时需要（新数据通过 raw_content 字段避免污染）。
    """
    while _RE_PREFIX.match(content):
        content = _RE_PREFIX.sub('', content)
    return content


def _estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    try:
        return len(ds_token.encode(text))
    except Exception as e:
        _log.warning("token 估算失败: %s", e)
        return 0


@dataclass
class ChatMessage:
    role: str
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
            "raw_content": self.content,
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
    def from_dict(data) -> "ChatMessage":
        if isinstance(data, str):
            _log.warning("from_dict 收到 str 而非 dict (len=%d)，按 user 消息兜底", len(data))
            return ChatMessage(role="user", content=data, timestamp=0.0)
        content = data.get("raw_content", data.get("content", ""))
        return ChatMessage(
            role=data.get("role", "user"),
            content=content,
            timestamp=data.get("timestamp", 0.0),
            message_id=data.get("message_id"),
            sender_id=data.get("sender_id"),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("tool_name"),
            tool_calls=data.get("tool_calls"),
            reasoning_content=data.get("reasoning_content"),
        )


def group_user_messages(messages: List["ChatMessage"]) -> List[List["ChatMessage"]]:
    """将连续同发送人的 user 消息分组。
    
    非 user 消息（assistant/tool）或 sender_id 为 None 的消息各自为一组。
    用户消息按连续 + 同 sender_id 合并为同一组，不关心时间窗口。
    时间窗口仅在格式化时使用。
    
    Returns:
        List of groups, each group is a list of ChatMessage objects.
        单元素组表示无需合并（非 user 或独立 user 消息）。
    """
    groups: List[List[ChatMessage]] = []
    buf: List[ChatMessage] = []

    for msg in messages:
        if msg.role != "user" or not msg.sender_id:
            if buf:
                groups.append(buf)
                buf = []
            groups.append([msg])
            continue

        if not buf:
            buf.append(msg)
            continue

        if msg.sender_id != buf[0].sender_id:
            groups.append(buf)
            buf = [msg]
            continue

        buf.append(msg)

    if buf:
        groups.append(buf)

    return groups
