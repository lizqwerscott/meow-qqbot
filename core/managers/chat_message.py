import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from deepseek_tokenizer import ds_token


def _estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(ds_token.encode(text))


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
