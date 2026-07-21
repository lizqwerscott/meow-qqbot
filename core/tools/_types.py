"""工具系统核心类型"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolContext:
    chat_id: str
    is_group: bool
    reply_to: str
    sender_id: str
    reply_callback: Callable
    delivery_channel: str = ""
    reply_to_message_id: str = ""


@dataclass
class ToolResult:
    content: str
    sent_emoji: bool = False


@dataclass
class ToolEntry:
    name: str
    description: str
    parameters: dict
    handler: Callable
    section: str = "other"
    cron_allowed: bool = False
