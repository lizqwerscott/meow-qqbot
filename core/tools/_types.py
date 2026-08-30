"""工具系统核心类型"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from core.engine.delivery_ledger import DeliveryReceipt
    from core.engine.turn_capabilities import TurnCapabilities


@dataclass
class ToolContext:
    chat_id: str
    is_group: bool
    reply_to: str
    sender_id: str
    reply_callback: Callable
    delivery_channel: str = ""
    reply_to_message_id: str = ""
    internal_control: bool = False
    turn_id: str = ""
    turn_revision: int = 0
    principal_id: str = ""
    planner_lease_id: str = ""
    planner_plan_id: str = ""
    consumer_evidence_callback: Optional[Callable[[str], Awaitable[None]]] = None
    transition_turn: Optional[Callable[..., Awaitable[Any]]] = None
    capabilities: Optional["TurnCapabilities"] = None
    turn_active_callback: Optional[Callable[[], Awaitable[bool]]] = None


@dataclass
class ToolResult:
    content: str
    sent_emoji: bool = False
    no_reply: bool = False
    delivery_receipt: Optional["DeliveryReceipt"] = None
    delivery_kind: str = ""


@dataclass
class ToolEntry:
    name: str
    description: str
    parameters: dict
    handler: Callable
    section: str = "other"
    cron_allowed: bool = False
    delivery_kind: str = ""
