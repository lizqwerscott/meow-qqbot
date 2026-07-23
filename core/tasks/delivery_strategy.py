"""DeliveryStrategy — per-source 投递策略。

每个 source 注册一个 strategy。不持有 session_key/chat_id，
通过 deliver() 的 delivery_target 参数获取投递目标。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class DeliveryStrategy(ABC):
    @abstractmethod
    async def deliver(self, result: Any, *, delivery_target: str) -> None:
        ...


class HeartbeatDeliveryStrategy(DeliveryStrategy):
    """心跳结果 → 通知抑制 → DM 给管理员。

    show_ok: 静默成功（notify=false）时是否发送确认消息
    show_alerts: 告警（notify=true）时是否发送
    """

    def __init__(self, heartbeat_manager: Any, show_ok: bool = False, show_alerts: bool = True):
        self._hb = heartbeat_manager
        self._show_ok = show_ok
        self._show_alerts = show_alerts

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        self._hb.record_delivery_start()
        if result.should_notify and result.notification_text:
            if not self._show_alerts:
                return
            text = result.notification_text.strip()
            if self._hb.should_suppress(text):
                return
            await self._hb.deliver_to_admin(text)
            self._hb.record_notification(text)
        elif self._show_ok:
            await self._hb.deliver_to_admin("一切正常，无需关注。")


class ChatReplyDeliveryStrategy(DeliveryStrategy):
    """系统事件结果 → 直接回复到 chat。

    delivery_target = QQ chat_id（真实的群聊或私聊 ID）。
    """

    def __init__(self, reply_callback: Callable, context_manager: Any = None):
        self._send = reply_callback
        self._ctx = context_manager

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        if not result.captured_replies or not delivery_target:
            return
        from .delivery_normalization import normalize_heartbeat_reply
        non_silent: list[str] = []
        for reply in result.captured_replies:
            cleaned, should_skip = normalize_heartbeat_reply(reply)
            if not should_skip:
                non_silent.append(cleaned)
        if not non_silent:
            return
        combined = "\n\n".join(non_silent)
        is_group = False
        if self._ctx:
            chat_type = self._ctx.get_chat_type(delivery_target)
            if chat_type is not None:
                is_group = chat_type
        await self._send(
            chat_id=delivery_target,
            content=combined,
            message_id="",
            is_group=is_group,
        )


class SilentDeliveryStrategy(DeliveryStrategy):
    """静默模式 — 不投递任何内容。"""

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        pass
