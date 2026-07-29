"""DeliveryStrategy — per-source 投递策略。

每个 source 注册一个 strategy。不持有 session_key/chat_id，
通过 deliver() 的 delivery_target 参数获取投递目标。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)


class DeliveryStrategy(ABC):
    @abstractmethod
    async def deliver(self, result: Any, *, delivery_target: str) -> None: ...


class HeartbeatDeliveryStrategy(DeliveryStrategy):
    """心跳结果 → 通知抑制 → DM 给管理员 或 deliver_to_user。

    result.deliver_to_user 有值时发到该用户的聊天，否则 DM 管理员。
    """

    def __init__(
        self,
        heartbeat_manager: Any,
        reply_callback: Any = None,
        context_manager: Any = None,
        show_ok: bool = False,
        show_alerts: bool = True,
    ):
        self._hb = heartbeat_manager
        self._send = reply_callback
        self._ctx = context_manager
        self._show_ok = show_ok
        self._show_alerts = show_alerts

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        if result.should_notify and result.notification_text:
            if not self._show_alerts:
                _log.debug("[Delivery] Heartbeat 跳过: show_alerts=false")
                return
            text = result.notification_text.strip()
            deliver_to_user = getattr(result, "deliver_to_user", "")
            if deliver_to_user and self._send:
                # 投递到用户聊天（系统事件结果）
                is_group = False
                if self._ctx:
                    chat_type = self._ctx.get_chat_type(deliver_to_user)
                    if chat_type is not None:
                        is_group = chat_type
                _log.info(
                    "[Delivery] Heartbeat 发送到用户: target=%s len=%d",
                    deliver_to_user[:16],
                    len(text),
                )
                await self._send(
                    chat_id=deliver_to_user,
                    content=text,
                    message_id="",
                    is_group=is_group,
                )
            else:
                # 投递给管理员 DM（心跳通知）
                if self._hb.should_suppress(text):
                    _log.info(
                        "[Delivery] Heartbeat 抑制: 文本重复 (cooldown=%.1fh)",
                        self._hb._cooldown_hours,
                    )
                    return
                self._hb.record_delivery_start()
                _log.info(
                    "[Delivery] Heartbeat 发送DM: len=%d text=%.60s", len(text), text
                )
                await self._hb.deliver_to_admin(text)
                self._hb.record_notification(text)
        elif self._show_ok:
            self._hb.record_delivery_start()
            _log.debug("[Delivery] Heartbeat 发送静默确认: ok")
            await self._hb.deliver_to_admin("一切正常，无需关注。")


class ChatReplyDeliveryStrategy(DeliveryStrategy):
    """系统事件结果 → 直接回复到 chat。

    delivery_target = QQ chat_id（真实的群聊或私聊 ID）。
    同时支持 normal chat 的 captured_replies 和 system event 的 notification_text。
    """

    def __init__(self, reply_callback: Callable, context_manager: Any = None):
        self._send = reply_callback
        self._ctx = context_manager

    async def _send_text(self, text: str, target: str) -> None:
        is_group = False
        if self._ctx:
            chat_type = self._ctx.get_chat_type(target)
            if chat_type is not None:
                is_group = chat_type
        await self._send(
            chat_id=target,
            content=text,
            message_id="",
            is_group=is_group,
        )

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        # System event path: AI 使用 heartbeat_respond，结果在 notification_text 中
        if getattr(result, "should_notify", False) and getattr(
            result, "notification_text", ""
        ):
            text = result.notification_text.strip()
            if text:
                target = getattr(result, "deliver_to_user", "") or delivery_target
                if not target:
                    _log.debug("[Delivery] ChatReply 跳过(通知): 无投递目标")
                    return
                _log.info(
                    "[Delivery] ChatReply 发送(通知): target=%s len=%d",
                    target[:16],
                    len(text),
                )
                await self._send_text(text, target)
                return

        # Normal chat path: AI 直接输出文本
        if not result.captured_replies or not delivery_target:
            _log.debug("[Delivery] ChatReply 跳过: 无回复或投递目标为空")
            return

        # Normal chat path: AI 直接输出文本
        if not result.captured_replies:
            _log.debug("[Delivery] ChatReply 跳过: 无回复")
            return
        from .delivery_normalization import normalize_heartbeat_reply

        non_silent: list[str] = []
        for reply in result.captured_replies:
            cleaned, should_skip = normalize_heartbeat_reply(reply)
            if not should_skip:
                non_silent.append(cleaned)
        if not non_silent:
            _log.debug("[Delivery] ChatReply 跳过: 全部被标准化过滤")
            return
        combined = "\n\n".join(non_silent)
        _log.info(
            "[Delivery] ChatReply 发送: target=%s len=%d is_group=%s",
            delivery_target[:16],
            len(combined),
        )
        await self._send_text(combined, delivery_target)


class SilentDeliveryStrategy(DeliveryStrategy):
    """静默模式 — 不投递任何内容。"""

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        pass
