"""DeliveryStrategy — per-source 投递策略。

每个 source 注册一个 strategy。不持有 session_key/chat_id，
通过 deliver() 的 delivery_target 参数获取投递目标。
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from core.engine.delivery_ledger import DeliveryController

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
        delivery_controller: Optional[DeliveryController] = None,
    ):
        self._hb = heartbeat_manager
        self._send = reply_callback
        self._ctx = context_manager
        self._show_ok = show_ok
        self._show_alerts = show_alerts
        self._delivery_controller = delivery_controller

    def _admin_delivery_id(self, result: Any, text: str, kind: str) -> str:
        turn_id = str(getattr(result, "turn_id", "") or "")
        if not turn_id:
            turn_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"heartbeat:{turn_id}:{kind}:admin"

    async def _deliver_to_admin(self, result: Any, text: str, *, kind: str) -> bool:
        """Deliver an administrator heartbeat through the receipt ledger."""
        if not self._delivery_controller:
            return await self._hb.deliver_to_admin(text)
        admin_id = getattr(self._hb, "admin_delivery_target", "")
        if not admin_id:
            admin_ids = getattr(self._hb, "_admin_ids", ())
            admin_id = admin_ids[0] if admin_ids else ""
        receipt_sender = getattr(self._hb, "deliver_to_admin_receipt", None)
        if not admin_id or receipt_sender is None:
            return await self._hb.deliver_to_admin(text)

        async def _transport(**_kwargs):
            return await receipt_sender(text)

        receipt = await self._delivery_controller.deliver_text(
            delivery_id=self._admin_delivery_id(result, text, kind),
            chat_id=admin_id,
            content=text,
            callback=_transport,
            message_id="",
            is_group=False,
        )
        return receipt.status == "accepted"

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
                if self._delivery_controller:
                    await self._delivery_controller.deliver_text(
                        delivery_id=f"heartbeat:{getattr(result, 'turn_id', '')}:user",
                        chat_id=deliver_to_user,
                        content=text,
                        callback=self._send,
                        message_id="",
                        is_group=is_group,
                    )
                else:
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
                delivered = await self._deliver_to_admin(
                    result, text, kind="notification"
                )
                if not delivered:
                    raise RuntimeError("heartbeat admin delivery was not confirmed")
                self._hb.record_notification(text)
        elif self._show_ok:
            self._hb.record_delivery_start()
            _log.debug("[Delivery] Heartbeat 发送静默确认: ok")
            text = "一切正常，无需关注。"
            delivered = await self._deliver_to_admin(result, text, kind="ok")
            if not delivered:
                raise RuntimeError("heartbeat admin delivery was not confirmed")


class ChatReplyDeliveryStrategy(DeliveryStrategy):
    """系统事件结果 → 直接回复到 chat。

    delivery_target = QQ chat_id（真实的群聊或私聊 ID）。
    同时支持 normal chat 的 captured_replies 和 system event 的 notification_text。
    """

    def __init__(
        self,
        reply_callback: Callable,
        context_manager: Any = None,
        delivery_controller: Optional[DeliveryController] = None,
        require_delivery: bool = False,
        allow_result_target: bool = True,
    ):
        self._send = reply_callback
        self._ctx = context_manager
        self._delivery_controller = delivery_controller
        self._require_delivery = require_delivery
        self._allow_result_target = allow_result_target

    async def _send_text(
        self, text: str, target: str, *, delivery_id: str = ""
    ) -> None:
        is_group = False
        if self._ctx:
            chat_type = self._ctx.get_chat_type(target)
            if chat_type is not None:
                is_group = chat_type
        if self._delivery_controller:
            receipt = await self._delivery_controller.deliver_text(
                delivery_id=delivery_id or f"chat-reply:{target}",
                chat_id=target,
                content=text,
                callback=self._send,
                message_id="",
                is_group=is_group,
            )
            if receipt.status != "accepted":
                raise RuntimeError(f"chat delivery was not confirmed: {receipt.status}")
            return
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
                target = delivery_target
                if self._allow_result_target:
                    target = getattr(result, "deliver_to_user", "") or delivery_target
                if not target:
                    if self._require_delivery:
                        raise RuntimeError("work-plan delivery has no target")
                    _log.debug("[Delivery] ChatReply 跳过(通知): 无投递目标")
                    return
                _log.info(
                    "[Delivery] ChatReply 发送(通知): target=%s len=%d",
                    target[:16],
                    len(text),
                )
                await self._send_text(
                    text,
                    target,
                    delivery_id=f"wake:{getattr(result, 'turn_id', '')}:notification",
                )
                return

        # Normal chat path: AI 直接输出文本
        if not result.captured_replies or not delivery_target:
            if self._require_delivery:
                if not delivery_target:
                    raise RuntimeError("work-plan delivery has no target")
                raise RuntimeError("work-plan wake produced no deliverable reply")
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
            if self._require_delivery:
                raise RuntimeError("work-plan wake produced only silent replies")
            _log.debug("[Delivery] ChatReply 跳过: 全部被标准化过滤")
            return
        combined = "\n\n".join(non_silent)
        _log.info(
            "[Delivery] ChatReply 发送: target=%s len=%d is_group=%s",
            delivery_target[:16],
            len(combined),
        )
        await self._send_text(
            combined,
            delivery_target,
            delivery_id=f"wake:{getattr(result, 'turn_id', '')}:captured",
        )


class SilentDeliveryStrategy(DeliveryStrategy):
    """静默模式 — 不投递任何内容。"""

    async def deliver(self, result: Any, *, delivery_target: str = "") -> None:
        pass
