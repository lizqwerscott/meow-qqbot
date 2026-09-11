"""Delivery seam for channel-neutral conversation execution."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from uuid import uuid4

from core.engine.conversation_delivery import ConversationDeliveryAdapter
from core.engine.delivery_ledger import DeliveryReceipt
from core.session_identity import (
    ApprovalPrompt,
    ChannelCapabilities,
    DeliveryTarget,
    DeliveryValidation,
    InboundEnvelope,
)

from .events import EventHub


class WebUiDeliveryAdapter:
    """Render accepted replies as WebUI events, never as channel messages."""

    channel = "webui"

    def __init__(
        self, *, session_id: str = "", account_id: str = "admin", hub: EventHub
    ) -> None:
        self._session_id = session_id
        self.account_id = account_id
        self._hub = hub
        self._delivery_indices: dict[tuple[str, str], int] = {}

    async def deliver(
        self, *, turn_id: str, session_id: str = "", **kwargs
    ) -> DeliveryReceipt:
        session_id = session_id or self._session_id
        if not session_id:
            raise ValueError("WebUI delivery requires a session_id")
        delivery_key = (session_id, turn_id)
        delivery_index = self._delivery_indices.get(delivery_key, 0) + 1
        self._delivery_indices[delivery_key] = delivery_index
        content = str(kwargs.get("content") or "")
        event_type = "message.created" if delivery_index == 1 else "message.delta"
        resources = []
        for resource in kwargs.get("resources") or ():
            if is_dataclass(resource):
                resource = asdict(resource)
            resource = dict(resource)
            media_id = str(resource.get("media_id") or "")
            media_uri = str(resource.get("media_uri") or "")
            if not media_id and media_uri.startswith("media://inbound/"):
                media_id = media_uri.removeprefix("media://inbound/")
            if media_id:
                resource["media_id"] = media_id
                resource.setdefault("preview_url", f"/media/{media_id}/content")
                resource.setdefault(
                    "download_url", f"/media/{media_id}/content?download=true"
                )
            extra = resource.get("extra")
            if isinstance(extra, dict) and extra.get("preview_url"):
                resource.setdefault("preview_url", str(extra["preview_url"]))
            resource_type = str(resource.get("resource_type") or "file")
            mime_type = str(resource.get("mime_type") or "")
            resource.setdefault(
                "is_image",
                resource_type in {"emoji", "image"} or mime_type.startswith("image/"),
            )
            resource.setdefault(
                "is_audio",
                resource_type in {"voice", "audio"} or mime_type.startswith("audio/"),
            )
            resources.append(resource)
        await self._hub.publish(
            session_id,
            event_type,
            turn_id=turn_id,
            payload={
                "role": "assistant",
                "content": content,
                "message_id": str(kwargs.get("message_id") or ""),
                "delivery_index": delivery_index,
                "resources": resources,
            },
        )
        return DeliveryReceipt(
            status="accepted",
            logical_delivery_id=(
                str(kwargs.get("delivery_id") or f"webui:{turn_id}:{delivery_index}")
            ),
            platform_message_id=f"webui-message-{uuid4().hex}",
        )

    async def parse_inbound(self, payload: object) -> InboundEnvelope:
        return InboundEnvelope(
            self.channel,
            self.account_id,
            payload,
            metadata={"session_id": self._session_id},
        )

    async def resolve_target(self, envelope: InboundEnvelope) -> DeliveryTarget:
        session_id = self._session_id or str(envelope.metadata.get("session_id") or "")
        if not session_id:
            raise ValueError("WebUI envelope has no session_id")
        return DeliveryTarget("webui", self.account_id, "direct", session_id)

    async def send_message(
        self, target: DeliveryTarget, content: str, *, reply_to: str = "", **options
    ) -> DeliveryReceipt:
        return await self.deliver(
            session_id=target.target_id,
            turn_id=str(options.pop("turn_id", "") or reply_to),
            content=content,
            **options,
        )

    async def send_approval(
        self, target: DeliveryTarget, prompt: ApprovalPrompt, *, reply_to: str = ""
    ) -> DeliveryReceipt:
        await self._hub.publish(
            target.target_id,
            "approval.requested",
            turn_id=prompt.session_key,
            payload={
                "session_key": prompt.session_key,
                "title": prompt.title,
                "description": prompt.description,
                "command_preview": prompt.command_preview,
                "cwd": prompt.cwd,
                "severity": prompt.severity,
                "timeout_sec": prompt.timeout_sec,
            },
        )
        return DeliveryReceipt(
            status="accepted",
            logical_delivery_id=f"webui:approval:{prompt.session_key}",
            platform_message_id=f"webui-approval-{uuid4().hex}",
        )

    async def validate_target(self, target: DeliveryTarget) -> DeliveryValidation:
        if target.channel != self.channel or target.account_id != self.account_id:
            return DeliveryValidation(False, "adapter_account_mismatch", target)
        if target.chat_type != "direct" or not target.target_id:
            return DeliveryValidation(False, "invalid_webui_target", target)
        return DeliveryValidation(True, "adapter_accepts_target", target)

    def capabilities(self) -> ChannelCapabilities:
        return ChannelCapabilities(
            supports_groups=False,
            supports_direct=True,
            supports_replies=True,
            supports_media=True,
        )
