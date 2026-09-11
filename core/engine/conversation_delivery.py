"""Channel-neutral delivery router for conversations and approvals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from core.engine.delivery_ledger import DeliveryReceipt
from core.session_identity import ApprovalPrompt, ChannelRegistry, DeliveryTarget


@dataclass(frozen=True, slots=True)
class DeliveryRequest:
    """Channel-neutral outbound message request."""

    target: DeliveryTarget
    content: str = ""
    reply_to: str = ""
    options: dict[str, Any] = field(default_factory=dict)


class ConversationDeliveryAdapter(Protocol):
    """The one delivery operation shared by WebUI and external channels."""

    async def deliver(self, *, turn_id: str, **kwargs) -> DeliveryReceipt: ...


class ChannelDeliveryRouter:
    """Deep seam routing every outbound message to its registered adapter."""

    def __init__(self, registry: ChannelRegistry) -> None:
        self._registry = registry

    async def deliver(self, request: DeliveryRequest) -> Any:
        adapter = self._registry.for_target(request.target)
        validation = await adapter.validate_target(request.target)
        if not validation.ok:
            raise ValueError(validation.reason or "delivery target rejected")
        return await adapter.send_message(
            request.target,
            request.content,
            reply_to=request.reply_to,
            **dict(request.options),
        )

    async def deliver_approval(
        self, target: DeliveryTarget, prompt: ApprovalPrompt, *, reply_to: str = ""
    ) -> Any:
        adapter = self._registry.for_target(target)
        validation = await adapter.validate_target(target)
        if not validation.ok:
            raise ValueError(validation.reason or "delivery target rejected")
        return await adapter.send_approval(target, prompt, reply_to=reply_to)
