"""Channel adapter protocol and account-scoped registry."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .identity import DeliveryTarget, DeliveryValidation, require_delivery_target


@dataclass(frozen=True, slots=True)
class ChannelCapabilities:
    supports_groups: bool = True
    supports_direct: bool = True
    supports_replies: bool = True
    supports_media: bool = False


@dataclass(slots=True)
class InboundEnvelope:
    channel: str
    account_id: str
    payload: Any
    target: DeliveryTarget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApprovalPrompt:
    """Channel-neutral approval prompt passed to a delivery adapter."""

    session_key: str
    title: str
    description: str
    command_preview: str = ""
    cwd: str = ""
    severity: str = "info"
    timeout_sec: int = 120


@runtime_checkable
class ChannelAdapter(Protocol):
    channel: str
    account_id: str

    async def parse_inbound(self, payload: object) -> InboundEnvelope: ...

    async def resolve_target(self, envelope: InboundEnvelope) -> DeliveryTarget: ...

    async def send_message(
        self, target: DeliveryTarget, content: str, *, reply_to: str = "", **options
    ) -> Any: ...

    async def send_approval(
        self, target: DeliveryTarget, prompt: ApprovalPrompt, *, reply_to: str = ""
    ) -> Any: ...

    async def validate_target(self, target: DeliveryTarget) -> DeliveryValidation: ...

    def capabilities(self) -> ChannelCapabilities: ...


class ChannelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], ChannelAdapter] = {}

    def register(self, adapter: ChannelAdapter, *, replace: bool = False) -> None:
        key = (adapter.channel, adapter.account_id)
        if key in self._adapters and not replace:
            raise ValueError(f"channel adapter already registered: {key}")
        self._adapters[key] = adapter

    def get(self, channel: str, account_id: str) -> ChannelAdapter:
        try:
            return self._adapters[(channel, account_id)]
        except KeyError as exc:
            raise KeyError(f"no adapter registered for {channel}/{account_id}") from exc

    def for_target(self, target: DeliveryTarget) -> ChannelAdapter:
        return self.get(target.channel, target.account_id)

    def list_accounts(self, channel: str | None = None) -> list[tuple[str, str]]:
        keys = sorted(self._adapters)
        if channel is None:
            return keys
        return [key for key in keys if key[0] == channel]


AsyncCallback = Callable[..., Awaitable[Any] | Any]


class QQAdapter:
    """Callback-backed QQ adapter used as the first concrete channel boundary."""

    channel = "qq"

    def __init__(
        self,
        *,
        account_id: str = "default",
        parse_callback: AsyncCallback | None = None,
        target_callback: AsyncCallback | None = None,
        send_callback: AsyncCallback | None = None,
        approval_callback: AsyncCallback | None = None,
        validate_callback: AsyncCallback | None = None,
        capabilities: ChannelCapabilities | None = None,
    ) -> None:
        self.account_id = account_id
        self._parse_callback = parse_callback
        self._target_callback = target_callback
        self._send_callback = send_callback
        self._approval_callback = approval_callback
        self._validate_callback = validate_callback
        self._capabilities = capabilities or ChannelCapabilities(supports_media=True)

    async def parse_inbound(self, payload: object) -> InboundEnvelope:
        if self._parse_callback is None:
            raise NotImplementedError("QQAdapter requires a parse_callback")
        result = self._parse_callback(payload)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, InboundEnvelope):
            return result
        return InboundEnvelope(
            self.channel, self.account_id, payload, metadata={"parsed": result}
        )

    async def resolve_target(self, envelope: InboundEnvelope) -> DeliveryTarget:
        if envelope.target is not None:
            return envelope.target
        if self._target_callback is None:
            raise NotImplementedError("QQAdapter requires a target_callback")
        result = self._target_callback(envelope)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, DeliveryTarget):
            raise TypeError("target_callback must return DeliveryTarget")
        return result

    async def send_message(
        self, target: DeliveryTarget, content: str, *, reply_to: str = "", **options
    ) -> Any:
        require_delivery_target(target)
        if target.channel != self.channel or target.account_id != self.account_id:
            raise ValueError("target does not belong to this QQ adapter account")
        if self._send_callback is None:
            raise NotImplementedError("QQAdapter requires a send_callback")
        options.pop("turn_id", None)
        result = self._send_callback(target, content, reply_to=reply_to, **options)
        return await result if inspect.isawaitable(result) else result

    async def send_approval(
        self, target: DeliveryTarget, prompt: ApprovalPrompt, *, reply_to: str = ""
    ) -> Any:
        require_delivery_target(target)
        if target.channel != self.channel or target.account_id != self.account_id:
            raise ValueError("target does not belong to this QQ adapter account")
        if self._approval_callback is None:
            raise NotImplementedError("QQAdapter requires an approval_callback")
        result = self._approval_callback(target, prompt, reply_to=reply_to)
        return await result if inspect.isawaitable(result) else result

    async def validate_target(self, target: DeliveryTarget) -> DeliveryValidation:
        require_delivery_target(target)
        if target.channel != self.channel or target.account_id != self.account_id:
            return DeliveryValidation(False, "adapter_account_mismatch", target)
        if self._validate_callback is None:
            return DeliveryValidation(True, "adapter_accepts_target", target)
        result = self._validate_callback(target)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, DeliveryValidation):
            raise TypeError("validate_callback must return DeliveryValidation")
        return result

    def capabilities(self) -> ChannelCapabilities:
        return self._capabilities
