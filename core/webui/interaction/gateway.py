"""Deep WebUI interaction module over the channel-neutral AgentEngine seam."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from core.engine.conversation_delivery import (
    ChannelDeliveryRouter,
    ConversationDeliveryAdapter,
    DeliveryRequest,
)
from core.engine.conversation_event_log import TurnStatus
from core.message import InputMessage, MessageType, ResourceMeta
from core.session_identity import (
    ChannelRegistry,
    DeliveryTarget,
    build_chat_session_key,
)

from .delivery import WebUiDeliveryAdapter
from .events import EventHub
from .models import SubmissionReceipt, WebUiSession
from .store import WebUiSessionStore


class WebUiConversationGateway:
    """Create isolated WebUI sessions and submit them through one route seam."""

    def __init__(
        self,
        *,
        operator_id: str,
        route_callback: Callable[..., Awaitable[None]],
        get_user_nickname: Callable[[str], str],
        event_log=None,
        media_service=None,
        store: WebUiSessionStore | None = None,
        store_path: str = "data/webui_sessions.sqlite3",
        hub: EventHub | None = None,
        channel_registry: ChannelRegistry | None = None,
        delivery_adapter_factory: (
            Callable[[str, EventHub], ConversationDeliveryAdapter] | None
        ) = None,
    ) -> None:
        operator_id = str(operator_id).strip().lower()
        if not operator_id or ":" in operator_id or "/" in operator_id:
            raise ValueError("operator_id must be a safe session segment")
        self.operator_id = operator_id
        self._route = route_callback
        self._get_user_nickname = get_user_nickname
        self._event_log = event_log
        self._media_service = media_service
        self._store = store or WebUiSessionStore(store_path)
        self._hub = hub or EventHub()
        self._delivery_router = None
        if channel_registry is not None:
            channel_registry.register(
                WebUiDeliveryAdapter(
                    account_id=self.operator_id,
                    hub=self._hub,
                ),
                replace=True,
            )
            self._delivery_router = ChannelDeliveryRouter(channel_registry)
        self._delivery_adapter_factory = delivery_adapter_factory or (
            lambda session_id, hub: WebUiDeliveryAdapter(session_id=session_id, hub=hub)
        )
        self._tasks: set[asyncio.Task] = set()

    @property
    def hub(self) -> EventHub:
        return self._hub

    async def create_session(
        self, *, title: str = "", mode: str = "agent"
    ) -> WebUiSession:
        if mode not in {"agent", "chat"}:
            raise ValueError("mode must be 'agent' or 'chat'")
        session_id = f"s-{uuid4().hex}"
        target = DeliveryTarget("webui", self.operator_id, "direct", session_id)
        session = self._store.create(
            session_id=session_id,
            session_key=build_chat_session_key(target),
            operator_id=self.operator_id,
            title=(str(title).strip()[:120] or "新会话"),
            mode=mode,
        )
        await self._hub.publish(
            session.session_id,
            "session.ready",
            payload={"session": session.to_dict()},
        )
        return session

    async def list_sessions(self) -> list[WebUiSession]:
        return self._store.list(self.operator_id)

    def get_session(self, session_id: str) -> WebUiSession:
        return self._require_session(session_id)

    async def rename_session(self, session_id: str, title: str) -> WebUiSession:
        session = self._require_session(session_id)
        renamed = self._store.update_title(
            session_id, self.operator_id, str(title).strip()[:120] or "新会话"
        )
        if renamed is None:
            raise KeyError(session_id)
        await self._hub.publish(
            session_id,
            "session.updated",
            payload={"session": renamed.to_dict()},
        )
        return renamed

    async def submit(
        self,
        session_id: str,
        *,
        content: str = "",
        resources: Sequence[dict] = (),
        request_id: str = "",
        mode: str | None = None,
    ) -> SubmissionReceipt:
        session = self._require_session(session_id)
        content = str(content or "")
        if len(content) > 100_000:
            raise ValueError("message content is too long")
        parsed_resources = await self._resources(
            resources, session_key=session.session_key
        )
        if not content.strip() and not parsed_resources:
            raise ValueError("content or resources is required")
        if mode is not None and mode not in {"agent", "chat"}:
            raise ValueError("mode must be 'agent' or 'chat'")
        requested_mode = mode or session.mode
        if session.mode == "agent" and requested_mode == "chat":
            raise ValueError("agent sessions cannot switch back to chat")
        if requested_mode != session.mode:
            updated = self._store.update_mode(
                session_id, self.operator_id, requested_mode
            )
            if updated is not None:
                session = updated
                await self._hub.publish(
                    session_id,
                    "session.updated",
                    payload={"session": session.to_dict()},
                )
        request_id = str(request_id or f"request-{uuid4().hex}").strip()
        if not request_id or len(request_id) > 160:
            raise ValueError("invalid request_id")
        turn_id = f"webui-turn-{uuid4().hex}"
        duplicate = self._store.reserve_submission(session_id, request_id, turn_id)
        if duplicate is not None:
            return duplicate
        self._store.touch(session_id, self.operator_id)
        target = DeliveryTarget("webui", self.operator_id, "direct", session_id)
        message = InputMessage(
            id=turn_id,
            sender_id=self.operator_id,
            chat_id=session.session_key,
            content=content,
            is_group=False,
            delivery_target=target,
            session_key=session.session_key,
            msg_type=(
                MessageType.TEXT
                if not parsed_resources
                else MessageType(parsed_resources[0].resource_type)
            ),
            resources=list(parsed_resources),
            session_mode=session.mode,
        )
        await self._hub.publish(
            session_id,
            "turn.accepted",
            turn_id=turn_id,
            payload={
                "request_id": request_id,
                "message_id": turn_id,
                "content": content,
                "resources": [
                    {
                        key: getattr(resource, key)
                        for key in resource.__dataclass_fields__
                        if getattr(resource, key) not in ("", 0, 0.0, None, {})
                    }
                    for resource in parsed_resources
                ],
            },
        )
        task = asyncio.create_task(self._run_turn(session, message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return SubmissionReceipt(session_id, turn_id, request_id, True)

    async def _run_turn(self, session: WebUiSession, message: InputMessage) -> None:
        delivery = self._delivery_adapter_factory(session.session_id, self._hub)
        target = DeliveryTarget("webui", self.operator_id, "direct", session.session_id)

        async def reply_callback(**kwargs):
            if self._delivery_router is None:
                return await delivery.deliver(turn_id=message.id, **kwargs)
            options = dict(kwargs)
            options.pop("content", None)
            options.pop("chat_id", None)
            options.pop("is_group", None)
            options["turn_id"] = message.id
            return await self._delivery_router.deliver(
                DeliveryRequest(
                    target=target,
                    content=str(kwargs.get("content") or ""),
                    reply_to=str(kwargs.get("message_id") or ""),
                    options=options,
                )
            )

        try:
            await self._route(
                input_message=message,
                reply_callback=reply_callback,
                get_user_nickname=self._get_user_nickname,
            )
            if message.session_mode == "agent" and session.mode != "agent":
                updated = self._store.update_mode(
                    session.session_id, self.operator_id, "agent"
                )
                if updated is not None:
                    await self._hub.publish(
                        session.session_id,
                        "session.updated",
                        payload={"session": updated.to_dict()},
                    )
            await self._wait_for_terminal(session, message.id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._hub.publish(
                session.session_id,
                "turn.failed",
                turn_id=message.id,
                payload={"error": str(exc)[:500]},
            )

    async def _wait_for_terminal(self, session: WebUiSession, turn_id: str) -> None:
        if self._event_log is None:
            await self._hub.publish(
                session.session_id, "turn.completed", turn_id=turn_id
            )
            return
        for _ in range(3600):
            report = await self._event_log.validate_turn(
                turn_id, chat_id=session.session_key
            )
            if report.reason == "turn_not_found":
                if await self._hub.has_turn_events(session.session_id, turn_id):
                    await self._hub.publish(
                        session.session_id,
                        "turn.completed",
                        turn_id=turn_id,
                        payload={"status": "completed", "ledger": False},
                    )
                    return
            if report.status in {
                TurnStatus.COMPLETED.value,
                TurnStatus.FAILED.value,
                TurnStatus.ABORTED.value,
                TurnStatus.BLOCKED.value,
                TurnStatus.INCOMPLETE.value,
            }:
                event_type = (
                    "turn.completed"
                    if report.status == TurnStatus.COMPLETED.value
                    else "turn.failed"
                )
                await self._hub.publish(
                    session.session_id,
                    event_type,
                    turn_id=turn_id,
                    payload={"status": report.status, "valid": report.valid},
                )
                return
            await asyncio.sleep(0.1)
        await self._hub.publish(
            session.session_id,
            "turn.failed",
            turn_id=turn_id,
            payload={"error": "turn completion timeout"},
        )

    def _require_session(self, session_id: str) -> WebUiSession:
        session_id = str(session_id).strip()
        if not session_id or "/" in session_id or "\x00" in session_id:
            raise ValueError("invalid session_id")
        session = self._store.get(session_id, self.operator_id)
        if session is None:
            raise KeyError(session_id)
        return session

    async def _resources(
        self, resources: Sequence[dict], *, session_key: str
    ) -> tuple[ResourceMeta, ...]:
        if not isinstance(resources, (list, tuple)) or len(resources) > 10:
            raise ValueError("resources must contain at most 10 items")
        allowed = {
            "resource_type",
            "resource_id",
            "media_id",
            "media_uri",
            "source_url",
            "storage_status",
            "hash",
            "mime_type",
            "width",
            "height",
            "size",
            "duration",
            "filename",
            "extra",
        }
        result = []
        for raw in resources:
            if not isinstance(raw, dict):
                raise ValueError("each resource must be an object")
            values = {key: raw[key] for key in allowed if key in raw}
            resource_type = str(values.get("resource_type") or "")
            if resource_type not in {
                item.value for item in MessageType if item != MessageType.TEXT
            }:
                raise ValueError("unsupported resource type")
            if values.get("source_url"):
                raise ValueError("WebUI resources must use authorized media_uri")
            media_uri = str(values.get("media_uri") or "")
            if resource_type != MessageType.EMOJI.value and not media_uri:
                raise ValueError("WebUI resources require media_uri")
            if media_uri:
                store = getattr(self._media_service, "store", None)
                authorize = getattr(store, "authorize", None)
                if not callable(authorize):
                    raise ValueError("media authorization is unavailable")
                record = await authorize(session_key, media_uri)
                if record is None:
                    raise ValueError("resource is not authorized for this session")
                values.update(
                    {
                        "media_id": record.media_id,
                        "media_uri": record.media_uri,
                        "resource_type": record.resource_type,
                        "storage_status": "ready",
                        "hash": record.sha256,
                        "mime_type": record.mime_type,
                        "size": record.size,
                        "filename": record.filename,
                    }
                )
            result.append(ResourceMeta(**values))
        return tuple(result)
