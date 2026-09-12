"""Deep WebUI interaction module over the channel-neutral AgentEngine seam."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from uuid import uuid4

from core.engine.conversation_delivery import (
    ChannelDeliveryRouter,
    ConversationDeliveryAdapter,
    DeliveryRequest,
)
from core.engine.conversation_event_log import TurnStatus
from core.engine.tool_events import ToolLifecycleEvent
from core.message import InputMessage, MessageType, ResourceMeta
from core.session_identity import (
    ChannelRegistry,
    DeliveryTarget,
    build_chat_session_key,
)

from .delivery import WebUiDeliveryAdapter
from .events import EventHub
from .models import SubmissionReceipt, WebUiSession, WebUiSessionPage
from .store import WebUiSessionStore


class WebUiConversationGateway:
    """Create isolated WebUI sessions and submit them through one route seam."""

    _DRAFT_UPLOAD_PREFIX = "webui-upload-"

    def __init__(
        self,
        *,
        operator_id: str,
        route_callback: Callable[..., Awaitable[None]],
        get_user_nickname: Callable[[str], str],
        event_log=None,
        media_service=None,
        model_registry=None,
        context_compact_callback: Callable[..., Awaitable[dict]] | None = None,
        approval_pending_callback: Callable[[str, str], list[dict]] | None = None,
        store: WebUiSessionStore | None = None,
        store_path: str = "data/webui_sessions.sqlite3",
        hub: EventHub | None = None,
        channel_registry: ChannelRegistry | None = None,
        delivery_adapter_factory: (
            Callable[[str, EventHub], ConversationDeliveryAdapter] | None
        ) = None,
        draft_upload_ttl_seconds: float = 86_400,
    ) -> None:
        operator_id = str(operator_id).strip().lower()
        if not operator_id or ":" in operator_id or "/" in operator_id:
            raise ValueError("operator_id must be a safe session segment")
        self.operator_id = operator_id
        self._route = route_callback
        self._get_user_nickname = get_user_nickname
        self._event_log = event_log
        self._media_service = media_service
        self._model_registry = model_registry
        self._context_compact = context_compact_callback
        self._approval_pending = approval_pending_callback
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
        self._draft_upload_ttl_seconds = max(0.0, float(draft_upload_ttl_seconds))
        self._tasks: set[asyncio.Task] = set()

    @property
    def hub(self) -> EventHub:
        return self._hub

    async def create_session(
        self, *, title: str = "", mode: str = "agent"
    ) -> WebUiSession:
        await self._cleanup_draft_uploads()
        if not isinstance(mode, str) or mode not in {"agent", "chat"}:
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
        self._store.record_audit(session.session_id, "session.created", mode=mode)
        return session

    async def list_sessions(
        self, *, limit: int = 50, cursor: str = ""
    ) -> WebUiSessionPage:
        await self._cleanup_draft_uploads()
        return self._store.list_page(self.operator_id, limit=limit, cursor=cursor)

    async def list_external_sessions(self, *, limit: int = 100) -> WebUiSessionPage:
        if self._event_log is None:
            return WebUiSessionPage((), False)
        webui_sessions = self._store.list(self.operator_id)
        known_keys = {session.session_key for session in webui_sessions}
        chat_ids = await self._event_log.chat_ids(
            visible_only=True, session_kinds=("chat", "group", "private")
        )
        external = []
        for chat_id in chat_ids:
            if (
                not chat_id
                or chat_id in known_keys
                or "/" in chat_id
                or "\x00" in chat_id
                or chat_id.startswith(("task:", "cron:", "heartbeat:", "work-plan:"))
            ):
                continue
            summary = await self._event_log.session_summary(chat_id)
            parts = chat_id.split(":")
            channel = (
                parts[2] if len(parts) == 6 and parts[0] == "agent" else "external"
            )
            last_activity = float(summary.get("last_activity") or 0)
            external.append(
                WebUiSession(
                    session_id=chat_id,
                    session_key=chat_id,
                    operator_id=self.operator_id,
                    title=f"{channel} · {chat_id[-32:]}",
                    mode="readonly",
                    created_at=last_activity,
                    updated_at=last_activity,
                    read_only=True,
                    channel=channel,
                )
            )
        external.sort(
            key=lambda session: (session.updated_at, session.session_id), reverse=True
        )
        return WebUiSessionPage(tuple(external[: max(1, min(int(limit), 100))]), False)

    def get_session(self, session_id: str) -> WebUiSession:
        return self._require_session(session_id)

    async def resolve_history_session(self, session_id: str) -> WebUiSession:
        try:
            return self._require_session(session_id)
        except KeyError:
            for session in (await self.list_external_sessions(limit=100)).items:
                if session.session_id == session_id:
                    return session
            raise

    def model_options(self) -> list[dict[str, object]]:
        registry = self._model_registry
        list_options = getattr(registry, "list_group_options", None)
        options = list_options() if callable(list_options) else []
        return [{"id": "auto", "label": "自动路由", "model_count": 0}, *options]

    def chat_options(self) -> dict[str, object]:
        list_efforts = getattr(
            self._model_registry, "list_reasoning_effort_options", None
        )
        reasoning_options = list_efforts() if callable(list_efforts) else []
        thinking_control = {
            "enabled": bool(reasoning_options),
            "default": "provider",
            "reason": (
                "可在当前 Turn 覆写，未选择时跟随模型配置"
                if reasoning_options
                else "当前模型服务按全局配置固定思考强度"
            ),
        }
        if reasoning_options:
            thinking_control["options"] = reasoning_options
        return {
            "model_groups": self.model_options(),
            "controls": {
                "thinking_effort": thinking_control,
                "quick_mode": {
                    "enabled": False,
                    "default": "off",
                    "reason": "快速模式尚未提供按 Turn 的安全覆写",
                },
                "context_compaction": {
                    "enabled": self._context_compact is not None,
                    "default": "manual",
                    "reason": (
                        "管理员可手动压缩模型上下文"
                        if self._context_compact is not None
                        else "当前运行时未启用上下文压缩"
                    ),
                },
            },
        }

    def list_audit(self, session_id: str, *, limit: int = 50) -> list[dict]:
        """Return content-free audit entries for one of this operator's sessions."""
        self._require_session(session_id)
        visible_details = {
            "session.created": {"mode"},
            "session.renamed": set(),
            "turn.submitted": {
                "mode",
                "model_group",
                "reasoning_effort",
                "resource_count",
            },
            "attachment.uploaded": {"resource_type", "size"},
            "attachment.discarded": set(),
            "context.compacted": {
                "changed",
                "tier",
                "operation",
                "before_tokens",
                "after_tokens",
                "saved_tokens",
                "reason",
            },
        }
        entries = []
        for entry in self._store.list_audit(session_id, limit=limit):
            allowed = visible_details.get(entry["action"], set())
            details = entry["details"]
            entries.append(
                {
                    "action": entry["action"],
                    "details": {key: details[key] for key in allowed if key in details},
                    "created_at": entry["created_at"],
                }
            )
        return entries

    def list_pending_approvals(self, session_id: str) -> list[dict]:
        self._require_session(session_id)
        if self._approval_pending is None:
            return []
        return list(self._approval_pending(self.operator_id, session_id))

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
        self._store.record_audit(session_id, "session.renamed")
        return renamed

    async def compact_session(self, session_id: str) -> dict:
        session = self._require_session(session_id)
        if self._context_compact is None:
            raise RuntimeError("context compaction is unavailable")
        result = await self._context_compact(
            chat_id=session.session_key,
            principal_id=self.operator_id,
            user_nickname=self._get_user_nickname(self.operator_id),
        )
        raw_reason = str(result.get("reason") or "")
        reason = {
            "no_scope": "no_scope",
            "disabled": "disabled",
            "unavailable": "unavailable",
            "tier3 summary": "summary",
            "summary rejected": "summary_rejected",
            "summary failed": "summary_failed",
        }.get(raw_reason, "not_changed" if not result.get("changed") else "completed")
        safe_result = {
            key: result.get(key)
            for key in (
                "changed",
                "tier",
                "operation",
                "before_tokens",
                "after_tokens",
                "saved_tokens",
                "reason",
                "scope_count",
            )
        }
        safe_result["reason"] = reason
        self._store.record_audit(
            session_id,
            "context.compacted",
            changed=bool(safe_result.get("changed")),
            tier=int(safe_result.get("tier") or 0),
            operation=str(safe_result.get("operation") or "none"),
            before_tokens=int(safe_result.get("before_tokens") or 0),
            after_tokens=int(safe_result.get("after_tokens") or 0),
            saved_tokens=int(safe_result.get("saved_tokens") or 0),
            reason=str(safe_result.get("reason") or ""),
        )
        await self._hub.publish(
            session_id,
            "context.compacted",
            payload=safe_result,
        )
        return safe_result

    async def submit(
        self,
        session_id: str,
        *,
        content: str = "",
        resources: Sequence[dict] = (),
        request_id: str = "",
        mode: str | None = None,
        model_group: str | None = None,
        reasoning_effort: str | None = None,
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
        if mode is not None and (
            not isinstance(mode, str) or mode not in {"agent", "chat"}
        ):
            raise ValueError("mode must be 'agent' or 'chat'")
        requested_mode = mode or session.mode
        if session.mode == "agent" and requested_mode == "chat":
            raise ValueError("agent sessions cannot switch back to chat")
        model_group = str(model_group or "").strip()
        model_chain = None
        if model_group and model_group != "auto":
            registry = self._model_registry
            has_group = getattr(registry, "has_group", None)
            get_group = getattr(registry, "get_group", None)
            if (
                not callable(has_group)
                or not callable(get_group)
                or not has_group(model_group)
            ):
                raise ValueError("unknown model group")
            model_chain = get_group(model_group)
            if not model_chain:
                raise ValueError("model group is unavailable")
        reasoning_effort = str(reasoning_effort or "").strip().lower()
        if reasoning_effort == "provider":
            reasoning_effort = ""
        if reasoning_effort:
            list_efforts = getattr(
                self._model_registry, "list_reasoning_effort_options", None
            )
            allowed_efforts = (
                list_efforts(model_chain) if callable(list_efforts) else []
            )
            if reasoning_effort not in allowed_efforts:
                raise ValueError("unsupported reasoning effort")
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
        await self._claim_draft_uploads(session, parsed_resources, turn_id)
        self._store.touch(session_id, self.operator_id)
        self._store.record_audit(
            session_id,
            "turn.submitted",
            mode=session.mode,
            resource_count=len(parsed_resources),
            model_group=model_group or "auto",
            reasoning_effort=reasoning_effort or "provider",
            turn_id=turn_id,
        )
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
            model_chain=model_chain,
            reasoning_effort=reasoning_effort or None,
        )
        await self._hub.publish(
            session_id,
            "turn.accepted",
            turn_id=turn_id,
            payload={
                "request_id": request_id,
                "message_id": turn_id,
                "sender_id": self.operator_id,
                "content": content,
                "mode": session.mode,
                "model_group": model_group or "auto",
                "reasoning_effort": reasoning_effort or "provider",
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

    async def upload_resource(
        self,
        session_id: str,
        *,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> dict:
        await self._cleanup_draft_uploads()
        session = self._require_session(session_id)
        if not data:
            raise ValueError("uploaded file is empty")
        media_service = self._media_service
        store = getattr(media_service, "store", None)
        if not getattr(media_service, "enabled", False) or store is None:
            raise ValueError("media upload is unavailable")
        mime_type = (
            str(mime_type or "application/octet-stream")
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        if mime_type == "image/svg+xml":
            raise ValueError("SVG uploads are not supported")
        resource_type = "image" if mime_type.startswith("image/") else "file"
        limit = int(
            getattr(
                media_service,
                "max_image_bytes" if resource_type == "image" else "max_file_bytes",
                10 * 1024 * 1024 if resource_type == "image" else 25 * 1024 * 1024,
            )
        )
        if len(data) > limit:
            raise ValueError(f"file exceeds the {limit} byte upload limit")
        safe_filename = re.sub(r"[\x00-\x1f\x7f/\\]+", "_", str(filename or "file"))
        safe_filename = safe_filename.strip(" .")[:120] or "file"
        upload_id = f"{self._DRAFT_UPLOAD_PREFIX}{uuid4().hex}"
        record = await store.save(
            chat_id=session.session_key,
            message_id=upload_id,
            sender_id=self.operator_id,
            resource_type=resource_type,
            source_url=f"webui-upload://{upload_id}",
            mime_type=mime_type,
            filename=safe_filename,
            data=data,
        )
        resource = {
            "resource_type": resource_type,
            "media_id": record.media_id,
            "media_uri": record.media_uri,
            "storage_status": "ready",
            "hash": record.sha256,
            "mime_type": record.mime_type,
            "size": record.size,
            "filename": record.filename,
            "preview_url": (
                f"/media/{record.media_id}/content" if resource_type == "image" else ""
            ),
            "download_url": f"/media/{record.media_id}/content?download=true",
            "extra": {"webui_upload_id": upload_id},
        }
        self._store.record_audit(
            session_id,
            "attachment.uploaded",
            media_id=record.media_id,
            resource_type=resource_type,
            size=record.size,
        )
        return resource

    async def discard_upload(
        self, session_id: str, *, media_uri: str, upload_id: str
    ) -> None:
        session = self._require_session(session_id)
        if not upload_id.startswith(self._DRAFT_UPLOAD_PREFIX):
            raise ValueError("invalid draft upload")
        store = getattr(self._media_service, "store", None)
        remove_reference = getattr(store, "remove_message_reference", None)
        if not callable(remove_reference):
            raise ValueError("media upload is unavailable")
        if not await remove_reference(
            chat_id=session.session_key,
            message_id=upload_id,
            media_uri=str(media_uri or ""),
        ):
            raise ValueError("draft upload is no longer available")
        self._store.record_audit(session_id, "attachment.discarded")

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

        async def tool_event_callback(event: ToolLifecycleEvent) -> None:
            safe_metadata = {}
            for key in ("reason", "error_code", "retryable"):
                value = event.metadata.get(key)
                if isinstance(value, (str, bool, int, float)):
                    safe_metadata[key] = value
            arguments = event.metadata.get("arguments")
            if isinstance(arguments, dict):
                safe_metadata["arguments"] = arguments
            result = event.metadata.get("result")
            if isinstance(result, str):
                safe_metadata["result"] = result[:4000]
            resources = event.metadata.get("resources")
            if isinstance(resources, (list, tuple)):
                safe_metadata["resources"] = [
                    dict(resource)
                    for resource in resources
                    if isinstance(resource, dict)
                ][:10]
            await self._hub.publish(
                session.session_id,
                event.event_type,
                turn_id=message.id,
                payload={
                    "tool_call_id": str(event.tool_call_id)[:200],
                    "tool_name": str(event.tool_name)[:120],
                    "status": str(event.status)[:40],
                    "elapsed_ms": event.elapsed_ms,
                    "attempt": event.attempt,
                    "metadata": safe_metadata,
                    "arguments": safe_metadata.get("arguments"),
                    "result": safe_metadata.get("result"),
                    "resources": safe_metadata.get("resources"),
                },
            )

        route_kwargs = {
            "input_message": message,
            "reply_callback": reply_callback,
            "get_user_nickname": self._get_user_nickname,
        }
        try:
            route_parameters = inspect.signature(self._route).parameters
        except (TypeError, ValueError):
            route_parameters = {}
        if "tool_event_callback" in route_parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in route_parameters.values()
        ):
            route_kwargs["tool_event_callback"] = tool_event_callback

        try:
            await self._route(**route_kwargs)
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
            if await self._hub.has_turn_event_type(
                session.session_id, turn_id, "turn.failed"
            ):
                return
            await self._hub.publish(
                session.session_id, "turn.completed", turn_id=turn_id
            )
            return
        for _ in range(3600):
            report = await self._event_log.validate_turn(
                turn_id, chat_id=session.session_key
            )
            if report.reason == "turn_not_found":
                if await self._hub.has_turn_event_type(
                    session.session_id, turn_id, "turn.failed"
                ):
                    return
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

    async def _cleanup_draft_uploads(self) -> None:
        store = getattr(self._media_service, "store", None)
        cleanup = getattr(store, "cleanup_message_references", None)
        if not callable(cleanup):
            return
        await cleanup(
            message_id_prefix=self._DRAFT_UPLOAD_PREFIX,
            older_than=time.time() - self._draft_upload_ttl_seconds,
        )

    async def _claim_draft_uploads(
        self,
        session: WebUiSession,
        resources: Sequence[ResourceMeta],
        turn_id: str,
    ) -> None:
        store = getattr(self._media_service, "store", None)
        move_reference = getattr(store, "move_message_reference", None)
        if not callable(move_reference):
            return
        upload_ids = set()
        for resource in resources:
            upload_id = str(resource.extra.get("webui_upload_id") or "")
            if (
                not resource.media_uri
                or not upload_id.startswith(self._DRAFT_UPLOAD_PREFIX)
                or upload_id in upload_ids
            ):
                continue
            upload_ids.add(upload_id)
            await move_reference(
                chat_id=session.session_key,
                source_message_id=upload_id,
                target_message_id=turn_id,
                media_uri=resource.media_uri,
            )

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
