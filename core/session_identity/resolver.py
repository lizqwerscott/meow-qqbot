"""High-level resolution of chat and internal session identities.

The resolver is the only place where callers should choose between the
canonical identity and the pre-migration legacy key.  Keeping that decision in
one small object lets the runtime stay on the legacy path until a cold
migration has recorded a cutover state.
"""

from __future__ import annotations

from typing import Literal

from .identity import (
    ConversationRef,
    DeliveryTarget,
    build_chat_session_key,
    build_internal_session_key,
    build_workspace_slug,
    parse_chat_session_key,
)
from .registry import SessionIdentityRegistry


class SessionIdentityResolver:
    """Resolve durable conversation references for runtime consumers.

    ``canonical_enabled`` is deliberately explicit.  It is off by default so
    a freshly deployed bot cannot silently orphan legacy stores before the
    offline migration has completed.  A migrated registry can opt in by
    enabling it at startup.
    """

    def __init__(
        self,
        registry: SessionIdentityRegistry | None = None,
        *,
        agent_id: str = "main",
        canonical_enabled: bool = False,
    ) -> None:
        self.registry = registry
        self.agent_id = agent_id
        self.canonical_enabled = bool(canonical_enabled)

    def resolve_inbound(
        self,
        target: DeliveryTarget,
        *,
        legacy_key: str | None = None,
    ) -> ConversationRef:
        """Return the stable reference for one channel delivery target."""
        existing = (
            self.registry.find_by_target(target, agent_id=self.agent_id)
            if self.registry
            else None
        )
        if existing is None and self.registry is not None and self.canonical_enabled:
            existing = self.registry.register_chat(
                target,
                agent_id=self.agent_id,
                legacy_key=legacy_key or target.target_id,
                state="cutover" if self.canonical_enabled else "planned",
            )
        canonical_key = (
            existing.session_key
            if existing
            else build_chat_session_key(target, agent_id=self.agent_id)
        )
        if self.canonical_enabled:
            if existing is not None and existing.session_kind == "chat":
                return existing
            return ConversationRef(canonical_key, target, "chat")

        legacy = legacy_key or target.target_id
        document_id = existing.hindsight_document_id if existing else ""
        return ConversationRef(
            session_key=legacy,
            target=target,
            session_kind="chat",
            legacy_session_keys=((canonical_key,) if existing is not None else ()),
            hindsight_document_id=(document_id if existing is not None else ""),
        )

    def resolve_internal(
        self,
        kind: Literal["cron", "heartbeat", "work_plan", "work-plan"],
        *,
        job_id: str = "",
        run_id: str = "",
        name: str = "",
        legacy_key: str | None = None,
    ) -> ConversationRef:
        """Resolve cron/heartbeat/work-plan execution identities."""
        session_kind = "work_plan" if kind == "work-plan" else kind
        segments: list[str]
        if kind == "cron":
            if job_id and run_id:
                segments = ["job", job_id, "run", run_id]
            elif run_id:
                segments = ["run", run_id]
            elif job_id:
                segments = ["job", job_id]
            elif name:
                segments = [name]
            else:
                segments = ["main"]
        elif kind == "heartbeat":
            segments = [name or "events"]
        else:
            segments = [name or job_id or run_id]
        wire_kind = "work-plan" if session_kind == "work_plan" else session_kind
        canonical_key = build_internal_session_key(
            wire_kind, *segments, agent_id=self.agent_id
        )
        legacy = legacy_key or self._legacy_internal_key(
            kind, job_id=job_id, run_id=run_id, name=name
        )
        existing = None
        document_id = ""
        if self.registry is not None:
            existing = self.registry.get(canonical_key)
            if existing is None and self.canonical_enabled:
                existing = self.registry.register_internal(
                    canonical_key,
                    session_kind=session_kind,
                    agent_id=self.agent_id,
                    legacy_key=legacy,
                    state="cutover" if self.canonical_enabled else "planned",
                )
            document_id = existing.hindsight_document_id if existing else ""
        return ConversationRef(
            session_key=canonical_key if self.canonical_enabled else legacy,
            target=None,
            session_kind=session_kind,
            legacy_session_keys=(
                (canonical_key,)
                if not self.canonical_enabled and existing is not None
                else (
                    (legacy,)
                    if self.canonical_enabled and legacy != canonical_key
                    else ()
                )
            ),
            hindsight_document_id=document_id,
        )

    def resolve_cron_execution(
        self,
        *,
        mode: str,
        job_id: str,
        task_id: str,
        custom_id: str = "",
    ) -> ConversationRef:
        """Resolve the three existing cron lifecycles without losing semantics."""
        normalized = "isolated" if mode == "current" else mode
        if normalized == "main":
            return self.resolve_internal("cron", name="main", legacy_key="cron:main")
        if normalized == "custom" and custom_id:
            canonical = build_internal_session_key(
                "cron", "custom", custom_id, agent_id=self.agent_id
            )
            return self._register_internal_ref(
                canonical,
                legacy_key=f"cron:{custom_id}",
            )
        return self.resolve_internal(
            "cron", job_id=job_id, run_id=task_id, legacy_key=f"task:{task_id}"
        )

    def _register_internal_ref(
        self, canonical_key: str, *, legacy_key: str
    ) -> ConversationRef:
        existing = self.registry.get(canonical_key) if self.registry else None
        if existing is None and self.registry is not None and self.canonical_enabled:
            existing = self.registry.register_internal(
                canonical_key,
                session_kind="cron",
                agent_id=self.agent_id,
                legacy_key=legacy_key,
                state="cutover" if self.canonical_enabled else "planned",
            )
        document_id = existing.hindsight_document_id if existing else ""
        return ConversationRef(
            canonical_key if self.canonical_enabled else legacy_key,
            None,
            "cron",
            (
                (canonical_key,)
                if not self.canonical_enabled
                else ((legacy_key,) if legacy_key != canonical_key else ())
            ),
            document_id,
        )

    def resolve_legacy(
        self,
        raw_key: str,
        *,
        is_group: bool | None = None,
    ) -> ConversationRef:
        """Resolve a legacy key through registry aliases, fail-closed on ambiguity."""
        if raw_key.startswith("agent:"):
            registered = self.registry.get(raw_key) if self.registry else None
            if registered is not None:
                return registered
            try:
                target = parse_chat_session_key(raw_key)
            except ValueError:
                target = None
            if target is not None:
                return self.resolve_inbound(target, legacy_key=target.target_id)
            parts = raw_key.split(":")
            if (
                len(parts) >= 4
                and parts[0] == "agent"
                and parts[2]
                in {
                    "cron",
                    "heartbeat",
                    "work_plan",
                    "work-plan",
                }
            ):
                session_kind = "work_plan" if parts[2] == "work-plan" else parts[2]
                return ConversationRef(
                    raw_key,
                    None,
                    session_kind,
                    (),
                    "",
                )
        matches = self.registry.find_by_legacy(raw_key) if self.registry else []
        if len(matches) > 1:
            if is_group is None:
                raise ValueError(f"ambiguous legacy session key: {raw_key}")
            matches = [
                item
                for item in matches
                if item.target
                and item.target.chat_type == ("group" if is_group else "direct")
            ]
            if len(matches) != 1:
                raise ValueError(f"ambiguous legacy session key: {raw_key}")
        if matches:
            ref = matches[0]
            return (
                ref
                if self.canonical_enabled
                else ConversationRef(
                    raw_key,
                    ref.target,
                    ref.session_kind,
                    (ref.session_key,),
                    ref.hindsight_document_id,
                )
            )
        if is_group is not None:
            target = DeliveryTarget(
                "qq", "default", "group" if is_group else "direct", raw_key
            )
            return self.resolve_inbound(target, legacy_key=raw_key)
        if raw_key.startswith("heartbeat:"):
            return ConversationRef(raw_key, None, "heartbeat", (), "")
        if raw_key.startswith(("task:", "cron:")):
            return ConversationRef(raw_key, None, "cron", (), "")
        raise ValueError(f"unknown legacy session key: {raw_key}")

    def storage_key(self, ref: ConversationRef) -> str:
        return ref.session_key

    def workspace_slug(self, ref: ConversationRef) -> str:
        if ref.target is None:
            raise ValueError("internal sessions do not have workspaces")
        return build_workspace_slug(ref.target)

    def recall_aliases(self, ref: ConversationRef) -> tuple[str, ...]:
        tags = [f"chat:{ref.session_key}"]
        tags.extend(f"chat:{key}" for key in ref.legacy_session_keys)
        return tuple(dict.fromkeys(tags))

    @staticmethod
    def _legacy_internal_key(
        kind: str, *, job_id: str = "", run_id: str = "", name: str = ""
    ) -> str:
        if kind == "heartbeat":
            return f"heartbeat:{name or 'events'}"
        if kind == "cron":
            if job_id and run_id:
                return f"task:{run_id}"
            return f"cron:{name or 'main'}"
        return f"work-plan:{name or job_id or run_id}"
