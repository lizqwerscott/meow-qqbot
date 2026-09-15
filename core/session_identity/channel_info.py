"""Channel information snapshots and the QQ read-only provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .identity import DeliveryTarget

_log = logging.getLogger(__name__)

REASON_SELF_INFO_UNAVAILABLE = "self_info_unavailable"
REASON_CHAT_INFO_UNAVAILABLE = "chat_info_unavailable"
REASON_SELF_MEMBERSHIP_UNAVAILABLE = "self_membership_unavailable"
REASON_CAPABILITY_UNAVAILABLE = "capability_unavailable"
REASON_AUTH_FAILED = "auth_failed"
REASON_REQUEST_TIMEOUT = "request_timeout"
REASON_UPSTREAM_5XX = "upstream_5xx"
REASON_SCHEMA_INVALID = "schema_invalid"
REASON_CACHE_CORRUPT = "cache_corrupt"


@dataclass(frozen=True, slots=True)
class ChannelActorInfo:
    actor_id: str
    display_name: str = ""
    username: str = ""
    is_bot: bool | None = None


@dataclass(frozen=True, slots=True)
class ChannelChatInfo:
    chat_id: str
    chat_type: str
    title: str = ""
    description: str = ""
    member_count: int | None = None


@dataclass(frozen=True, slots=True)
class ChannelMembershipInfo:
    actor_id: str
    role: str = ""
    joined_at: str = ""
    can_send: bool | None = None
    can_receive: bool | None = None
    can_manage: bool | None = None


@dataclass(frozen=True, slots=True)
class ChannelInfoSnapshot:
    self_info: ChannelActorInfo | None = None
    chat_info: ChannelChatInfo | None = None
    self_membership: ChannelMembershipInfo | None = None
    subject: ChannelActorInfo | None = None
    subject_membership: ChannelMembershipInfo | None = None
    availability: str = "unavailable"
    unavailable_reasons: tuple[str, ...] = ()
    fetched_at: float = 0.0
    stale: bool = False
    schema_version: int = 1
    channel: str = ""
    channel_name: str = ""
    account_id: str = ""

    @classmethod
    def unavailable(
        cls,
        *reasons: str,
        channel: str = "",
        channel_name: str = "",
        account_id: str = "",
    ) -> "ChannelInfoSnapshot":
        return cls(
            channel=channel,
            channel_name=channel_name,
            account_id=account_id,
            unavailable_reasons=tuple(dict.fromkeys(reasons)),
        )

    def prompt_summary(self, *, include_runtime_status: bool = True) -> dict[str, Any]:
        """Return a bounded, platform-ID-free prompt projection."""
        result: dict[str, Any] = {
            "availability": self.availability,
        }
        if self.channel:
            result["channel"] = self.channel
        if self.channel_name:
            result["channel_name"] = _safe_text(self.channel_name, 80)
        if include_runtime_status:
            result["stale"] = self.stale
        if self.self_info:
            result["bot"] = {
                "display_name": _safe_text(self.self_info.display_name, 80),
                "username": _safe_text(self.self_info.username, 80),
                "is_bot": self.self_info.is_bot,
            }
        if self.chat_info:
            result["chat"] = {
                "type": self.chat_info.chat_type,
                "title": _safe_text(self.chat_info.title, 120),
                "description": _safe_text(self.chat_info.description, 240),
                "member_count": self.chat_info.member_count,
            }
        if self.self_membership:
            result["bot_membership"] = {
                "role": _safe_text(self.self_membership.role, 60),
                "can_send": self.self_membership.can_send,
                "can_receive": self.self_membership.can_receive,
                "can_manage": self.self_membership.can_manage,
            }
        return result


class ChannelInfoProvider(Protocol):
    async def get_info(
        self,
        target: DeliveryTarget | None = None,
        *,
        subject_id: str | None = None,
        refresh: bool = False,
    ) -> ChannelInfoSnapshot: ...

    def capabilities(self) -> frozenset[str]: ...


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    payload: dict[str, Any]
    fetched_at: float
    expires_at: float
    last_error_reason: str = ""


class QQChannelInfoProvider:
    """Deep provider hiding QQ HTTP, cache, and error normalization."""

    channel = "qq"

    def __init__(
        self,
        api_client: Any,
        *,
        account_id: str = "default",
        db_path: str | Path = "data/channel_info.sqlite3",
        timeout: float = 10.0,
    ) -> None:
        self.api_client = api_client
        self.account_id = account_id
        self.timeout = timeout
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS channel_info_cache (
                scope TEXT NOT NULL,
                channel TEXT NOT NULL,
                account_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                fetched_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                last_error_reason TEXT NOT NULL DEFAULT '',
                schema_version INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY(scope, channel, account_id, target_id)
            );
            """)
        self._conn.commit()
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def capabilities(self) -> frozenset[str]:
        return frozenset({"self_info", "chat_info", "self_membership"})

    def cache_status(self) -> list[dict[str, Any]]:
        now = time.time()
        rows = self._conn.execute("""
            SELECT scope, target_id, fetched_at, expires_at, last_error_reason
            FROM channel_info_cache
            ORDER BY scope, target_id
            """).fetchall()
        return [
            {
                "scope": str(row["scope"]),
                "target_fingerprint": _target_fingerprint(
                    str(row["scope"]), str(row["target_id"])
                ),
                "fetched_at": float(row["fetched_at"]),
                "expires_at": float(row["expires_at"]),
                "stale": float(row["expires_at"]) <= now,
                "last_error_reason": str(row["last_error_reason"] or ""),
            }
            for row in rows
        ]

    def list_cached_chats(self, *, limit: int = 500) -> list[dict[str, Any]]:
        """Return cached group metadata for local administrative views.

        This reads the local cache only and never refreshes QQ data.
        """
        now = time.time()
        rows = self._conn.execute(
            """
            SELECT channel, account_id, target_id, payload, fetched_at,
                   expires_at, last_error_reason
            FROM channel_info_cache
            WHERE scope = 'chat'
            ORDER BY fetched_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
        cached_chats = []
        for row in rows:
            channel = str(row["channel"])
            account_id = str(row["account_id"])
            target_id = str(row["target_id"])
            reason = str(row["last_error_reason"] or "")
            title = ""
            description = ""
            member_count = None
            if not reason:
                try:
                    payload = json.loads(row["payload"])
                    if not isinstance(payload, dict):
                        raise ValueError("cache payload is not an object")
                    chat_info = self._parse_chat(
                        payload,
                        DeliveryTarget(channel, account_id, "group", target_id),
                    )
                    title = _safe_text(chat_info.title, 120)
                    description = _safe_text(chat_info.description, 240)
                    member_count = chat_info.member_count
                except (TypeError, ValueError, json.JSONDecodeError):
                    reason = REASON_CACHE_CORRUPT
            cached_chats.append(
                {
                    "channel": channel,
                    "channel_name": "QQ" if channel == self.channel else channel,
                    "account_id": account_id,
                    "target_id": target_id,
                    "target_fingerprint": _target_fingerprint("chat", target_id),
                    "title": title,
                    "description": description,
                    "member_count": member_count,
                    "fetched_at": float(row["fetched_at"]),
                    "expires_at": float(row["expires_at"]),
                    "stale": float(row["expires_at"]) <= now,
                    "last_error_reason": reason,
                }
            )
        return cached_chats

    async def check_health(self) -> ChannelActorInfo:
        data = await self.api_client.request("GET", "/users/@me", timeout=self.timeout)
        return self._parse_actor(data, require_id=True, is_bot=True)

    async def get_info(
        self,
        target: DeliveryTarget | None = None,
        *,
        subject_id: str | None = None,
        refresh: bool = False,
    ) -> ChannelInfoSnapshot:
        if target is not None and (
            target.channel != self.channel or target.account_id != self.account_id
        ):
            return ChannelInfoSnapshot.unavailable(
                REASON_CAPABILITY_UNAVAILABLE,
                channel=self.channel,
                channel_name="QQ",
                account_id=self.account_id,
            )

        self_info, self_stale, self_reasons = await self._get_component(
            scope="account",
            target_id=self.account_id,
            path="/users/@me",
            ttl=3600.0,
            reason=REASON_SELF_INFO_UNAVAILABLE,
            parser=lambda data: self._parse_actor(data, require_id=True, is_bot=True),
            refresh=refresh,
        )

        chat_info = None
        self_membership = None
        stale = self_stale
        reasons = list(self_reasons)
        if target is not None and target.chat_type == "group":
            chat_info, chat_stale, chat_reasons = await self._get_component(
                scope="chat",
                target_id=target.target_id,
                path=f"/v2/groups/{target.target_id}/info",
                ttl=600.0,
                reason=REASON_CHAT_INFO_UNAVAILABLE,
                parser=lambda data: self._parse_chat(data, target),
                refresh=refresh,
            )
            self_membership, membership_stale, membership_reasons = (
                await self._get_component(
                    scope="membership",
                    target_id=target.target_id,
                    path=f"/v2/groups/{target.target_id}/bot_state",
                    ttl=60.0,
                    reason=REASON_SELF_MEMBERSHIP_UNAVAILABLE,
                    parser=lambda data: self._parse_membership(data, self_info),
                    refresh=refresh,
                )
            )
            stale = stale or chat_stale or membership_stale
            reasons.extend(chat_reasons)
            reasons.extend(membership_reasons)

        if subject_id:
            reasons.append(REASON_CAPABILITY_UNAVAILABLE)

        available = sum(
            value is not None for value in (self_info, chat_info, self_membership)
        )
        expected = 1 if target is None or target.chat_type == "direct" else 3
        availability = (
            "complete"
            if available == expected
            else "partial" if available else "unavailable"
        )
        return ChannelInfoSnapshot(
            channel=self.channel,
            channel_name="QQ",
            account_id=self.account_id,
            self_info=self_info,
            chat_info=chat_info,
            self_membership=self_membership,
            availability=availability,
            unavailable_reasons=tuple(dict.fromkeys(reasons)),
            fetched_at=time.time(),
            stale=stale,
        )

    async def _get_component(
        self,
        *,
        scope: str,
        target_id: str,
        path: str,
        ttl: float,
        reason: str,
        parser: Callable[[dict[str, Any]], Any],
        refresh: bool,
    ) -> tuple[Any | None, bool, tuple[str, ...]]:
        key = (scope, target_id)
        lock = await self._get_lock(key)
        async with lock:
            now = time.time()
            cached, cache_reason = self._read_cache(scope, target_id)
            if cached and cached.last_error_reason and cached.expires_at > now:
                return None, False, (cached.last_error_reason,)
            if cached and not refresh and cached.expires_at > now:
                if cached.last_error_reason and not cached.payload:
                    return None, False, (cached.last_error_reason,)
                try:
                    return parser(cached.payload), False, ()
                except Exception:
                    cache_reason = REASON_CACHE_CORRUPT
                    cached = None

            try:
                data = await self.api_client.request("GET", path, timeout=self.timeout)
                parsed = parser(data)
                payload = _encode_payload(parsed)
                self._write_cache(scope, target_id, payload, now + ttl, "")
                return parsed, False, ()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_reason = _reason_for_exception(exc, reason)
                if cache_reason == REASON_CACHE_CORRUPT:
                    error_reason = REASON_CACHE_CORRUPT
                if cached:
                    try:
                        return (
                            parser(cached.payload),
                            True,
                            (error_reason,),
                        )
                    except Exception:
                        pass
                self._write_cache(
                    scope, target_id, {}, now + _cooldown(error_reason), error_reason
                )
                _log.warning(
                    "QQ channel info unavailable scope=%s reason=%s",
                    scope,
                    error_reason,
                )
                return None, False, (error_reason,)

    async def _get_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(key, asyncio.Lock())

    def _read_cache(self, scope: str, target_id: str) -> tuple[_CacheEntry | None, str]:
        row = self._conn.execute(
            """
            SELECT payload, fetched_at, expires_at, last_error_reason
            FROM channel_info_cache
            WHERE scope = ? AND channel = ? AND account_id = ? AND target_id = ?
            """,
            (scope, self.channel, self.account_id, target_id),
        ).fetchone()
        if row is None:
            return None, ""
        try:
            payload = json.loads(row["payload"])
            if not isinstance(payload, dict):
                raise ValueError("cache payload is not an object")
            return (
                _CacheEntry(
                    payload=payload,
                    fetched_at=float(row["fetched_at"]),
                    expires_at=float(row["expires_at"]),
                    last_error_reason=str(row["last_error_reason"] or ""),
                ),
                str(row["last_error_reason"] or ""),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, REASON_CACHE_CORRUPT

    def _write_cache(
        self,
        scope: str,
        target_id: str,
        payload: dict[str, Any],
        expires_at: float,
        reason: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO channel_info_cache (
                    scope, channel, account_id, target_id, payload,
                    fetched_at, expires_at, last_error_reason, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(scope, channel, account_id, target_id) DO UPDATE SET
                    payload = excluded.payload,
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at,
                    last_error_reason = excluded.last_error_reason,
                    schema_version = 1
                """,
                (
                    scope,
                    self.channel,
                    self.account_id,
                    target_id,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                    expires_at,
                    reason,
                ),
            )

    @staticmethod
    def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
        value = data.get("data") if isinstance(data, dict) else None
        return value if isinstance(value, dict) else data

    @classmethod
    def _parse_actor(
        cls, data: dict[str, Any], *, require_id: bool, is_bot: bool | None
    ) -> ChannelActorInfo:
        value = cls._unwrap(data)
        actor_id = str(
            value.get("id") or value.get("user_id") or value.get("actor_id") or ""
        )
        if require_id and not actor_id:
            raise ValueError("actor id missing")
        return ChannelActorInfo(
            actor_id=actor_id,
            display_name=str(value.get("display_name") or value.get("nickname") or ""),
            username=str(value.get("username") or value.get("name") or ""),
            is_bot=value.get("is_bot", is_bot),
        )

    @classmethod
    def _parse_chat(
        cls, data: dict[str, Any], target: DeliveryTarget
    ) -> ChannelChatInfo:
        value = cls._unwrap(data)
        return ChannelChatInfo(
            chat_id=str(
                value.get("group_openid")
                or value.get("id")
                or value.get("chat_id")
                or target.target_id
            ),
            chat_type=str(value.get("chat_type") or "group"),
            title=str(
                value.get("group_name") or value.get("name") or value.get("title") or ""
            ),
            description=str(value.get("description") or ""),
            member_count=_optional_int(value.get("member_count")),
        )

    @classmethod
    def _parse_membership(
        cls, data: dict[str, Any], self_info: ChannelActorInfo | None
    ) -> ChannelMembershipInfo:
        value = cls._unwrap(data)
        return ChannelMembershipInfo(
            actor_id=(
                self_info.actor_id
                if self_info
                else str(value.get("member_openid") or value.get("actor_id") or "")
            ),
            role=str(value.get("role") or value.get("member_role") or ""),
            joined_at=str(value.get("joined_at") or ""),
            can_send=_optional_bool(
                value.get("allow_proactive_msg"), value.get("can_send")
            ),
            can_receive=_optional_bool(
                value.get("receive_message"), value.get("can_receive")
            ),
            can_manage=_optional_bool(value.get("is_admin"), value.get("can_manage")),
        )

    def close(self) -> None:
        self._conn.close()


def _safe_text(value: str, limit: int) -> str:
    return "".join(ch for ch in str(value) if ord(ch) >= 32 or ch in "\n\t")[:limit]


def _target_fingerprint(scope: str, target_id: str) -> str:
    if scope == "account":
        return "account"
    return hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:10]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_bool(*values: Any) -> bool | None:
    for value in values:
        if value is not None:
            return bool(value)
    return None


def _encode_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, ChannelActorInfo):
        return {
            "kind": "actor",
            "actor_id": value.actor_id,
            "display_name": value.display_name,
            "username": value.username,
            "is_bot": value.is_bot,
        }
    if isinstance(value, ChannelChatInfo):
        return {
            "kind": "chat",
            "chat_id": value.chat_id,
            "chat_type": value.chat_type,
            "title": value.title,
            "description": value.description,
            "member_count": value.member_count,
        }
    if isinstance(value, ChannelMembershipInfo):
        return {
            "kind": "membership",
            "actor_id": value.actor_id,
            "role": value.role,
            "joined_at": value.joined_at,
            "can_send": value.can_send,
            "can_receive": value.can_receive,
            "can_manage": value.can_manage,
        }
    raise TypeError(f"unsupported cache value: {type(value).__name__}")


def _reason_for_exception(exc: Exception, default: str) -> str:
    text = str(exc).lower()
    if "11253" in text or "40012010" in text or "11001" in text or "40011002" in text:
        return REASON_CAPABILITY_UNAVAILABLE
    if "timeout" in text:
        return REASON_REQUEST_TIMEOUT
    if " 5" in text or "500" in text or "502" in text or "503" in text:
        return REASON_UPSTREAM_5XX
    if "auth" in text or "401" in text or "403" in text:
        return REASON_AUTH_FAILED
    if "schema" in text or "missing" in text:
        return REASON_SCHEMA_INVALID
    return default


def _cooldown(reason: str) -> float:
    if reason == REASON_CAPABILITY_UNAVAILABLE:
        return 3600.0
    if reason == REASON_SCHEMA_INVALID:
        return 300.0
    return 30.0
