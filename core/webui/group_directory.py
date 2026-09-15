"""Read-only group directory for the WebUI."""

from __future__ import annotations

from typing import Any

from core.session_identity.identity import parse_chat_session_key


class GroupDirectory:
    """Combines observed group activity with locally cached group metadata."""

    def __init__(self, identity_manager=None, channel_info_provider=None) -> None:
        self._identity_manager = identity_manager
        self._channel_info_provider = channel_info_provider

    def list_groups(self) -> list[dict[str, Any]]:
        observed_chats = self._list_observed_chats()
        cached_chats = self._list_cached_chats()
        groups: dict[tuple[str, str, str], dict[str, Any]] = {}

        for cached in cached_chats:
            key = self._key(cached)
            groups[key] = {
                "channel": cached["channel"],
                "channel_name": cached.get("channel_name") or cached["channel"],
                "account_id": cached["account_id"],
                "chat_fingerprint": cached["target_fingerprint"],
                "title": cached.get("title") or "",
                "description": cached.get("description") or "",
                "platform_member_count": cached.get("member_count"),
                "observed_member_count": 0,
                "message_count": 0,
                "last_observed_at": None,
                "fetched_at": cached.get("fetched_at"),
                "stale": bool(cached.get("stale")),
                "last_error_reason": cached.get("last_error_reason") or "",
            }

        for observed in observed_chats:
            key = self._key(observed)
            group = groups.setdefault(
                key,
                {
                    "channel": observed["channel"],
                    "channel_name": observed["channel"],
                    "account_id": observed["account_id"],
                    "chat_fingerprint": observed["chat_fingerprint"],
                    "title": "",
                    "description": "",
                    "platform_member_count": None,
                    "observed_member_count": 0,
                    "message_count": 0,
                    "last_observed_at": None,
                    "fetched_at": None,
                    "stale": False,
                    "last_error_reason": "",
                },
            )
            group["chat_fingerprint"] = observed["chat_fingerprint"]
            group["observed_member_count"] = observed["member_count"]
            group["message_count"] = observed["message_count"]
            group["last_observed_at"] = observed["last_seen"]

        return sorted(
            groups.values(),
            key=lambda group: (
                group["title"] == "",
                group["title"].casefold(),
                group["chat_fingerprint"],
            ),
        )

    def describe_sessions(self, session_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return safe channel/group labels for canonical session keys."""
        cached_chats = self._list_cached_chats()
        observed_chats = self._list_observed_chats()
        groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for cached in cached_chats:
            groups[
                (
                    str(cached["channel"]),
                    str(cached["account_id"]),
                    "group",
                    str(cached["target_id"]),
                )
            ] = {
                "channel_name": cached.get("channel_name") or cached["channel"],
                "group_name": cached.get("title") or "",
            }
        for observed in observed_chats:
            groups.setdefault(
                (
                    str(observed["channel"]),
                    str(observed["account_id"]),
                    str(observed.get("chat_type") or "group"),
                    str(observed["chat_id"]),
                ),
                {
                    "channel_name": observed["channel"],
                    "group_name": "",
                },
            )

        descriptions: dict[str, dict[str, Any]] = {}
        for session_id in session_ids:
            try:
                target = parse_chat_session_key(session_id)
            except (TypeError, ValueError):
                descriptions[session_id] = {
                    "channel_name": "",
                    "channel": "",
                    "account_id": "",
                    "chat_type": "",
                    "group_name": "",
                }
                continue
            group = groups.get(target.catalog_key, {})
            descriptions[session_id] = {
                "channel_name": group.get("channel_name") or target.channel,
                "channel": target.channel,
                "account_id": target.account_id,
                "chat_type": target.chat_type,
                "group_name": group.get("group_name") or "",
            }
        return descriptions

    def _list_observed_chats(self) -> list[dict[str, Any]]:
        list_chats = getattr(self._identity_manager, "list_chats", None)
        return list_chats() if callable(list_chats) else []

    def _list_cached_chats(self) -> list[dict[str, Any]]:
        list_cached_chats = getattr(
            self._channel_info_provider, "list_cached_chats", None
        )
        return list_cached_chats() if callable(list_cached_chats) else []

    @staticmethod
    def _key(group: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(group["channel"]),
            str(group["account_id"]),
            str(group.get("target_id") or group.get("chat_id") or ""),
        )
