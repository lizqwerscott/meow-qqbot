"""Persistent channel identities, chat aliases, and recent activity."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.session_identity import DeliveryTarget


@dataclass(frozen=True, slots=True)
class ChannelUserRef:
    identity_ref: str
    person_ref: str = ""
    current_name: str = ""
    historical_names: tuple[str, ...] = ()
    chat_name: str = ""
    scope: str = ""

    def prompt_line(self) -> str:
        parts = [f"- identity_ref: {self.identity_ref}"]
        if self.chat_name or self.current_name:
            parts.append(f"  当前称呼：{self.chat_name or self.current_name}")
        if self.historical_names:
            parts.append(f"  历史名字：{'、'.join(self.historical_names)}")
        if self.person_ref:
            parts.append(f"  统一人物：{self.person_ref}")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class _RosterSnapshot:
    refs: tuple[ChannelUserRef, ...]
    refreshed_at: float
    turns_since_refresh: int = 0


class IdentityManager:
    """Deep identity module with a small observe/resolve/search interface."""

    def __init__(self, db_path: str | Path = "data/identity.sqlite3") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._roster_snapshots: dict[tuple[str, str, str, str], _RosterSnapshot] = {}
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS channel_identities (
                channel TEXT NOT NULL,
                account_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                identity_ref TEXT NOT NULL UNIQUE,
                current_name TEXT NOT NULL DEFAULT '',
                username TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY(channel, account_id, actor_id)
            );
            CREATE TABLE IF NOT EXISTS persons (
                person_ref TEXT PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS identity_links (
                identity_ref TEXT PRIMARY KEY,
                person_ref TEXT NOT NULL REFERENCES persons(person_ref),
                linked_at REAL NOT NULL,
                linked_by TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_memberships (
                channel TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                chat_type TEXT NOT NULL DEFAULT 'group',
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                mention_count INTEGER NOT NULL DEFAULT 0,
                reply_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(channel, account_id, chat_id, chat_type, actor_id)
            );
            CREATE TABLE IF NOT EXISTS aliases (
                identity_ref TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                alias TEXT NOT NULL,
                is_manual INTEGER NOT NULL DEFAULT 0,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                PRIMARY KEY(identity_ref, scope, scope_key, alias)
            );
            CREATE INDEX IF NOT EXISTS idx_aliases_lookup
                ON aliases(scope, scope_key, alias);
            CREATE TABLE IF NOT EXISTS identity_suggestions (
                suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_ref TEXT NOT NULL,
                candidate_identity_ref TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                decided_at REAL,
                decided_by TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_identity_suggestions_status
                ON identity_suggestions(status, created_at);
            """)
        table_info = self._conn.execute(
            "PRAGMA table_info(chat_memberships)"
        ).fetchall()
        columns = {row["name"] for row in table_info}
        if "chat_type" not in columns:
            self._conn.execute(
                "ALTER TABLE chat_memberships ADD COLUMN chat_type TEXT NOT NULL DEFAULT 'group'"
            )
            table_info = self._conn.execute(
                "PRAGMA table_info(chat_memberships)"
            ).fetchall()
        primary_key = [row["name"] for row in table_info if row["pk"]]
        if "chat_type" not in primary_key:
            self._conn.execute(
                "ALTER TABLE chat_memberships RENAME TO chat_memberships_legacy"
            )
            self._conn.execute("""
                CREATE TABLE chat_memberships (
                    channel TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    chat_type TEXT NOT NULL DEFAULT 'group',
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    mention_count INTEGER NOT NULL DEFAULT 0,
                    reply_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(channel, account_id, chat_id, chat_type, actor_id)
                )
                """)
            self._conn.execute("""
                INSERT INTO chat_memberships (
                    channel, account_id, chat_id, actor_id, chat_type,
                    first_seen, last_seen, message_count, mention_count, reply_count
                )
                SELECT channel, account_id, chat_id, actor_id,
                       COALESCE(chat_type, 'group'), first_seen, last_seen,
                       message_count, mention_count, reply_count
                FROM chat_memberships_legacy
                """)
            self._conn.execute("DROP TABLE chat_memberships_legacy")
        self._conn.commit()

    def observe(
        self,
        target: DeliveryTarget,
        actor_id: str,
        name: str = "",
        username: str = "",
        *,
        mentioned: bool = False,
        replied: bool = False,
        observed_at: float | None = None,
    ) -> ChannelUserRef:
        if not actor_id:
            raise ValueError("actor_id is required")
        now = time.time() if observed_at is None else float(observed_at)
        identity = self._ensure_identity(
            target.channel, target.account_id, actor_id, name, username, now
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO chat_memberships (
                    channel, account_id, chat_id, actor_id, chat_type,
                    first_seen, last_seen, message_count, mention_count, reply_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel, account_id, chat_id, chat_type, actor_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    message_count = chat_memberships.message_count + excluded.message_count,
                    mention_count = chat_memberships.mention_count + excluded.mention_count,
                    reply_count = chat_memberships.reply_count + excluded.reply_count
                """,
                (
                    target.channel,
                    target.account_id,
                    target.target_id,
                    actor_id,
                    target.chat_type,
                    now,
                    now,
                    1 if not mentioned and not replied else 0,
                    1 if mentioned else 0,
                    1 if replied else 0,
                ),
            )
        if name:
            self._record_alias(
                identity, "channel", self._channel_key(target), name, False, now
            )
            self._record_alias(
                identity, "chat", self._chat_key(target), name, False, now
            )
        return self.get_ref(target, actor_id) or identity

    def observe_many(
        self,
        target: DeliveryTarget,
        entries: Iterable[tuple[str, str, bool, bool]],
    ) -> None:
        for actor_id, name, mentioned, replied in entries:
            if actor_id:
                self.observe(
                    target,
                    actor_id,
                    name,
                    mentioned=mentioned,
                    replied=replied,
                )

    def observe_legacy_history(
        self, target: DeliveryTarget, messages: Iterable[dict]
    ) -> int:
        """Backfill identities and activity from the retired chat history."""
        observed = 0
        for message in messages:
            if message.get("role") != "user":
                continue
            actor_id = str(message.get("sender_id") or "").strip()
            if not actor_id:
                continue
            timestamp = message.get("timestamp")
            try:
                observed_at = float(timestamp) if timestamp is not None else None
            except (TypeError, ValueError):
                observed_at = None
            existing = self._conn.execute(
                """
                SELECT 1 FROM chat_memberships
                WHERE channel = ? AND account_id = ? AND chat_id = ?
                  AND chat_type = ? AND actor_id = ?
                """,
                (
                    target.channel,
                    target.account_id,
                    target.target_id,
                    target.chat_type,
                    actor_id,
                ),
            ).fetchone()
            if existing is not None:
                ref = self.get_ref(target, actor_id)
                if ref is not None and message.get("name"):
                    self._record_alias(
                        ref,
                        "chat",
                        self._chat_key(target),
                        str(message["name"]),
                        False,
                        observed_at or time.time(),
                    )
                continue
            self.observe(
                target,
                actor_id,
                str(message.get("name") or ""),
                observed_at=observed_at,
            )
            observed += 1
        return observed

    def get_ref(self, target: DeliveryTarget, actor_id: str) -> ChannelUserRef | None:
        row = self._conn.execute(
            """
            SELECT ci.identity_ref, ci.current_name, ci.username,
                   il.person_ref, p.display_name
            FROM channel_identities ci
            LEFT JOIN identity_links il ON il.identity_ref = ci.identity_ref
            LEFT JOIN persons p ON p.person_ref = il.person_ref
            WHERE ci.channel = ? AND ci.account_id = ? AND ci.actor_id = ?
            """,
            (target.channel, target.account_id, actor_id),
        ).fetchone()
        if row is None:
            return None
        chat_name = self._latest_alias(
            row["identity_ref"], "chat", self._chat_key(target)
        )
        names = self._aliases(row["identity_ref"], "channel", self._channel_key(target))
        for alias in self._chat_aliases(target, row["identity_ref"]):
            if alias not in names:
                names.append(alias)
        if row["current_name"] and row["current_name"] not in names:
            names.insert(0, row["current_name"])
        return ChannelUserRef(
            identity_ref=row["identity_ref"],
            person_ref=str(row["person_ref"] or ""),
            current_name=str(
                row["display_name"] or row["current_name"] or row["username"] or ""
            ),
            historical_names=tuple(names),
            chat_name=chat_name,
            scope=self._chat_key(target),
        )

    def get_channel_name(
        self,
        actor_id: str,
        *,
        channel: str = "qq",
        account_id: str = "default",
    ) -> str:
        """Return the best channel-level display name for internal formatting."""
        row = self._conn.execute(
            """
            SELECT identity_ref, current_name, username
            FROM channel_identities
            WHERE channel = ? AND account_id = ? AND actor_id = ?
            """,
            (channel, account_id, actor_id),
        ).fetchone()
        if row is None:
            return str(actor_id or "")
        aliases = self._aliases(
            row["identity_ref"], "channel", f"{channel}:{account_id}"
        )
        return str(aliases[0] or row["current_name"] or row["username"] or actor_id)

    def ensure_legacy_ref(
        self, target: DeliveryTarget, actor_id: str
    ) -> ChannelUserRef | None:
        """Backfill an identity and membership from legacy conversation data."""
        actor_id = str(actor_id or "")
        if not actor_id:
            return None
        now = time.time()
        self._ensure_identity(
            target.channel,
            target.account_id,
            actor_id,
            "",
            "",
            now,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO chat_memberships (
                    channel, account_id, chat_id, actor_id, chat_type,
                    first_seen, last_seen, message_count, mention_count, reply_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0)
                ON CONFLICT(channel, account_id, chat_id, chat_type, actor_id) DO NOTHING
                """,
                (
                    target.channel,
                    target.account_id,
                    target.target_id,
                    actor_id,
                    target.chat_type,
                    now,
                    now,
                ),
            )
        return self.get_ref(target, actor_id)

    def resolve_identity(self, target: DeliveryTarget, identity_ref: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT actor_id FROM channel_identities
            WHERE channel = ? AND account_id = ? AND identity_ref = ?
            """,
            (target.channel, target.account_id, identity_ref),
        ).fetchone()
        if row is None:
            return None
        membership = self._conn.execute(
            """
            SELECT 1 FROM chat_memberships
            WHERE channel = ? AND account_id = ? AND chat_id = ?
              AND chat_type = ? AND actor_id = ?
            """,
            (
                target.channel,
                target.account_id,
                target.target_id,
                target.chat_type,
                row["actor_id"],
            ),
        ).fetchone()
        return str(row["actor_id"]) if membership else None

    def project_text(self, target: DeliveryTarget, text: str) -> str:
        """Replace known platform IDs with anonymous refs at token boundaries."""
        rows = self._conn.execute(
            """
            SELECT ci.actor_id, ci.identity_ref
            FROM channel_identities ci
            JOIN chat_memberships cm
              ON cm.channel = ci.channel AND cm.account_id = ci.account_id
             AND cm.actor_id = ci.actor_id
            WHERE cm.channel = ? AND cm.account_id = ? AND cm.chat_id = ?
              AND cm.chat_type = ?
            """,
            (target.channel, target.account_id, target.target_id, target.chat_type),
        ).fetchall()
        projected = str(text)
        for row in sorted(
            rows, key=lambda item: len(str(item["actor_id"])), reverse=True
        ):
            actor_id = str(row["actor_id"])
            identity_ref = str(row["identity_ref"])
            escaped_actor_id = re.escape(actor_id)
            projected = re.sub(
                rf"(?<![\w-])(@?){escaped_actor_id}(?![\w-])",
                lambda match: f"{match.group(1)}{identity_ref}",
                projected,
            )
        return projected

    def search(
        self, target: DeliveryTarget, query: str, limit: int = 10
    ) -> list[ChannelUserRef]:
        query = query.strip().casefold()
        if not query:
            return []
        rows = self._conn.execute(
            """
            SELECT DISTINCT ci.actor_id
            FROM channel_identities ci
            JOIN chat_memberships cm
              ON cm.channel = ci.channel AND cm.account_id = ci.account_id
             AND cm.actor_id = ci.actor_id
            LEFT JOIN aliases a ON a.identity_ref = ci.identity_ref
            WHERE cm.channel = ? AND cm.account_id = ? AND cm.chat_id = ?
              AND cm.chat_type = ?
              AND (lower(ci.identity_ref) LIKE ? OR lower(ci.current_name) LIKE ?
                   OR lower(ci.username) LIKE ? OR lower(a.alias) LIKE ?)
            ORDER BY cm.message_count DESC, cm.last_seen DESC, ci.actor_id
            LIMIT ?
            """,
            (
                target.channel,
                target.account_id,
                target.target_id,
                target.chat_type,
                f"%{query}%",
                f"%{query}%",
                f"%{query}%",
                f"%{query}%",
                max(1, min(limit, 50)),
            ),
        ).fetchall()
        return [
            self.get_ref(target, row["actor_id"])
            for row in rows
            if self.get_ref(target, row["actor_id"])
        ]

    def project_recent(
        self,
        target: DeliveryTarget,
        activity: dict[str, float],
        *,
        forced_actor_ids: Iterable[str] = (),
        limit: int = 30,
    ) -> list[ChannelUserRef]:
        roster = self.project_roster(target, activity, limit=limit)
        roster_refs = {ref.identity_ref for ref in roster}
        forced_refs = self.project_forced(target, forced_actor_ids)
        return roster + [
            ref for ref in forced_refs if ref.identity_ref not in roster_refs
        ]

    def project_roster(
        self,
        target: DeliveryTarget,
        activity: dict[str, float],
        *,
        limit: int = 30,
        refresh: bool = False,
    ) -> list[ChannelUserRef]:
        """Return a temporarily frozen, activity-ranked identity roster."""
        limit = max(1, min(limit, 100))
        key = target.catalog_key
        now = time.monotonic()
        snapshot = self._roster_snapshots.get(key)
        if (
            snapshot is not None
            and not refresh
            and now - snapshot.refreshed_at < 300.0
            and snapshot.turns_since_refresh < 10
        ):
            self._roster_snapshots[key] = _RosterSnapshot(
                refs=snapshot.refs,
                refreshed_at=snapshot.refreshed_at,
                turns_since_refresh=snapshot.turns_since_refresh + 1,
            )
            return list(snapshot.refs[:limit])

        refs = self._rank_roster_candidates(target, activity, max(limit, 30))
        self._roster_snapshots[key] = _RosterSnapshot(
            refs=tuple(refs), refreshed_at=now
        )
        return refs[:limit]

    def project_forced(
        self,
        target: DeliveryTarget,
        forced_actor_ids: Iterable[str],
    ) -> list[ChannelUserRef]:
        """Return current-turn identities observed in this exact chat."""
        forced = list(
            dict.fromkeys(actor_id for actor_id in forced_actor_ids if actor_id)
        )
        refs = []
        for actor_id in forced:
            ref = self.get_ref(target, actor_id)
            if ref is not None and self.resolve_identity(target, ref.identity_ref):
                refs.append(ref)
        return refs

    def _rank_roster_candidates(
        self,
        target: DeliveryTarget,
        activity: dict[str, float],
        limit: int,
    ) -> list[ChannelUserRef]:
        candidates = set(activity)
        rows = {}
        for actor_id in candidates:
            ref = self.get_ref(target, actor_id)
            if ref is not None:
                rows[actor_id] = ref
        if len(rows) < limit:
            fallback_rows = self._conn.execute(
                """
                SELECT actor_id FROM chat_memberships
                WHERE channel = ? AND account_id = ? AND chat_id = ? AND chat_type = ?
                ORDER BY message_count DESC, mention_count DESC, reply_count DESC, last_seen DESC, actor_id
                LIMIT ?
                """,
                (
                    *target.catalog_key[:2],
                    target.target_id,
                    target.chat_type,
                    max(limit, 30),
                ),
            ).fetchall()
            for row in fallback_rows:
                actor_id = str(row["actor_id"])
                if actor_id not in rows:
                    ref = self.get_ref(target, actor_id)
                    if ref is not None:
                        rows[actor_id] = ref
        ranked = sorted(
            rows.values(),
            key=lambda ref: (
                -self._activity_score(
                    target,
                    self.actor_for(target, ref.identity_ref),
                    activity,
                ),
                ref.identity_ref,
            ),
        )
        return ranked[:limit]

    def actor_for(self, target: DeliveryTarget, identity_ref: str) -> str:
        row = self._conn.execute(
            """
            SELECT actor_id FROM channel_identities
            WHERE channel = ? AND account_id = ? AND identity_ref = ?
            """,
            (target.channel, target.account_id, identity_ref),
        ).fetchone()
        return str(row["actor_id"]) if row else ""

    def _activity_score(
        self, target: DeliveryTarget, actor_id: str, recent_activity: dict[str, float]
    ) -> float:
        if actor_id in recent_activity:
            return recent_activity[actor_id]
        row = self._conn.execute(
            "SELECT message_count, mention_count, reply_count, last_seen FROM chat_memberships WHERE channel = ? AND account_id = ? AND chat_id = ? AND chat_type = ? AND actor_id = ?",
            (*target.catalog_key[:2], target.target_id, target.chat_type, actor_id),
        ).fetchone()
        if row is None:
            return 0.0
        age = max(0.0, time.time() - float(row["last_seen"] or time.time()))
        decay = pow(2.718281828, -age / 3600.0)
        return (
            float(row["message_count"])
            + 2.0 * float(row["mention_count"])
            + 2.0 * float(row["reply_count"])
        ) * decay

    def set_chat_alias(
        self, target: DeliveryTarget, actor_id: str, alias: str
    ) -> ChannelUserRef:
        ref = self.observe(target, actor_id, alias)
        self._record_alias(
            ref, "chat", self._chat_key(target), alias, True, time.time()
        )
        return self.get_ref(target, actor_id) or ref

    def set_chat_alias_for_identity(
        self, target: DeliveryTarget, identity_ref: str, alias: str
    ) -> ChannelUserRef:
        actor_id = self.resolve_identity(target, identity_ref)
        if not actor_id:
            raise ValueError("identity is not observed in this chat")
        now = time.time()
        self._record_alias(
            identity_ref,
            "chat",
            self._chat_key(target),
            alias,
            True,
            now,
        )
        return self.get_ref(target, actor_id) or ChannelUserRef(
            identity_ref, chat_name=alias, scope=self._chat_key(target)
        )

    def create_person(self, display_name: str = "") -> str:
        person_ref = f"person_{secrets.token_hex(4)}"
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO persons(person_ref, display_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (person_ref, display_name, now, now),
            )
        return person_ref

    def list_members(self, *, limit: int = 500) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT cm.channel, cm.account_id, cm.chat_id, cm.actor_id, cm.last_seen,
                   cm.message_count, cm.mention_count, cm.reply_count,
                   ci.identity_ref, ci.current_name, ci.username,
                   il.person_ref, p.display_name
            FROM chat_memberships cm
            JOIN channel_identities ci
              ON ci.channel = cm.channel AND ci.account_id = cm.account_id
             AND ci.actor_id = cm.actor_id
            LEFT JOIN identity_links il ON il.identity_ref = ci.identity_ref
            LEFT JOIN persons p ON p.person_ref = il.person_ref
            WHERE cm.chat_type = 'group'
            ORDER BY cm.last_seen DESC, cm.message_count DESC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
        result = []
        for row in rows:
            target = DeliveryTarget(
                channel=str(row["channel"]),
                account_id=str(row["account_id"]),
                chat_type="group",
                target_id=str(row["chat_id"]),
            )
            ref = self.get_ref(target, str(row["actor_id"]))
            result.append(
                {
                    "identity_ref": row["identity_ref"],
                    "chat_id": row["chat_id"],
                    "person_ref": row["person_ref"] or "",
                    "person_name": row["display_name"] or "",
                    "current_chat_name": ref.chat_name if ref else "",
                    "historical_names": list(ref.historical_names) if ref else [],
                    "current_name": row["current_name"] or row["username"] or "",
                    "chat_fingerprint": self.chat_fingerprint(
                        row["channel"], row["account_id"], row["chat_id"]
                    ),
                    "channel": row["channel"],
                    "message_count": int(row["message_count"]),
                    "mention_count": int(row["mention_count"]),
                    "reply_count": int(row["reply_count"]),
                    "last_seen": float(row["last_seen"]),
                }
            )
        return result

    def list_direct_peers(self, *, limit: int = 500) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT cm.channel, cm.account_id, cm.chat_id, cm.actor_id, cm.last_seen,
                   cm.message_count, cm.mention_count, cm.reply_count,
                   ci.identity_ref, ci.current_name, ci.username,
                   il.person_ref, p.display_name
            FROM chat_memberships cm
            JOIN channel_identities ci
              ON ci.channel = cm.channel AND ci.account_id = cm.account_id
             AND ci.actor_id = cm.actor_id
            LEFT JOIN identity_links il ON il.identity_ref = ci.identity_ref
            LEFT JOIN persons p ON p.person_ref = il.person_ref
            WHERE cm.chat_type = 'direct'
            ORDER BY cm.last_seen DESC, cm.message_count DESC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
        result = []
        for row in rows:
            target = DeliveryTarget(
                channel=str(row["channel"]),
                account_id=str(row["account_id"]),
                chat_type="direct",
                target_id=str(row["chat_id"]),
            )
            ref = self.get_ref(target, str(row["actor_id"]))
            result.append(
                {
                    "identity_ref": row["identity_ref"],
                    "chat_id": row["chat_id"],
                    "person_ref": row["person_ref"] or "",
                    "person_name": row["display_name"] or "",
                    "current_chat_name": ref.chat_name if ref else "",
                    "historical_names": list(ref.historical_names) if ref else [],
                    "current_name": row["current_name"] or row["username"] or "",
                    "chat_fingerprint": self.chat_fingerprint(
                        row["channel"], row["account_id"], row["chat_id"]
                    ),
                    "channel": row["channel"],
                    "message_count": int(row["message_count"]),
                    "mention_count": int(row["mention_count"]),
                    "reply_count": int(row["reply_count"]),
                    "last_seen": float(row["last_seen"]),
                }
            )
        return result

    def list_chats(self, *, limit: int = 500) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT channel, account_id, chat_id, chat_type,
                   COUNT(*) AS member_count,
                   MAX(last_seen) AS last_seen,
                   SUM(message_count) AS message_count
            FROM chat_memberships
            WHERE chat_type = 'group'
            GROUP BY channel, account_id, chat_id, chat_type
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
        return [
            {
                "channel": row["channel"],
                "account_id": row["account_id"],
                "chat_id": row["chat_id"],
                "chat_type": row["chat_type"],
                "chat_fingerprint": self.chat_fingerprint(
                    row["channel"], row["account_id"], row["chat_id"]
                ),
                "member_count": int(row["member_count"]),
                "message_count": int(row["message_count"] or 0),
                "last_seen": float(row["last_seen"]),
            }
            for row in rows
        ]

    def set_channel_alias(
        self,
        actor_id: str,
        alias: str,
        *,
        channel: str = "qq",
        account_id: str = "default",
        manual: bool = True,
    ) -> ChannelUserRef:
        """Set a channel-level alias without fabricating chat membership."""
        target = DeliveryTarget(channel, account_id, "direct", actor_id)
        now = time.time()
        identity = self._ensure_identity(channel, account_id, actor_id, "", "", now)
        with self._conn:
            self._conn.execute(
                "DELETE FROM aliases WHERE identity_ref = ? AND scope = 'channel' AND scope_key = ? AND is_manual = 1",
                (identity.identity_ref, f"{channel}:{account_id}"),
            )
        self._record_alias(
            identity,
            "channel",
            self._channel_key(target),
            alias,
            manual,
            now,
        )
        return identity

    def remove_automatic_channel_aliases(
        self,
        actor_id: str,
        *,
        channel: str = "qq",
        account_id: str = "default",
    ) -> None:
        row = self._conn.execute(
            "SELECT identity_ref FROM channel_identities WHERE channel = ? AND account_id = ? AND actor_id = ?",
            (channel, account_id, actor_id),
        ).fetchone()
        if row is None:
            return
        with self._conn:
            self._conn.execute(
                "DELETE FROM aliases WHERE identity_ref = ? AND scope = 'channel' AND scope_key = ? AND is_manual = 0",
                (row["identity_ref"], f"{channel}:{account_id}"),
            )

    def promote_channel_alias(
        self,
        actor_id: str,
        alias: str | None = None,
        *,
        channel: str = "qq",
        account_id: str = "default",
    ) -> bool:
        row = self._conn.execute(
            "SELECT identity_ref FROM channel_identities WHERE channel = ? AND account_id = ? AND actor_id = ?",
            (channel, account_id, actor_id),
        ).fetchone()
        if row is None:
            return False
        scope_key = f"{channel}:{account_id}"
        if alias is None:
            selected = self._conn.execute(
                "SELECT alias FROM aliases WHERE identity_ref = ? AND scope = 'channel' AND scope_key = ? AND is_manual = 0 ORDER BY last_seen DESC LIMIT 1",
                (row["identity_ref"], scope_key),
            ).fetchone()
            alias = str(selected["alias"]) if selected else ""
        if not alias:
            return False
        self.set_channel_alias(
            actor_id,
            alias,
            channel=channel,
            account_id=account_id,
            manual=True,
        )
        self.remove_automatic_channel_aliases(
            actor_id, channel=channel, account_id=account_id
        )
        return True

    def list_channel_aliases(
        self,
        *,
        channel: str = "qq",
        account_id: str = "default",
        limit: int = 500,
    ) -> dict[str, dict[str, object]]:
        """Return legacy-shaped manual/automatic views backed by SQLite aliases."""
        rows = self._conn.execute(
            """
            SELECT ci.actor_id, a.alias, a.is_manual, a.last_seen
            FROM channel_identities ci
            JOIN aliases a ON a.identity_ref = ci.identity_ref
            WHERE ci.channel = ? AND ci.account_id = ?
              AND a.scope = 'channel' AND a.scope_key = ?
            ORDER BY a.is_manual DESC, a.last_seen DESC, ci.actor_id
            LIMIT ?
            """,
            (channel, account_id, f"{channel}:{account_id}", max(1, min(limit, 2000))),
        ).fetchall()
        manual: dict[str, str] = {}
        automatic: dict[str, dict[str, object]] = {}
        for row in rows:
            actor_id = str(row["actor_id"])
            alias = str(row["alias"])
            if row["is_manual"]:
                manual.setdefault(actor_id, alias)
            else:
                entry = automatic.setdefault(
                    actor_id,
                    {"aliases": [], "updated_at": float(row["last_seen"])},
                )
                entry["aliases"].append(alias)
                entry["updated_at"] = max(
                    float(entry["updated_at"]), float(row["last_seen"])
                )
        return {"manual": manual, "auto": automatic}

    def remove_channel_alias(
        self,
        actor_id: str,
        *,
        channel: str = "qq",
        account_id: str = "default",
    ) -> None:
        row = self._conn.execute(
            "SELECT identity_ref FROM channel_identities WHERE channel = ? AND account_id = ? AND actor_id = ?",
            (channel, account_id, actor_id),
        ).fetchone()
        if row is None:
            return
        with self._conn:
            self._conn.execute(
                "DELETE FROM aliases WHERE identity_ref = ? AND scope = 'channel' AND scope_key = ? AND is_manual = 1",
                (row["identity_ref"], f"{channel}:{account_id}"),
            )

    def list_persons(self, *, limit: int = 500) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT person_ref, display_name, created_at, updated_at FROM persons ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(limit, 2000)),),
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def chat_fingerprint(channel: str, account_id: str, chat_id: str) -> str:
        digest = hashlib.sha256(
            f"{channel}:{account_id}:{chat_id}".encode("utf-8")
        ).hexdigest()
        return digest[:10]

    def link_person(self, identity_ref: str, person_ref: str, linked_by: str) -> None:
        identity = self._conn.execute(
            "SELECT 1 FROM channel_identities WHERE identity_ref = ?",
            (identity_ref,),
        ).fetchone()
        person = self._conn.execute(
            "SELECT 1 FROM persons WHERE person_ref = ?", (person_ref,)
        ).fetchone()
        if identity is None or person is None:
            raise ValueError("identity or person does not exist")
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO identity_links(identity_ref, person_ref, linked_at, linked_by) VALUES (?, ?, ?, ?)",
                (identity_ref, person_ref, time.time(), linked_by),
            )
            self._conn.execute(
                "UPDATE persons SET updated_at = ? WHERE person_ref = ?",
                (time.time(), person_ref),
            )

    def suggest_link(
        self, identity_ref: str, candidate_identity_ref: str, reason: str = ""
    ) -> int:
        if identity_ref == candidate_identity_ref:
            raise ValueError("an identity cannot suggest itself")
        rows = self._conn.execute(
            "SELECT identity_ref FROM channel_identities WHERE identity_ref IN (?, ?)",
            (identity_ref, candidate_identity_ref),
        ).fetchall()
        if len(rows) != 2:
            raise ValueError("identity does not exist")
        now = time.time()
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO identity_suggestions(
                    identity_ref, candidate_identity_ref, reason, status, created_at
                ) VALUES (?, ?, ?, 'pending', ?)
                """,
                (identity_ref, candidate_identity_ref, str(reason)[:240], now),
            )
        return int(cursor.lastrowid)

    def suggest_matching_identities(self, *, limit: int = 100) -> int:
        rows = self._conn.execute(
            """
            SELECT identity_ref, channel, account_id, current_name, username
            FROM channel_identities
            ORDER BY identity_ref
            LIMIT ?
            """,
            (max(2, min(limit, 500)),),
        ).fetchall()
        candidates = []
        for row in rows:
            names = [row["current_name"], row["username"]]
            names.extend(
                self._aliases(
                    row["identity_ref"],
                    "channel",
                    f"{row['channel']}:{row['account_id']}",
                )
            )
            normalized = {
                "".join(str(name).split()).casefold()
                for name in names
                if str(name or "").strip()
            }
            normalized = {name for name in normalized if len(name) >= 2}
            if normalized:
                candidates.append((row, normalized))

        created = 0
        for index, (first, first_names) in enumerate(candidates):
            for second, second_names in candidates[index + 1 :]:
                if first["channel"] == second["channel"]:
                    continue
                if not first_names.intersection(second_names):
                    continue
                first_person = self.get_identity_person(first["identity_ref"])
                second_person = self.get_identity_person(second["identity_ref"])
                if first_person and first_person == second_person:
                    continue
                existing = self._conn.execute(
                    """
                    SELECT 1 FROM identity_suggestions
                    WHERE status = 'pending'
                      AND ((identity_ref = ? AND candidate_identity_ref = ?)
                       OR (identity_ref = ? AND candidate_identity_ref = ?))
                    """,
                    (
                        first["identity_ref"],
                        second["identity_ref"],
                        second["identity_ref"],
                        first["identity_ref"],
                    ),
                ).fetchone()
                if existing:
                    continue
                self.suggest_link(
                    first["identity_ref"],
                    second["identity_ref"],
                    f"名称匹配：{sorted(first_names.intersection(second_names))[0]}",
                )
                created += 1
        return created

    def list_suggestions(
        self, *, status: str = "pending", limit: int = 500
    ) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT suggestion_id, identity_ref, candidate_identity_ref, reason,
                   status, created_at, decided_at, decided_by
            FROM identity_suggestions
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, max(1, min(limit, 2000))),
        ).fetchall()
        return [dict(row) for row in rows]

    def resolve_suggestion(
        self, suggestion_id: int, *, accepted: bool, decided_by: str
    ) -> None:
        row = self._conn.execute(
            "SELECT * FROM identity_suggestions WHERE suggestion_id = ?",
            (suggestion_id,),
        ).fetchone()
        if row is None or row["status"] != "pending":
            raise ValueError("pending suggestion does not exist")
        if accepted:
            first = self.get_identity_person(row["identity_ref"])
            second = self.get_identity_person(row["candidate_identity_ref"])
            person_ref = first or second or self.create_person()
            if first and second and first != second:
                raise ValueError("identities are linked to different persons")
            self.link_person(row["identity_ref"], person_ref, decided_by)
            self.link_person(row["candidate_identity_ref"], person_ref, decided_by)
        now = time.time()
        with self._conn:
            self._conn.execute(
                "UPDATE identity_suggestions SET status = ?, decided_at = ?, decided_by = ? WHERE suggestion_id = ?",
                (
                    "accepted" if accepted else "rejected",
                    now,
                    decided_by,
                    suggestion_id,
                ),
            )

    def get_identity_person(self, identity_ref: str) -> str:
        row = self._conn.execute(
            "SELECT person_ref FROM identity_links WHERE identity_ref = ?",
            (identity_ref,),
        ).fetchone()
        return str(row["person_ref"]) if row else ""

    def migrate_legacy_nicknames(
        self,
        *,
        manual_path: str | Path = "config/nicknames.json",
        auto_path: str | Path = "data/nicknames.json",
        backup: bool = True,
    ) -> dict[str, int]:
        """Import legacy channel-level names without inventing chat membership."""
        imported_manual = self._migrate_legacy_file(
            manual_path, manual=True, backup=backup
        )
        imported_auto = self._migrate_legacy_file(
            auto_path, manual=False, backup=backup
        )
        return {"manual": imported_manual, "auto": imported_auto}

    def _migrate_legacy_file(
        self, path: str | Path, *, manual: bool, backup: bool
    ) -> int:
        source = Path(path)
        if not source.is_file():
            return 0
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid nickname file: {source}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"nickname file must be an object: {source}")
        if backup:
            backup_path = Path(f"{source}.legacy.bak")
            if not backup_path.exists():
                shutil.copy2(source, backup_path)

        now = time.time()
        imported = 0
        for actor_id, raw_value in data.items():
            actor_id = str(actor_id).strip()
            if not actor_id:
                continue
            if manual:
                aliases = [str(raw_value).strip()] if raw_value else []
            elif isinstance(raw_value, dict):
                aliases = [
                    str(value).strip()
                    for value in raw_value.get("aliases", [])
                    if str(value).strip()
                ]
            elif raw_value:
                aliases = [str(raw_value).strip()]
            else:
                aliases = []
            aliases = list(dict.fromkeys(alias for alias in aliases if alias))
            if not aliases:
                continue
            identity = self._ensure_identity(
                "qq", "default", actor_id, aliases[-1], "", now
            )
            for alias in aliases:
                self._record_alias(
                    identity, "channel", "qq:default", alias, manual, now
                )
            imported += 1
        return imported

    def _ensure_identity(
        self,
        channel: str,
        account_id: str,
        actor_id: str,
        name: str,
        username: str,
        now: float,
    ) -> ChannelUserRef:
        row = self._conn.execute(
            "SELECT identity_ref FROM channel_identities WHERE channel = ? AND account_id = ? AND actor_id = ?",
            (channel, account_id, actor_id),
        ).fetchone()
        if row:
            with self._conn:
                self._conn.execute(
                    "UPDATE channel_identities SET current_name = COALESCE(NULLIF(?, ''), current_name), username = COALESCE(NULLIF(?, ''), username), last_seen = ? WHERE channel = ? AND account_id = ? AND actor_id = ?",
                    (name, username, now, channel, account_id, actor_id),
                )
            return ChannelUserRef(
                str(row["identity_ref"]), current_name=name, scope=account_id
            )
        identity_ref = self._new_identity_ref()
        with self._conn:
            self._conn.execute(
                "INSERT INTO channel_identities(channel, account_id, actor_id, identity_ref, current_name, username, created_at, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (channel, account_id, actor_id, identity_ref, name, username, now, now),
            )
        return ChannelUserRef(identity_ref, current_name=name, scope=account_id)

    def _new_identity_ref(self) -> str:
        while True:
            value = f"member_{secrets.token_hex(4)}"
            row = self._conn.execute(
                "SELECT 1 FROM channel_identities WHERE identity_ref = ?", (value,)
            ).fetchone()
            if row is None:
                return value

    def _record_alias(
        self,
        identity: ChannelUserRef | str,
        scope: str,
        scope_key: str,
        alias: str,
        manual: bool,
        now: float,
    ) -> None:
        identity_ref = (
            identity.identity_ref if isinstance(identity, ChannelUserRef) else identity
        )
        alias = str(alias).strip()
        if not alias:
            return
        with self._conn:
            self._conn.execute(
                "INSERT INTO aliases(identity_ref, scope, scope_key, alias, is_manual, first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(identity_ref, scope, scope_key, alias) DO UPDATE SET is_manual = MAX(aliases.is_manual, excluded.is_manual), last_seen = excluded.last_seen",
                (identity_ref, scope, scope_key, alias, int(manual), now, now),
            )

    def _aliases(self, identity_ref: str, scope: str, scope_key: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT alias FROM aliases WHERE identity_ref = ? AND scope = ? AND scope_key = ? ORDER BY is_manual DESC, last_seen DESC",
            (identity_ref, scope, scope_key),
        ).fetchall()
        return [str(row["alias"]) for row in rows]

    def _latest_alias(self, identity_ref: str, scope: str, scope_key: str) -> str:
        values = self._aliases(identity_ref, scope, scope_key)
        return values[0] if values else ""

    @staticmethod
    def _channel_key(target: DeliveryTarget) -> str:
        return f"{target.channel}:{target.account_id}"

    @staticmethod
    def _chat_key(target: DeliveryTarget) -> str:
        return f"{target.channel}:{target.account_id}:{target.chat_type}:{target.target_id}"

    @staticmethod
    def _legacy_chat_key(target: DeliveryTarget) -> str:
        return f"{target.channel}:{target.account_id}:{target.target_id}"

    def _chat_aliases(self, target: DeliveryTarget, identity_ref: str) -> list[str]:
        aliases = self._aliases(identity_ref, "chat", self._chat_key(target))
        if target.chat_type == "group":
            for alias in self._aliases(
                identity_ref, "chat", self._legacy_chat_key(target)
            ):
                if alias not in aliases:
                    aliases.append(alias)
        return aliases

    def close(self) -> None:
        self._conn.close()
