"""Durable discovery and validation catalog for delivery targets."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .identity import DeliveryTarget, DeliveryValidation


@dataclass(frozen=True, slots=True)
class KnownTarget:
    target: DeliveryTarget
    first_seen: float
    last_seen: float
    status: str
    source: str


@dataclass(frozen=True, slots=True)
class TargetPrincipal:
    principal_id: str
    is_admin: bool = False


@dataclass(frozen=True, slots=True)
class TargetFilters:
    channel: str | None = None
    account_id: str | None = None
    chat_type: str | None = None
    status: str | None = None


class DeliveryTargetCatalog:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS delivery_targets (
                channel TEXT NOT NULL,
                account_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                PRIMARY KEY(channel, account_id, chat_type, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_delivery_targets_status
                ON delivery_targets(status);
            """)
        self._conn.commit()

    def observe(
        self,
        target: DeliveryTarget,
        *,
        observed_at: float | None = None,
        source: str = "inbound",
        status: str = "verified",
    ) -> KnownTarget:
        observed_at = time.time() if observed_at is None else float(observed_at)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO delivery_targets (
                    channel, account_id, chat_type, target_id, first_seen,
                    last_seen, status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel, account_id, chat_type, target_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    status = excluded.status,
                    source = excluded.source
                """,
                (*target.catalog_key, observed_at, observed_at, status, source),
            )
        return self.get(target)  # type: ignore[return-value]

    def get(self, target: DeliveryTarget) -> KnownTarget | None:
        row = self._conn.execute(
            """
            SELECT * FROM delivery_targets
            WHERE channel = ? AND account_id = ? AND chat_type = ? AND target_id = ?
            """,
            target.catalog_key,
        ).fetchone()
        return self._row_to_known(row) if row is not None else None

    def list_visible(
        self,
        principal: TargetPrincipal | None = None,
        filters: TargetFilters | None = None,
        *,
        include_inactive: bool = False,
    ) -> list[KnownTarget]:
        """List targets visible to an operator.

        The catalog intentionally exposes all known targets only to admins;
        non-admin callers may still resolve/validate an explicitly supplied
        target but cannot enumerate the global directory.
        """
        if principal is not None and not principal.is_admin:
            return []
        filters = filters or TargetFilters()
        clauses = []
        values: list[str] = []
        if not include_inactive:
            clauses.append("status != 'inactive'")
        for column, value in (
            ("channel", filters.channel),
            ("account_id", filters.account_id),
            ("chat_type", filters.chat_type),
            ("status", filters.status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            "SELECT * FROM delivery_targets"
            + where
            + " ORDER BY channel, account_id, chat_type, target_id",
            values,
        ).fetchall()
        return [self._row_to_known(row) for row in rows]

    def resolve_for_cron(
        self,
        reference: str,
        principal: TargetPrincipal,
        current: DeliveryTarget,
    ) -> DeliveryTarget | None:
        """Resolve ``current`` or an explicitly catalogued target for Cron."""
        if reference == "current":
            return current
        if not principal.is_admin:
            return None
        candidates = [
            known
            for known in self.list_visible(principal, include_inactive=False)
            if known.target.target_id == reference
        ]
        if len(candidates) != 1:
            return None
        return candidates[0].target

    def mark_inactive(self, target: DeliveryTarget) -> None:
        with self._conn:
            updated = self._conn.execute(
                """
                UPDATE delivery_targets SET status = 'inactive'
                WHERE channel = ? AND account_id = ? AND chat_type = ? AND target_id = ?
                """,
                target.catalog_key,
            ).rowcount
        if not updated:
            raise KeyError(target.catalog_key)

    def validate(self, target: DeliveryTarget) -> DeliveryValidation:
        known = self.get(target)
        if known is None:
            return DeliveryValidation(False, "target_not_observed", target)
        if known.status == "inactive":
            return DeliveryValidation(False, "target_inactive", target)
        return DeliveryValidation(True, "ok", target)

    def purge(self, target: DeliveryTarget, *, referenced: bool = False) -> None:
        if referenced:
            raise ValueError("cannot purge a referenced delivery target")
        with self._conn:
            self._conn.execute(
                """
                DELETE FROM delivery_targets
                WHERE channel = ? AND account_id = ? AND chat_type = ? AND target_id = ?
                """,
                target.catalog_key,
            )

    @staticmethod
    def _row_to_known(row: sqlite3.Row) -> KnownTarget:
        return KnownTarget(
            target=DeliveryTarget(
                row["channel"], row["account_id"], row["chat_type"], row["target_id"]
            ),
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            status=row["status"],
            source=row["source"],
        )

    def close(self) -> None:
        self._conn.close()
