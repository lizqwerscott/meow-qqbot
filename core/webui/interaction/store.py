"""Small durable store for WebUI session metadata and submit idempotency."""

from __future__ import annotations

import base64
import binascii
import json
import math
import sqlite3
import time
from pathlib import Path

from .models import SubmissionReceipt, WebUiSession, WebUiSessionPage


class WebUiSessionStore:
    def __init__(self, path: str = "data/webui_sessions.sqlite3") -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def _ensure_open(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS webui_sessions (
                session_id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL UNIQUE,
                operator_id TEXT NOT NULL,
                title TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS webui_submissions (
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (session_id, request_id)
            );
            CREATE TABLE IF NOT EXISTS webui_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_webui_audit_session
            ON webui_audit(session_id, created_at DESC);
            """)
        self._conn.commit()
        return self._conn

    @staticmethod
    def _session(row: sqlite3.Row) -> WebUiSession:
        return WebUiSession(
            session_id=str(row["session_id"]),
            session_key=str(row["session_key"]),
            operator_id=str(row["operator_id"]),
            title=str(row["title"]),
            mode=str(row["mode"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def create(
        self,
        *,
        session_id: str,
        session_key: str,
        operator_id: str,
        title: str,
        mode: str,
    ) -> WebUiSession:
        now = time.time()
        conn = self._ensure_open()
        conn.execute(
            "INSERT INTO webui_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, session_key, operator_id, title, mode, now, now),
        )
        conn.commit()
        return self._session(
            conn.execute(
                "SELECT * FROM webui_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        )

    def get(self, session_id: str, operator_id: str) -> WebUiSession | None:
        row = (
            self._ensure_open()
            .execute(
                "SELECT * FROM webui_sessions WHERE session_id = ? AND operator_id = ?",
                (session_id, operator_id),
            )
            .fetchone()
        )
        return self._session(row) if row is not None else None

    @staticmethod
    def _encode_session_cursor(session: WebUiSession) -> str:
        value = json.dumps(
            {"updated_at": session.updated_at, "session_id": session.session_id},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_session_cursor(cursor: str) -> tuple[float, str]:
        try:
            padding = "=" * (-len(cursor) % 4)
            value = json.loads(
                base64.b64decode(
                    (cursor + padding).encode("ascii"),
                    altchars=b"-_",
                    validate=True,
                )
            )
            if not isinstance(value, dict):
                raise ValueError("cursor payload must be an object")
            updated_at = float(value["updated_at"])
            session_id = value["session_id"]
            if not isinstance(session_id, str):
                raise ValueError("cursor session_id must be a string")
        except (
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
            binascii.Error,
            UnicodeError,
        ):
            raise ValueError("invalid session cursor") from None
        if not math.isfinite(updated_at) or not session_id:
            raise ValueError("invalid session cursor")
        return updated_at, session_id

    def list_page(
        self, operator_id: str, *, limit: int = 50, cursor: str = ""
    ) -> WebUiSessionPage:
        limit = max(1, min(int(limit), 100))
        conn = self._ensure_open()
        params: list[object] = [operator_id]
        where = "operator_id = ?"
        if cursor:
            updated_at, session_id = self._decode_session_cursor(cursor)
            where += " AND (updated_at < ? OR (updated_at = ? AND session_id < ?))"
            params.extend([updated_at, updated_at, session_id])
        rows = conn.execute(
            "SELECT * FROM webui_sessions WHERE "
            f"{where} ORDER BY updated_at DESC, session_id DESC LIMIT ?",
            (*params, limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        sessions = tuple(self._session(row) for row in rows[:limit])
        next_cursor = self._encode_session_cursor(sessions[-1]) if has_more else None
        return WebUiSessionPage(sessions, has_more, next_cursor)

    def list(self, operator_id: str) -> list[WebUiSession]:
        return list(self.list_page(operator_id, limit=100).items)

    def update_title(
        self, session_id: str, operator_id: str, title: str
    ) -> WebUiSession | None:
        conn = self._ensure_open()
        updated = conn.execute(
            "UPDATE webui_sessions SET title = ?, updated_at = ? "
            "WHERE session_id = ? AND operator_id = ?",
            (title, time.time(), session_id, operator_id),
        ).rowcount
        conn.commit()
        return self.get(session_id, operator_id) if updated else None

    def update_mode(
        self, session_id: str, operator_id: str, mode: str
    ) -> WebUiSession | None:
        conn = self._ensure_open()
        updated = conn.execute(
            "UPDATE webui_sessions SET mode = ?, updated_at = ? "
            "WHERE session_id = ? AND operator_id = ?",
            (mode, time.time(), session_id, operator_id),
        ).rowcount
        conn.commit()
        return self.get(session_id, operator_id) if updated else None

    def touch(self, session_id: str, operator_id: str) -> None:
        conn = self._ensure_open()
        conn.execute(
            "UPDATE webui_sessions SET updated_at = ? "
            "WHERE session_id = ? AND operator_id = ?",
            (time.time(), session_id, operator_id),
        )
        conn.commit()

    def reserve_submission(
        self, session_id: str, request_id: str, turn_id: str
    ) -> SubmissionReceipt | None:
        conn = self._ensure_open()
        existing = conn.execute(
            "SELECT turn_id FROM webui_submissions WHERE session_id = ? "
            "AND request_id = ?",
            (session_id, request_id),
        ).fetchone()
        if existing is not None:
            return SubmissionReceipt(
                session_id=session_id,
                turn_id=str(existing["turn_id"]),
                request_id=request_id,
                accepted=True,
                duplicate=True,
            )
        conn.execute(
            "INSERT INTO webui_submissions VALUES (?, ?, ?, ?)",
            (session_id, request_id, turn_id, time.time()),
        )
        conn.commit()
        return None

    def record_audit(self, session_id: str, action: str, **details: object) -> None:
        conn = self._ensure_open()
        conn.execute(
            "INSERT INTO webui_audit(session_id, action, details, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                session_id,
                action,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
                time.time(),
            ),
        )
        conn.commit()

    def list_audit(self, session_id: str, limit: int = 100) -> list[dict]:
        rows = (
            self._ensure_open()
            .execute(
                "SELECT action, details, created_at FROM webui_audit WHERE session_id=? "
                "ORDER BY audit_id DESC LIMIT ?",
                (session_id, max(1, min(int(limit), 1_000))),
            )
            .fetchall()
        )
        return [
            {
                "action": str(row["action"]),
                "details": json.loads(str(row["details"])),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]
