import asyncio
import hashlib
import mimetypes
import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

from core.media.models import MediaRecord


class MediaStore:
    """受控媒体文件与 SQLite 授权索引。"""

    def __init__(self, root: str | Path, retention_days: int | None = None):
        self.root = Path(root)
        self.inbound = self.root / "inbound"
        self.index_path = self.root / "index.sqlite3"
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    async def open(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbound.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_sync()
        self._repair_sync()

    def _open_sync(self):
        conn = sqlite3.connect(self.index_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS media_objects (
                media_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
                local_path TEXT NOT NULL, mime_type TEXT NOT NULL,
                size INTEGER NOT NULL, created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0, filename TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '', summary_model TEXT NOT NULL DEFAULT '',
                summary_version TEXT NOT NULL DEFAULT '',
                file_summary TEXT NOT NULL DEFAULT '',
                file_summary_model TEXT NOT NULL DEFAULT '',
                file_summary_version TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS media_transcripts (
                media_id TEXT PRIMARY KEY,
                transcript TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS media_messages (
                chat_id TEXT NOT NULL, message_id TEXT NOT NULL,
                sender_id TEXT NOT NULL, media_id TEXT NOT NULL,
                resource_type TEXT NOT NULL DEFAULT '',
                position INTEGER NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (chat_id, message_id, media_id)
            );
            CREATE INDEX IF NOT EXISTS idx_media_messages_recent
            ON media_messages(chat_id, created_at DESC);
            """)
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(media_messages)")
        }
        if "resource_type" not in columns:
            conn.execute(
                "ALTER TABLE media_messages "
                "ADD COLUMN resource_type TEXT NOT NULL DEFAULT ''"
            )
        object_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(media_objects)")
        }
        if "summary_version" not in object_columns:
            conn.execute(
                "ALTER TABLE media_objects "
                "ADD COLUMN summary_version TEXT NOT NULL DEFAULT ''"
            )
        for column in ("file_summary", "file_summary_model", "file_summary_version"):
            if column not in object_columns:
                conn.execute(
                    f"ALTER TABLE media_objects ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )
        conn.commit()
        return conn

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def repair(self) -> None:
        if self._conn is None:
            return
        async with self._lock:
            self._repair_sync()

    def _repair_sync(self) -> None:
        conn = self._conn
        if conn is None:
            return
        rows = conn.execute("SELECT media_id, local_path FROM media_objects").fetchall()
        indexed = {Path(row["local_path"]).resolve() for row in rows}
        for path in self.inbound.rglob("*"):
            if path.is_file() and path.resolve() not in indexed:
                try:
                    path.unlink()
                except OSError:
                    pass
        conn.commit()

    async def save(
        self,
        *,
        chat_id: str,
        message_id: str,
        sender_id: str,
        resource_type: str,
        source_url: str,
        mime_type: str,
        filename: str,
        data: bytes,
        position: int = 0,
    ) -> MediaRecord:
        if self._conn is None:
            await self.open()
        digest = hashlib.sha256(data).hexdigest()
        media_id = digest
        now = time.time()
        suffix = (
            Path(urlparse(source_url).path).suffix.lower()
            or Path(filename).suffix.lower()
        )
        suffix = (
            suffix if len(suffix) <= 10 and suffix.replace(".", "").isalnum() else ""
        )
        path = self.inbound / digest[:2] / f"{digest}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            self._save_sync(
                media_id,
                path,
                digest,
                mime_type,
                filename,
                data,
                chat_id,
                message_id,
                sender_id,
                resource_type,
                position,
                now,
            )
        return MediaRecord(
            media_id,
            f"media://inbound/{media_id}",
            chat_id,
            message_id,
            sender_id,
            resource_type,
            mime_type
            or mimetypes.guess_type(str(path))[0]
            or "application/octet-stream",
            len(data),
            digest,
            path,
            now,
            0,
            filename,
        )

    def _save_sync(
        self,
        media_id,
        path,
        digest,
        mime_type,
        filename,
        data,
        chat_id,
        message_id,
        sender_id,
        resource_type,
        position,
        now,
    ):
        if not path.exists():
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(data)
            os.replace(temp, path)
        conn = self._conn
        assert conn is not None
        conn.execute(
            "INSERT OR IGNORE INTO media_objects(media_id,sha256,local_path,mime_type,size,created_at,expires_at,filename) VALUES(?,?,?,?,?,?,?,?)",
            (
                media_id,
                digest,
                str(path),
                mime_type or "",
                len(data),
                now,
                0,
                filename or "",
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO media_messages("
            "chat_id,message_id,sender_id,media_id,resource_type,position,created_at"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                chat_id,
                message_id,
                sender_id,
                media_id,
                resource_type,
                position,
                now,
            ),
        )
        conn.commit()

    async def authorize(
        self, chat_id: str, media_uri: str, image_only: bool = False
    ) -> MediaRecord | None:
        record, _ = await self.authorize_with_reason(chat_id, media_uri, image_only)
        return record

    async def authorize_with_reason(
        self, chat_id: str, media_uri: str, image_only: bool = False
    ) -> tuple[MediaRecord | None, str]:
        if self._conn is None:
            return None, "MEDIA_NOT_AVAILABLE"
        if not media_uri.startswith(
            "media://inbound/"
        ) or "/" in media_uri.removeprefix("media://inbound/"):
            return None, "INVALID_MEDIA_URI"
        media_id = media_uri.removeprefix("media://inbound/")
        async with self._lock:
            row = self._conn.execute(
                "SELECT o.*, m.message_id, m.chat_id, m.sender_id, m.resource_type, "
                "m.created_at AS message_created_at "
                "FROM media_objects o JOIN media_messages m ON m.media_id=o.media_id "
                "WHERE o.media_id=? AND m.chat_id=? ORDER BY m.created_at DESC LIMIT 1",
                (media_id, chat_id),
            ).fetchone()
        if not row:
            exists = self._conn.execute(
                "SELECT 1 FROM media_objects WHERE media_id=?", (media_id,)
            ).fetchone()
            return (None, "MEDIA_FORBIDDEN" if exists else "MEDIA_NOT_AVAILABLE")
        if not Path(row["local_path"]).is_file():
            return None, "MEDIA_NOT_AVAILABLE"
        if image_only and not row["mime_type"].startswith("image/"):
            return None, "UNSUPPORTED_MEDIA_TYPE"
        return (
            MediaRecord(
                row["media_id"],
                media_uri,
                row["chat_id"],
                row["message_id"],
                row["sender_id"],
                self._resource_type_from_row(row),
                row["mime_type"],
                row["size"],
                row["sha256"],
                Path(row["local_path"]),
                row["message_created_at"],
                row["expires_at"],
                row["filename"],
                row["summary"],
                row["summary_model"],
                row["summary_version"],
                row["file_summary"],
                row["file_summary_model"],
                row["file_summary_version"],
            ),
            "",
        )

    async def find_message_media(
        self, chat_id: str, message_id: str
    ) -> list[MediaRecord]:
        if self._conn is None or not message_id:
            return []
        async with self._lock:
            rows = self._conn.execute(
                "SELECT o.*, m.message_id, m.chat_id, m.sender_id, m.resource_type, "
                "m.created_at AS message_created_at FROM media_objects o "
                "JOIN media_messages m ON m.media_id=o.media_id "
                "WHERE m.chat_id=? AND m.message_id=? "
                "ORDER BY m.position ASC",
                (chat_id, message_id),
            ).fetchall()
        return [
            self._record_from_row(row)
            for row in rows
            if Path(row["local_path"]).is_file()
        ]

    @staticmethod
    def _resource_type_from_row(row) -> str:
        return row["resource_type"] or row["mime_type"].split("/")[0]

    @staticmethod
    def _record_from_row(row) -> MediaRecord:
        return MediaRecord(
            row["media_id"],
            f"media://inbound/{row['media_id']}",
            row["chat_id"],
            row["message_id"],
            row["sender_id"],
            MediaStore._resource_type_from_row(row),
            row["mime_type"],
            row["size"],
            row["sha256"],
            Path(row["local_path"]),
            row["message_created_at"],
            row["expires_at"],
            row["filename"],
            row["summary"],
            row["summary_model"],
            row["summary_version"],
            row["file_summary"],
            row["file_summary_model"],
            row["file_summary_version"],
        )

    async def recent(
        self, chat_id: str, window_seconds: int, limit: int
    ) -> list[MediaRecord]:
        if self._conn is None:
            return []
        cutoff = time.time() - max(0, window_seconds)
        async with self._lock:
            rows = self._conn.execute(
                "SELECT o.*, m.message_id, m.chat_id, m.sender_id, m.resource_type, "
                "m.created_at AS message_created_at "
                "FROM media_objects o JOIN media_messages m ON m.media_id=o.media_id "
                "WHERE m.chat_id=? AND m.created_at>=? GROUP BY o.media_id "
                "ORDER BY MAX(m.created_at) DESC LIMIT ?",
                (chat_id, cutoff, max(0, limit)),
            ).fetchall()
        return [
            MediaRecord(
                r["media_id"],
                f"media://inbound/{r['media_id']}",
                r["chat_id"],
                r["message_id"],
                r["sender_id"],
                self._resource_type_from_row(r),
                r["mime_type"],
                r["size"],
                r["sha256"],
                Path(r["local_path"]),
                r["message_created_at"],
                r["expires_at"],
                r["filename"],
                r["summary"],
                r["summary_model"],
                r["summary_version"],
                r["file_summary"],
                r["file_summary_model"],
                r["file_summary_version"],
            )
            for r in rows
        ]

    async def update_summary(
        self, media_id: str, summary: str, model: str = "", version: str = ""
    ) -> None:
        if self._conn is None:
            return
        async with self._lock:
            self._conn.execute(
                "UPDATE media_objects SET summary=?, summary_model=?, summary_version=? WHERE media_id=?",
                (summary, model, version, media_id),
            )
            self._conn.commit()

    async def update_file_summary(
        self, media_id: str, summary: str, model: str = "", version: str = ""
    ) -> None:
        if self._conn is None:
            return
        async with self._lock:
            self._conn.execute(
                "UPDATE media_objects SET file_summary=?, file_summary_model=?, file_summary_version=? WHERE media_id=?",
                (summary, model, version, media_id),
            )
            self._conn.commit()

    async def get_transcript(self, media_id: str) -> tuple[str, str] | None:
        if self._conn is None:
            return None
        async with self._lock:
            row = self._conn.execute(
                "SELECT transcript, model FROM media_transcripts WHERE media_id=?",
                (media_id,),
            ).fetchone()
        return (row["transcript"], row["model"]) if row else None

    async def update_transcript(
        self, media_id: str, transcript: str, model: str = ""
    ) -> None:
        if self._conn is None:
            return
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO media_transcripts(media_id,transcript,model,updated_at) VALUES(?,?,?,?)",
                (media_id, transcript, model, time.time()),
            )
            self._conn.commit()

    async def usage(self) -> tuple[int, int]:
        if self._conn is None:
            return 0, 0
        async with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size), 0), COUNT(*) FROM media_objects"
            ).fetchone()
        return int(row[0]), int(row[1])

    async def list_objects(
        self, descending: bool = True, limit: int = 200
    ) -> list[dict]:
        if self._conn is None:
            return []
        order = "DESC" if descending else "ASC"
        async with self._lock:
            rows = self._conn.execute(
                f"SELECT media_id,mime_type,size,created_at,filename,summary,summary_model,local_path FROM media_objects ORDER BY created_at {order} LIMIT ?",
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return [self._web_record_from_row(row) for row in rows]

    async def get_object(self, media_id: str) -> dict | None:
        if self._conn is None or not media_id or "/" in media_id:
            return None
        async with self._lock:
            row = self._conn.execute(
                "SELECT o.*, MIN(m.chat_id) AS source_chat_id, MIN(m.message_id) AS source_message_id, "
                "MIN(m.sender_id) AS source_sender_id, GROUP_CONCAT(m.chat_id || ':' || m.message_id || ':' || m.sender_id) AS references_ "
                "FROM media_objects o LEFT JOIN media_messages m ON m.media_id=o.media_id "
                "WHERE o.media_id=? GROUP BY o.media_id",
                (media_id,),
            ).fetchone()
        return self._web_record_from_row(row) if row else None

    async def get_content_info(self, media_id: str) -> tuple[Path, dict] | None:
        if self._conn is None or not media_id or "/" in media_id:
            return None
        async with self._lock:
            row = self._conn.execute(
                "SELECT * FROM media_objects WHERE media_id=?", (media_id,)
            ).fetchone()
        if not row:
            return None
        path = Path(row["local_path"])
        try:
            path = path.resolve(strict=True)
            if not path.is_relative_to(self.inbound.resolve()):
                return None
        except (OSError, RuntimeError):
            return None
        return path, dict(row)

    @staticmethod
    def _web_record_from_row(row) -> dict:
        item = dict(row)
        path = Path(item.pop("local_path"))
        item["storage_status"] = "ready" if path.is_file() else "missing"
        item["references"] = [
            value for value in (item.pop("references_", "") or "").split(",") if value
        ]
        item["references_count"] = len(item["references"])
        item["has_summary"] = bool(item.get("summary"))
        item.setdefault("source_chat_id", "")
        item.setdefault("source_message_id", "")
        item.setdefault("source_sender_id", "")
        return item

    async def delete_media(self, media_id: str) -> bool:
        if self._conn is None:
            return False
        async with self._lock:
            row = self._conn.execute(
                "SELECT local_path FROM media_objects WHERE media_id=?", (media_id,)
            ).fetchone()
            if not row:
                return False
            try:
                Path(row["local_path"]).unlink(missing_ok=True)
            except OSError:
                return False
            self._conn.execute(
                "DELETE FROM media_messages WHERE media_id=?", (media_id,)
            )
            self._conn.execute(
                "DELETE FROM media_transcripts WHERE media_id=?", (media_id,)
            )
            self._conn.execute(
                "DELETE FROM media_objects WHERE media_id=?", (media_id,)
            )
            self._conn.commit()
            return True

    async def cleanup(self, max_total_bytes: int = 2_147_483_648) -> int:
        """Remove unreferenced objects; live media is never evicted automatically."""
        if self._conn is None:
            return 0
        async with self._lock:
            return self._cleanup_sync(max_total_bytes)

    async def clear_all(self) -> int:
        if self._conn is None:
            return 0
        async with self._lock:
            rows = self._conn.execute("SELECT local_path FROM media_objects").fetchall()
            removed_ids = []
            for row in rows:
                try:
                    Path(row["local_path"]).unlink(missing_ok=True)
                except OSError:
                    continue
                removed_ids.append(row["local_path"])
            if removed_ids:
                placeholders = ",".join("?" for _ in removed_ids)
                self._conn.execute(
                    f"DELETE FROM media_messages WHERE media_id IN (SELECT media_id FROM media_objects WHERE local_path IN ({placeholders}))",
                    removed_ids,
                )
                self._conn.execute(
                    f"DELETE FROM media_transcripts WHERE media_id IN (SELECT media_id FROM media_objects WHERE local_path IN ({placeholders}))",
                    removed_ids,
                )
                self._conn.execute(
                    f"DELETE FROM media_objects WHERE local_path IN ({placeholders})",
                    removed_ids,
                )
            self._conn.commit()
            return len(removed_ids)

    def _cleanup_sync(self, max_total_bytes):
        conn = self._conn
        removed = 0
        rows = conn.execute(
            "SELECT media_id,local_path,size FROM media_objects WHERE media_id NOT IN (SELECT media_id FROM media_messages)",
        ).fetchall()
        for row in rows:
            try:
                Path(row["local_path"]).unlink(missing_ok=True)
            except OSError:
                continue
            conn.execute(
                "DELETE FROM media_objects WHERE media_id=?", (row["media_id"],)
            )
            removed += 1
        conn.commit()
        return removed
