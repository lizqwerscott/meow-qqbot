import asyncio
import functools
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


def deduplicate_history(messages: List[dict]) -> List[dict]:
    """Remove only logically identical persisted messages.

    User messages use their platform message ID and tool results use their tool
    call ID. Records without a stable identity are intentionally preserved,
    including repeated assistant replies.
    """
    seen: set[tuple[str, str]] = set()
    result: List[dict] = []
    for message in messages:
        key = _message_identity(message)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        result.append(message)
    return result


def _message_identity(message: Any) -> Optional[tuple[str, str]]:
    if not isinstance(message, dict):
        return None
    role = str(message.get("role") or "")
    if role == "user" and message.get("message_id"):
        return ("user", str(message["message_id"]))
    if role == "tool" and message.get("tool_call_id"):
        return ("tool", str(message["tool_call_id"]))
    return None


class ContextStore(ABC):

    @abstractmethod
    def flush(self, chat_id: str, messages: List[dict]) -> None:
        """持久化 chat_id 的消息列表。实现可使用增量追加优化。"""

    @abstractmethod
    def load(self, chat_id: str) -> Optional[List[dict]]:
        """加载已持久化的消息。从未保存过的返回 None。"""

    async def load_async(self, chat_id: str) -> Optional[List[dict]]:
        """异步加载已持久化的消息。默认委托给同步 load()，子类可覆盖。"""
        return await asyncio.to_thread(
            functools.partial(self.load, chat_id),
        )

    @abstractmethod
    def delete(self, chat_id: str) -> None:
        """删除 chat_id 的所有持久化数据。"""

    @abstractmethod
    def archive(self, chat_id: str, archive_ts: str) -> Optional[str]:
        """归档当前数据。返回归档标识符（如文件路径），无可归档数据则返回 None。"""

    @abstractmethod
    def archive_messages(
        self, chat_id: str, archive_ts: str, messages: List[dict]
    ) -> Optional[str]:
        """仅归档指定消息，活跃数据由调用方另行刷新。"""

    # ── 聊天类型元数据 ──

    @abstractmethod
    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        """返回群聊（True）/私聊（False）/未知（None）。"""

    @abstractmethod
    async def set_chat_type(self, chat_id: str, is_group: bool) -> None:
        """记录 chat_id 的聊天类型。"""

    # ── 归档查询 ──

    @abstractmethod
    def list_archives(self, chat_id: str) -> List[dict]:
        """返回 chat_id 的所有归档文件信息列表。"""

    @abstractmethod
    def read_archive(self, file_path: str, max_messages: int = 200) -> List[dict]:
        """读取归档文件中的消息。"""

    @abstractmethod
    def get_all_disk_ids(self) -> List[str]:
        """返回持久化层发现的所有 chat_id（含磁盘但未必在内存中）。"""

    @abstractmethod
    def get_archived_summary(self) -> Dict[str, int]:
        """返回 {chat_id: 归档文件数} 的摘要。"""

    @abstractmethod
    def cleanup_stale_archives(self, retention_seconds: float) -> int:
        """清理超过 retention_seconds 的归档文件，返回删除的文件数。"""

    def release_file_lock(self, chat_id: str) -> None:
        """释放 chat_id 的文件级锁（默认无操作，JSONLContextStore 覆盖）。"""


class JSONLContextStore(ContextStore):

    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._lock = threading.Lock()
        self._flushed: Dict[str, int] = {}

        self._file_locks: Dict[str, threading.Lock] = {}
        self._file_lock_guard = threading.Lock()

        self._chat_types: Dict[str, bool] = {}
        self._chat_types_lock = asyncio.Lock()
        self._chat_types_path = self._base_dir / "meta" / "chat_types.json"
        self._load_chat_types()

    # ── 内部工具 ──

    def _get_path(self, chat_id: str) -> Path:
        return self._base_dir / f"{chat_id}.jsonl"

    def _get_old_path(self, chat_id: str) -> Path:
        return self._base_dir / f"{chat_id}.json"

    def _acquire_file_lock(self, chat_id: str) -> threading.Lock:
        with self._file_lock_guard:
            if chat_id not in self._file_locks:
                self._file_locks[chat_id] = threading.Lock()
            return self._file_locks[chat_id]

    def release_file_lock(self, chat_id: str) -> None:
        with self._file_lock_guard:
            self._file_locks.pop(chat_id, None)

    def _archive_path(self, chat_id: str, archive_ts: str) -> Path:
        """在 chat 文件锁内为 archive 分配不覆盖既有文件的路径。"""
        path = self._get_path(chat_id)
        base = path.parent / f"{path.name}.archived.{archive_ts}"
        archive_path = base
        suffix = 1
        while archive_path.exists():
            archive_path = base.with_name(f"{base.name}.{suffix}")
            suffix += 1
        return archive_path

    # ── 消息持久化 ──

    def flush(self, chat_id: str, messages: List[dict]) -> None:
        with self._acquire_file_lock(chat_id):
            path = self._get_path(chat_id)
            if not messages:
                return

            with self._lock:
                flushed = self._flushed.get(chat_id, 0)
            path.parent.mkdir(parents=True, exist_ok=True)

            if len(messages) < flushed:
                self._write_full(path, messages)
            else:
                new_msgs = messages[flushed:]
                if new_msgs:
                    lines = [
                        json.dumps(msg, ensure_ascii=False) + "\n" for msg in new_msgs
                    ]
                    with open(path, "a", encoding="utf-8") as f:
                        f.writelines(lines)

            with self._lock:
                self._flushed[chat_id] = len(messages)

    def _write_full(self, path: Path, messages: List[dict]) -> None:
        lines = [json.dumps(msg, ensure_ascii=False) + "\n" for msg in messages]
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load(self, chat_id: str) -> Optional[List[dict]]:
        with self._acquire_file_lock(chat_id):
            path = self._get_path(chat_id)
            old_path = self._get_old_path(chat_id)

            if path and path.exists():
                try:
                    data = self._load_jsonl(path)
                except Exception as e:
                    _log.warning("加载 JSONL 缓存失败 [%s..]: %s", chat_id[:12], e)
                    return None
            elif old_path and old_path.exists():
                try:
                    data = json.loads(old_path.read_text(encoding="utf-8"))
                    if data:
                        self._write_full(path, data)
                        with self._lock:
                            self._flushed[chat_id] = len(data)
                        old_path.unlink(missing_ok=True)
                        _log.info(
                            "已迁移旧 JSON 缓存 → JSONL [%s..] (%d 条)",
                            chat_id[:12],
                            len(data),
                        )
                    return data or None
                except Exception as e:
                    _log.warning(
                        "加载/迁移旧 JSON 缓存失败 [%s..]: %s", chat_id[:12], e
                    )
                    return None
            else:
                return None

            if not data:
                return None

            with self._lock:
                self._flushed[chat_id] = len(data)
            return data

    def _load_jsonl(self, path: Path) -> List[dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    _log.warning(
                        "跳过损坏的 JSONL 行 [%s..]: %s",
                        path.stem[:12],
                        line[:80],
                    )
        return deduplicate_history(data)

    def repair_duplicates(self, chat_id: str) -> int:
        """Rewrite one legacy JSONL file without logical duplicates."""
        with self._acquire_file_lock(chat_id):
            path = self._get_path(chat_id)
            if not path.exists():
                return 0
            raw = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            repaired = deduplicate_history(raw)
            removed = len(raw) - len(repaired)
            if removed:
                self._write_full(path, repaired)
            with self._lock:
                self._flushed[chat_id] = len(repaired)
            return removed

    def delete(self, chat_id: str) -> None:
        with self._acquire_file_lock(chat_id):
            with self._lock:
                self._flushed.pop(chat_id, None)
            path = self._get_path(chat_id)
            if path:
                path.unlink(missing_ok=True)

    def archive(self, chat_id: str, archive_ts: str) -> Optional[str]:
        with self._acquire_file_lock(chat_id):
            path = self._get_path(chat_id)
            if path.exists():
                archive_path = self._archive_path(chat_id, archive_ts)
                path.rename(archive_path)
                with self._lock:
                    self._flushed.pop(chat_id, None)
                _log.info(
                    "已归档 [%s..] → %s",
                    chat_id[:12],
                    archive_path.name,
                )
                return str(archive_path)
            return None

    def archive_messages(
        self, chat_id: str, archive_ts: str, messages: List[dict]
    ) -> Optional[str]:
        if not messages:
            return None
        with self._acquire_file_lock(chat_id):
            archive_path = self._archive_path(chat_id, archive_ts)
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_full(archive_path, messages)
            _log.info(
                "已归档 %d 条消息 [%s..] → %s",
                len(messages),
                chat_id[:12],
                archive_path.name,
            )
            return str(archive_path)

    # ── 聊天类型 ──

    def _load_chat_types(self) -> None:
        if self._chat_types_path.exists():
            try:
                data = json.loads(self._chat_types_path.read_text())
                self._chat_types = {k: bool(v) for k, v in data.items()}
                _log.info("已加载 %d 个聊天类型记录 (meta/)", len(self._chat_types))
            except Exception:
                _log.warning("meta/chat_types.json 加载失败，使用空映射")
            return

        self._migrate_chat_types()
        if self._chat_types:
            return

    def _migrate_chat_types(self) -> None:
        candidates = [
            ("data/chat_types.json", self._base_dir.parent / "chat_types.json"),
            ("data/sessions/chat_types.json", self._base_dir / "chat_types.json"),
        ]
        for label, path in candidates:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    parsed = {k: bool(v) for k, v in data.items()}
                    if parsed:
                        self._chat_types_path.parent.mkdir(parents=True, exist_ok=True)
                        self._chat_types_path.write_text(
                            json.dumps(parsed, ensure_ascii=False)
                        )
                        path.unlink(missing_ok=True)
                        self._chat_types = parsed
                        _log.info("已迁移 %s → meta/ (%d 条)", label, len(parsed))
                        return
                except Exception as e:
                    _log.warning("%s 加载/迁移失败: %s", label, e)
                    continue

    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        return self._chat_types.get(chat_id)

    async def set_chat_type(self, chat_id: str, is_group: bool) -> None:
        async with self._chat_types_lock:
            if self._chat_types.get(chat_id) == is_group:
                return
            self._chat_types[chat_id] = is_group
            await self._save_chat_types()

    async def _save_chat_types(self) -> None:
        try:
            self._chat_types_path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(self._chat_types, ensure_ascii=False)
            await asyncio.to_thread(self._chat_types_path.write_text, data)
        except Exception as e:
            _log.warning("保存聊天类型失败: %s", e)

    # ── 归档查询 ──

    def list_archives(self, chat_id: str) -> List[dict]:
        files = []
        if not self._base_dir.is_dir():
            return files
        for f in self._base_dir.iterdir():
            if f.name.startswith(f"{chat_id}.jsonl.archived."):
                ts_str = f.name.split(".jsonl.archived.")[-1]
                files.append(
                    {
                        "path": str(f),
                        "timestamp_str": ts_str,
                        "size": f.stat().st_size,
                        "mtime": f.stat().st_mtime,
                    }
                )
        files.sort(key=lambda x: x["mtime"], reverse=True)
        return files

    def read_archive(self, file_path: str, max_messages: int = 200) -> List[dict]:
        path = Path(file_path).resolve()
        if not path.exists():
            return []
        try:
            path.relative_to(self._base_dir.resolve())
        except ValueError:
            _log.warning("拒绝越界读取归档: %s", file_path)
            return []
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    pass
        if max_messages and len(data) > max_messages:
            data = data[-max_messages:]
        return data

    def get_all_disk_ids(self) -> List[str]:
        disk_ids: set[str] = set()
        if self._base_dir.is_dir():
            for f in self._base_dir.glob("*.jsonl"):
                disk_ids.add(f.stem)
            for f in self._base_dir.glob("*.json"):
                disk_ids.add(f.stem)
        return sorted(disk_ids)

    def get_archived_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not self._base_dir.is_dir():
            return counts
        for f in self._base_dir.iterdir():
            if ".archived." in f.name:
                chat_id = f.name.split(".jsonl.archived.")[0]
                counts[chat_id] = counts.get(chat_id, 0) + 1
        return counts

    def cleanup_stale_archives(self, retention_seconds: float) -> int:
        cutoff = time.time() - retention_seconds
        removed = 0
        if not self._base_dir.is_dir():
            return removed
        for f in self._base_dir.iterdir():
            if ".archived." in f.name:
                try:
                    mtime = f.stat().st_mtime
                    if mtime < cutoff:
                        f.unlink()
                        removed += 1
                except Exception as e:
                    _log.warning("清理归档文件失败 %s: %s", f.name, e)
        if removed:
            _log.info("归档清理完成: 移除了 %d 个文件", removed)
        return removed


class SQLiteContextStore(ContextStore):
    """Durable active context store with JSONL kept only for archives."""

    def __init__(self, db_path: str, archive_dir: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._archive_store = JSONLContextStore(archive_dir)
        self._lock = threading.RLock()
        self._chat_types_lock = asyncio.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS context_messages (
                chat_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                message_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (chat_id, seq),
                UNIQUE (chat_id, message_key)
            )
            """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS context_chat_types (
                chat_id TEXT PRIMARY KEY,
                is_group INTEGER NOT NULL
            )
            """)
        self._conn.commit()

    @staticmethod
    def _storage_key(message: dict, index: int) -> str:
        identity = _message_identity(message)
        if identity is not None:
            kind, value = identity
            return f"{kind}:{value}"
        payload = json.dumps(
            message, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return f"record:{index}:{digest}"

    def flush(self, chat_id: str, messages: List[dict]) -> None:
        if not chat_id:
            return
        messages = deduplicate_history(messages)
        rows = [
            (
                chat_id,
                index,
                self._storage_key(message, index),
                json.dumps(message, ensure_ascii=False, sort_keys=True),
            )
            for index, message in enumerate(messages)
        ]
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM context_messages WHERE chat_id = ?", (chat_id,)
            )
            self._conn.executemany(
                """
                INSERT INTO context_messages (chat_id, seq, message_key, payload)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def load(self, chat_id: str) -> Optional[List[dict]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT payload FROM context_messages
                 WHERE chat_id = ? ORDER BY seq
                """,
                (chat_id,),
            ).fetchall()
        if rows:
            return [json.loads(row["payload"]) for row in rows]

        self._archive_store.repair_duplicates(chat_id)
        legacy = self._archive_store.load(chat_id)
        if legacy:
            self.flush(chat_id, legacy)
            return legacy
        return None

    async def load_async(self, chat_id: str) -> Optional[List[dict]]:
        return await asyncio.to_thread(self.load, chat_id)

    def migrate_legacy(self) -> dict[str, int]:
        migrated = 0
        removed_duplicates = 0
        for chat_id in self._archive_store.get_all_disk_ids():
            removed_duplicates += self._archive_store.repair_duplicates(chat_id)
            messages = self._archive_store.load(chat_id)
            if messages:
                self.flush(chat_id, messages)
                migrated += 1
        return {
            "sessions": migrated,
            "removed_duplicates": removed_duplicates,
        }

    def delete(self, chat_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM context_messages WHERE chat_id = ?", (chat_id,)
            )
        self._archive_store.delete(chat_id)

    def archive(self, chat_id: str, archive_ts: str) -> Optional[str]:
        messages = self.load(chat_id)
        if not messages:
            return None
        path = self.archive_messages(chat_id, archive_ts, messages)
        self.delete(chat_id)
        return path

    def archive_messages(
        self, chat_id: str, archive_ts: str, messages: List[dict]
    ) -> Optional[str]:
        return self._archive_store.archive_messages(
            chat_id, archive_ts, deduplicate_history(messages)
        )

    def list_archives(self, chat_id: str) -> List[dict]:
        return self._archive_store.list_archives(chat_id)

    def read_archive(self, file_path: str, max_messages: int = 200) -> List[dict]:
        return self._archive_store.read_archive(file_path, max_messages)

    def get_all_disk_ids(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT chat_id FROM context_messages ORDER BY chat_id"
            ).fetchall()
        return list(
            dict.fromkeys(
                [str(row["chat_id"]) for row in rows]
                + self._archive_store.get_all_disk_ids()
            )
        )

    def get_archived_summary(self) -> Dict[str, int]:
        return self._archive_store.get_archived_summary()

    def cleanup_stale_archives(self, retention_seconds: float) -> int:
        return self._archive_store.cleanup_stale_archives(retention_seconds)

    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        with self._lock:
            row = self._conn.execute(
                "SELECT is_group FROM context_chat_types WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        if row is not None:
            return bool(row["is_group"])
        return self._archive_store.get_chat_type(chat_id)

    async def set_chat_type(self, chat_id: str, is_group: bool) -> None:
        async with self._chat_types_lock:
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    INSERT INTO context_chat_types(chat_id, is_group)
                    VALUES (?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET is_group = excluded.is_group
                    """,
                    (chat_id, int(is_group)),
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class MemoryContextStore(ContextStore):

    def __init__(self):
        self._data: Dict[str, List[dict]] = {}
        self._chat_types: Dict[str, bool] = {}
        self._chat_types_lock = asyncio.Lock()

    def flush(self, chat_id: str, messages: List[dict]) -> None:
        if messages:
            self._data[chat_id] = list(messages)

    def load(self, chat_id: str) -> Optional[List[dict]]:
        data = self._data.get(chat_id)
        return list(data) if data is not None else None

    def delete(self, chat_id: str) -> None:
        self._data.pop(chat_id, None)

    def archive(self, chat_id: str, archive_ts: str) -> Optional[str]:
        self._data.pop(chat_id, None)
        return None

    def archive_messages(
        self, chat_id: str, archive_ts: str, messages: List[dict]
    ) -> Optional[str]:
        return None

    def get_chat_type(self, chat_id: str) -> Optional[bool]:
        return self._chat_types.get(chat_id)

    async def set_chat_type(self, chat_id: str, is_group: bool) -> None:
        async with self._chat_types_lock:
            self._chat_types[chat_id] = is_group

    def list_archives(self, chat_id: str) -> List[dict]:
        return []

    def read_archive(self, file_path: str, max_messages: int = 200) -> List[dict]:
        return []

    def get_all_disk_ids(self) -> List[str]:
        return list(self._data.keys())

    def get_archived_summary(self) -> Dict[str, int]:
        return {}

    def cleanup_stale_archives(self, retention_seconds: float) -> int:
        return 0
