import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


class ContextStore(ABC):

    @abstractmethod
    def flush(self, chat_id: str, messages: List[dict]) -> None:
        """持久化 chat_id 的消息列表。实现可使用增量追加优化。"""

    @abstractmethod
    def load(self, chat_id: str) -> Optional[List[dict]]:
        """加载已持久化的消息。从未保存过的返回 None。"""

    @abstractmethod
    def delete(self, chat_id: str) -> None:
        """删除 chat_id 的所有持久化数据。"""

    @abstractmethod
    def archive(self, chat_id: str, archive_ts: str) -> Optional[str]:
        """归档当前数据。返回归档标识符（如文件路径），无可归档数据则返回 None。"""

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
        self._chat_types_path = self._base_dir / "chat_types.json"
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
                        json.dumps(msg, ensure_ascii=False) + "\n"
                        for msg in new_msgs
                    ]
                    with open(path, "a", encoding="utf-8") as f:
                        f.writelines(lines)

            with self._lock:
                self._flushed[chat_id] = len(messages)

    def _write_full(self, path: Path, messages: List[dict]) -> None:
        lines = [
            json.dumps(msg, ensure_ascii=False) + "\n" for msg in messages
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def load(self, chat_id: str) -> Optional[List[dict]]:
        with self._acquire_file_lock(chat_id):
            path = self._get_path(chat_id)
            old_path = self._get_old_path(chat_id)

            if path and path.exists():
                try:
                    data = self._load_jsonl(path)
                except Exception as e:
                    _log.warning(
                        "加载 JSONL 缓存失败 [%s..]: %s", chat_id[:12], e
                    )
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
                            chat_id[:12], len(data),
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
                        path.stem[:12], line[:80],
                    )
        return data

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
            archive_path = path.parent / f"{path.name}.archived.{archive_ts}"
            if path.exists():
                path.rename(archive_path)
                with self._lock:
                    self._flushed.pop(chat_id, None)
                _log.info(
                    "已归档 [%s..] → %s", chat_id[:12], archive_path.name,
                )
                return str(archive_path)
            return None

    # ── 聊天类型 ──

    def _load_chat_types(self) -> None:
        if not self._chat_types_path.exists():
            old_path = self._base_dir.parent / "chat_types.json"
            if old_path.exists():
                try:
                    old_data = json.loads(old_path.read_text())
                    self._chat_types = {k: bool(v) for k, v in old_data.items()}
                    self._chat_types_path.write_text(
                        json.dumps(self._chat_types, ensure_ascii=False)
                    )
                    old_path.unlink(missing_ok=True)
                    _log.info(
                        "已迁移 chat_types.json → data/cache/ (%d 条)",
                        len(self._chat_types),
                    )
                except Exception as e:
                    _log.warning("chat_types.json 迁移失败: %s", e)
            return

        try:
            data = json.loads(self._chat_types_path.read_text())
            self._chat_types = {k: bool(v) for k, v in data.items()}
            _log.info("已加载 %d 个聊天类型记录", len(self._chat_types))
        except Exception:
            _log.warning("chat_types.json 加载失败，使用空映射")

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
        path = Path(file_path)
        if not path.exists():
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
