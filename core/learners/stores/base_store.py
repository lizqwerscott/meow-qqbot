"""JsonStore — 通用 JSON 文件持久化基类。

模式完全复用 EmojiStorage (emoji_manager.py):
- asyncio.Lock 保护写
- 线程池执行同步刷盘
- 临时文件 + 原子替换
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)


class JsonStore:
    """通用 JSON 文件持久化存储。

    Args:
        path: JSON 文件路径。
        default: 初始化时的默认数据结构。
    """

    def __init__(self, path: str, default: Optional[dict] = None):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._default = default or {}
        self._data: dict = self._load()

    # ── 公开接口 ──

    def get(self, key: str) -> Optional[dict]:
        return self._data.get("items", {}).get(key)

    def get_all(self) -> List[dict]:
        return list(self._data.get("items", {}).values())

    def count(self) -> int:
        return len(self._data.get("items", {}))

    def keys(self) -> List[str]:
        return list(self._data.get("items", {}).keys())

    async def save(self, key: str, record: dict) -> None:
        async with self._lock:
            self._data.setdefault("items", {})[key] = record
            await self._flush_async()

    async def update(self, key: str, **kwargs) -> bool:
        async with self._lock:
            items = self._data.setdefault("items", {})
            if key not in items:
                return False
            items[key].update(kwargs)
            items[key]["updated_at"] = time.time()
            await self._flush_async()
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            items = self._data.setdefault("items", {})
            if key not in items:
                return False
            del items[key]
            await self._flush_async()
            return True

    async def clear(self) -> None:
        async with self._lock:
            self._data = dict(self._default)
            self._data.setdefault("items", {})
            await self._flush_async()

    # ── 内部 IO ──

    def _load(self) -> dict:
        if not self._path.exists():
            _log.info(f"创建新数据文件: {self._path}")
            data = dict(self._default)
            data.setdefault("items", {})
            return data
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                _log.warning(f"数据文件格式不正确，重置: {self._path}")
                data = dict(self._default)
                data.setdefault("items", {})
            return data
        except (json.JSONDecodeError, OSError) as e:
            _log.error(f"读取数据文件失败: {e}，备份并重置: {self._path}")
            backup = self._path.with_suffix(".json.bak")
            try:
                if self._path.exists():
                    import shutil

                    shutil.copy2(self._path, backup)
                    _log.info(f"已备份到: {backup}")
            except Exception as e:
                _log.warning(f"备份损坏文件失败 [{self._path}]: {e}")
            data = dict(self._default)
            data.setdefault("items", {})
            return data

    def _flush_sync(self) -> None:
        try:
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
        except OSError as e:
            _log.error(f"写入数据文件失败: {e}")

    async def _flush_async(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._flush_sync)
