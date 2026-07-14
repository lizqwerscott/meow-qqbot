import json
import logging
import os
from typing import Dict

_log = logging.getLogger(__name__)


class NicknameManager:
    """
    统一昵称管理器。

    负责：
    - 昵称数据的手动/自动加载与持久化
    - 按优先级查询（手动 > 自动 > 原始 ID）
    - 运行时采集用户昵称（实时持久化）

    所有 nickname 相关的数据读取/写入都通过此类的实例进行，
    各模块通过持有同一实例引用来共享数据。
    """

    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        self.nicknames: Dict[str, str] = {}
        self.auto_nicknames: Dict[str, str] = {}
        self._initial_loaded = False

    # ── 生命周期 ──

    def load_all(self) -> None:
        if self._initial_loaded:
            _log.debug("已加载过昵称，跳过重复加载")
            return
        self.nicknames = self._load_nicknames()
        self.auto_nicknames = self._load_auto_nicknames()
        self._initial_loaded = True
        _log.info(
            f"已加载 {len(self.nicknames)} 个手动昵称 + "
            f"{len(self.auto_nicknames)} 个自动昵称"
        )

    async def flush_save(self) -> None:
        pass

    def save_auto(self) -> None:
        self._save_auto_nicknames()

    # ── 核心 API ──

    def get(self, user_id: str) -> str:
        if user_id in self.nicknames:
            return self.nicknames[user_id]
        if user_id in self.auto_nicknames:
            return self.auto_nicknames[user_id]
        return user_id

    def all_merged(self) -> Dict[str, str]:
        merged = dict(self.nicknames)
        for uid, name in self.auto_nicknames.items():
            if uid not in merged:
                merged[uid] = name
        return merged

    def collect(self, user_id: str, username: str) -> None:
        if not user_id or not username:
            return
        if user_id == self.bot_id:
            return
        if user_id in self.nicknames:
            return
        if self.auto_nicknames.get(user_id) == username:
            return
        self.auto_nicknames[user_id] = username
        _log.debug(f"已采集昵称: {username} ({user_id[:12]}..)")
        self._save_auto_nicknames()

    # ── 内部文件操作 ──

    def _load_nicknames(self) -> Dict[str, str]:
        nicknames_file = "nicknames.json"
        if os.path.exists(nicknames_file):
            try:
                with open(nicknames_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log.error(f"加载昵称文件失败: {e}")
                return {}
        _log.warning(f"昵称文件 {nicknames_file} 不存在")
        return {}

    def _load_auto_nicknames(self) -> Dict[str, str]:
        path = "data/nicknames.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log.error(f"加载自动昵称文件失败: {e}")
        return {}

    def _save_auto_nicknames(self) -> None:
        path = "data/nicknames.json"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.auto_nicknames, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _log.error(f"保存自动昵称失败: {e}")


