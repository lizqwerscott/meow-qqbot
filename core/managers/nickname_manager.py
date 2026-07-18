import json
import logging
import os
import time
from typing import Dict, Iterator, List, Tuple

_log = logging.getLogger(__name__)


class NicknameManager:
    """
    统一昵称管理器。

    负责：
    - 昵称数据的手动/自动加载与持久化
    - 按优先级查询（手动 > 自动 > 原始 ID）
    - 运行时采集用户昵称（实时持久化）
    - 同一用户的多别名历史管理

    数据格式：
    - manual: {id: name}
    - auto:   {id: {aliases: [name1, name2, ...], updated_at: timestamp}}
    - 自动加载时兼容旧格式 {id: name} → 自动迁移
    """

    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        self.nicknames: Dict[str, str] = {}
        self.auto_nicknames: Dict[str, dict] = {}
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
        """返回用户最新昵称，兜底 user_id。"""
        if user_id in self.nicknames:
            return self.nicknames[user_id]
        entry = self.auto_nicknames.get(user_id)
        if entry and entry.get("aliases"):
            return entry["aliases"][-1]
        return user_id

    def get_aliases(self, user_id: str) -> List[str]:
        """返回用户全部别名（手动 + 自动），保序去重。"""
        names: List[str] = []
        if user_id in self.nicknames:
            names.append(self.nicknames[user_id])
        entry = self.auto_nicknames.get(user_id)
        if entry:
            for a in entry.get("aliases", []):
                if a not in names:
                    names.append(a)
        return names

    def iter_users(self) -> Iterator[Tuple[str, List[str]]]:
        """迭代所有已知用户 (user_id, aliases)，不包含 bot 自身。"""
        seen = set()
        for uid in self.nicknames:
            if uid != self.bot_id:
                seen.add(uid)
                yield uid, self.get_aliases(uid)
        for uid in self.auto_nicknames:
            if uid != self.bot_id and uid not in seen:
                yield uid, self.get_aliases(uid)

    def all_merged(self) -> Dict[str, str]:
        """返回 {id: 最新昵称} — 保持向下兼容。"""
        merged = dict(self.nicknames)
        for uid, entry in self.auto_nicknames.items():
            if uid not in merged:
                aliases = entry.get("aliases", [])
                merged[uid] = aliases[-1] if aliases else uid
        return merged

    def collect(self, user_id: str, username: str) -> None:
        if not user_id or not username:
            return
        if user_id == self.bot_id:
            return
        if user_id in self.nicknames:
            return
        entry = self.auto_nicknames.get(user_id)
        aliases = entry["aliases"] if entry else []
        if username not in aliases:
            aliases.append(username)
            self.auto_nicknames[user_id] = {"aliases": aliases, "updated_at": time.time()}
            _log.debug(f"已采集昵称: {username} ({user_id[:12]}..)")
            self._save_auto_nicknames()
        elif aliases and aliases[-1] != username:
            aliases.remove(username)
            aliases.append(username)
            self.auto_nicknames[user_id] = {"aliases": aliases, "updated_at": time.time()}
            self._save_auto_nicknames()

    # ── 内部文件操作 ──

    def _load_nicknames(self) -> Dict[str, str]:
        nicknames_file = "config/nicknames.json"
        if os.path.exists(nicknames_file):
            try:
                with open(nicknames_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _log.error(f"加载昵称文件失败: {e}")
                return {}
        _log.warning(f"昵称文件 {nicknames_file} 不存在")
        return {}

    def _load_auto_nicknames(self) -> Dict[str, dict]:
        path = "data/nicknames.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result: Dict[str, dict] = {}
                for uid, val in data.items():
                    if isinstance(val, dict):
                        result[uid] = val
                    else:
                        result[uid] = {"aliases": [str(val)], "updated_at": 0}
                return result
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


