"""俚语存储 — JargonEntry 数据模型 + JargonStore 持久化。"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from core.learners.stores.base_store import JsonStore


@dataclass
class JargonEntry:
    term: str
    definition: str
    examples: List[str] = field(default_factory=list)
    origin_sessions: List[str] = field(default_factory=list)
    group_variants: Dict[str, str] = field(default_factory=dict)
    inference_level: int = 0
    frequency: int = 0
    source: str = "auto"  # "manual" | "auto"
    added_by: str = ""
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "JargonEntry":
        return JargonEntry(
            term=d.get("term", ""),
            definition=d.get("definition", ""),
            examples=d.get("examples", []),
            origin_sessions=d.get("origin_sessions", []),
            group_variants=d.get("group_variants", {}),
            inference_level=d.get("inference_level", 0),
            frequency=d.get("frequency", 0),
            source=d.get("source", "auto"),
            added_by=d.get("added_by", ""),
            first_seen_at=d.get("first_seen_at", 0.0),
            last_seen_at=d.get("last_seen_at", 0.0),
        )


_DEFAULT = {
    "version": 1,
    "items": {},
}


class JargonStore:
    """俚语持久化存储。

    key = term（俚语词汇本身，小写）。
    """

    def __init__(self, path: str = "data/learners/jargons.json"):
        self._store = JsonStore(path, default=_DEFAULT)

    # ── 查询 ──

    def get(self, term: str) -> Optional[JargonEntry]:
        d = self._store.get(term.lower().strip())
        return JargonEntry.from_dict(d) if d else None

    def get_all_entries(self) -> List[JargonEntry]:
        return [JargonEntry.from_dict(d) for d in self._store.get_all()]

    def count(self) -> int:
        return self._store.count()

    def search(self, query: str) -> List[JargonEntry]:
        """按 term 或 definition 模糊搜索。"""
        q = query.lower().strip()
        results = []
        for d in self._store.get_all():
            if q in d.get("term", "").lower() or q in d.get("definition", "").lower():
                results.append(JargonEntry.from_dict(d))
        return results

    def get_active_for_chat(self, chat_id: str, min_level: int = 1) -> List[JargonEntry]:
        """获取某群活跃的俚语（level >= min_level）。

        - 手动添加的词条（source=manual）全局可见
        - 自动挖掘的词条仅在其出现过的群可见
        """
        results = []
        for d in self._store.get_all():
            entry = JargonEntry.from_dict(d)
            if entry.inference_level >= min_level:
                if entry.source == "manual":
                    results.append(entry)
                elif chat_id in entry.origin_sessions or chat_id in entry.group_variants:
                    results.append(entry)
        return results

    def get_level3_entries(self) -> List[JargonEntry]:
        return [
            JargonEntry.from_dict(d)
            for d in self._store.get_all()
            if d.get("inference_level", 0) >= 3
        ]

    # ── 写入 ──

    async def save(self, entry: JargonEntry) -> None:
        await self._store.save(entry.term.lower().strip(), entry.to_dict())

    async def delete(self, term: str) -> bool:
        return await self._store.delete(term.lower().strip())

    async def update(self, term: str, **kwargs) -> bool:
        return await self._store.update(term.lower().strip(), **kwargs)

    # ── 内部 ──

    def _term_exists(self, term: str) -> bool:
        return self._store.get(term.lower().strip()) is not None
