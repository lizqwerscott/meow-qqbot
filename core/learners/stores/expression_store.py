"""表情映射存储 — ExpressionMapping 数据模型 + ExpressionStore 持久化。"""

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from core.learners.stores.base_store import JsonStore


@dataclass
class ExpressionMapping:
    trigger_keywords: List[str] = field(default_factory=list)
    expression_hash: str = ""
    context_tags: List[str] = field(default_factory=list)
    review_status: str = "pending"  # pending / approved / rejected / rescue
    frequency: int = 0
    confidence_weight: float = 0.5
    source_rooms: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ExpressionMapping":
        return ExpressionMapping(
            trigger_keywords=d.get("trigger_keywords", []),
            expression_hash=d.get("expression_hash", ""),
            context_tags=d.get("context_tags", []),
            review_status=d.get("review_status", "pending"),
            frequency=d.get("frequency", 0),
            confidence_weight=d.get("confidence_weight", 0.5),
            source_rooms=d.get("source_rooms", []),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


_DEFAULT = {
    "version": 1,
    "items": {},
}


class ExpressionStore:
    """表情映射持久化存储。key = expression_hash。"""

    def __init__(self, path: str = "data/learners/expressions.json"):
        self._store = JsonStore(path, default=_DEFAULT)

    # ── 查询 ──

    def get(self, key: str) -> Optional[ExpressionMapping]:
        d = self._store.get(key)
        return ExpressionMapping.from_dict(d) if d else None

    def get_all_mappings(self) -> List[ExpressionMapping]:
        return [ExpressionMapping.from_dict(d) for d in self._store.get_all()]

    def get_all(self) -> List[dict]:
        return self._store.get_all()

    def keys(self) -> List[str]:
        return self._store.keys()

    async def delete(self, key: str) -> bool:
        return await self._store.delete(key)

    def count(self) -> int:
        return self._store.count()

    def get_approved(self) -> List[ExpressionMapping]:
        return [
            ExpressionMapping.from_dict(d)
            for d in self._store.get_all()
            if d.get("review_status") == "approved"
        ]

    def get_pending(self) -> List[ExpressionMapping]:
        return [
            ExpressionMapping.from_dict(d)
            for d in self._store.get_all()
            if d.get("review_status") == "pending"
        ]

    def find_by_keyword(self, keyword: str) -> List[ExpressionMapping]:
        """查找触发词包含 keyword 的已审核映射。"""
        kw = keyword.lower().strip()
        results = []
        for d in self._store.get_all():
            if d.get("review_status") != "approved":
                continue
            triggers = [t.lower().strip() for t in d.get("trigger_keywords", [])]
            if kw in triggers or any(kw in t for t in triggers):
                results.append(ExpressionMapping.from_dict(d))
        return results

    # ── 写入 ──

    async def save(self, mapping: ExpressionMapping) -> None:
        key = mapping.expression_hash
        await self._store.save(key, mapping.to_dict())

    async def update_status(self, expression_hash: str, status: str) -> bool:
        return await self._store.update(
            expression_hash, review_status=status, updated_at=time.time()
        )

    async def update_weight(self, expression_hash: str, weight: float) -> bool:
        return await self._store.update(
            expression_hash, confidence_weight=weight, updated_at=time.time()
        )
