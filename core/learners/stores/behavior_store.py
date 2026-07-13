"""行为模式存储 — BehaviorPattern 数据模型 + BehaviorStore 持久化。"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from core.learners.stores.base_store import JsonStore


@dataclass
class BehaviorPattern:
    scene_descriptor: str
    suggested_action: str
    expected_effect: str
    action_keywords: List[str] = field(default_factory=list)
    confidence: float = 0.5
    last_triggered: float = 0.0
    source_rooms: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BehaviorPattern":
        return BehaviorPattern(
            scene_descriptor=d.get("scene_descriptor", ""),
            suggested_action=d.get("suggested_action", ""),
            expected_effect=d.get("expected_effect", ""),
            action_keywords=d.get("action_keywords", []),
            confidence=d.get("confidence", 0.5),
            last_triggered=d.get("last_triggered", 0.0),
            source_rooms=d.get("source_rooms", []),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


_DEFAULT = {
    "version": 1,
    "items": {},
}


class BehaviorStore:
    """行为模式持久化存储。key = scene_descriptor 的 hash。"""

    def __init__(self, path: str = "data/learners/behaviors.json"):
        self._store = JsonStore(path, default=_DEFAULT)

    # ── 查询 ──

    def get(self, key: str) -> Optional[BehaviorPattern]:
        d = self._store.get(key)
        return BehaviorPattern.from_dict(d) if d else None

    def get_all_patterns(self) -> List[BehaviorPattern]:
        return [BehaviorPattern.from_dict(d) for d in self._store.get_all()]

    def get_all(self) -> List[dict]:
        return self._store.get_all()

    def keys(self) -> List[str]:
        return self._store.keys()

    async def delete(self, key: str) -> bool:
        return await self._store.delete(key)

    def count(self) -> int:
        return self._store.count()

    def search_by_scene(self, keywords: List[str]) -> List[BehaviorPattern]:
        """按场景关键词模糊匹配。"""
        results = []
        for d in self._store.get_all():
            pattern = BehaviorPattern.from_dict(d)
            score = 0
            desc = pattern.scene_descriptor.lower()
            for k in keywords:
                if k.lower() in desc:
                    score += 1
                if k.lower() in " ".join(pattern.action_keywords).lower():
                    score += 0.5
            if score > 0:
                results.append((pattern, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results]

    # ── 写入 ──

    async def save(self, pattern: BehaviorPattern) -> None:
        key = pattern.scene_descriptor[:64]
        await self._store.save(key, pattern.to_dict())

    async def delete(self, key: str) -> bool:
        return await self._store.delete(key)

    async def update_confidence(self, key: str, confidence: float) -> bool:
        return await self._store.update(key, confidence=confidence, updated_at=time.time())
