"""场景聚类存储 — TagCluster 数据模型 + SceneStore 持久化。"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from core.learners.stores.base_store import JsonStore


@dataclass
class TagCluster:
    cluster_id: str
    tags: Dict[str, float] = field(default_factory=dict)
    prototype_utterance: str = ""
    member_count: int = 0
    centroid_vector: Optional[List[float]] = None
    last_updated: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TagCluster":
        return TagCluster(
            cluster_id=d.get("cluster_id", ""),
            tags=d.get("tags", {}),
            prototype_utterance=d.get("prototype_utterance", ""),
            member_count=d.get("member_count", 0),
            centroid_vector=d.get("centroid_vector"),
            last_updated=d.get("last_updated", 0.0),
        )


_DEFAULT = {
    "version": 1,
    "items": {},
    "_next_id": 1,
}


class SceneStore:
    """场景簇持久化存储。key = cluster_id。"""

    def __init__(self, path: str = "data/learners/scenes.json"):
        self._store = JsonStore(path, default=_DEFAULT)

    # ── 查询 ──

    def get(self, cluster_id: str) -> Optional[TagCluster]:
        d = self._store.get(cluster_id)
        return TagCluster.from_dict(d) if d else None

    def get_all_clusters(self) -> List[TagCluster]:
        return [TagCluster.from_dict(d) for d in self._store.get_all()]

    def get_all(self) -> List[dict]:
        return self._store.get_all()

    def keys(self) -> List[str]:
        return self._store.keys()

    def count(self) -> int:
        return self._store.count()

    # ── 写入 ──

    async def save(self, cluster: TagCluster) -> None:
        await self._store.save(cluster.cluster_id, cluster.to_dict())

    async def delete(self, cluster_id: str) -> bool:
        return await self._store.delete(cluster_id)

    async def _next_id(self) -> str:
        from core.learners.stores.base_store import _log
        nid = self._store._data.get("_next_id", 1)
        self._store._data["_next_id"] = nid + 1
        return f"sc_{nid:04d}"
