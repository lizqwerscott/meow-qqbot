"""SceneClusterer — 场景聚类器

将对话片段聚类为可复用的"场景标签簇"。
为 BehaviorLearner 提供底层的语义支撑。

流程：
  累积对话 → 满阈值 → LLM 三维编码 (态度/领域/需求)
  → 与现有簇计算相似度 → 归入现有簇或创建新簇
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from core.learners.stores.scene_store import SceneStore, TagCluster
from core.learners.base import cosine_similarity

_log = logging.getLogger(__name__)

# ── 标签 → 向量编码（简单 one-hot 用于相似度计算）──

_ATTITUDES = ["正面", "负面", "中性", "嘲讽", "焦虑", "兴奋"]
_DOMAINS = ["职场", "技术", "生活", "娱乐", "情感", "其他"]
_NEEDS = ["寻求帮助", "情感宣泄", "信息获取", "社交互动", "闲聊"]


def _label_vector(attitude: str, domain: str, need: str) -> List[float]:
    """将三维标签编码为向量。"""
    vec = [0.0] * (len(_ATTITUDES) + len(_DOMAINS) + len(_NEEDS))
    if attitude in _ATTITUDES:
        vec[_ATTITUDES.index(attitude)] = 1.0
    if domain in _DOMAINS:
        vec[len(_ATTITUDES) + _DOMAINS.index(domain)] = 1.0
    if need in _NEEDS:
        vec[len(_ATTITUDES) + len(_DOMAINS) + _NEEDS.index(need)] = 1.0
    return vec


def _tags_from_labels(attitude: str, domain: str, need: str) -> Dict[str, float]:
    """将三维标签转为带权重的 tag dict。"""
    tags = {}
    if attitude:
        tags[attitude] = 0.9
    if domain:
        tags[domain] = 0.85
    if need:
        tags[need] = 0.8
    return tags


def _cluster_tag_overlap(a: Dict[str, float], b: Dict[str, float]) -> float:
    """计算两个 tag 集合的加权重叠度（Jaccard-like）。"""
    if not a or not b:
        return 0.0
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        return 0.0
    min_sum = sum(min(a.get(k, 0), b.get(k, 0)) for k in common_keys)
    max_sum = sum(max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b))
    if max_sum == 0:
        return 0.0
    return min_sum / max_sum


_LLM_PROMPT_TEMPLATE = """你是一个对话场景分析器。分析以下对话片段的三维度：

1. 态度 (Attitude)：正面 / 负面 / 中性 / 嘲讽 / 焦虑 / 兴奋
2. 领域 (Domain)：职场 / 技术 / 生活 / 娱乐 / 情感
3. 需求 (Need)：寻求帮助 / 情感宣泄 / 信息获取 / 社交互动 / 闲聊

每个片段返回一个 JSON 对象。仅返回 JSON 数组，不要其他文字：

{dialogs}

返回格式：[{{"attitude": "...", "domain": "...", "need": "..."}}, ...]"""


class SceneClusterer:
    """场景聚类器。

    使用方式：
        clusterer = SceneClusterer(store, ai_service, config)
        await clusterer.observe("真的受不了了，天天加班到十点")
        labels = await clusterer.get_labels("抱怨加班")
    """

    def __init__(
        self,
        store: SceneStore,
        ai_service: Any = None,
        config: Optional[dict] = None,
    ):
        self._store = store
        self._ai = ai_service
        cfg = config or {}

        self._window_size: int = cfg.get("window_size", 50)
        self._reuse_threshold: float = cfg.get("reuse_threshold", 0.7)
        self._max_clusters: int = cfg.get("max_clusters", 1000)

        # 全局对话缓冲
        self._buffer: List[dict] = []
        self._buffer_lock = asyncio.Lock()
        # 缓存 label → cluster_id 映射
        self._label_cache: Dict[str, str] = {}

        _log.info(
            f"SceneClusterer 已初始化 (window={self._window_size}, "
            f"reuse_threshold={self._reuse_threshold})"
        )

    @property
    def store(self) -> "SceneStore":
        return self._store

    # ── 核心入口 ──

    async def observe(self, message_text: str) -> None:
        """累积对话片段，满阈值触发聚类。"""
        if not message_text or len(message_text) < 5:
            return

        async with self._buffer_lock:
            self._buffer.append({
                "text": message_text[:200],
                "ts": time.time(),
            })

            if len(self._buffer) >= self._window_size:
                batch = self._buffer[:]
                self._buffer.clear()
                asyncio.create_task(self._cluster_batch(batch))

    async def get_labels(self, content: str) -> Dict[str, float]:
        """获取某条消息匹配的场景标签。

        先用 LLM 即时分析，再匹配已有簇。
        返回 tag → weight dict。
        """
        # 先查缓存
        cache_key = content[:80].strip()
        if cache_key in self._label_cache:
            cluster = self._store.get(self._label_cache[cache_key])
            if cluster:
                return dict(cluster.tags)

        # 即时 LLM 分析
        labels = await self._analyze_single(content)
        if not labels:
            return {}

        tags = _tags_from_labels(
            labels.get("attitude", ""),
            labels.get("domain", ""),
            labels.get("need", ""),
        )

        # 匹配已有簇
        best_cluster, similarity = self._find_best_cluster(tags)
        if best_cluster and similarity >= self._reuse_threshold:
            self._label_cache[cache_key] = best_cluster.cluster_id
            return dict(best_cluster.tags)

        return tags

    # ── 批量聚类 ──

    async def _cluster_batch(self, batch: List[dict]) -> None:
        """对一批对话执行聚类分析。"""
        texts = [b["text"] for b in batch]

        labels_list = await self._batch_analyze(texts)
        if not labels_list:
            return

        for text, labels in zip(texts, labels_list):
            if not labels:
                continue
            tags = _tags_from_labels(
                labels.get("attitude", ""),
                labels.get("domain", ""),
                labels.get("need", ""),
            )
            await self._assign_to_cluster(text, tags)

    async def _assign_to_cluster(self, text: str, tags: Dict[str, float]) -> None:
        """将分析结果归入已有簇或创建新簇。"""
        best_cluster, similarity = self._find_best_cluster(tags)

        if best_cluster and similarity >= self._reuse_threshold:
            new_tags = dict(best_cluster.tags)
            for k, v in tags.items():
                new_tags[k] = new_tags.get(k, 0) * 0.9 + v * 0.1
            best_cluster.tags = new_tags
            best_cluster.member_count += 1
            best_cluster.last_updated = time.time()
            await self._store.save(best_cluster)
        else:
            if self._store.count() >= self._max_clusters:
                return
            cid = await self._store._next_id()
            cluster = TagCluster(
                cluster_id=cid,
                tags=tags,
                prototype_utterance=text[:200],
                member_count=1,
                last_updated=time.time(),
            )
            await self._store.save(cluster)
            _log.info(f"SceneClusterer 新簇: {cid} tags={tags}")

    def _find_best_cluster(self, tags: Dict[str, float]) -> Tuple[Optional[TagCluster], float]:
        """从已有簇中找到最佳匹配。"""
        best_cluster = None
        best_sim = 0.0
        for c in self._store.get_all_clusters():
            sim = _cluster_tag_overlap(tags, c.tags)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c
        return best_cluster, best_sim

    # ── LLM 分析 ──

    async def _batch_analyze(self, texts: List[str]) -> List[Optional[dict]]:
        """批量 LLM 分析。"""
        if not self._ai:
            return [None] * len(texts)

        dialogs_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = _LLM_PROMPT_TEMPLATE.format(dialogs=dialogs_text)

        try:
            messages = [{"role": "user", "content": prompt}]
            response_text, _ = await self._ai.chat_completion(
                messages=messages,
                max_tokens=200 * len(texts),
            )
            if not response_text:
                return [None] * len(texts)

            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if not isinstance(data, list):
                return [None] * len(texts)

            results = []
            for item in data[:len(texts)]:
                if isinstance(item, dict):
                    results.append(item)
                else:
                    results.append(None)
            return results
        except Exception as e:
            _log.warning(f"SceneClusterer 批量分析失败: {e!r}")
            return [None] * len(texts)

    async def _analyze_single(self, text: str) -> Optional[dict]:
        """单条 LLM 分析（用于实时的 get_labels）。"""
        if not self._ai:
            return None

        prompt = _LLM_PROMPT_TEMPLATE.format(dialogs=f"1. {text}")

        try:
            messages = [{"role": "user", "content": prompt}]
            response_text, _ = await self._ai.chat_completion(
                messages=messages,
                max_tokens=200,
            )
            if not response_text:
                return None

            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            if isinstance(data, list) and data:
                return data[0]
            if isinstance(data, dict):
                return data
            return None
        except Exception as e:
            _log.warning(f"SceneClusterer 单条分析失败: {e!r}")
            return None

    # ── 查询 ──

    def get_all_clusters(self) -> List[TagCluster]:
        return self._store.get_all_clusters()

    def get_cluster_count(self) -> int:
        return self._store.count()
