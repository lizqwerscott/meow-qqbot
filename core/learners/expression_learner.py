"""ExpressionLearner — 表情学习器

学习用户在特定语境下倾向于发送哪类表情，以及这些表情背后的触发词。

流程：
  监控表情使用频率 → 频率达标 → LLM 验证语义一致性
  → 语义一致 + 安全 → approved / 否则 pending

与 EmojiManager 联动：
  学到的映射按 confidence_weight 影响 emoji_selector 的选择概率
"""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from core.learners.stores.expression_store import ExpressionMapping, ExpressionStore

_log = logging.getLogger(__name__)

# ── 中文分词辅助（简单关键词提取）──

import re as _re

# 标点 → 空格
_CJK_PUNCT_RE = _re.compile(
    r"[\u3000-\u303f\uff00-\uffef\u2000-\u206f\u2100-\u214f\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65,.:;!?\"'()\[\]{}…—·]"
)


def _extract_keywords(text: str, max_words: int = 5) -> List[str]:
    """从文本中提取关键词（最简实现：去停用词后取前 N 个词）。"""
    text = _CJK_PUNCT_RE.sub(" ", text)
    tokens = [t.strip() for t in re.split(r"\s+", text) if t.strip()]
    stopwords = {
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "你",
        "他",
        "她",
        "它",
        "们",
        "这",
        "那",
        "什么",
        "怎么",
        "这个",
        "那个",
        "可以",
        "知道",
        "觉得",
        "应该",
        "可能",
        "已经",
        "没有",
        "不是",
        "就是",
        "因为",
        "所以",
        "但是",
        "而且",
        "虽然",
        "然后",
        "还是",
        "或者",
        "如果",
        "现在",
        "时候",
        "今天",
        "明天",
        "昨天",
        "刚刚",
        "正在",
        "哈",
        "啊",
        "吧",
        "吗",
        "呢",
        "哦",
        "嗯",
        "啦",
        "呀",
        "嘛",
        "嘿",
        "一个",
        "我们",
        "你们",
        "他们",
        "自己",
    }
    keywords = [t for t in tokens if t.lower() not in stopwords and len(t) >= 2]
    return keywords[:max_words]


_LLM_CONSISTENCY_PROMPT = """分析以下含表情的消息样本，判断表情的使用语义是否一致。

表情描述: {emoji_desc}
表情标签: {emoji_tags}

消息样本:
{samples}

请判断：
1. 这些语境中表情的语义是否一致？(consistent: true/false)
2. 如果用关键词描述这个表情在这些语境中的用途，会是什么？（2-3个关键词）

仅返回 JSON：
{{"consistent": true/false, "reason": "简短理由", "suggested_keywords": ["...", "..."]}}"""


class ExpressionLearner:
    """表情学习器。

    使用方式：
        learner = ExpressionLearner(store, ai_service, config)
        await learner.observe(message_text, emoji_hash, emoji_desc, emoji_tags, chat_id)
    """

    def __init__(
        self,
        store: ExpressionStore,
        ai_service: Any = None,
        emoji_manager: Any = None,
        config: Optional[dict] = None,
    ):
        self._store = store
        self._ai = ai_service
        self._emoji_manager = emoji_manager
        cfg = config or {}

        self._min_frequency: int = cfg.get("min_frequency", 5)
        self._observation_window: int = cfg.get("observation_window", 30)
        self._max_concurrent_batches: int = cfg.get("max_concurrent_batches", 3)
        self._review_required: bool = cfg.get("review_required", True)
        self._auto_approve_threshold: float = cfg.get("auto_approve_threshold", 0.9)

        # 滑动窗口追踪: {emoji_hash: deque[(keyword, chat_id, timestamp)]}
        self._emoji_log: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._observation_window * 2)
        )
        # 并发控制
        self._semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        _log.info(
            f"ExpressionLearner 已初始化 "
            f"(min_freq={self._min_frequency}, window={self._observation_window})"
        )

    @property
    def store(self) -> "ExpressionStore":
        return self._store

    # ── 核心入口 ──

    async def observe(
        self,
        message_text: str,
        emoji_hash: str,
        emoji_desc: str = "",
        emoji_tags: Optional[List[str]] = None,
        chat_id: str = "",
    ) -> None:
        """观察一条含表情的消息。"""
        if not emoji_hash or not message_text:
            return

        keywords = _extract_keywords(message_text)
        if not keywords:
            return

        now = time.time()
        for kw in keywords:
            self._emoji_log[emoji_hash].append((kw, chat_id, now))

        # 检查触发条件
        if not self._should_learn(emoji_hash):
            return

        existing = self._store.get(emoji_hash)
        if existing:
            return  # 已经学过了

        async with self._semaphore:
            await self._learn_mapping(
                emoji_hash=emoji_hash,
                emoji_desc=emoji_desc,
                emoji_tags=emoji_tags or [],
                chat_id=chat_id,
            )

    # ── 学习 ──

    async def _learn_mapping(
        self,
        emoji_hash: str,
        emoji_desc: str,
        emoji_tags: List[str],
        chat_id: str,
    ) -> None:
        """执行学习（LLM 验证 + 存储）。"""
        samples = self._get_samples(emoji_hash)
        if not samples:
            return

        sample_texts = []
        seen = set()
        for kw, cid, ts in samples:
            sample_key = f"{kw} @ {cid}"
            if sample_key not in seen:
                seen.add(sample_key)
                sample_texts.append(f'- "{kw}" (in {cid})')

        samples_str = "\n".join(sample_texts[:10])

        result = await self._verify_consistency(emoji_desc, emoji_tags, samples_str)
        if not result:
            return

        if not result.get("consistent"):
            _log.debug(f"ExpressionLearner 语义不一致，跳过: {emoji_hash[:12]}..")
            return

        suggested = result.get("suggested_keywords", [])
        context_tags = emoji_tags or suggested

        # 合并关键词
        all_keywords = list(set(suggested + [kw for kw, _, _ in samples[:5]]))

        status = "approved"
        if (
            self._review_required
            and result.get("confidence", 1.0) < self._auto_approve_threshold
        ):
            status = "pending"

        mapping = ExpressionMapping(
            trigger_keywords=all_keywords[:10],
            expression_hash=emoji_hash,
            context_tags=context_tags,
            review_status=status,
            frequency=len(samples),
            confidence_weight=min(len(samples) / self._observation_window, 1.0),
            source_rooms=[chat_id],
            created_at=time.time(),
            updated_at=time.time(),
        )

        await self._store.save(mapping)
        _log.info(
            f"ExpressionLearner 新映射 [{emoji_hash[:12]}..]: "
            f"keywords={all_keywords}, status={status}"
        )

    # ── LLM 验证 ──

    async def _verify_consistency(
        self,
        emoji_desc: str,
        emoji_tags: List[str],
        samples_str: str,
    ) -> Optional[dict]:
        """LLM 验证表情语义一致性。"""
        if not self._ai:
            return {"consistent": True, "suggested_keywords": [], "confidence": 1.0}

        prompt = _LLM_CONSISTENCY_PROMPT.format(
            emoji_desc=emoji_desc or "(无描述)",
            emoji_tags=", ".join(emoji_tags) if emoji_tags else "(无标签)",
            samples=samples_str or "(无样本)",
        )

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
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            return json.loads(cleaned)
        except Exception as e:
            _log.warning(f"ExpressionLearner 验证失败: {e!r}")
            return None

    # ── 查询 ──

    def get_mappings_for_keywords(self, keywords: List[str]) -> List[ExpressionMapping]:
        """获取与关键词匹配的已审核表情映射。"""
        results = []
        seen = set()
        for kw in keywords:
            for mapping in self._store.find_by_keyword(kw):
                if mapping.expression_hash not in seen:
                    seen.add(mapping.expression_hash)
                    results.append(mapping)
        results.sort(key=lambda m: m.confidence_weight, reverse=True)
        return results

    def get_approved_mappings(self) -> List[ExpressionMapping]:
        return self._store.get_approved()

    def get_pending_mappings(self) -> List[ExpressionMapping]:
        return self._store.get_pending()

    def get_all_mappings(self) -> List[ExpressionMapping]:
        return self._store.get_all_mappings()

    def get_mapping_count(self) -> int:
        return self._store.count()

    # ── 审核操作 ──

    async def approve(self, expression_hash: str) -> bool:
        return await self._store.update_status(expression_hash, "approved")

    async def reject(self, expression_hash: str) -> bool:
        return await self._store.update_status(expression_hash, "rejected")

    async def rescue(self, expression_hash: str) -> bool:
        return await self._store.update_status(expression_hash, "rescue")

    # ── 内部 ──

    def _should_learn(self, emoji_hash: str) -> bool:
        """检查表情是否达到学习频率。"""
        log = self._emoji_log.get(emoji_hash)
        if not log or len(log) < self._min_frequency:
            return False

        now = time.time()
        cutoff = now - 86400  # 24h 窗口
        recent = [x for x in log if x[2] >= cutoff]
        return len(recent) >= self._min_frequency

    def _get_samples(self, emoji_hash: str) -> List[tuple]:
        """获取表情的上下文样本。"""
        log = self._emoji_log.get(emoji_hash)
        if not log:
            return []
        return list(log)[-self._observation_window :]
