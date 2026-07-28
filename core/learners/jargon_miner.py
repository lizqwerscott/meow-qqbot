"""JargonMiner — 俚语挖掘器

从社群对话中自动识别黑话/自定义术语，分四级推理：
  Level 0: 首次出现，记录频率
  Level 1: 同群 ≥3 次，追踪上下文
  Level 2: 跨群 + 语义一致 → LLM 推理含义
  Level 3: Bot 主动使用 + 正向反馈 → 固化为主动词汇
"""

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Set

from core.learners.stores.jargon_store import JargonEntry, JargonStore

_log = logging.getLogger(__name__)

# ── 基本停用词（过滤常见词/虚词）──
_STOP_WORDS: Set[str] = {
    # 中文单字虚词
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "可以", "这个", "那个", "什么", "怎么", "我们", "你们", "他们",
    "因为", "所以", "但是", "而且", "如果", "虽然", "然后", "还是", "或者",
    "没有", "不是", "就是", "自己", "知道", "觉得", "可以", "应该", "可能",
    "已经", "这么", "那么", "这样", "那样", "这里", "那里", "这些", "那些",
    "时候", "现在", "今天", "明天", "昨天", "刚刚", "已经", "正在",
    "哈", "啊", "吧", "吗", "呢", "哦", "嗯", "啦", "呀", "嘛", "嘿",
    # 英文常见词
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "own", "same", "so", "too", "very", "just", "also", "well", "now",
    "here", "there", "then", "than", "as", "at", "by", "for", "from",
    "in", "into", "of", "on", "to", "with", "up", "down", "out", "off",
    "over", "under", "again", "further", "once",
    # 数字/短符号
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
}

# 额外高频中文 2-gram（几乎不可能是俚语）
_COMMON_CJK_2GRAM: Set[str] = {
    "我们", "他们", "你们", "它们", "自己", "什么", "怎么", "这个", "那个",
    "可以", "知道", "觉得", "应该", "可能", "已经", "没有", "不是", "就是",
    "因为", "所以", "但是", "而且", "虽然", "然后", "还是", "或者", "如果",
    "现在", "时候", "今天", "明天", "昨天", "刚刚", "正在", "马上", "立刻",
    "这样", "那样", "这里", "那里", "这些", "那些", "这么", "那么", "多么",
    "一直", "一起", "一下", "一会儿", "有点", "非常", "特别", "比较",
    "真的", "真是", "但是", "不过", "结果", "其实", "当然", "肯定", "一定",
    "看到", "听到", "想到", "说道", "认为", "感觉", "开始", "继续", "准备",
    "希望", "想要", "需要", "必须", "能够", "成为", "作为", "进行", "通过",
    "然而", "此外", "否则", "否则", "不然", "总之", "以及", "关于", "对于",
    "根据", "按照", "除了", "包括", "属于", "来自", "以及", "还是", "为了",
}

# 按标点/空格分割中文段
_CJK_SEGMENT = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+(?:[\.\'\u2022][\u4e00-\u9fff\u3400-\u4dbf]+)*")
# 英文/数字/符号 token（含常见网络用语写法如 "6", "yyds"）
_EN_TOKEN = re.compile(r"[a-zA-Z0-9_\-\+\.#@\u00b7]+")


def _tokenize(text: str) -> List[str]:
    """从文本中提取候选俚语 token。

    策略：
    - 英文/数字词直接提取（≥2 字符，或全是字母缩写如 "yyds"）
    - 中文段整体提取（2-6 字符，过滤停用词 / 常见 2-gram）
    - 不做内部 n-gram 滑动，避免碎片化噪声
    """
    tokens: List[str] = []

    # 英文/数字/符号 token
    for m in _EN_TOKEN.finditer(text):
        t = m.group()
        # 纯数字跳过
        if t.isdigit():
            continue
        # 至少 2 字符但允许 "6" 这种特殊情况—不过太短容易误判，还是 ≥2
        if len(t) >= 2 and t.lower() not in _STOP_WORDS:
            tokens.append(t)

    # 中文段 — 按标点/空白自然分割后的完整片段
    for m in _CJK_SEGMENT.finditer(text):
        t = m.group()
        if len(t) < 2 or len(t) > 6:
            continue
        if t in _STOP_WORDS or t in _COMMON_CJK_2GRAM:
            continue
        # 排除纯数字中文
        if all(c in "零一二三四五六七八九十百千万亿" for c in t):
            continue
        tokens.append(t)

    return tokens


class JargonMiner:
    """俚语挖掘器。

    使用方式：
        miner = JargonMiner(store, ai_service, config)
        await miner.observe("YBB 吃饭了没", "group_1001")
    """

    def __init__(
        self,
        store: JargonStore,
        ai_service: Any = None,
        config: Optional[dict] = None,
    ):
        self._store = store
        self._ai = ai_service
        cfg = config or {}

        self._inference_thresholds: List[int] = cfg.get("inference_thresholds", [1, 3, 5, 10])
        self._cross_group_min: int = cfg.get("cross_group_min", 2)
        self._max_per_room: int = cfg.get("max_jargon_per_room", 500)

        # 并发锁（保护共享内存状态）
        self._lock = asyncio.Lock()
        # 频率追踪: LRU，上限 10000 条目
        self._freq: OrderedDict = OrderedDict()
        self._max_freq = 10000
        # 上下文缓存: {term: [context_msg, ...]}
        self._contexts: Dict[str, List[str]] = defaultdict(list)
        # 去重缓存：LRU，避免同一消息重复触发
        self._seen_terms: OrderedDict = OrderedDict()
        self._max_seen = 10000

        _log.info(
            "JargonMiner 已初始化 "
            f"(thresholds={self._inference_thresholds}, "
            f"cross_group_min={self._cross_group_min})"
        )

    @property
    def store(self) -> JargonStore:
        return self._store

    # ── 核心入口 ──

    async def observe(self, message_text: str, chat_id: str) -> None:
        """观察一条消息，更新俚语频率统计。

        轻量方法，仅维护内存计数器 + 存储。
        独立追踪 per-chat 频率 + 跨群计数，不混用。
        """
        if not message_text or len(message_text) < 3:
            return

        tokens = _tokenize(message_text)
        if not tokens:
            return

        now = time.time()

        async with self._lock:
            for term in set(tokens):
                if len(term) > 20:
                    continue

                # 去重 (LRU)
                dedup_key = (term, chat_id, message_text[:50])
                if dedup_key in self._seen_terms:
                    continue
                self._seen_terms[dedup_key] = None
                self._seen_terms.move_to_end(dedup_key)
                if len(self._seen_terms) > self._max_seen:
                    self._seen_terms.popitem(last=False)

                # 递增 per-chat 频率 (LRU)
                freq_key = (term, chat_id)
                current = self._freq.get(freq_key, 0)
                self._freq[freq_key] = current + 1
                self._freq.move_to_end(freq_key)
                if len(self._freq) > self._max_freq:
                    self._freq.popitem(last=False)
                chat_count = self._freq[freq_key]

        # 锁外进行 store/AI 调用（不阻塞并发）
        for term in set(tokens):
            if len(term) > 20:
                continue
            freq_key = (term, chat_id)
            async with self._lock:
                chat_count = self._freq.get(freq_key, 0)
            existing = self._store.get(term)
            if existing:
                await self._update_entry(existing, chat_id, message_text, chat_count, now)
            else:
                await self._create_entry(term, chat_id, message_text, chat_count, now)

    # ── 条目创建与更新 ──

    async def _create_entry(
        self, term: str, chat_id: str, context: str, chat_count: int, now: float
    ) -> None:
        """创建新俚语候选。仅在满足最低频率时创建。"""
        # 至少同群出现 3 次才创建候选（Level 1 条件）
        if chat_count < self._inference_thresholds[1]:
            return

        entry = JargonEntry(
            term=term,
            definition="",
            examples=[context] if context else [],
            origin_sessions=[chat_id],
            inference_level=1,
            frequency=1,
            source="auto",
            first_seen_at=now,
            last_seen_at=now,
        )

        # 检查跨群 Level 2
        await self._check_promote_to_level2(entry)
        await self._store.save(entry)
        _log.info(f"JargonMiner 新候选: {term} (level={entry.inference_level})")

    async def _update_entry(
        self, entry: JargonEntry, chat_id: str, context: str, chat_count: int, now: float
    ) -> None:
        """更新已有条目并检查晋升。"""
        changed = False

        if chat_id not in entry.origin_sessions:
            entry.origin_sessions.append(chat_id)
            changed = True

        if context not in entry.examples:
            entry.examples.append(context)
            if len(entry.examples) > 50:
                entry.examples = entry.examples[-50:]
            changed = True

        entry.frequency += 1
        entry.last_seen_at = now
        changed = True

        # 晋升 Level 2：跨群检测（独立于频率）
        if entry.inference_level < 2:
            if await self._check_promote_to_level2(entry):
                changed = True

        # 晋升 Level 1：同群频率达标
        if entry.inference_level < 1 and chat_count >= self._inference_thresholds[1]:
            entry.inference_level = 1
            changed = True

        if changed:
            await self._store.save(entry)

    async def _check_promote_to_level2(self, entry: JargonEntry) -> bool:
        """检查并执行 Level 2 晋升（跨群 + LLM 推理）。"""
        if entry.source == "manual":
            return False

        source_rooms = set(entry.origin_sessions)
        if len(source_rooms) < self._cross_group_min:
            return False

        if entry.inference_level >= 2:
            return False

        llm_entry = entry  # 保存引用供异步推理使用
        asyncio.create_task(self._delayed_promote(llm_entry))
        return True

    async def _delayed_promote(self, entry: JargonEntry) -> None:
        """异步执行 LLM 推理和晋升，不阻塞消息热路径。"""
        try:
            await self._infer_meaning(entry)
            entry.inference_level = 2
            source_rooms = set(entry.origin_sessions)
            _log.info(f"JargonMiner Level 2 晋升: {entry.term} (rooms={source_rooms})")
            await self._store.save(entry)
        except Exception as e:
            _log.warning(f"JargonMiner 异步晋升失败 [{entry.term}]: {e}")

    # ── LLM 推理 ──

    async def _infer_meaning(self, entry: JargonEntry) -> None:
        """调用 LLM 推理俚语含义。"""
        if not self._ai:
            entry.definition = "(未配置 AI 推理)"
            return

        # 已经有手动定义则跳过
        if entry.source == "manual" and entry.definition:
            return

        examples_text = "\n".join(f"- {e}" for e in entry.examples[:5])
        groups_text = ", ".join(list(set(entry.origin_sessions))[:5])

        prompt = (
            "你是一个社群俚语分析器。以下是一个在社群中出现的候选俚语。\n\n"
            f"俚语: {entry.term}\n"
            f"出现群组: {groups_text}\n"
            f"使用示例:\n{examples_text}\n\n"
            "请推断这个俚语的可能含义。仅返回 JSON：\n"
            '{"definition": "含义描述", "explanation": "为什么这样推断"}'
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response_text, _ = await self._ai.chat_completion(
                messages=messages,
                max_tokens=300,
            )
            if response_text:
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(cleaned)
                entry.definition = parsed.get("definition", "").strip()
                _log.info(
                    f"JargonMiner 推理 [{entry.term}]: {entry.definition}"
                )
            else:
                _log.warning(f"JargonMiner 推理返回空: {entry.term}")
        except Exception as e:
            _log.warning(f"JargonMiner 推理失败 [{entry.term}]: {e!r}")

    # ── 手动添加 ──

    async def add_manual(
        self,
        term: str,
        definition: str,
        examples: Optional[List[str]] = None,
        added_by: str = "",
        chat_id: str = "",
    ) -> JargonEntry:
        """用户/管理员手动添加俚语。"""
        now = time.time()
        entry = JargonEntry(
            term=term,
            definition=definition,
            examples=examples or [],
            origin_sessions=[chat_id] if chat_id else [],
            inference_level=3,
            frequency=0,
            source="manual",
            added_by=added_by,
            first_seen_at=now,
            last_seen_at=now,
        )
        await self._store.save(entry)
        _log.info(f"JargonMiner 手动添加: {term} = {definition}")
        return entry

    # ── Level 3 晋升 ──

    async def promote_to_level3(self, term: str) -> bool:
        """将俚语提升到 Level 3（Bot 主动词汇）。"""
        entry = self._store.get(term)
        if not entry:
            return False
        await self._store.update(term, inference_level=3)
        _log.info(f"JargonMiner 晋升 Level 3: {term}")
        return True

    # ── 查询 ──

    def get_active_jargons(self, chat_id: str) -> List[JargonEntry]:
        """获取某群活跃的俚语（用于注入 prompt）。"""
        entries = self._store.get_active_for_chat(chat_id, min_level=1)
        # 按 Level 降序，manual 优先
        entries.sort(key=lambda e: (e.inference_level, e.source != "manual"), reverse=True)
        return entries[:20]

    def get_all_entries(self) -> List[JargonEntry]:
        return self._store.get_all_entries()

    def search(self, query: str) -> List[JargonEntry]:
        return self._store.search(query)

    # ── 词条管理 ──

    async def delete_entry(self, term: str) -> bool:
        return await self._store.delete(term)
