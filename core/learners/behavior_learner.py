"""BehaviorLearner — 行为学习器

捕捉 Bot 与用户交互的深层模式。
学习"在什么场景下，采取什么行为，会得到什么结果"。

核心机制：
- 通过 LLM 解析聊天历史 → 生成"场景-行为-结果"三元组
- 正反馈 → 增加置信度；负反馈/衰减 → 降低置信度
- SceneClusterer 提供场景标签支撑
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from core.learners.stores.behavior_store import BehaviorPattern, BehaviorStore
from core.learners.base import exponential_decay

_log = logging.getLogger(__name__)

_LLM_SUMMARIZE_PROMPT = """分析以下对话记录，总结一条行为模式。

用户消息: {user_msg}
Bot回复: {bot_reply}
效果: {effect}

请从以下三个维度总结：
1. 场景描述 (scene)：用户在什么情境下说了那句话
2. 建议行为 (action)：Bot 的什么行为/语气/策略起到了{effect}效果
3. 预期效果 (effect_desc)：这种行为在类似场景中预期达到什么效果
4. 行为关键词 (keywords)：3-5个关键词描述这种行为风格

仅返回 JSON：
{{"scene": "...", "action": "...", "effect_desc": "...", "keywords": ["...", "..."]}}"""


class BehaviorLearner:
    """行为学习器。

    使用方式：
        learner = BehaviorLearner(store, scene_clusterer, ai_service, config)
        await learner.observe(user_msg, bot_reply, effect, chat_id)
    """

    def __init__(
        self,
        store: BehaviorStore,
        scene_clusterer: Any = None,
        ai_service: Any = None,
        config: Optional[dict] = None,
    ):
        self._store = store
        self._scene = scene_clusterer
        self._ai = ai_service
        cfg = config or {}

        self._trigger_frequency: int = cfg.get("trigger_frequency", 3)
        self._trigger_window_hours: float = cfg.get("trigger_window_hours", 24)
        self._decay_rate: float = cfg.get("decay_rate", 0.1)
        self._decay_threshold: float = cfg.get("decay_threshold", 0.2)
        self._similarity_threshold: float = cfg.get("similarity_threshold", 0.75)
        self._max_patterns_per_room: int = cfg.get("max_patterns_per_room", 200)

        # 场景-行为计数器: {(scene_hash, action_hash): [timestamps]}
        self._scene_action_log: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        _log.info(
            f"BehaviorLearner 已初始化 "
            f"(trigger_freq={self._trigger_frequency}, "
            f"decay_rate={self._decay_rate})"
        )

    # ── 核心入口 ──

    async def observe(
        self,
        user_message: str,
        bot_reply: str,
        effect: str,
        chat_id: str,
    ) -> None:
        """观察一次交互。

        effect: "positive" | "negative" | "neutral"
        """
        if not user_message or not bot_reply:
            return

        scene_labels = await self._get_scene_labels(user_message)
        if not scene_labels:
            return

        scene_text = ", ".join(f"{k}" for k in scene_labels.keys())
        action_text = self._extract_action_style(bot_reply)

        pair_key = (scene_text[:64], action_text[:64])
        now = time.time()

        self._scene_action_log[pair_key].append(now)

        # 检查触发条件
        if not self._should_trigger(pair_key):
            return

        # 触发学习
        asyncio.create_task(self._learn_pattern(
            user_msg=user_message,
            bot_reply=bot_reply,
            effect=effect,
            scene_text=scene_text,
            action_text=action_text,
            chat_id=chat_id,
        ))

    async def report_effect(
        self,
        scene_summary: str,
        action_taken: str,
        effect: str,
        chat_id: str,
    ) -> None:
        """AI 主动报告行为效果。"""
        scene_text = scene_summary[:64]
        action_text = action_taken[:64]
        pair_key = (scene_text, action_text)
        now = time.time()

        self._scene_action_log[pair_key].append(now)

        if not self._should_trigger(pair_key):
            return

        asyncio.create_task(self._learn_pattern(
            user_msg=scene_summary,
            bot_reply=action_taken,
            effect=effect,
            scene_text=scene_text,
            action_text=action_text,
            chat_id=chat_id,
        ))

    # ── 学习 ──

    async def _learn_pattern(
        self,
        user_msg: str,
        bot_reply: str,
        effect: str,
        scene_text: str,
        action_text: str,
        chat_id: str,
    ) -> None:
        """后台执行模式学习（LLM 总结 + 存储）。"""
        pattern = await self._summarize_pattern(user_msg, bot_reply, effect, scene_text)

        if not pattern:
            return

        pattern.source_rooms.append(chat_id)
        pattern.source_rooms = list(set(pattern.source_rooms))

        # 检查相似模式
        existing = self._find_similar(pattern)
        if existing:
            existing.confidence = min(existing.confidence + 0.1, 1.0)
            existing.last_triggered = time.time()
            existing.updated_at = time.time()
            if chat_id not in existing.source_rooms:
                existing.source_rooms.append(chat_id)
            await self._store.save(existing)
            _log.info(f"BehaviorLearner 更新模式: {existing.scene_descriptor[:40]}..")
        else:
            await self._store.save(pattern)
            _log.info(f"BehaviorLearner 新模式: {pattern.scene_descriptor[:40]}..")

    async def _summarize_pattern(
        self,
        user_msg: str,
        bot_reply: str,
        effect: str,
        scene_text: str = "",
    ) -> Optional[BehaviorPattern]:
        """用 LLM 总结行为模式。"""
        if not self._ai:
            return BehaviorPattern(
                scene_descriptor=scene_text or user_msg[:64],
                suggested_action=bot_reply[:64],
                expected_effect=effect,
                action_keywords=[],
                confidence=0.5,
                last_triggered=time.time(),
                created_at=time.time(),
                updated_at=time.time(),
            )

        prompt = _LLM_SUMMARIZE_PROMPT.format(
            user_msg=user_msg[:200],
            bot_reply=bot_reply[:200],
            effect=effect,
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response_text, _ = await self._ai.chat_completion(
                messages=messages,
                max_tokens=300,
            )
            if not response_text:
                return None

            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(cleaned)
            now = time.time()
            return BehaviorPattern(
                scene_descriptor=data.get("scene", scene_text),
                suggested_action=data.get("action", bot_reply[:64]),
                expected_effect=data.get("effect_desc", effect),
                action_keywords=data.get("keywords", []),
                confidence=0.6,
                last_triggered=now,
                created_at=now,
                updated_at=now,
            )
        except Exception as e:
            _log.warning(f"BehaviorLearner LLM 总结失败: {e!r}")
            return None

    # ── Prompt 注入 ──

    async def get_relevant_patterns(
        self,
        scene_labels: Dict[str, float],
        top_k: int = 3,
    ) -> List[BehaviorPattern]:
        """获取与当前场景相关的行为模式。"""
        if not scene_labels:
            return []

        keywords = list(scene_labels.keys())
        patterns = self._store.search_by_scene(keywords)
        patterns = [p for p in patterns if p.confidence >= self._decay_threshold]
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns[:top_k]

    # ── 衰减 ──

    async def decay_patterns(self) -> int:
        """执行一次衰减扫描，返回软删除的模式数。"""
        now = time.time()
        removed = 0

        for pattern in self._store.get_all_patterns():
            delta_hours = (now - pattern.last_triggered) / 3600
            if delta_hours < 24:
                continue

            new_confidence = exponential_decay(pattern.confidence, self._decay_rate, delta_hours)
            if new_confidence < self._decay_threshold:
                await self._store.delete(pattern.scene_descriptor[:64])
                removed += 1
                _log.debug(f"BehaviorLearner 衰减删除: {pattern.scene_descriptor[:40]}..")
            else:
                await self._store.update_confidence(pattern.scene_descriptor[:64], new_confidence)

        return removed

    # ── 内部 ──

    def _should_trigger(self, pair_key: Tuple[str, str]) -> bool:
        """检查场景-行为对是否达到触发阈值。"""
        timestamps = self._scene_action_log.get(pair_key, [])
        now = time.time()
        window_start = now - self._trigger_window_hours * 3600
        recent = [t for t in timestamps if t >= window_start]
        return len(recent) >= self._trigger_frequency

    def _find_similar(self, pattern: BehaviorPattern) -> Optional[BehaviorPattern]:
        """查找描述相似的模式。"""
        for existing in self._store.get_all_patterns():
            if existing.scene_descriptor == pattern.scene_descriptor:
                return existing
        return None

    @staticmethod
    def _extract_action_style(reply: str) -> str:
        """简单提取回复的风格特征。"""
        if not reply:
            return ""
        # 提取前 60 字符作为行为风格摘要
        return reply[:60]

    async def _get_scene_labels(self, text: str) -> Dict[str, float]:
        """获取场景标签。"""
        if self._scene:
            return await self._scene.get_labels(text)
        return {}

    # ── 查询 ──

    def get_all_patterns(self) -> List[BehaviorPattern]:
        return self._store.get_all_patterns()

    def get_pattern_count(self) -> int:
        return self._store.count()
