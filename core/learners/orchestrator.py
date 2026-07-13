"""LearningOrchestrator — 学习系统中央调度器。

统一入口：
- on_message(): dispatch() 中调用，异步非阻塞
- enrich_prompt_context(): prompt_builder.py 中调用
- on_bot_reply(): 记录 Bot 回复用于 BehaviorLearner
- 各子学习器的查询方法
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.learners.jargon_miner import JargonMiner
from core.learners.scene_clusterer import SceneClusterer
from core.learners.behavior_learner import BehaviorLearner
from core.learners.expression_learner import ExpressionLearner
from core.learners.stores.jargon_store import JargonEntry, JargonStore
from core.learners.stores.scene_store import SceneStore
from core.learners.stores.behavior_store import BehaviorPattern, BehaviorStore
from core.learners.stores.expression_store import ExpressionMapping, ExpressionStore
from core.learners.base import LearnerConfig, config_from_dict

_log = logging.getLogger(__name__)


class LearningOrchestrator:
    """学习系统中央调度器。"""

    def __init__(
        self,
        config: dict,
        ai_service: Any = None,
        data_dir: str = "data/learners/",
        emoji_manager: Any = None,
    ):
        self._cfg: LearnerConfig = config_from_dict(config)
        self._ai = ai_service

        self.jargon: JargonMiner
        self.scene: SceneClusterer
        self.behavior: BehaviorLearner
        self.expression: ExpressionLearner

        self._init_all(data_dir, emoji_manager)

        if self._cfg.enabled:
            self._decay_task = asyncio.create_task(self._decay_loop())

        _log.info("LearningOrchestrator 已初始化 (enabled=%s)", self._cfg.enabled)

    def _init_all(self, data_dir: str, emoji_manager: Any) -> None:
        # Jargon
        jargon_store = JargonStore(path=f"{data_dir}jargons.json")
        self.jargon = JargonMiner(
            store=jargon_store,
            ai_service=self._ai,
            config={
                "inference_thresholds": self._cfg.inference_thresholds,
                "cross_group_min": self._cfg.cross_group_min,
                "max_jargon_per_room": self._cfg.max_jargon_per_room,
            },
        )

        # Scene
        scene_store = SceneStore(path=f"{data_dir}scenes.json")
        self.scene = SceneClusterer(
            store=scene_store,
            ai_service=self._ai,
            config={
                "window_size": self._cfg.cluster_window_size,
                "reuse_threshold": self._cfg.cluster_reuse_threshold,
                "max_clusters": self._cfg.max_clusters,
            },
        )

        # Behavior
        behavior_store = BehaviorStore(path=f"{data_dir}behaviors.json")
        self.behavior = BehaviorLearner(
            store=behavior_store,
            scene_clusterer=self.scene,
            ai_service=self._ai,
            config={
                "trigger_frequency": self._cfg.trigger_frequency,
                "trigger_window_hours": self._cfg.trigger_window_hours,
                "decay_rate": self._cfg.decay_rate,
                "decay_threshold": self._cfg.decay_threshold,
                "similarity_threshold": self._cfg.similarity_threshold,
                "max_patterns_per_room": self._cfg.max_patterns_per_room,
            },
        )

        # Expression
        expression_store = ExpressionStore(path=f"{data_dir}expressions.json")
        self.expression = ExpressionLearner(
            store=expression_store,
            ai_service=self._ai,
            emoji_manager=emoji_manager,
            config={
                "min_frequency": self._cfg.min_frequency,
                "observation_window": self._cfg.observation_window,
                "max_concurrent_batches": self._cfg.max_concurrent_batches,
                "review_required": self._cfg.review_required,
                "auto_approve_threshold": self._cfg.auto_approve_threshold,
            },
        )

    # ── 消息观察 ──

    async def on_message(
        self,
        message_text: str,
        chat_id: str,
        sender_id: str = "",
    ) -> None:
        """轻量消息观察，dispatch() 中调用。"""
        if not self._cfg.enabled:
            return

        # Scene — 累积对话片段
        await self.scene.observe(message_text)

        # Jargon — 俚语挖掘
        await self.jargon.observe(message_text, chat_id)

    async def on_emoji_seen(
        self,
        message_text: str,
        emoji_hash: str,
        emoji_desc: str = "",
        emoji_tags: Optional[List[str]] = None,
        chat_id: str = "",
    ) -> None:
        """观察含表情的消息（由表情处理路径调用）。"""
        if not self._cfg.enabled:
            return
        await self.expression.observe(
            message_text, emoji_hash, emoji_desc, emoji_tags or [], chat_id,
        )

    async def on_bot_reply(
        self,
        user_message: str,
        bot_reply: str,
        effect: str,
        chat_id: str,
    ) -> None:
        """记录 Bot 回复，用于 BehaviorLearner。

        effect: "positive" | "negative" | "neutral"
        """
        if not self._cfg.enabled:
            return
        await self.behavior.observe(user_message, bot_reply, effect, chat_id)

    # ── Prompt 注入 ──

    async def enrich_prompt_context(
        self,
        chat_id: str,
        sender_id: str = "",
        message_text: str = "",
    ) -> str:
        """返回需要注入到动态上下文的学习结果文本。"""
        if not self._cfg.enabled:
            return ""

        parts = []

        # 俚语词典
        jargons = self.jargon.get_active_jargons(chat_id)
        if jargons:
            lines = ["【社群俚语词典】"]
            for e in jargons[:10]:
                source_tag = "[手动添加] " if e.source == "manual" else ""
                def_text = e.definition or "(含义待推理)"
                lines.append(f"- {source_tag}\"{e.term}\": {def_text}")
            parts.append("\n".join(lines))

        # 行为模式（需场景标签）
        if message_text:
            scene_labels = await self.scene.get_labels(message_text)
            patterns = await self.behavior.get_relevant_patterns(scene_labels, top_k=2)
            if patterns:
                lines = ["【习得的行为参考】"]
                for p in patterns:
                    conf = int(p.confidence * 100)
                    lines.append(
                        f"- 场景: {p.scene_descriptor[:60]} → "
                        f"行为: {p.suggested_action[:60]} "
                        f"(置信度 {conf}%)"
                    )
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    # ── 衰减循环 ──

    async def _decay_loop(self):
        """每小时执行一次行为模式衰减。"""
        while True:
            await asyncio.sleep(3600)
            try:
                removed = await self.behavior.decay_patterns()
                if removed:
                    _log.info(f"BehaviorLearner 衰减: 移除了 {removed} 个过期模式")
            except Exception as e:
                _log.warning(f"衰减循环异常: {e!r}")

    # ── 查询 / 管理 ──

    # Jargon
    def get_jargon_entries(self) -> List[JargonEntry]:
        return self.jargon.get_all_entries()

    def search_jargon(self, query: str) -> List[JargonEntry]:
        return self.jargon.search(query)

    async def add_jargon(self, term: str, definition: str, examples: Optional[List[str]] = None,
                         added_by: str = "", chat_id: str = "") -> JargonEntry:
        return await self.jargon.add_manual(term, definition, examples, added_by, chat_id)

    async def delete_jargon(self, term: str) -> bool:
        return await self.jargon.delete_entry(term)

    # Expression
    def get_expression_mappings(self) -> List[ExpressionMapping]:
        return self.expression.get_all_mappings()

    def get_expression_approved(self) -> List[ExpressionMapping]:
        return self.expression.get_approved_mappings()

    def get_expression_pending(self) -> List[ExpressionMapping]:
        return self.expression.get_pending_mappings()

    async def approve_expression(self, expression_hash: str) -> bool:
        return await self.expression.approve(expression_hash)

    async def reject_expression(self, expression_hash: str) -> bool:
        return await self.expression.reject(expression_hash)

    # Behavior
    def get_behavior_patterns(self) -> List[BehaviorPattern]:
        return self.behavior.get_all_patterns()

    # Scene
    def get_scene_clusters(self) -> list:
        return self.scene.get_all_clusters()

    # ── 统计 ──

    def get_stats(self) -> dict:
        if not self._cfg.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "jargon_count": self.jargon._store.count(),
            "scene_count": self.scene.get_cluster_count(),
            "behavior_count": self.behavior.get_pattern_count(),
            "expression_count": self.expression.get_mapping_count(),
        }

    async def stop(self):
        if self._cfg.enabled and hasattr(self, "_decay_task"):
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass
