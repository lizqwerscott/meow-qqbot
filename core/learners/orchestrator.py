"""LearningOrchestrator — 学习系统中央调度器。

当前只保留黑话 (Jargon) 学习系统。
"""

import logging
from typing import Any, Dict, List, Optional

from core.learners.jargon_miner import JargonMiner
from core.learners.stores.jargon_store import JargonEntry, JargonStore
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

        self._init_all(data_dir)

        _log.info("LearningOrchestrator 已初始化 (enabled=%s)", self._cfg.enabled)

    def _init_all(self, data_dir: str) -> None:
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
        try:
            await self.jargon.observe(message_text, chat_id)
        except Exception as e:
            _log.warning("学习系统观察消息失败 [%s..]: %s", chat_id[:12], e)

    # ── Prompt 注入 ──

    async def enrich_prompt_context(
        self,
        chat_id: str,
        sender_id: str = "",
        message_text: str = "",
    ) -> str:
        """返回需要注入到动态上下文的俚语词典。"""
        if not self._cfg.enabled:
            return ""

        jargons = self.jargon.get_active_jargons(chat_id)
        if not jargons:
            return ""

        lines = ["【社群俚语词典】"]
        for e in jargons[:10]:
            source_tag = "[手动添加] " if e.source == "manual" else ""
            def_text = e.definition or "(含义待推理)"
            lines.append(f"- {source_tag}\"{e.term}\": {def_text}")
        return "\n".join(lines)

    # ── 查询 / 管理 ──

    def get_jargon_entries(self) -> List[JargonEntry]:
        return self.jargon.get_all_entries()

    def search_jargon(self, query: str) -> List[JargonEntry]:
        return self.jargon.search(query)

    async def add_jargon(self, term: str, definition: str, examples: Optional[List[str]] = None,
                         added_by: str = "", chat_id: str = "") -> JargonEntry:
        return await self.jargon.add_manual(term, definition, examples, added_by, chat_id)

    async def delete_jargon(self, term: str) -> bool:
        return await self.jargon.delete_entry(term)

    # ── 统计 ──

    def get_stats(self) -> dict:
        if not self._cfg.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "jargon_count": self.jargon._store.count(),
        }

    async def stop(self):
        pass