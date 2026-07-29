import logging
import re
from typing import Optional

from core.message import InputMessage

_log = logging.getLogger(__name__)

_DIRTY_PATTERNS = (
    "<available_skills", "<skill>", "<description>",
    "<name>", "【工具配合原则】", "【记忆系统】",
    "--- 技能系统 ---", "工具配合原则",
)

_DIRTY_REGEX = re.compile(
    r"(?:\b\d{17}[\dXx]\b)"  # 中国身份证
    r"|(?:\b1[3-9]\d{9}\b)"  # 手机号
    r"|(?:\b[\w.-]+@[\w.-]+\.\w{2,}\b)",  # 邮箱
)


def _is_dirty(text: str) -> bool:
    for p in _DIRTY_PATTERNS:
        if p in text:
            return True
    return bool(_DIRTY_REGEX.search(text))


class MemoryBlockBuilder:
    """构建记忆、学习、归档摘要动态块。"""

    def __init__(
        self,
        hindsight,
        search_top_k: int,
        learners,
        archive_manager,
    ) -> None:
        self.hindsight = hindsight
        self._search_top_k = search_top_k
        self.learners = learners
        self._archive_manager = archive_manager

    async def build(
        self,
        *,
        chat_id: str,
        sender_id: str,
        input_message: InputMessage,
        max_archive_chars: int = 0,
    ) -> Optional[str]:
        parts = []

        memory_text = await self.build_memory_context(sender_id, input_message)
        if memory_text:
            parts.append(memory_text)

        if self.learners:
            try:
                learning_ctx = await self.learners.enrich_prompt_context(
                    chat_id=chat_id,
                    sender_id=sender_id,
                    message_text=input_message.content,
                )
                if learning_ctx:
                    parts.append(learning_ctx)
            except Exception as e:
                _log.warning(
                    "学习上下文注入失败 [%s..]: %s", chat_id[:12], e
                )

        if self._archive_manager:
            try:
                summary_text = self._archive_manager.consume_summary(chat_id)
            except Exception as e:
                _log.warning(
                    "归档摘要注入失败 [%s..]: %s", chat_id[:12], e
                )
                summary_text = None
            if summary_text:
                original_len = len(summary_text)
                if max_archive_chars > 0 and original_len > max_archive_chars:
                    summary_text = summary_text[:max_archive_chars] + "\n...(截断)"
                _log.info(
                    "注入归档摘要 [%s..] (%d 字符，截断前 %d)",
                    chat_id[:12], len(summary_text), original_len,
                )
                parts.append(
                    "以下内容来自过去几天的对话记录，"
                    "帮助你了解之前聊过什么：\n" + summary_text
                )

        if not parts:
            return None
        return "\n\n".join(parts)

    async def build_memory_context(
        self,
        sender_id: str,
        input_message: InputMessage,
    ) -> str:
        if not self.hindsight:
            return ""

        query = input_message.content.strip()
        if not query:
            return ""

        try:
            result = await self.hindsight.search(
                user_id=sender_id,
                query=query,
                top_k=self._search_top_k,
                include_profile=True,
            )

            episodes = result.get("episodes", [])
            profiles = result.get("profiles", [])

            if not episodes and not profiles:
                return ""

            parts = ["--- 相关记忆 ---"]

            if profiles:
                for p in profiles[:1]:
                    pd = p.get("profile_data", {})
                    if isinstance(pd, dict):
                        for k, v in pd.items():
                            if isinstance(v, str) and _is_dirty(v):
                                continue
                            parts.append(f"- [{k}]: {str(v)[:150]}")

            if episodes:
                count = 0
                for e in episodes:
                    if count >= 3:
                        break
                    summary = (
                        e.get("summary", "") or e.get("episode", "")
                    ).strip()
                    if not summary:
                        continue
                    if _is_dirty(summary):
                        continue
                    parts.append(f"- {summary[:150]}")
                    count += 1

            if len(parts) == 1:
                return ""

            parts.append("--- 相关记忆结束 ---")

            _log.info(
                f"自动记忆注入: sender={sender_id[:16]}.. "
                f"注入{len(episodes)}条经历, {len(profiles)}条画像"
            )
            return "\n".join(parts)
        except Exception as e:
            _log.warning(f"自动记忆注入失败: {e!r}")
            return ""
