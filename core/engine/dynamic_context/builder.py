import logging
from typing import List, Optional

from core.message import InputMessage

from .emoji import EmojiBlockBuilder
from .memory import MemoryBlockBuilder
from .skill import SkillBlockBuilder
from .social import SocialBlockBuilder
from .system_event import SystemEventBlockBuilder
from .time import TimeBlockBuilder
from .workspace import WorkspaceBlockBuilder

_log = logging.getLogger(__name__)


class DynamicContextBuilder:
    """动态上下文编排器。

    收集所有动态块构建器的输出，组装为一条 system 消息内容。
    """

    def __init__(
        self,
        *,
        hindsight,
        search_top_k: int,
        learners,
        archive_manager,
        system_events,
        skill_managers,
        workspace_manager,
        perm,
        admin_ids,
        nm,
        bot_id: str,
        emoji_manager,
    ) -> None:
        self._memory = MemoryBlockBuilder(
            hindsight=hindsight,
            search_top_k=search_top_k,
            learners=learners,
            archive_manager=archive_manager,
        )
        self._sys_evt = SystemEventBlockBuilder(system_events)
        self._skill = SkillBlockBuilder(skill_managers)
        self._time = TimeBlockBuilder()
        self._workspace = WorkspaceBlockBuilder(workspace_manager, perm, admin_ids)
        self._social = SocialBlockBuilder(nm, bot_id)
        self._emoji = EmojiBlockBuilder(emoji_manager)

    @property
    def memory_builder(self) -> MemoryBlockBuilder:
        return self._memory

    async def build(
        self,
        *,
        chat_id: str,
        is_group: bool,
        sender_id: str,
        input_message: InputMessage,
        has_emojis: bool,
        has_users: bool,
    ) -> Optional[str]:
        parts: List[str] = []

        # 系统事件
        txt = await self._sys_evt.build(chat_id=chat_id)
        if txt:
            parts.append(txt)

        # 记忆 + 学习 + 归档
        txt = await self._memory.build(
            chat_id=chat_id,
            sender_id=sender_id,
            input_message=input_message,
            max_archive_chars=3000,
        )
        if txt:
            parts.append(txt)

        # 当前时间
        parts.append(self._time.build())

        # 消息投递提示（始终添加）
        parts.append(
            "【消息投递】你的工具调用之间的文本正常展示给用户。"
            "如果需要在最终回复中使用 send_message 工具，"
            "send_message 投递后你的后续文本将不再自动发送。"
        )

        # 工作区 + HEARTBEAT.md
        txt = await self._workspace.build(
            chat_id=chat_id,
            is_group=is_group,
            sender_id=sender_id,
        )
        if txt:
            parts.append(txt)

        # 技能条目
        txt = await self._skill.build(max_skills=20, max_desc_chars=1000)
        if txt:
            parts.append(txt)

        # 表情标签
        txt = await self._emoji.build(has_emojis=has_emojis)
        if txt:
            parts.append(txt)

        # 社交上下文（Bot ID + 群友列表）
        txt = await self._social.build(
            chat_id=chat_id,
            is_group=is_group,
            has_users=has_users,
            max_users=30,
        )
        if txt:
            parts.append(txt)

        if not parts:
            return None

        text = "\n\n".join(parts)
        if len(text) > 8000:
            _log.warning(
                "动态 system prompt 较长: %d 字符", len(text),
            )
        return text
