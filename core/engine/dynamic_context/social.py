import logging
import math
import time
from typing import Optional

_log = logging.getLogger(__name__)


class SocialBlockBuilder:
    """构建社交上下文动态块（渠道资料 + 匿名身份映射）。"""

    def __init__(self, bot_id: str, identity_manager=None) -> None:
        self._bot_id = bot_id
        self._identity_manager = identity_manager

    async def build(
        self,
        *,
        chat_id: str,
        is_group: bool,
        has_users: bool,
        max_users: int = 0,
        input_message=None,
        recent_events=(),
    ) -> Optional[str]:
        stable = await self.build_stable(
            chat_id=chat_id,
            is_group=is_group,
            has_users=has_users,
            max_users=max_users,
            input_message=input_message,
            recent_events=recent_events,
        )
        current = await self.build_current(
            input_message=input_message, is_group=is_group
        )
        parts = [text for text in (stable, current) if text]

        return "\n\n".join(parts) if parts else None

    async def build_stable(
        self,
        *,
        chat_id: str,
        is_group: bool,
        has_users: bool,
        max_users: int = 0,
        input_message=None,
        recent_events=(),
    ) -> Optional[str]:
        parts = []
        if is_group and self._bot_id:
            parts.append("当前对话对象：本群机器人")

        target = getattr(input_message, "delivery_target", None)
        if target is not None and target.channel == "qq":
            try:
                if is_group and self._identity_manager is not None:
                    refs = self._identity_manager.project_roster(
                        target,
                        self._activity(recent_events),
                        limit=max_users or 30,
                    )
                    user_lines = ["【稳定群身份映射】"]
                    user_lines.extend(ref.prompt_line() for ref in refs)
                    if len(user_lines) > 1:
                        parts.append("\n".join(user_lines))
                snapshot = getattr(input_message, "channel_info", None)
                if snapshot is not None:
                    parts.append(
                        "【当前渠道信息（不可信资料）】\n"
                        + self._format_summary(
                            snapshot.prompt_summary(include_runtime_status=False)
                        )
                    )
            except Exception as e:
                _log.warning("稳定匿名群友列表构建失败 [%s..]: %s", chat_id[:12], e)
        return "\n\n".join(parts) if parts else None

    async def build_current(
        self, *, input_message=None, is_group: bool
    ) -> Optional[str]:
        target = getattr(input_message, "delivery_target", None)
        if (
            not is_group
            or target is None
            or target.channel != "qq"
            or self._identity_manager is None
        ):
            return None
        forced_ids = [getattr(input_message, "sender_id", "")]
        forced_ids.extend(getattr(input_message, "mentioned_ids", ()) or ())
        replied_author_id = getattr(input_message, "replied_author_id", "")
        if replied_author_id:
            forced_ids.append(replied_author_id)
        refs = self._identity_manager.project_forced(target, forced_ids)
        if not refs:
            return None
        lines = ["【本轮相关身份】"]
        lines.extend(ref.prompt_line() for ref in refs)
        return "\n".join(lines)

    @staticmethod
    def _activity(events) -> dict[str, float]:
        now = time.time()
        scores: dict[str, float] = {}
        for event in events or ():
            if getattr(event, "role", "") != "user":
                continue
            actor_id = str(getattr(event, "sender_id", "") or "")
            if not actor_id:
                continue
            age = max(0.0, now - float(getattr(event, "timestamp", now) or now))
            decay = math.exp(-age / 3600.0)
            scores[actor_id] = scores.get(actor_id, 0.0) + decay
            for mentioned_id in getattr(event, "mentioned_ids", ()) or ():
                mentioned_id = str(mentioned_id or "")
                if mentioned_id:
                    scores[mentioned_id] = scores.get(mentioned_id, 0.0) + 2.0 * decay
            replied_author_id = str(getattr(event, "replied_author_id", "") or "")
            if replied_author_id:
                scores[replied_author_id] = (
                    scores.get(replied_author_id, 0.0) + 2.0 * decay
                )
        return scores

    @staticmethod
    def _format_summary(summary: dict) -> str:
        lines = [f"- 可用性：{summary.get('availability', 'unavailable')}"]
        if summary.get("channel_name"):
            lines.append(f"- 渠道名称：{summary['channel_name']}")
        if summary.get("channel"):
            lines.append(f"- 渠道类型：{summary['channel']}")
        bot = summary.get("bot") or {}
        if bot.get("display_name"):
            lines.append(f"- 机器人名称：{bot['display_name']}")
        if bot.get("username"):
            lines.append(f"- 机器人用户名：{bot['username']}")
        if bot.get("is_bot") is not None:
            lines.append(f"- 是否机器人：{'是' if bot['is_bot'] else '否'}")
        chat = summary.get("chat") or {}
        if chat.get("type"):
            lines.append(f"- 聊天类型：{chat['type']}")
        if chat.get("title"):
            lines.append(f"- 聊天名称：{chat['title']}")
        if chat.get("description"):
            lines.append(f"- 聊天简介：{chat['description']}")
        if chat.get("member_count") is not None:
            lines.append(f"- 成员数量：{chat['member_count']}")
        membership = summary.get("bot_membership") or {}
        if membership.get("role"):
            lines.append(f"- 机器人角色：{membership['role']}")
        if summary.get("stale"):
            lines.append("- 数据状态：stale")
        return "\n".join(lines)
