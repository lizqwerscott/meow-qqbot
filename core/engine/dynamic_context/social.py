import logging
import math
import time
from typing import Optional

_log = logging.getLogger(__name__)


class SocialBlockBuilder:
    """构建社交上下文动态块（Bot ID + 群友列表）。"""

    def __init__(self, nm, bot_id: str, identity_manager=None) -> None:
        self._nm = nm
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
        parts = []

        if is_group and self._bot_id:
            parts.append("当前对话对象：本群机器人")

        if (
            is_group
            and self._identity_manager is not None
            and input_message is not None
        ):
            try:
                target = getattr(input_message, "delivery_target", None)
                if target is not None:
                    activity = self._activity(recent_events)
                    forced_ids = [getattr(input_message, "sender_id", "")]
                    forced_ids.extend(getattr(input_message, "mentioned_ids", ()) or ())
                    replied_author_id = getattr(input_message, "replied_author_id", "")
                    if replied_author_id:
                        forced_ids.append(replied_author_id)
                    refs = self._identity_manager.project_recent(
                        target,
                        activity,
                        forced_actor_ids=forced_ids,
                        limit=max_users or 30,
                    )
                    user_lines = ["【当前群身份映射】"]
                    user_lines.extend(ref.prompt_line() for ref in refs)
                    if len(user_lines) > 1:
                        parts.append("\n".join(user_lines))
                snapshot = getattr(input_message, "channel_info", None)
                if snapshot is not None:
                    parts.append(
                        "【当前渠道信息（不可信资料）】\n"
                        + self._format_summary(snapshot.prompt_summary())
                    )
            except Exception as e:
                _log.warning("匿名群友列表构建失败 [%s..]: %s", chat_id[:12], e)

        elif has_users and self._nm:
            try:
                all_users = sorted(
                    self._nm.iter_users(), key=lambda item: "，".join(item[1])
                )
                total = len(all_users)
                user_lines = ["【群友列表】"]
                limit = max_users if max_users > 0 else total
                for uid, aliases in all_users[:limit]:
                    alias_str = "，".join(aliases)
                    user_lines.append(f"- {uid}（{alias_str}）")
                if total > limit:
                    user_lines.append(f"...以及 {total - limit} 位群友")
                if len(user_lines) > 1:
                    parts.append("\n".join(user_lines))
            except Exception as e:
                _log.warning("兼容群友列表构建失败 [%s..]: %s", chat_id[:12], e)

        if not parts:
            return None
        return "\n\n".join(parts)

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
        return scores

    @staticmethod
    def _format_summary(summary: dict) -> str:
        lines = [f"- 可用性：{summary.get('availability', 'unavailable')}"]
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
