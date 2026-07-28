import logging
from typing import Optional

_log = logging.getLogger(__name__)


class SocialBlockBuilder:
    """构建社交上下文动态块（Bot ID + 群友列表）。"""

    def __init__(self, nm, bot_id: str) -> None:
        self._nm = nm
        self._bot_id = bot_id

    async def build(
        self,
        *,
        chat_id: str,
        is_group: bool,
        has_users: bool,
    ) -> Optional[str]:
        parts = []

        if is_group and self._bot_id:
            parts.append(
                f"你的 ID: {self._bot_id}（群友 @ 你时显示为 @{self._bot_id}）"
            )

        if has_users and self._nm:
            try:
                user_lines = ["【群友列表】"]
                for uid, aliases in sorted(
                    self._nm.iter_users(), key=lambda x: "，".join(x[1])
                ):
                    alias_str = "，".join(aliases)
                    user_lines.append(f"- {uid}（{alias_str}）")
                if len(user_lines) > 1:
                    parts.append("\n".join(user_lines))
            except Exception as e:
                _log.warning(
                    "群友列表构建失败 [%s..]: %s", chat_id[:12], e
                )

        if not parts:
            return None
        return "\n\n".join(parts)
