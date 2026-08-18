import logging
import time
from typing import Optional

_log = logging.getLogger(__name__)


class SystemEventBlockBuilder:
    """构建系统事件动态块。"""

    def __init__(self, system_events) -> None:
        self._system_events = system_events

    async def build(self, *, chat_id: str) -> Optional[str]:
        if not self._system_events:
            return None
        events = self._system_events.peek(chat_id)
        if not events:
            return None
        lines = []
        for e in events:
            ts = time.strftime("%H:%M:%S", time.localtime(e.ts))
            lines.append(f"System: [{ts}] {e.text}")
        lines.append("")
        lines.append(
            "处理完成后，如果没有需要关注的事项，使用 heartbeat_respond(notify=false) "
            "或回复 NO_REPLY 静默结束。如果有需要通知用户的事项，"
            '使用 heartbeat_respond(notify=true, notification_text="...")。'
            "仅回复文本时，非 NO_REPLY 的内容会被转发给用户。"
        )
        return "【系统事件】\n" + "\n".join(lines)
