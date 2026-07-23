"""SystemEventPrompt — 将系统事件队列格式化为此轮 wake 的 prompt。"""

import time
from typing import Any, Optional


def build_system_events_prompt(
    system_events: Any,
    session_key: str,
) -> Optional[str]:
    """Peek 指定 session 的系统事件，格式化为带时间戳的纯文本。

    返回 None 表示无事件。调用方需在 AI 执行成功后
    调用 system_events.consume_snapshot(session_key)。
    """
    if not system_events:
        return None

    events = system_events.peek_and_snapshot(session_key)
    if not events:
        return None

    lines: list[str] = []
    for e in events:
        ts_str = time.strftime("%H:%M:%S", time.localtime(e.ts))
        lines.append(f"- [{ts_str}] {e.text}")

    return "以下系统事件需要关注：\n" + "\n".join(lines)


def get_pending_event_count(
    system_events: Any,
    session_key: str,
) -> int:
    if not system_events:
        return 0
    return len(system_events.peek(session_key))
