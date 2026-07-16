import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="心跳", aliases=["heartbeat", "hb"], permission="admin", description="手动触发心跳检查。可加额外检查指令")
class HeartbeatCommand:
    def __init__(self, agent_engine, heartbeat_manager):
        self._agent_engine = agent_engine
        self._heartbeat_manager = heartbeat_manager

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        from datetime import datetime, timezone, timedelta
        tz_name = "CST (UTC+8)"
        now_str = datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y-%m-%d %H:%M"
        )

        extra = args.strip()
        if extra:
            prompt = f"现在时间是 {now_str}（{tz_name}）。额外检查指令：{extra}"
        else:
            prompt = f"现在时间是 {now_str}（{tz_name}）。"

        _log.info(f"手动触发心跳: extra={extra or '(无)'}")

        if self._heartbeat_manager:
            should_notify, text = await self._heartbeat_manager.trigger_heartbeat(prompt)
        else:
            should_notify, text = await self._agent_engine.execute_heartbeat(prompt)

        if should_notify and text:
            return make_reply(input_message, f"[❤️ 心跳提醒]\n{text}")
        return make_reply(input_message, "✅ 心跳检查完成，无需关注。")
