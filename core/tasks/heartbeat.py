"""HeartbeatManager — 周期心跳。

独立 asyncio 循环，不创建 TaskRecord/CronJob。
让 AI 定期检查是否有需要关注的事项，有提醒则发给管理员。
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

_log = logging.getLogger(__name__)

HEARTBEAT_DEFAULT_PROMPT = """现在是 {time}，{tz}。

你是一个群聊助手的心跳检查。
请检查是否有需要关注的事项：
- 有没有什么需要提醒的？
- 群里的气氛或状态有没有异常？
- 有没有待办的任务超时了？

如果没有需要关注的事项，只回复 HEARTBEAT_OK。
如果有需要提醒的，简短说明即可，不要超过 100 字。"""


class HeartbeatManager:
    """周期心跳管理器。

    不依赖 CronJobScheduler / TaskStore。
    独立 asyncio 循环，不创建 TaskRecord。

    Args:
        config: {
            "enabled": bool,
            "every": int,              # 分钟
            "use_router_model": bool,
            "prompt": str (optional),
            "active_hours": {
                "start": "09:00",
                "end": "24:00",
            },
            "skip_when_busy": bool,
            "busy_idle_minutes": int,
        }
        router_model: RouterModel 实例（use_router_model=True 时需要）
        bot_id: 机器人 ID
        admin_ids: 管理员 ID 列表
        api_client: QQApiClient 实例（用于发消息）
        agent_engine: AgentEngine 实例（用于读取 last_active_time）
    """

    def __init__(
        self,
        config: dict,
        router_model: Any,
        model_registry: Any = None,
        bot_id: str = "",
        admin_ids: Optional[list] = None,
        api_client: Any = None,
        agent_engine: Any = None,
    ):
        self._enabled = config.get("enabled", False)
        self._every_minutes = config.get("every", 30)
        self._model_name = config.get("model", "")
        self._custom_prompt = config.get("prompt", "")

        active_hours = config.get("active_hours", {})
        self._active_start = active_hours.get("start", "00:00")
        self._active_end = active_hours.get("end", "24:00")
        self._timezone_str = active_hours.get("timezone", "Asia/Shanghai")

        self._skip_when_busy = config.get("skip_when_busy", True)
        self._busy_idle_minutes = config.get("busy_idle_minutes", 10)

        self._router_model = router_model
        self._model_registry = model_registry
        self._bot_id = bot_id
        self._admin_ids = admin_ids if isinstance(admin_ids, list) else []
        self._api = api_client
        self._agent_engine = agent_engine

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_lock = asyncio.Lock()

    async def start(self):
        if not self._enabled:
            _log.info("心跳系统未启用")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        _log.info(
            f"心跳系统已启动: every={self._every_minutes}min "
            f"active={self._active_start}-{self._active_end} "
            f"admin={self._admin_ids[0] if self._admin_ids else 'none'}"
        )

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        _log.info("心跳系统已停止")

    async def _loop(self):
        while self._running:
            try:
                await asyncio.sleep(self._every_minutes * 60)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error(f"心跳循环异常: {e}", exc_info=True)

    async def _tick(self):
        """一次心跳周期。"""
        # 防止重叠
        if self._heartbeat_lock.locked():
            _log.debug("心跳还在执行，跳过本次")
            return

        async with self._heartbeat_lock:
            # 1. 活跃时段检查
            if not self._in_active_hours():
                _log.debug("心跳跳过：不在活跃时段")
                return

            # 2. 忙碌检查
            if self._skip_when_busy and self._is_busy():
                _log.debug("心跳跳过：用户活跃中")
                return

            # 3. 执行心跳
            await self._do_heartbeat()

    def _in_active_hours(self) -> bool:
        """检查当前是否在活跃时段内。"""
        try:
            tz = timezone(timedelta(hours=8))
            now = datetime.now(tz)
            cur = now.hour * 60 + now.minute

            start_parts = self._active_start.split(":")
            end_parts = self._active_end.split(":")
            start_min = int(start_parts[0]) * 60 + int(start_parts[1])
            end_min = int(end_parts[0]) * 60 + int(end_parts[1])

            if end_min <= start_min:
                # 跨天（如 22:00 ~ 02:00）
                return cur >= start_min or cur < end_min
            return start_min <= cur < end_min
        except Exception:
            return True  # 解析失败默认放行

    def _is_busy(self) -> bool:
        """检查是否有用户最近活跃。"""
        if not self._agent_engine:
            return False
        last_time = getattr(self._agent_engine, "last_active_time", 0.0)
        if last_time <= 0:
            return False
        elapsed = time.time() - last_time
        return elapsed < self._busy_idle_minutes * 60

    async def _do_heartbeat(self):
        """执行心跳：调用模型 → 判断 → 投递。"""
        # 构造 prompt
        tz_name = "CST (UTC+8)"
        now_str = datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y-%m-%d %H:%M"
        )
        if self._custom_prompt:
            prompt = self._custom_prompt.format(time=now_str, tz=tz_name)
        else:
            prompt = HEARTBEAT_DEFAULT_PROMPT.format(time=now_str, tz=tz_name)

        # 调用模型
        result = ""
        if self._model_registry and self._model_name:
            result = await self._model_registry.simple_chat(
                model_name=self._model_name,
                messages=[
                    {"role": "system", "content": "你是一个群聊助手的心跳检查器。请检查是否有需要关注的事项。\n\n如果没有，只回复 HEARTBEAT_OK。如果有提醒，简短说明，不超过 100 字。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
        elif self._router_model:
            result = await self._router_model.simple_chat(prompt)
        else:
            _log.warning("心跳跳过：无可用模型")
            return

        if not result:
            _log.debug("心跳无返回")
            return

        # HEARTBEAT_OK 检测
        stripped = result.strip()
        if stripped == "HEARTBEAT_OK":
            _log.debug("心跳 HEARTBEAT_OK，静默")
            return
        # 如果开头或结尾是 HEARTBEAT_OK，移除后检查剩余
        if stripped.startswith("HEARTBEAT_OK"):
            stripped = stripped[len("HEARTBEAT_OK"):].strip()
        if stripped.endswith("HEARTBEAT_OK"):
            stripped = stripped[:-len("HEARTBEAT_OK")].strip()
        if not stripped or len(stripped) < 5:
            _log.debug("心跳内容过短（HEARTBEAT_OK 变体），静默")
            return

        # 投递到管理员
        await self._deliver_to_admin(stripped)

    async def _deliver_to_admin(self, text: str):
        """发送心跳提醒给第一个管理员。"""
        if not self._admin_ids:
            _log.warning("心跳无法投递：未配置管理员 ID")
            return
        if not self._api:
            _log.warning("心跳无法投递：API 客户端未就绪")
            return

        admin_id = self._admin_ids[0]
        content = f"[❤️ 心跳提醒]\n{text[:500]}"
        try:
            await self._api.send_text("c2c", admin_id, content, reply_to=None)
            _log.info(f"心跳提醒已投递到管理员 {admin_id[:12]}..")
        except Exception as e:
            _log.error(f"心跳投递失败: {e}")
