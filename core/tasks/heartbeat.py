"""HeartbeatManager — 周期心跳（OpenClaw 风格工具调用流）。

独立 asyncio 循环，不创建 TaskRecord/CronJob。
让 AI 通过工具调用循环定期检查是否有需要关注的事项，有提醒则发给管理员。

AI 在工具循环中可以：
- read_file(HEARTBEAT.md) 读取心跳检查清单
- search_memory 搜索待办事项和提醒
- 检查工作区文件
- 最终通过 heartbeat_respond(notify, notification_text) 工具回应
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

_log = logging.getLogger(__name__)


class HeartbeatManager:
    """周期心跳管理器。

    不依赖 CronJobScheduler / TaskStore。
    独立 asyncio 循环，不创建 TaskRecord。
    使用 AgentEngine.execute_heartbeat() 走完整工具调用循环。

    Args:
        config: {
            "enabled": bool,
            "every": int,              # 分钟
            "active_hours": {
                "start": "09:00",
                "end": "24:00",
            },
            "skip_when_busy": bool,
            "busy_idle_minutes": int,
        }
        agent_engine: AgentEngine 实例（用于 execute_heartbeat 和 last_active_time）
        api_client: QQApiClient 实例（用于发消息）
        admin_ids: 管理员 ID 列表
    """

    def __init__(
        self,
        config: dict,
        router_model: Any = None,
        ai_service: Any = None,
        model_registry: Any = None,
        bot_id: str = "",
        admin_ids: Optional[list] = None,
        api_client: Any = None,
        agent_engine: Any = None,
        heartbeat_path: str = "",
    ):
        self._enabled = config.get("enabled", False)
        self._every_minutes = config.get("every", 30)

        raw_models = config.get("model", "")
        if isinstance(raw_models, str):
            self._model_names = [raw_models] if raw_models else []
        elif isinstance(raw_models, list):
            self._model_names = raw_models
        else:
            self._model_names = []

        self._heartbeat_path = heartbeat_path

        active_hours = config.get("active_hours", {})
        self._active_start = active_hours.get("start", "00:00")
        self._active_end = active_hours.get("end", "24:00")
        self._timezone_str = active_hours.get("timezone", "Asia/Shanghai")

        self._skip_when_busy = config.get("skip_when_busy", True)
        self._busy_idle_minutes = config.get("busy_idle_minutes", 10)

        self._router_model = router_model
        self._ai_service = ai_service
        self._model_registry = model_registry
        self._bot_id = bot_id
        self._admin_ids = admin_ids if isinstance(admin_ids, list) else []
        self._api = api_client
        self._agent_engine = agent_engine

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_lock = asyncio.Lock()
        self._last_heartbeat_time: float = 0.0

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
                elapsed = time.time() - self._last_heartbeat_time
                sleep_for = max(0, self._every_minutes * 60 - elapsed)
                await asyncio.sleep(sleep_for)
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                _log.error(f"心跳循环异常: {e}", exc_info=True)

    async def _tick(self):
        """一次自动心跳周期。"""
        if self._heartbeat_lock.locked():
            _log.debug("心跳还在执行，跳过本次")
            return

        async with self._heartbeat_lock:
            if not self._in_active_hours():
                _log.debug("心跳跳过：不在活跃时段")
                return
            if self._skip_when_busy and self._is_busy():
                _log.debug("心跳跳过：用户活跃中")
                return
            await self._run_heartbeat()

    async def trigger_heartbeat(self, prompt: str = "") -> tuple[bool, str | None]:
        """公开接口：手动触发心跳，带锁保护，重置计时器。

        Args:
            prompt: 自定义 prompt，为空时使用默认时间 prompt

        Returns:
            (should_notify, notification_text)
        """
        async with self._heartbeat_lock:
            return await self._run_heartbeat(prompt)

    async def _run_heartbeat(self, prompt: str = "") -> tuple[bool, str | None]:
        """内部执行心跳（调用方需持有 _heartbeat_lock）。"""
        if not self._agent_engine:
            _log.warning("心跳跳过：AgentEngine 未就绪")
            return False, None

        if not prompt:
            tz_name = "CST (UTC+8)"
            now_str = datetime.now(timezone(timedelta(hours=8))).strftime(
                "%Y-%m-%d %H:%M"
            )
            prompt = f"现在时间是 {now_str}（{tz_name}）。"

        should_notify, text = await self._agent_engine.execute_heartbeat(prompt)
        self._last_heartbeat_time = time.time()

        if should_notify and text:
            await self._deliver_to_admin(text)
            _log.info(f"心跳提醒已投递: text={text[:80]!r}")
        else:
            _log.debug("心跳无需投递")

        return should_notify, text

    def _in_active_hours(self) -> bool:
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
