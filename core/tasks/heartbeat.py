"""HeartbeatManager — 周期心跳（OpenClaw 风格工具调用流）。

独立 asyncio 循环，不创建 TaskRecord/CronJob。
让 AI 通过工具调用循环定期检查是否有需要关注的事项，有提醒则发给管理员。

HEARTBEAT.md 由框架预读后直接注入 user message（AI 不需要 tool call 读取）。
通知 DM 同时写入 context_manager，保证后续心跳能感知反馈链路。
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

_log = logging.getLogger(__name__)

HEARTBEAT_DEFAULT_PROMPT = (
    "请进行心跳检查。检查记忆和任务系统中是否有待办事项、提醒或需要关注的事情。"
    "如果没有需要关注的事项，调用 heartbeat_respond(notify=false) 静默结束。"
    "不需要汇报正常状态，只需要在确实需要提醒时才通知管理员。"
)


class HeartbeatManager:
    """周期心跳管理器。

    不依赖 CronJobScheduler / TaskStore。
    独立 asyncio 循环，不创建 TaskRecord。
    使用 AgentEngine.execute_heartbeat() 走完整工具调用循环。

    Args:
        config: heartbeat 配置段
        agent_engine: AgentEngine 实例
        api_client: QQApiClient 实例
        admin_ids: 管理员 ID 列表
        context_manager: ChatContextManager（用于记录通知 DM）
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
        context_manager: Any = None,
        system_events: Any = None,
        task_manager: Any = None,
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
        self._context_manager = context_manager
        self._system_events = system_events
        self._task_manager = task_manager

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_lock = asyncio.Lock()
        self._last_heartbeat_time: float = 0.0
        self._session = config.get("session", "isolated")
        self._system_prompt_mode = config.get("system_prompt", "minimal")
        self._config_prompt = config.get("prompt", "")
        self._last_notification_text: str = ""
        self._last_notification_sent_at: float = 0.0
        self._notification_cooldown_hours: float = config.get("notification_cooldown_hours", 12.0)

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

    async def _run_heartbeat(self, manual_prompt: str = "") -> tuple[bool, str | None]:
        """内部执行心跳（调用方需持有 _heartbeat_lock）。

        user message 组装优先级：
        manual_prompt（手动触发）>
        config.prompt >
        HEARTBEAT.md 文件内容 >
        HEARTBEAT_DEFAULT_PROMPT
        """
        if not self._agent_engine:
            _log.warning("心跳跳过：AgentEngine 未就绪")
            return False, None

        # 自动心跳：无待办事项则跳过
        if not manual_prompt:
            if not self._has_pending_events() and not self._has_pending_tasks():
                _log.debug("心跳跳过：无待办事项")
                return False, None

        hb_content = await self._load_heartbeat_content(manual_prompt)
        tz_name = "CST (UTC+8)"
        now_str = datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y-%m-%d %H:%M"
        )
        user_prompt = f"{hb_content}\n\n当前时间：{now_str}（{tz_name}）。"

        # 为本次心跳生成唯一 chat_id，与上下文存储解耦
        chat_id = f"heartbeat:{int(time.time())}"

        # 解析模型链（支持组名如 "cheap" 转完整 fallback 链）
        model_chain = None
        if self._model_registry and self._model_names:
            all_models = []
            for name in self._model_names:
                chain = self._model_registry.get_chain(name)
                if chain:
                    all_models.extend(chain)
            if all_models:
                model_chain = all_models

        try:
            should_notify, text = await self._agent_engine.execute_heartbeat(
                prompt=user_prompt,
                session=self._session,
                system_prompt_mode=self._system_prompt_mode,
                model_chain=model_chain,
                chat_id=chat_id,
            )
            self._last_heartbeat_time = time.time()

            if should_notify and text:
                if self._should_suppress(text):
                    _log.info("心跳抑制：与上次通知相同（冷却期内）")
                else:
                    await self._deliver_to_admin(text)
                    self._last_notification_text = text
                    self._last_notification_sent_at = time.time()
                    _log.info(f"心跳提醒已投递: text={text[:80]!r}")
            else:
                _log.debug("心跳无需投递")

            return should_notify, text
        finally:
            # 清理旧心跳上下文，防止 context_manager 膨胀
            await self._cleanup_old_heartbeat_contexts(keep_last=3)

    async def _load_heartbeat_content(self, manual_prompt: str = "") -> str:
        """加载心跳指令内容。
        优先级：manual_prompt > config.prompt > HEARTBEAT.md > HEARTBEAT_DEFAULT_PROMPT
        """
        if manual_prompt and manual_prompt.strip():
            return manual_prompt.strip()

        if self._config_prompt and self._config_prompt.strip():
            return self._config_prompt.strip()

        content = await self._read_heartbeat_file()
        if content and content.strip():
            return content.strip()

        return HEARTBEAT_DEFAULT_PROMPT

    async def _read_heartbeat_file(self) -> str:
        """异步读取 HEARTBEAT.md 文件内容。"""
        if not self._heartbeat_path:
            return ""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._sync_read_heartbeat_file)
        except Exception as e:
            _log.warning(f"异步读取 HEARTBEAT 文件失败: {e}")
            return ""

    def _sync_read_heartbeat_file(self) -> str:
        try:
            with open(self._heartbeat_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content.strip() if content.strip() else ""
        except FileNotFoundError:
            return ""
        except Exception as e:
            _log.warning(f"同步读取 HEARTBEAT 文件失败: {e}")
            return ""

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
        except Exception as e:
            _log.warning(f"活跃时间解析失败 [{self._active_start}-{self._active_end}]: {e}")
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

    def _has_pending_events(self) -> bool:
        return bool(self._system_events and self._system_events.has_events("heartbeat:events"))

    def _has_pending_tasks(self) -> bool:
        if not self._task_manager:
            return False
        from core.tasks.models import TaskStatus
        failed = self._task_manager.list_tasks(limit=1, status=TaskStatus.FAILED)
        if failed:
            return True
        running = self._task_manager.list_tasks(limit=1, status=TaskStatus.RUNNING)
        if running:
            return True
        return False

    def _should_suppress(self, text: str) -> bool:
        """检查是否应抑制本次通知（同文本 + 冷却期）。

        Args:
            text: 本次要通知的文本（来自 heartbeat_respond 的原始文本）

        Returns:
            True 表示抑制，不投递
        """
        if not self._last_notification_text or self._last_notification_sent_at <= 0:
            return False
        if text != self._last_notification_text:
            return False
        if self._notification_cooldown_hours <= 0:
            return False
        elapsed = time.time() - self._last_notification_sent_at
        return elapsed < self._notification_cooldown_hours * 3600

    async def _deliver_to_admin(self, text: str):
        """发送心跳提醒给第一个管理员，同时写入 context_manager 保证反馈链路。"""
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

            # 写入 context_manager，后续心跳（session=main）能感知反馈链路
            if self._context_manager:
                await self._context_manager.add_assistant_message_async(
                    admin_id, content, f"hb_{int(time.time())}"
                )
        except Exception as e:
            _log.error(f"心跳投递失败: {e}")

    async def _cleanup_old_heartbeat_contexts(self, keep_last: int = 3):
        """清理旧的心跳上下文，只保留最近 keep_last 次。

        匹配 pattern heartbeat:\\d+ 的 chat_id，按时间戳排序，
        移除超出 keep_last 的上下文。
        同时删除磁盘上的 .jsonl 缓存文件。
        """
        if not self._context_manager:
            return
        pattern = re.compile(r"^heartbeat:\d+$")
        heartbeat_ids = [
            cid for cid in self._context_manager.get_all_chat_ids()
            if pattern.match(cid)
        ]
        if len(heartbeat_ids) <= keep_last:
            return
        heartbeat_ids.sort(key=lambda x: int(x.split(":")[1]))
        for cid in heartbeat_ids[:-keep_last]:
            try:
                await self._context_manager.clear_chat_history_async(cid)
                self._context_manager.remove_context(cid)
                _log.debug("已清理旧心跳上下文: %s", cid)
            except Exception as e:
                _log.warning("清理心跳上下文失败 [%s]: %s", cid, e)
