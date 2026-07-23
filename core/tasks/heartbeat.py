"""HeartbeatManager — 心跳投递管理器。

不负责调度和 AI 执行（由 WakeDispatcher 统一处理），仅负责：
- 注册 interval schedule
- 接收执行结果并决定是否投递给管理员
- 通知抑制（相同文本冷却期）、上下文清理
"""

import asyncio
import logging
import re
import time
from typing import Any, Optional

from core.engine.wake_dispatcher import WakeDispatcher, SOURCE_INTERVAL, SOURCE_MANUAL

_log = logging.getLogger(__name__)


class HeartbeatManager:
    """心跳投递管理器。

    不再包含调度循环，所有 wake 请求通过 WakeDispatcher 统一处理。
    """

    def __init__(
        self,
        config: dict,
        api_client: Any = None,
        admin_ids: Optional[list] = None,
        context_manager: Any = None,
        agent_engine: Any = None,
        wake_dispatcher: Optional[WakeDispatcher] = None,
        heartbeat_path: str = "",
    ):
        self._enabled = config.get("enabled", False)
        self._every_minutes = config.get("every", 30)
        self._admin_ids = admin_ids if isinstance(admin_ids, list) else []
        self._api = api_client
        self._context_manager = context_manager
        self._agent_engine = agent_engine
        self._wake = wake_dispatcher

        self._running = False
        self._task: Optional[asyncio.Task] = None

        self._last_notification_text: str = ""
        self._last_notification_sent_at: float = 0.0
        self._notification_cooldown_hours: float = config.get("notification_cooldown_hours", 12.0)

        self._heartbeat_path = heartbeat_path
        self._config_prompt = config.get("prompt", "")

    async def start(self):
        if not self._enabled:
            _log.info("心跳系统未启用")
            return
        if self._running:
            return
        self._running = True

        hb_prompt = await self._load_heartbeat_content()

        # 注册 delivery callback
        if self._wake:
            self._wake.set_delivery_callback(SOURCE_MANUAL, self._on_heartbeat_result)
            self._wake.set_delivery_callback(SOURCE_INTERVAL, self._on_heartbeat_result)
            await self._wake.start_interval(self._every_minutes, prompt=hb_prompt)

        _log.info(
            f"心跳已启动: every={self._every_minutes}min "
            f"admin={self._admin_ids[0] if self._admin_ids else 'none'}"
        )

    async def stop(self):
        self._running = False
        if self._wake:
            await self._wake.stop_interval()

    async def trigger_heartbeat(self, prompt: str = "") -> tuple[bool, str | None]:
        """手动触发心跳，直接走 WakeDispatcher.request(MANUAL)。"""
        full_prompt = prompt or await self._load_heartbeat_content()
        result = None
        if self._wake:
            result = await self._wake.request(
                source=SOURCE_MANUAL, intent="manual",
                session_key="heartbeat:events",
                reason="manual-trigger",
                extra_prompt=full_prompt,
                coalesce_ms=0,
            )
        if result and result.should_notify:
            return True, result.notification_text or None
        return False, None

    async def _on_heartbeat_result(self, result: Any) -> None:
        """从 WakeDispatcher 接收执行结果，决定是否投递。"""
        from core.engine.wake_dispatcher import WakeResult
        if not isinstance(result, WakeResult):
            _log.warning("_on_heartbeat_result 收到非 WakeResult 类型: %s", type(result))
            return
        if not result.should_notify or not result.notification_text:
            return
        if not self._running:
            return
        text = result.notification_text.strip()

        if self._should_suppress(text):
            _log.debug("心跳抑制：与上次通知相同（冷却期内）")
            return

        await self._deliver_to_admin(text)
        self._last_notification_text = text
        self._last_notification_sent_at = time.time()

        # 清理旧心跳 context，防止内存泄漏
        await self._cleanup_heartbeat_contexts(keep_last=3)

    # ── HEARTBEAT.md 内容加载 ──

    async def _load_heartbeat_content(self) -> str:
        content = await self._read_heartbeat_file()
        if content and content.strip():
            return content.strip()
        if self._config_prompt and self._config_prompt.strip():
            return self._config_prompt.strip()
        return (
            "请进行心跳检查。检查记忆和任务系统中是否有待办事项、提醒或需要关注的事情。"
            "如果没有需要关注的事项，调用 heartbeat_respond(notify=false) 静默结束。"
        )

    async def _read_heartbeat_file(self) -> str:
        if not self._heartbeat_path:
            return ""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._sync_read_heartbeat_file)
        except Exception as e:
            _log.warning(f"读取 HEARTBEAT 文件失败: {e}")
            return ""

    def _sync_read_heartbeat_file(self) -> str:
        import os
        try:
            with open(self._heartbeat_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""
        except Exception as e:
            _log.warning(f"读取心跳文件失败: {e}")
            return ""

    # ── 通知抑制 ──

    def _should_suppress(self, text: str) -> bool:
        if not self._last_notification_text or self._last_notification_sent_at <= 0:
            return False
        if self._notification_cooldown_hours <= 0:
            return False
        normalized_prev = re.sub(r"\s+", " ", self._last_notification_text).strip()
        normalized_cur = re.sub(r"\s+", " ", text).strip()
        if normalized_cur != normalized_prev:
            return False
        elapsed = time.time() - self._last_notification_sent_at
        return elapsed < self._notification_cooldown_hours * 3600

    # ── 投递 ──

    async def _deliver_to_admin(self, text: str):
        if not self._running:
            return
        if not self._admin_ids:
            _log.warning("心跳无法投递：未配置管理员 ID")
            return
        if not self._api:
            _log.warning("心跳无法投递：API 客户端未就绪")
            return

        admin_id = self._admin_ids[0]
        content = f"[❤️ 心跳提醒]\n{text}"
        try:
            await self._api.send_text("c2c", admin_id, content, reply_to=None)
            _log.info(f"心跳提醒已投递到管理员 {admin_id[:12]}..")
            if self._context_manager:
                await self._context_manager.add_assistant_message_async(
                    admin_id, content, f"hb_{int(time.time())}"
                )
        except Exception as e:
            _log.error(f"心跳投递失败: {e}")

    async def _cleanup_heartbeat_contexts(self, keep_last: int = 3):
        """清理旧的心跳上下文，只保留最近 keep_last 次。

        匹配 pattern heartbeat:\\d+ 的 chat_id，按时间戳排序，
        移除超出 keep_last 的上下文。
        """
        if not self._context_manager:
            return
        try:
            import re
            pattern = re.compile(r"^heartbeat:\d+$")
            hb_ids = [
                cid for cid in self._context_manager.get_all_chat_ids()
                if pattern.match(cid)
            ]
            if len(hb_ids) <= keep_last:
                return
            hb_ids.sort(key=lambda x: int(x.split(":")[1]))
            for cid in hb_ids[:-keep_last]:
                await self._context_manager.clear_chat_history_async(cid)
                self._context_manager.remove_context(cid)
                _log.debug("已清理旧心跳上下文: %s", cid)
        except Exception as e:
            _log.warning("清理心跳上下文失败: %s", e)
