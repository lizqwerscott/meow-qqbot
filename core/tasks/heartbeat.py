"""HeartbeatManager — 心跳管理器。

重构后职责：
1. 维护 interval 定时器，周期性调用 wake_coalescer.request_wake()
2. 提供 HEARTBEAT.md 内容加载 + task 解析
3. 提供 HeartbeatDeliveryStrategy 的 suppression 状态

不再持有调度循环本身（WakeCoalescer + WakeRunner 负责）。
不再引用 WakeDispatcher（改为直接使用 wake_coalescer module）。
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import core.tasks.wake_coalescer as _coalescer

_log = logging.getLogger(__name__)

# ── HeartbeatTask 解析 ──


@dataclass
class HeartbeatTask:
    name: str = ""
    interval_seconds: int = 3600
    prompt: str = ""
    command: str = ""


def _yaml_get_key_ci(data: dict, key: str) -> Any:
    """从 YAML dict 中大小写不敏感地获取 key。"""
    for k, v in data.items():
        if k.lower() == key.lower():
            return v
    return None


def _find_front_matter(content: str) -> tuple[str, Optional[str]]:
    """提取 YAML front matter（--- 分隔块）。

    返回 (去除 front matter 后的纯文本, YAML 文本或 None)
    """
    if not content.startswith("---"):
        return content.strip(), None
    m = re.search(r"\n---[ \t]*$", content[3:], re.MULTILINE)
    if not m:
        return content.strip(), None
    end = 3 + m.start() + 1
    yaml_text = content[3:end].strip()
    clean = content[end + 3 :].strip()
    return clean, yaml_text


def _find_tasks_yaml(content: str) -> tuple[str, Optional[str]]:
    """HEARTBEAT.md 中寻找 YAML tasks 块。

    支持两种格式（按优先级）:
    1. YAML front matter (--- ... ---) — 推荐
    2. 行内 tasks: 块 — 向后兼容

    返回 (去除 YAML 后的纯文本, 解析后的 YAML 文本或 None)
    """
    # 优先 front matter
    front_clean, front_yaml = _find_front_matter(content)
    if front_yaml:
        return front_clean, front_yaml
    # 回退到行内 tasks: 块
    m = re.search(
        r"^(?:#+\s*)?tasks\s*:",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return content.strip(), None
    clean = content[: m.start()].strip()
    raw_yaml = content[m.start() :]
    lines = raw_yaml.split("\n")
    if lines and lines[0].lstrip().startswith("#"):
        lines[0] = re.sub(r"^#+\s*", "", lines[0])
    yaml_text = "\n".join(lines).strip()
    return clean, yaml_text


def parse_heartbeat_tasks(content: str) -> list[HeartbeatTask]:
    """从 HEARTBEAT.md 中解析 YAML 格式的 tasks 块。

    格式示例:
        tasks:
          - name: check_morning
            interval_seconds: 3600
            prompt: 检查早间状态
          - name: check_evening
            interval_seconds: 7200
            command: status
    """
    _, yaml_text = _find_tasks_yaml(content)
    if not yaml_text:
        return []
    import yaml

    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return []
        items = _yaml_get_key_ci(data, "tasks")
        if not isinstance(items, list):
            return []
        result = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "")
            if not name:
                continue
            hb = HeartbeatTask(name=name)
            hb.interval_seconds = int(
                item.get("interval_seconds", item.get("interval", 3600))
            )
            hb.prompt = str(item.get("prompt", "") or "")
            hb.command = str(item.get("command", "") or "")
            result.append(hb)
        return result
    except Exception as e:
        _log.warning("HEARTBEAT.md YAML tasks 解析失败: %s", e)
        return []


def filter_due_tasks(
    tasks: list[HeartbeatTask], last_run: dict[str, float]
) -> list[HeartbeatTask]:
    now = time.time()
    return [t for t in tasks if now - last_run.get(t.name, 0) >= t.interval_seconds]


# ── HeartbeatManager ──


class HeartbeatManager:
    def __init__(
        self,
        config: dict,
        api_client: Any = None,
        admin_ids: Optional[list] = None,
        context_manager: Any = None,
        agent_engine: Any = None,
        wake_dispatcher: Any = None,  # 保留兼容，内部不再使用
        heartbeat_path: str = "",
        cooldown: Any = None,  # HeartbeatCooldown 实例（用于设置 next_due_ms）
    ):
        self._config = config
        self._cooldown = cooldown
        self._enabled = config.get("enabled", False)
        self._every = config.get("every", 30)
        self._isolated_session = config.get("isolated_session", True)
        self._admin_ids = admin_ids if isinstance(admin_ids, list) else []
        self._api = api_client
        self._context_manager = context_manager
        self._agent_engine = agent_engine

        self._heartbeat_path = heartbeat_path
        self._config_prompt = config.get("prompt", "")

        # 通知抑制状态（供 HeartbeatDeliveryStrategy 读取）
        self._last_text: str = ""
        self._last_sent: float = 0.0
        self._cooldown_hours: float = config.get("notification_cooldown_hours", 12.0)

        # 定时器
        self._running = False
        self._interval_task: Optional[asyncio.Task] = None

        # 任务定时追踪
        self._task_last_run: dict[str, float] = {}

        # pending delivery 状态（供 preflight deferral 使用）
        self._last_delivery_started_at: float = 0.0

    def record_delivery_start(self) -> None:
        self._last_delivery_started_at = time.time()

    def is_delivery_pending(self, window_ms: int = 30_000) -> bool:
        if self._last_delivery_started_at <= 0:
            return False
        return (time.time() - self._last_delivery_started_at) * 1000 < window_ms

    def resolve_isolated_session_key(self, base_key: str) -> str:
        """折叠 :heartbeat 链，返回真正的隔离 session key。"""
        collapsed = re.sub(r"(:heartbeat)+$", "", base_key)
        return f"{collapsed}:heartbeat"

    async def start(self):
        if not self._enabled:
            _log.info("心跳系统未启用")
            return
        if self._running:
            return
        self._running = True
        self._interval_task = asyncio.create_task(self._interval_loop())
        _log.info(
            f"心跳已启动: every={self._every}min "
            f"admin={self._admin_ids[0][:12] if self._admin_ids else 'none'}"
        )

    async def stop(self):
        self._running = False
        if self._interval_task and not self._interval_task.done():
            self._interval_task.cancel()
            try:
                await self._interval_task
            except asyncio.CancelledError:
                pass

    # ── Interval loop（相位对齐 + 活跃时段 seek） ──

    async def _interval_loop(self):
        from .heartbeat_schedule import (
            compute_next_phase_due_ms,
            is_in_active_hours_ts,
            resolve_phase_ms,
            seek_next_active_phase,
        )

        interval_ms = self._every * 60 * 1000
        # 持久化 seed：优先配置项，其次机器标识（重启不变），最后 fallback
        stable_seed = str(
            hashlib.sha256(
                (__import__("platform").node() or "default").encode()
            ).hexdigest()
        )
        seed = self._config.get("scheduler_seed", stable_seed)
        phase_ms = resolve_phase_ms(seed, "default", interval_ms)
        ah = self._config.get("active_hours", {})
        ah_start = ah.get("start")
        ah_end = ah.get("end")
        ah_tz = ah.get("timezone", "Asia/Shanghai")

        def is_active(ts_ms: float) -> bool:
            return is_in_active_hours_ts(ts_ms / 1000, ah_start, ah_end, ah_tz)

        cycle_count = 0
        while self._running:
            now_ms = time.time() * 1000
            next_due_ms = compute_next_phase_due_ms(now_ms, interval_ms, phase_ms)
            actual_ms = seek_next_active_phase(
                next_due_ms, interval_ms, phase_ms, is_active
            )
            delay = max(0, (actual_ms - time.time() * 1000) / 1000)
            # 将实际触发时间传给 cooldown，使 scheduled wake 的 nextDueMs 检查生效
            if self._cooldown:
                self._cooldown.set_next_due(actual_ms)
            await asyncio.sleep(delay)
            if not self._running:
                break
            prompt = await self._load_heartbeat_content()
            _coalescer.request_wake(
                source="interval",
                intent="scheduled",
                session_key="heartbeat:events",
                reason="定时心跳",
                extra_prompt=prompt,
                coalesce_ms=100,
            )
            cycle_count += 1
            if cycle_count % 10 == 0:
                await self._cleanup_heartbeat_contexts(keep_last=3)

    # ── 手动触发 ──

    async def trigger_heartbeat(self, prompt: str = "") -> tuple[bool, Optional[str]]:
        """手动触发心跳。返回 (有通知, 通知文本)。"""
        full_prompt = prompt or await self._load_heartbeat_content()
        wr = await _coalescer.execute_immediate(
            source="manual",
            intent="manual",
            session_key="heartbeat:events",
            reason="manual-trigger",
            extra_prompt=full_prompt,
        )
        result = wr.result
        if result and result.should_notify:
            return True, result.notification_text or None
        return False, None

    # ── HEARTBEAT.md ──

    async def _load_heartbeat_content(self) -> str:
        content = await self._read_heartbeat_file()
        if content and content.strip():
            tasks = parse_heartbeat_tasks(content)
            clean, _ = _find_tasks_yaml(content)
            due = filter_due_tasks(tasks, self._task_last_run)
            for t in due:
                self._task_last_run[t.name] = time.time()
            if due:
                lines = "\n".join(f"- {t.name}: {t.prompt or t.command}" for t in due)
                return f"{clean}\n\n当前需要关注的子任务：\n{lines}"
            return clean
        if self._config_prompt and self._config_prompt.strip():
            return self._config_prompt.strip()
        return (
            "请进行心跳检查。检查记忆和任务系统中是否有待办事项、"
            "提醒或需要关注的事情。"
            "如果没有需要关注的事项，调用 heartbeat_respond(notify=false) 静默结束。"
        )

    async def _read_heartbeat_file(self) -> str:
        if not self._heartbeat_path:
            return ""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._sync_read)
        except Exception as e:
            _log.warning(f"读取 HEARTBEAT 文件失败: {e}")
            return ""

    def _sync_read(self) -> str:
        try:
            with open(self._heartbeat_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""
        except Exception as e:
            _log.warning(f"读取心跳文件失败: {e}")
            return ""

    # ── 通知抑制（供 HeartbeatDeliveryStrategy 调用） ──

    def should_suppress(self, text: str) -> bool:
        if not self._last_text or self._last_sent <= 0:
            return False
        if self._cooldown_hours <= 0:
            return False
        from .delivery_normalization import strip_heartbeat_token

        norm_prev, _ = strip_heartbeat_token(self._last_text, ack_max_chars=0)
        norm_cur, _ = strip_heartbeat_token(text, ack_max_chars=0)
        normalized_prev = re.sub(r"\s+", " ", norm_prev).strip()
        normalized_cur = re.sub(r"\s+", " ", norm_cur).strip()
        if normalized_cur != normalized_prev:
            return False
        return (time.time() - self._last_sent) < self._cooldown_hours * 3600

    def record_notification(self, text: str) -> None:
        self._last_text = text
        self._last_sent = time.time()

    async def deliver_to_admin(self, text: str) -> bool:
        if not self._running or not self._admin_ids or not self._api:
            return False
        admin_id = self._admin_ids[0]
        content = f"[❤️ 心跳提醒]\n{text}"
        try:
            await self._api.send_text("c2c", admin_id, content, reply_to=None)
        except Exception as e:
            _log.error(f"心跳投递失败: {e}")
            return False

        _log.info(f"心跳提醒已投递到管理员 {admin_id[:12]}..")
        if self._context_manager:
            try:
                await self._context_manager.add_assistant_message_async(
                    admin_id, content, f"hb_{int(time.time())}"
                )
            except Exception as e:
                _log.warning(f"心跳投递后的上下文记录失败: {e}")
        return True

    # ── 上下文清理 ──

    async def _cleanup_heartbeat_contexts(self, keep_last: int = 3):
        if not self._context_manager:
            return
        try:
            pattern = re.compile(
                r"^heartbeat:\d+(:heartbeat)?$|^heartbeat:events:heartbeat$"
            )
            hb_ids = [
                cid
                for cid in await self._context_manager.get_all_chat_ids_async()
                if pattern.match(cid)
            ]
            if len(hb_ids) <= keep_last:
                return

            def _sort_key(cid: str) -> int:
                ts_part = cid.split(":")[1]
                try:
                    return int(ts_part)
                except ValueError:
                    return 0

            hb_ids.sort(key=_sort_key)
            for cid in hb_ids[:-keep_last]:
                await self._context_manager.clear_chat_history_async(cid)
                await self._context_manager.remove_context_async(cid)
        except Exception as e:
            _log.warning(f"清理心跳上下文失败: {e}")
