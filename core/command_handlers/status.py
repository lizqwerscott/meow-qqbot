import logging
import time
from typing import Any, Dict, List, Optional

import psutil

from core.command_handlers.base import command, make_reply, session_key_for_message
from core.engine.agent_engine import AgentEngine
from core.message import InputMessage

_log = logging.getLogger(__name__)


def _hindsight_status_line(health: dict) -> str:
    status = health.get("status", "unknown")
    if status == "disabled":
        return "未启用 🚫"
    if status == "ok":
        latency = health.get("latency_ms")
        if latency is not None:
            return f"已连接 ✅ ({latency}ms)"
        return "已连接 ✅"
    if status == "unknown":
        return "待检查 ⏳"
    error = health.get("error", "未知错误")
    return f"不可达 ❌ ({error})"


@command(
    name="状态",
    aliases=["status"],
    permission="admin",
    description="查看系统状态（管理员专用）",
)
class StatusCommand:
    def __init__(
        self,
        agent_engine: AgentEngine,
        approval_manager=None,
        channel_info_provider=None,
        identity_manager=None,
        token_usage_log=None,
    ):
        self.agent_engine = agent_engine
        self.approval_manager = approval_manager  # 2.4：审批白名单状态行
        self.channel_info_provider = channel_info_provider
        self.identity_manager = identity_manager
        self.token_usage_log = token_usage_log

    def _usage_lines(self) -> List[str]:
        """Brief AI consumption from the durable ledger (falls back to memory)."""
        log = self.token_usage_log
        if log is None:
            cost_tracker = getattr(self.agent_engine, "cost_tracker", None)
            stats = cost_tracker.get_global_stats() if cost_tracker else None
            if stats is None or stats.turn_count == 0:
                return []
            return [
                "",
                "**AI 消耗**（内存统计，账本未启用；重启清零）",
                f"- API 调用: `{stats.turn_count}` 次",
                f"- 输入 `{stats.prompt_tokens:,}` / 输出 `{stats.completion_tokens:,}` tokens",
                f"- 总费用: **¥{stats.cost:.4f}**",
            ]
        try:
            total = log.summary()
            today_since = time.mktime(
                time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d")
            )
            today = log.summary(since=today_since)
        except Exception as exc:
            _log.warning("读取用量账本失败: %s", exc)
            return []
        if int(total.get("record_count", 0)) == 0:
            return ["", "**AI 消耗**（持久账本）", "- 暂无记录"]
        return [
            "",
            "**AI 消耗**（持久账本，跨重启累计）",
            f"- 累计: 调用 `{total['record_count']}` 次，输入 `{total['prompt_tokens']:,}`"
            f" / 输出 `{total['completion_tokens']:,}` tokens，费用 **¥{total['cost']:.4f}**",
            f"- 今日: 调用 `{today['record_count']}` 次，费用 **¥{today['cost']:.4f}**",
            "- 明细/图表: WebUI `/usage`；本会话历史: `猫猫消耗`",
        ]

    @staticmethod
    def _plugin_count() -> int:
        from core.plugins.manager import _current as _pm

        return _pm.count if _pm else 0

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage("/")
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)
            process_cpu = process.cpu_percent(interval=0.1)

            stats = await self.agent_engine.get_stats()
            queue_sizes = stats.get("queue_sizes", {})
            total_queue = sum(queue_sizes.values())
            active_chats = stats.get("active_chats", 0)
            hindsight_health = stats.get("hindsight_health", {})
            learner_stats = stats.get("learners", {})
            sm = self.agent_engine._skill_managers
            engagement_status = {}
            get_engagement_status = getattr(
                self.agent_engine, "get_engagement_status", None
            )
            if get_engagement_status is not None:
                engagement_status = await get_engagement_status()
            engagement_metrics = engagement_status.get("engagement", {})
            delivery_counts = engagement_status.get("delivery", {})
            history_integrity = {}
            get_ledger_integrity = getattr(
                self.agent_engine, "get_ledger_integrity_async", None
            )
            if get_ledger_integrity is not None:
                history_integrity = await get_ledger_integrity(
                    session_key_for_message(input_message, self.agent_engine)
                )
            model_context = stats.get("model_context", {})
            prompt_projection = stats.get("prompt_projection", {})
            prompt_reports = stats.get("prompt_reports", {})
            archive_stats = stats.get("archive", {})
            model_context_lines = []
            if model_context:
                model_context_lines = [
                    "",
                    "**模型上下文投影**",
                    f"- compaction: `done={model_context.get('compaction_committed_count', 0)}` `failed={model_context.get('compaction_failed_count', 0)}` `abandoned={model_context.get('compaction_abandoned_count', 0)}`",
                    f"- schema: `{model_context.get('schema_version', 0)}`，scope: `{model_context.get('scope_count', 0)}`，events: `{model_context.get('event_count', 0)}`",
                    f"- usage: `observed={model_context.get('usage_observation_count', 0)}` `missing={model_context.get('usage_missing_count', 0)}` `hit={model_context.get('cache_hit_tokens', 0)}` `miss={model_context.get('cache_miss_tokens', 0)}` `rate={model_context.get('cache_hit_rate', 0)}%`，repair fallback: `{model_context.get('fallback_count', 0)}`",
                    f"- summary: `prompt={model_context.get('summary_prompt_tokens', 0)}` `completion={model_context.get('summary_completion_tokens', 0)}` `elapsed={model_context.get('summary_elapsed_ms', 0)}ms`",
                    f"- mode: `read={model_context.get('read_enabled', False)}` `write={model_context.get('write_enabled', False)}` `shadow={model_context.get('shadow', False)}`",
                    f"- overflow: `detected={model_context.get('overflow_count', 0)}` `recovered={model_context.get('overflow_recovery_count', 0)}`",
                ]
            engagement_lines = [
                "",
                "**会话参与**",
                f"- Shadow 候选: `{engagement_metrics.get('shadow_candidates', 0)}`",
                f"- Active 预留: `{engagement_metrics.get('active_reserved', 0)}`",
                f"- Delivery: `prepared={delivery_counts.get('prepared', 0)}` `sent={delivery_counts.get('sent', 0)}` `failed={delivery_counts.get('failed', 0)}`",
            ]
            history_lines = [
                "",
                "**账本完整性**",
                f"- identity 冲突: `{history_integrity.get('legacy_conflict_count', 0)}`",
            ]
            projection_lines = []
            if prompt_projection or prompt_reports or archive_stats:
                projection_lines = [
                    "",
                    "**账本投影观测**",
                    f"- Prompt visibility: `total={prompt_projection.get('visibility_count', 0)}` `visible={prompt_projection.get('visible_count', 0)}` `hidden={prompt_projection.get('hidden_count', 0)}` `lag={prompt_projection.get('projection_lag', 0)}`",
                    f"- Prompt reports: `total={prompt_reports.get('report_count', 0)}` `fallback={prompt_reports.get('fallback_count', 0)}` `degraded={prompt_reports.get('degraded_count', 0)}` `historical_excluded={prompt_reports.get('historical_exclusion_count', 0)}`",
                    f"- Archive: `batches={archive_stats.get('batch_count', 0)}` `pending={archive_stats.get('pending_count', 0)}` `events={archive_stats.get('event_count', 0)}` `export_failed={archive_stats.get('export_failed_count', 0)}`",
                ]
            event_integrity = history_integrity.get("event_integrity", {})
            if event_integrity:
                history_lines.append(
                    f"- 账本 turn: `total={event_integrity.get('turn_count', 0)}` `invalid={event_integrity.get('invalid_turn_count', 0)}` `incomplete={event_integrity.get('incomplete_turn_count', 0)}` `open={event_integrity.get('open_turn_count', 0)}` `waiting_tool={event_integrity.get('waiting_tool_turn_count', 0)}`"
                )
            skill_count = len(sm.list_skill_names()) if sm and sm.has_skills else 0

            cost_lines = self._usage_lines()

            channel_info_lines = []
            cache_status = getattr(self.channel_info_provider, "cache_status", None)
            if callable(cache_status):
                entries = cache_status()
                stale_count = sum(1 for entry in entries if entry.get("stale"))
                error_count = sum(
                    1 for entry in entries if entry.get("last_error_reason")
                )
                channel_info_lines = [
                    "",
                    "**渠道信息缓存**",
                    f"- 条目: `{len(entries)}`，stale: `{stale_count}`，错误: `{error_count}`",
                ]
                option = args.strip().split()[0] if args.strip() else ""
                if option in {"刷新", "refresh"}:
                    target = getattr(input_message, "delivery_target", None)
                    get_info = getattr(self.channel_info_provider, "get_info", None)
                    if target is not None and callable(get_info):
                        snapshot = await get_info(target, refresh=True)
                        reasons = "、".join(snapshot.unavailable_reasons) or "无"
                        channel_info_lines.extend(
                            [
                                f"- 当前目标刷新: `{snapshot.availability}`，stale=`{snapshot.stale}`",
                                f"- 原因: `{reasons}`",
                            ]
                        )
                elif option in {"全部", "all"}:
                    list_chats = getattr(self.identity_manager, "list_chats", None)
                    if callable(list_chats):
                        chats = list_chats()
                        channel_info_lines.append(f"- 已知群: `{len(chats)}` 个")
                        for chat in chats:
                            channel_info_lines.append(
                                "  - "
                                f"{chat.get('channel')}/{chat.get('account_id')} "
                                f"群指纹=`{chat.get('chat_fingerprint')}` "
                                f"成员=`{chat.get('member_count', 0)}` "
                                f"最近观察=`{time.strftime('%m-%d %H:%M', time.localtime(chat.get('last_seen', 0)))}`"
                            )

            status_text = [
                "**系统状态**",
                f"`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
                "",
                "**系统资源**",
                f"- CPU: `{cpu_percent:.1f}%`",
                f"- 内存: `{memory.percent:.1f}%` (`{memory.used / 1024**3:.1f}GB` / `{memory.total / 1024**3:.1f}GB`)",
                f"- 磁盘: `{disk.percent:.1f}%` (`{disk.used / 1024**3:.1f}GB` / `{disk.total / 1024**3:.1f}GB`)",
                "",
                "**进程状态**",
                f"- 内存: `{process_memory:.1f}MB`",
                f"- CPU: `{process_cpu:.1f}%`",
                "",
                "**机器人状态**",
                f"- 消息队列: `{total_queue}` 条 (`{len(queue_sizes)}` 会话)",
                f"- 活跃聊天: `{active_chats}` 个",
                f"- 技能: `{skill_count}` 个",
                f"- 插件: `{self._plugin_count()}` 个",
                *self._approval_line(),
                *engagement_lines,
                *history_lines,
                *projection_lines,
                *channel_info_lines,
                "",
                "**记忆系统**",
                f"- Hindsight: {_hindsight_status_line(hindsight_health)}",
                *model_context_lines,
                *cost_lines,
            ]

            if learner_stats.get("enabled"):
                jargon_count = learner_stats.get("jargon_count", 0)
                status_text.append("")
                status_text.append("**学习系统**")
                status_text.append(f"- 俚语词典: `{jargon_count}` 条")

            return make_reply(input_message, "\n".join(status_text))
        except ImportError:
            return make_reply(input_message, "无法获取系统状态信息，请安装psutil库。")
        except Exception as e:
            _log.error(f"状态命令处理失败: {e}")
            return []

    def _approval_line(self) -> List[str]:
        """2.4：审批白名单规模 + 最近一次 allow-always 时间（未注入时为空）。"""
        if self.approval_manager is None:
            return []
        wl = self.approval_manager.whitelist_stats()
        last = wl.get("last_allow_always_at") or ""
        last_text = f"（最近: {last[:16].replace('T', ' ')}） " if last else ""
        return [f"- 审批白名单: `{wl.get('count', 0)}` 条 {last_text}"]
