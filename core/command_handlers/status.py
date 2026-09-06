import logging
import time
from typing import Any, Dict, List, Optional

import psutil

from core.command_handlers.base import command, make_reply
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
    def __init__(self, agent_engine: AgentEngine, approval_manager=None):
        self.agent_engine = agent_engine
        self.approval_manager = approval_manager  # 2.4：审批白名单状态行

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
            history_migration = {}
            get_history_migration_status = getattr(
                self.agent_engine, "get_history_migration_status", None
            )
            if get_history_migration_status is not None:
                history_migration = await get_history_migration_status(
                    input_message.chat_id
                )
            history_migration_summary = {}
            get_history_migration_summary = getattr(
                self.agent_engine, "get_history_migration_summary", None
            )
            if get_history_migration_summary is not None:
                history_migration_summary = await get_history_migration_summary()
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
                "**历史迁移**",
                f"- 可见消息: `legacy={history_migration.get('legacy_visible_count', 0)}` `timeline={history_migration.get('timeline_visible_count', 0)}`",
                f"- 缺口: `{history_migration.get('missing_legacy_visible_count', 0)}`，legacy protocol: `{history_migration.get('legacy_protocol_count', 0)}`",
                f"- 可移除 legacy read: `{'yes' if history_migration.get('ready_for_legacy_read_removal') else 'no'}`",
                f"- 旧历史迁移水位: `{'complete' if history_migration.get('legacy_migration_complete', True) else 'pending'}`",
                f"- 全局会话: `{history_migration_summary.get('session_count', 0)}`，可退出 fallback: `{history_migration_summary.get('sessions_ready_for_legacy_read_removal', 0)}`",
                f"- 全局缺口会话: `{history_migration_summary.get('sessions_with_missing_legacy_visible', 0)}`，协议残留会话: `{history_migration_summary.get('sessions_with_legacy_protocol', 0)}`",
                f"- identity 冲突: `chat={history_migration.get('legacy_conflict_count', 0)}` `global={history_migration_summary.get('legacy_conflict_count', 0)}`",
            ]
            projection_lines = []
            if prompt_projection or prompt_reports or archive_stats:
                projection_lines = [
                    "",
                    "**账本投影观测**",
                    f"- Prompt visibility: `total={prompt_projection.get('visibility_count', 0)}` `visible={prompt_projection.get('visible_count', 0)}` `hidden={prompt_projection.get('hidden_count', 0)}` `lag={prompt_projection.get('projection_lag', 0)}`",
                    f"- Prompt reports: `total={prompt_reports.get('report_count', 0)}` `fallback={prompt_reports.get('fallback_count', 0)}` `degraded={prompt_reports.get('degraded_count', 0)}`",
                    f"- Archive: `batches={archive_stats.get('batch_count', 0)}` `pending={archive_stats.get('pending_count', 0)}` `events={archive_stats.get('event_count', 0)}` `export_failed={archive_stats.get('export_failed_count', 0)}`",
                ]
            event_integrity = history_migration.get("event_integrity", {})
            if event_integrity:
                history_lines.append(
                    f"- 账本 turn: `total={event_integrity.get('turn_count', 0)}` `invalid={event_integrity.get('invalid_turn_count', 0)}` `incomplete={event_integrity.get('incomplete_turn_count', 0)}` `open={event_integrity.get('open_turn_count', 0)}` `waiting_tool={event_integrity.get('waiting_tool_turn_count', 0)}`"
                )
            skill_count = len(sm.list_skill_names()) if sm and sm.has_skills else 0

            cost = stats.get("cost", {})
            cost_lines = []
            if cost.get("turn_count", 0) > 0:
                cost_lines = [
                    "",
                    "**AI 消耗**",
                    f"- API 调用: `{cost['turn_count']}` 次",
                    f"- 输入 tokens: `{cost.get('prompt_tokens', 0):,}` (命中 `{cost.get('cache_hit_rate', 0)}%`，hit `{cost.get('cache_hit_tokens', 0):,}` / miss `{cost.get('cache_miss_tokens', 0):,}`)",
                    f"- 缓存观测: `{cost.get('cache_observation_count', 0)}` 次（provider 未提供字段 `{cost.get('cache_usage_missing_count', 0)}` 次）",
                    f"- 输出 tokens: `{cost.get('completion_tokens', 0):,}`",
                    f"- 总费用: **¥{cost.get('total_cost', 0):.4f}**",
                ]

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
