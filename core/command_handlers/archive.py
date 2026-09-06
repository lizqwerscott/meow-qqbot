import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.managers.archive_manager import ArchiveManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


def _is_date(value: str) -> bool:
    if len(value) != 10:
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return False
    return parsed.strftime("%Y-%m-%d") == value


@command(
    name="归档", aliases=["archive"], permission="admin", description="会话归档管理"
)
class ArchiveCommand:
    def __init__(self, archive_manager: Optional[ArchiveManager] = None):
        self.archive_manager = archive_manager

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if not self.archive_manager:
            return make_reply(input_message, "归档系统未启用。")

        parts = args.strip().split(maxsplit=1) if args.strip() else []
        subcmd = parts[0] if parts else ""
        subargs = parts[1] if len(parts) > 1 else ""

        if not subcmd or subcmd in ("当前", "this", "."):
            return await self._show_status(input_message)
        if subcmd in ("查看", "list", "ls"):
            return await self._list_archives(input_message, subargs)
        if subcmd in ("执行日切", "daily"):
            return await self._run_archive(input_message, subargs)
        if subcmd in ("快照", "snapshot"):
            return await self._run_snapshot(input_message, subargs)
        if subcmd in ("执行", "run", "do"):
            _log.warning("归档命令“执行”已弃用，请使用“执行日切”")
            return await self._run_archive(input_message, subargs)
        if subcmd in ("摘要", "summary"):
            return await self._show_summary(input_message, subargs)
        if subcmd in ("完整性", "integrity", "校验"):
            return await self._show_integrity(input_message, subargs)
        if subcmd in ("迁移", "migration", "冲突"):
            return await self._show_migration_audit(input_message, subargs)
        if subcmd in ("修复", "repair"):
            return await self._run_repair(input_message, subargs)
        if subcmd in ("修订", "revision", "repair-note"):
            return await self._record_repair_revision(input_message, subargs)
        if subcmd in ("清理", "clean"):
            return await self._clean_archives(input_message)
        return make_reply(
            input_message,
            "未知子命令。可用: 当前, 查看, 执行日切, 快照, 摘要, 完整性, 迁移, 修复, 修订, 清理",
        )

    async def _show_status(self, input_message: InputMessage) -> List[Dict[str, Any]]:
        chat_id = input_message.chat_id
        status = await self.archive_manager.get_session_status_async(chat_id)
        count = status["message_count"]
        last_act = (
            time.strftime("%H:%M", time.localtime(status["last_activity"]))
            if status["last_activity"]
            else "无"
        )
        async_status = getattr(
            self.archive_manager, "get_archive_operation_status_async", None
        )
        if callable(async_status):
            operation_status = await async_status(chat_id)
        else:
            operation_status = getattr(
                self.archive_manager, "get_archive_operation_status", lambda _: {}
            )(chat_id)
        return make_reply(
            input_message,
            f"会话: {chat_id[:24]}…\n"
            f"当前消息: {count} 条\n"
            f"最后活跃: {last_act}\n"
            f"归档摘要: {status['archive_count']} 个\n"
            f"归档触发: 跨天首条消息（按消息时间戳）\n"
            f"回放: 昨天最后一个连续片段（间隔 {self.archive_manager.replay_gap_seconds} 秒）\n"
            f"摘要: {self.archive_manager.summary_count} 条\n"
            f"已提交 batch: {operation_status.get('committed_batches', 0)} 个\n"
            f"最近 batch: {operation_status.get('latest_committed_batch') or '无'}\n"
            f"待恢复事务: {operation_status.get('pending_operations', 0)} 个",
        )

    async def _list_archives(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        list_batches = getattr(self.archive_manager, "list_archive_batches_async", None)
        if callable(list_batches):
            batches = await list_batches(target)
            committed = [
                batch for batch in batches if batch.get("state") == "committed"
            ]
            if not committed:
                return make_reply(input_message, f"会话 {target[:24]}… 没有归档记录。")
            lines = [
                f"{batch['batch_id']} ({batch['event_count']} 事件, "
                f"{batch['turn_count']} turns, "
                f"{', '.join(batch.get('source_dates', ())) or '无日期'})"
                for batch in committed[:20]
            ]
            reply = f"归档 batch ({len(committed)} 个):\n" + "\n".join(lines)
            if len(committed) > 20:
                reply += f"\n... (还有 {len(committed) - 20} 个)"
            return make_reply(input_message, reply)
        files = await self.archive_manager.list_archives_async(target)
        if not files:
            return make_reply(input_message, f"会话 {target[:24]}… 没有归档记录。")
        files = sorted(files, key=lambda item: item.get("path", ""), reverse=True)
        lines = [
            f"{Path(item['path']).stem} ({item.get('size', 0)} 字节)"
            for item in files[:20]
        ]
        reply = f"归档摘要 ({len(files)} 个):\n" + "\n".join(lines)
        if len(files) > 20:
            reply += f"\n... (还有 {len(files) - 20} 个)"
        return make_reply(input_message, reply)

    async def _show_summary(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        text = await self.archive_manager.load_recent_summaries_async(target)
        if not text:
            return make_reply(
                input_message, f"会话 {target[:24]}… 没有可用的归档摘要。"
            )
        preview = text[:1500]
        if len(text) > 1500:
            preview += "\n…(过长已截断)"
        return make_reply(input_message, f"最近摘要:\n{preview}")

    async def _show_integrity(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        get_integrity = getattr(
            self.archive_manager, "get_event_integrity_async", None
        )
        if not callable(get_integrity):
            return make_reply(input_message, "账本完整性检查不可用。")
        try:
            summary = await get_integrity(target)
        except Exception as exc:
            return make_reply(input_message, f"完整性检查失败: {exc}")
        if summary.get("error"):
            return make_reply(input_message, f"账本完整性检查不可用: {summary['error']}")
        invalid_turns = summary.get("invalid_turns", [])
        lines = [
            f"会话 {target[:24]}… turn 完整性:",
            f"总数: {summary.get('turn_count', 0)}",
            f"异常: {summary.get('invalid_turn_count', 0)}",
            f"未完成: {summary.get('incomplete_turn_count', 0)}",
            f"开放: {summary.get('open_turn_count', 0)}",
        ]
        reasons = summary.get("invalid_reasons", {})
        if reasons:
            lines.append(
                "原因: "
                + ", ".join(
                    f"{reason}={count}" for reason, count in reasons.items()
                )
            )
        get_revisions = getattr(
            self.archive_manager, "get_turn_repair_revisions_async", None
        )
        if callable(get_revisions):
            try:
                revisions = await get_revisions(target)
            except Exception as exc:
                _log.warning("读取 turn 修订记录失败 [%s..]: %s", target[:12], exc)
                revisions = ()
            lines.append(f"追加式修订记录: {len(revisions)}")
            for revision in revisions[:10]:
                lines.append(
                    f"- {revision.original_turn_id[:18]}… "
                    f"revision={revision.revision_id} "
                    f"原状态={revision.original_status}/{revision.original_reason} "
                    f"说明={revision.reason[:120]}"
                )
        for report in invalid_turns[:20]:
            lines.append(
                f"- {report.get('turn_id', '')[:18]}… "
                f"{report.get('status', '')}/{report.get('reason', 'invalid_turn')}"
            )
        if len(invalid_turns) > 20:
            lines.append(f"… 其余 {len(invalid_turns) - 20} 个异常 turn 未展开")
        return make_reply(input_message, "\n".join(lines))

    @staticmethod
    def _parse_repair_args(
        input_message: InputMessage, args: str
    ) -> tuple[str, str]:
        tokens = args.strip().split()
        dates = [token for token in tokens if _is_date(token)]
        if len(dates) != 1 or len(tokens) > 2:
            raise ValueError("用法: 修复 <YYYY-MM-DD> [chat_id]")
        before_date = dates[0]
        target = next((token for token in tokens if token != before_date), None)
        return target or input_message.chat_id, before_date

    async def _run_repair(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        repair = getattr(self.archive_manager, "repair_event_log_archives", None)
        if not callable(repair):
            return make_reply(input_message, "账本归档修复不可用。")
        try:
            target, before_date = self._parse_repair_args(input_message, args)
            result = await repair(target, before_date=before_date)
        except Exception as exc:
            return make_reply(input_message, f"归档修复失败: {exc}")
        archived_events = sum(batch.event_count for batch in result.batches)
        archived_turns = sum(batch.unit_count for batch in result.batches)
        skipped = getattr(result, "skipped_turns", [])
        lines = [
            f"会话 {target[:24]}… 静态归档修复完成。",
            f"范围: {before_date} 之前",
            f"新增归档: {archived_events} 事件 / {archived_turns} turns",
            f"跳过: {len(skipped)} turns（不完整或协议异常不会强行归档）",
        ]
        if skipped:
            reasons: Dict[str, int] = {}
            for item in skipped:
                reason = str(item.get("reason") or "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
            lines.append(
                "跳过原因: "
                + ", ".join(
                    f"{reason}={count}" for reason, count in reasons.items()
                )
            )
        return make_reply(input_message, "\n".join(lines))

    async def _show_migration_audit(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        reader = getattr(
            self.archive_manager, "get_legacy_migration_audit_async", None
        )
        if not callable(reader):
            return make_reply(input_message, "旧归档迁移报告不可用。")
        try:
            report = await reader(target)
        except Exception as exc:
            return make_reply(input_message, f"读取迁移报告失败: {exc}")
        if report.get("status") == "not_found":
            return make_reply(input_message, f"会话 {target[:24]}… 没有迁移冲突报告。")
        if report.get("status") == "invalid_report":
            return make_reply(input_message, "迁移冲突报告损坏，请重新执行静态迁移。")
        lines = [
            f"会话 {target[:24]}… 旧归档迁移报告:",
            f"状态: {report.get('status', 'unknown')}",
            f"来源文件: {len(report.get('source_files', ()))}",
            f"重复记录: {report.get('duplicate_record_count', 0)}",
            f"非法记录: {report.get('invalid_record_count', 0)}",
            f"identity 冲突: {report.get('conflict_event_count', 0)}",
            f"读取/导入错误: {report.get('error_count', 0)}",
        ]
        path = report.get("conflict_report_path")
        if path:
            lines.append(f"报告文件: {path}")
        return make_reply(input_message, "\n".join(lines))

    async def _record_repair_revision(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        """Record an append-only repair note for one invalid turn."""
        tokens = args.strip().split(maxsplit=2)
        if len(tokens) != 3:
            return make_reply(
                input_message,
                "用法: 修订 <turn_id> <revision_id> <原因>；只记录修订，不修改原始 turn。",
            )
        record = getattr(
            self.archive_manager, "record_turn_repair_revision_async", None
        )
        if not callable(record):
            return make_reply(input_message, "账本修订记录不可用。")
        turn_id, revision_id, reason = tokens
        try:
            revision = await record(
                input_message.chat_id,
                turn_id,
                revision_id,
                reason,
                operator=input_message.sender_id,
            )
        except Exception as exc:
            return make_reply(input_message, f"修订记录失败: {exc}")
        return make_reply(
            input_message,
            f"已记录 turn {revision.original_turn_id[:24]}… 的修订 {revision.revision_id}。\n"
            "原始事件与完整性状态未修改，也未补写工具结果。",
        )

    async def _run_archive(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        try:
            target_is_group = (
                input_message.is_group if target == input_message.chat_id else None
            )
            result = await self.archive_manager.archive_manual(target, target_is_group)
            has_archived = bool(
                result.archive_paths
                or any(batch.event_count > 0 for batch in result.batches)
            )
            has_summary = bool(
                result.summary_path
                or any(batch.summary_path for batch in result.batches)
                or has_archived
            )
            status_line = (
                f"会话 {target[:24]}… 日切完成。\n"
                if has_archived
                else f"会话 {target[:24]}… 没有待日切消息。\n"
            )
            return make_reply(
                input_message,
                status_line + f"保留: {result.replay_count} 条（含今天全部）\n"
                f"摘要: {'已生成' if has_summary else '无'}\n"
                f"归档: {'已写入核心账本' if has_archived else '无'}",
            )
        except Exception as e:
            return make_reply(input_message, f"归档失败: {e}")

    async def _run_snapshot(
        self, input_message: InputMessage, chat_id: str
    ) -> List[Dict[str, Any]]:
        target = chat_id or input_message.chat_id
        try:
            target_is_group = (
                input_message.is_group if target == input_message.chat_id else None
            )
            result = await self.archive_manager.archive_snapshot(
                target, target_is_group
            )
            return make_reply(
                input_message,
                f"会话 {target[:24]}… 快照完成。\n"
                f"活动事件: {result.replay_count} 条\n"
                "核心账本与当前 active history 未改变。",
            )
        except Exception as e:
            return make_reply(input_message, f"快照失败: {e}")

    async def _clean_archives(
        self, input_message: InputMessage
    ) -> List[Dict[str, Any]]:
        try:
            removed = await self.archive_manager.cleanup_old_archives_async()
            return make_reply(input_message, f"归档清理完成，移除了 {removed} 个文件。")
        except Exception as e:
            return make_reply(input_message, f"清理失败: {e}")
