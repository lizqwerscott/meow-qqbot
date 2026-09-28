import logging
import re
import time
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply, session_key_for_message
from core.engine.agent_engine import AgentEngine
from core.engine.token_usage_log import TokenUsageLog
from core.message import InputMessage

_log = logging.getLogger(__name__)

_GLOBAL_TOKENS = {"全局", "总", "all", "global"}
_DAYS_PATTERN = re.compile(r"^(\d+)\s*(?:d|天)$", re.IGNORECASE)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_ts(value: Optional[float]) -> str:
    if not value:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))


def _summary_lines(title: str, summary: dict, *, scope: str = "") -> List[str]:
    hit = int(summary.get("cache_hit_tokens", 0))
    miss = int(summary.get("cache_miss_tokens", 0))
    total_cache = hit + miss
    hit_rate = f"{hit / total_cache:.1%}" if total_cache else "-"
    lines = [f"**{title}**", ""]
    if scope:
        lines.append(f"- 会话: `{scope}`")
    lines.extend(
        [
            f"- 调用: `{summary.get('record_count', 0)}` 次",
            f"- 输入 tokens: `{_fmt_tokens(int(summary.get('prompt_tokens', 0)))}`"
            f"（命中 `{hit_rate}`，hit `{_fmt_tokens(hit)}` / miss `{_fmt_tokens(miss)}`）",
            f"- 输出 tokens: `{_fmt_tokens(int(summary.get('completion_tokens', 0)))}`",
            f"- 费用: **¥{float(summary.get('cost', 0.0)):.4f}**",
            f"- 范围: {_fmt_ts(summary.get('earliest'))} → {_fmt_ts(summary.get('latest'))}",
        ]
    )
    return lines


def _breakdown_lines(rows: List[dict], empty: str = "  (暂无数据)") -> List[str]:
    if not rows:
        return [empty]
    lines = []
    for row in rows:
        key = str(row["key"])
        short = key[:20] + ".." if len(key) > 22 else key
        lines.append(
            f"- `{short}`  调用 `{row['record_count']}`  "
            f"tks `{_fmt_tokens(row['prompt_tokens'] + row['completion_tokens'])}`  "
            f"费用 **¥{row['cost']:.4f}**"
        )
    return lines


@command(
    name="消耗",
    aliases=["tokens", "cost"],
    permission="admin",
    description="查看 token 消耗（默认当前会话全部历史；`7d`=最近 7 天，`全局`=所有会话）",
)
class CostCommand:
    def __init__(
        self,
        agent_engine: AgentEngine,
        token_usage_log: Optional[TokenUsageLog] = None,
    ):
        self.agent_engine = agent_engine
        self.token_usage_log = token_usage_log

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        argument = args.strip()
        if self.token_usage_log is None:
            return make_reply(
                input_message, self._in_memory_report(input_message, argument)
            )

        session_key = session_key_for_message(input_message, self.agent_engine)

        if argument.lower() in _GLOBAL_TOKENS:
            lines = _summary_lines(
                "AI 消耗总览（全部历史）", self.token_usage_log.summary()
            )
            lines.extend(["", "**各会话 Top**"])
            lines.extend(
                _breakdown_lines(self.token_usage_log.breakdown("chat", limit=10))
            )
            lines.extend(
                ["", "按 日期/模型/来源/provider 的汇总与趋势见 WebUI `/usage`。"]
            )
            return make_reply(input_message, "\n".join(lines))

        days_match = _DAYS_PATTERN.match(argument)
        if days_match:
            days = max(1, int(days_match.group(1)))
            since = time.time() - days * 86400
            summary = self.token_usage_log.summary(since=since, chat_id=session_key)
            lines = _summary_lines(
                f"本会话 token 消耗（最近 {days} 天）", summary, scope=session_key
            )
            lines.extend(["", "**按来源**"])
            lines.extend(
                _breakdown_lines(
                    self.token_usage_log.breakdown(
                        "source", since=since, chat_id=session_key
                    )
                )
            )
        else:
            if argument:
                key = self._resolve_key(input_message, argument)
                scope = f"{argument} → `{key}`" if key != argument else argument
                title = "会话 token 消耗（全部历史）"
            else:
                key = scope = session_key
                title = "本会话 token 消耗（全部历史）"
            summary = self.token_usage_log.summary(chat_id=key)
            if summary["record_count"] == 0:
                return make_reply(
                    input_message, f"会话 `{argument or key}` 暂无用量记录。"
                )
            lines = _summary_lines(title, summary, scope=scope)
            lines.extend(["", "**按来源**"])
            lines.extend(
                _breakdown_lines(self.token_usage_log.breakdown("source", chat_id=key))
            )

        lines.extend(["", "明细与图表见 WebUI `/usage`；`猫猫消耗 全局` 看所有会话。"])
        return make_reply(input_message, "\n".join(lines))

    def _resolve_key(self, input_message: InputMessage, argument: str) -> str:
        """Resolve a legacy chat id to its ledger (session) key when possible."""
        resolver = getattr(self.agent_engine, "session_identity_resolver", None)
        if resolver is None:
            return argument
        try:
            return resolver.resolve_legacy(
                argument,
                is_group=(
                    input_message.is_group
                    if argument == input_message.chat_id
                    else None
                ),
            ).session_key
        except (TypeError, ValueError):
            return argument

    def _in_memory_report(self, input_message: InputMessage, argument: str) -> str:
        """Fallback when the durable ledger is disabled (process-lifetime stats)."""
        ct = self.agent_engine.cost_tracker
        global_stats = ct.get_global_stats()

        lines = ["**AI 消耗总览**（内存统计，账本未启用；重启清零）", ""]
        lines.extend(
            [
                f"- API 调用: `{global_stats.turn_count}` 次",
                f"- 输入 tokens: `{_fmt_tokens(global_stats.prompt_tokens)}`",
                f"  - 缓存命中: `{_fmt_tokens(global_stats.cache_hit_tokens)}` "
                f"({global_stats.cache_hit_rate:.1%})",
                f"  - 缓存未命中: `{_fmt_tokens(global_stats.cache_miss_tokens)}` "
                f"({1 - global_stats.cache_hit_rate:.1%})",
                f"- 输出 tokens: `{_fmt_tokens(global_stats.completion_tokens)}`",
                f"- 总费用: **¥{global_stats.cost:.4f}**",
                "",
                "**各会话消耗**",
            ]
        )

        if argument:
            lookup_key = self._resolve_key(input_message, argument)
            session = ct.get_session_stats(lookup_key)
            if session is None:
                lines.append(f"\n未找到会话 `{argument}`")
            else:
                lines.extend(
                    [
                        f"\n`{argument}`"
                        + (f" → `{lookup_key}`" if lookup_key != argument else ""),
                        f"  调用: `{session.turn_count}` 次",
                        f"  输入: `{_fmt_tokens(session.prompt_tokens)}` "
                        f"(命中 {session.cache_hit_rate:.1%})",
                        f"  输出: `{_fmt_tokens(session.completion_tokens)}`",
                        f"  费用: **¥{session.cost:.4f}**",
                    ]
                )
            return "\n".join(lines)

        sessions = ct.get_all_sessions()
        if not sessions:
            lines.append("  (暂无数据)")
        else:
            for cid, stats in sorted(
                sessions.items(), key=lambda item: item[1].cost, reverse=True
            ):
                cid_short = cid[:20] + ".." if len(cid) > 22 else cid
                lines.append(
                    f"- `{cid_short}`  调用 `{stats.turn_count}`  "
                    f"tks `{_fmt_tokens(stats.total_tokens)}`  "
                    f"命中 `{stats.cache_hit_rate:.0%}`  "
                    f"费用 **¥{stats.cost:.4f}**"
                )
        return "\n".join(lines)
