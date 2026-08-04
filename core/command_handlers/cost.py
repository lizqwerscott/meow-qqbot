import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.engine.agent_engine import AgentEngine
from core.message import InputMessage

_log = logging.getLogger(__name__)


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


@command(
    name="消耗",
    aliases=["tokens", "cost"],
    permission="admin",
    description="查看 token 消耗（管理员专用）",
)
class CostCommand:
    def __init__(self, agent_engine: AgentEngine):
        self.agent_engine = agent_engine

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        ct = self.agent_engine.cost_tracker
        global_stats = ct.get_global_stats()

        lines = ["**AI 消耗总览**", ""]

        lines.extend(
            [
                f"- API 调用: `{global_stats.turn_count}` 次",
                f"- 输入 tokens: `{_fmt_tokens(global_stats.prompt_tokens)}`",
                f"  - 缓存命中: `{_fmt_tokens(global_stats.cache_hit_tokens)}` ({global_stats.cache_hit_rate:.1%})",
                f"  - 缓存未命中: `{_fmt_tokens(global_stats.cache_miss_tokens)}` ({1 - global_stats.cache_hit_rate:.1%})",
                f"- 输出 tokens: `{_fmt_tokens(global_stats.completion_tokens)}`",
                f"- 总费用: **¥{global_stats.cost:.4f}**",
                "",
                "**各会话消耗**",
            ]
        )

        if args.strip():
            chat_id_arg = args.strip()
            session = ct.get_session_stats(chat_id_arg)
            if session is None:
                lines.append(f"\n未找到会话 `{chat_id_arg}`")
            else:
                lines.extend(
                    [
                        f"\n`{chat_id_arg}`",
                        f"  调用: `{session.turn_count}` 次",
                        f"  输入: `{_fmt_tokens(session.prompt_tokens)}` (命中 {session.cache_hit_rate:.1%})",
                        f"  输出: `{_fmt_tokens(session.completion_tokens)}`",
                        f"  费用: **¥{session.cost:.4f}**",
                    ]
                )
            return make_reply(input_message, "\n".join(lines))

        sessions = ct.get_all_sessions()
        if not sessions:
            lines.append("  (暂无数据)")
        else:
            sorted_sessions = sorted(
                sessions.items(), key=lambda x: x[1].cost, reverse=True
            )
            for cid, s in sorted_sessions:
                cid_short = cid[:20] + ".." if len(cid) > 22 else cid
                lines.append(
                    f"- `{cid_short}`  "
                    f"调用: `{s.turn_count}`  "
                    f"tks: `{_fmt_tokens(s.total_tokens)}`  "
                    f"命中: `{s.cache_hit_rate:.0%}`  "
                    f"费用: **¥{s.cost:.4f}**"
                )

        return make_reply(input_message, "\n".join(lines))
