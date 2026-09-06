import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from core.managers.chat_message import ChatMessage, _estimate_tokens

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    messages: List[ChatMessage]
    usage: Optional[Dict]


class ContextCompactor:
    """Legacy message-list compactor kept only for migration compatibility.

    Production sessions wired to ``ConversationEventLog`` never call this
    adapter. New compaction must go through ``ModelContextTranscript`` or the
    bounded prompt projection; this class can be removed after legacy rollout.
    """

    legacy_only = True

    def __init__(
        self,
        ai_service,
        compact_threshold_tokens: int,
        keep_recent_tokens: int,
        max_summary_tokens: int = 500,
    ):
        self.ai_service = ai_service
        self.compact_threshold_tokens = compact_threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.max_summary_tokens = max_summary_tokens

    async def compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        force: bool = False,
    ) -> CompactionResult:
        original = list(messages)
        estimated = self._estimate_tokens(original)
        if not force and estimated < self.compact_threshold_tokens:
            return CompactionResult(False, original, None)

        old_msgs, recent_msgs = self._split_by_token_budget(original)
        if not old_msgs:
            return CompactionResult(False, original, None)

        text = self._format_for_summary(old_msgs)
        try:
            summary, usage = await self.ai_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个对话摘要助手。请将以下对话内容压缩为一段"
                            "简洁的摘要，保留重要的事实、决定、用户偏好、约定等"
                            "关键信息。不要添加原文没有的内容。"
                        ),
                    },
                    {"role": "user", "content": f"请总结以下对话：\n\n{text}"},
                ],
                max_tokens=self.max_summary_tokens,
            )
        except Exception as e:
            _log.warning("压缩失败: %r", e)
            return CompactionResult(False, original, None)

        if not summary:
            _log.warning("压缩返回空结果，跳过")
            return CompactionResult(False, original, usage)
        summary = summary.strip()
        if not summary:
            _log.warning("压缩返回空结果，跳过")
            return CompactionResult(False, original, usage)

        compacted = [
            ChatMessage(
                role="assistant",
                content=f"【历史对话摘要】\n{summary}",
                timestamp=old_msgs[0].timestamp,
                name="系统",
            ),
            *recent_msgs,
        ]
        return CompactionResult(True, compacted, usage)

    @staticmethod
    def _estimate_tokens(messages: Sequence[ChatMessage]) -> int:
        total = 0
        for msg in messages:
            total += _estimate_tokens(msg.content)
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    function = tool_call.get("function", {})
                    total += _estimate_tokens(function.get("name"))
                    total += _estimate_tokens(function.get("arguments"))
            if msg.reasoning_content:
                total += _estimate_tokens(msg.reasoning_content)
        return total

    def _split_by_token_budget(
        self, messages: Sequence[ChatMessage]
    ) -> Tuple[List[ChatMessage], List[ChatMessage]]:
        recent: List[ChatMessage] = []
        total = 0
        recent_tool_ids: set = set()
        for message in reversed(messages):
            message_tokens = self._estimate_tokens([message])
            if total + message_tokens > self.keep_recent_tokens and recent:
                if (
                    message.role == "tool"
                    and message.tool_call_id
                    and message.tool_call_id in recent_tool_ids
                ):
                    recent.insert(0, message)
                    total += message_tokens
                    continue
                if message.tool_calls:
                    call_ids = {
                        call.get("id") for call in message.tool_calls if call.get("id")
                    }
                    if call_ids & recent_tool_ids:
                        recent.insert(0, message)
                        total += message_tokens
                        recent_tool_ids.difference_update(call_ids)
                        continue
                break
            recent.insert(0, message)
            total += message_tokens
            if message.role == "tool" and message.tool_call_id:
                recent_tool_ids.add(message.tool_call_id)

        old = list(messages[: -len(recent)]) if recent else list(messages[:-1])
        return old, recent

    @staticmethod
    def _format_for_summary(messages: Sequence[ChatMessage]) -> str:
        lines = []
        for message in messages:
            time_str = time.strftime("%m-%d %H:%M", time.localtime(message.timestamp))
            if message.role == "user":
                display_name = message.name or message.sender_id or "用户"
                lines.append(f"[{time_str}] {display_name}: {message.content}")
            elif message.role == "assistant":
                if message.tool_calls:
                    tools = ", ".join(
                        tc["function"]["name"] for tc in message.tool_calls
                    )
                    lines.append(
                        f"[{time_str}] 助手(调用工具: {tools}): {message.content}"
                    )
                else:
                    lines.append(f"[{time_str}] 助手: {message.content}")
            elif message.role == "tool":
                tool_name = message.tool_name or "工具"
                preview = message.content[:100].replace("\n", " ")
                lines.append(f"[{time_str}] {tool_name} 返回: {preview}...")
        return "\n".join(lines)
