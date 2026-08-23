"""Trusted prompt contract for a turn's user-visible delivery semantics."""

from dataclasses import dataclass

from core.managers.session_manager import InboundIntent


@dataclass(frozen=True)
class DeliveryPromptContract:
    """A runtime-owned delivery contract rendered separately from user content."""

    intent: InboundIntent
    delivery_mode: str
    reply_target: str

    @staticmethod
    def _delivery_tool_names(tools: list[dict] | None) -> list[str]:
        return sorted(
            {
                tool.get("function", {}).get("name", "")
                for tool in tools or []
                if tool.get("function", {}).get("name")
                in {"send_message", "send_emoji"}
            }
        )

    def fingerprint(self, tools: list[dict] | None) -> dict[str, object]:
        return {
            "intent": str(self.intent),
            "delivery_mode": self.delivery_mode,
            "delivery_tools": self._delivery_tool_names(tools),
        }

    def render(self, tools: list[dict] | None) -> str:
        delivery_tools = self._delivery_tool_names(tools)
        tool_names = "、".join(delivery_tools) if delivery_tools else "无"
        if self.delivery_mode == "message_tool_only":
            mode_rules = (
                "本轮为 message_tool_only 交付。任何 assistant 文本，包括最终文本，"
                "都不会自动发送给用户。若决定回复，必须调用可见投递工具；"
                "否则唯一输出 NO_REPLY，且不要调用投递工具。"
            )
        else:
            mode_rules = (
                "本轮为 automatic 交付。只有不含 tool call 的最终 assistant 文本"
                "会自动发送给用户。带 tool call 的 assistant 文本、tool arguments 和"
                "tool results 都只供内部使用；需要工具时先调用工具，再单独输出最终回答。"
                "若不应回复，唯一输出 NO_REPLY。"
            )
        ambient_rule = (
            "本轮是群聊主动参与；只有确实能补充当前讨论时才投递，否则保持 NO_REPLY。"
            if self.intent is InboundIntent.GROUP_AMBIENT
            else ""
        )
        return "\n".join(
            (
                "<delivery_contract>",
                mode_rules,
                f"可见显式投递工具：{tool_names}。",
                ambient_rule,
                "</delivery_contract>",
            )
        )

    def render_target(self) -> str:
        """Render the per-turn target after the stable context prefix."""
        target = self.reply_target or "当前会话"
        return f"固定投递目标：{target}。不得选择、推断或改写其他 chat 的投递目标。"
