"""Immutable capability contract for one provider turn."""

from dataclasses import dataclass
from typing import Optional, Sequence

from core.managers.session_manager import InboundIntent


@dataclass(frozen=True)
class TurnCapabilities:
    """Fail-closed execution and delivery permissions for a turn."""

    intent: InboundIntent
    allow_tools: bool = True
    allowed_tool_names: Optional[frozenset[str]] = None
    allow_automatic_reply: bool = True
    chat_id: str = ""
    sender_id: str = ""
    reply_to: str = ""
    allowed_media_uris: Optional[frozenset[str]] = None
    allowed_delivery_kinds: Optional[frozenset[str]] = None
    cancellation_generation: int = 0

    _AMBIENT_TOOL_NAMES = frozenset(
        {"send_message", "send_emoji", "image", "pdf", "read_file"}
    )

    @classmethod
    def for_intent(
        cls,
        intent: InboundIntent,
        *,
        chat_id: str = "",
        sender_id: str = "",
        reply_to: str = "",
        allowed_media_uris: Optional[frozenset[str]] = None,
    ) -> "TurnCapabilities":
        fixed_context = {
            "chat_id": chat_id,
            "sender_id": sender_id,
            "reply_to": reply_to,
        }
        if intent is InboundIntent.GROUP_AMBIENT:
            return cls(
                intent=intent,
                allow_tools=True,
                allowed_tool_names=cls._AMBIENT_TOOL_NAMES,
                allow_automatic_reply=False,
                allowed_media_uris=allowed_media_uris or frozenset(),
                allowed_delivery_kinds=frozenset({"message", "emoji"}),
                **fixed_context,
            )
        return cls(intent=intent, **fixed_context)

    def allows_context(self, *, chat_id: str, sender_id: str, reply_to: str) -> bool:
        """Reject tools built for a different runtime-owned conversation context."""
        return (
            (not self.chat_id or self.chat_id == chat_id)
            and (not self.sender_id or self.sender_id == sender_id)
            and (not self.reply_to or self.reply_to == reply_to)
        )

    def allows_tool(self, name: str) -> bool:
        if not self.allow_tools:
            return False
        if self.allowed_tool_names is None:
            return True
        return name in self.allowed_tool_names

    def allows_tool_args(self, name: str, args: dict) -> bool:
        """Require ambient media operations to stay within runtime-authorized URIs."""
        if self.allowed_media_uris is None:
            return True
        if name not in {"image", "pdf", "read_file"}:
            return True
        media_uri = args.get("media_uri")
        if not isinstance(media_uri, str) or media_uri not in self.allowed_media_uris:
            return False
        # ``read_file`` also reads the workspace. Ambient may use only its
        # attachment branch, never the file_path branch.
        return name != "read_file" or not args.get("file_path")

    def allows_delivery_kind(self, kind: str) -> bool:
        """Reject a tool whose externally visible delivery is not authorized."""
        return not kind or (
            self.allowed_delivery_kinds is None or kind in self.allowed_delivery_kinds
        )

    def filter_tools(self, tools: Optional[Sequence[dict]]) -> Optional[list[dict]]:
        if tools is None:
            return None
        if not self.allow_tools:
            return []
        if self.allowed_tool_names is None:
            return list(tools)
        return [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") in self.allowed_tool_names
        ]
