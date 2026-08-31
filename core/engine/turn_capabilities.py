"""Immutable capability contract for one provider turn."""

from dataclasses import dataclass
from typing import Optional, Sequence

from core.engine.prompt_snapshot import PromptMode
from core.managers.session_manager import InboundIntent


@dataclass(frozen=True)
class TurnCapabilities:
    """Fail-closed execution and delivery permissions for a turn."""

    intent: InboundIntent
    mode: PromptMode = PromptMode.AGENT
    capability_profile: str = "agent_full"
    planner_actions: frozenset[str] = frozenset()
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
    _CHAT_TOOL_NAMES = frozenset(
        {
            "send_emoji",
            "memory",
            "mark_important",
            "web_search",
            "web_fetch",
            "work_plan",
        }
    )
    _CHAT_CONVERSATION_TOOL_NAMES = _CHAT_TOOL_NAMES - {"work_plan"}
    _ALL_PLANNER_ACTIONS = frozenset({"wait", "no_reply", "request_agent"})
    _LIMITED_PLANNER_ACTIONS = frozenset({"wait", "no_reply"})

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

    @classmethod
    def for_mode(
        cls,
        *,
        mode: PromptMode,
        capability_profile: str,
        intent: InboundIntent,
        chat_id: str = "",
        sender_id: str = "",
        reply_to: str = "",
        allowed_media_uris: Optional[frozenset[str]] = None,
    ) -> "TurnCapabilities":
        """Build a fail-closed capability contract from an audited mode profile."""
        mode = PromptMode(mode)
        fixed_context = {
            "chat_id": chat_id,
            "sender_id": sender_id,
            "reply_to": reply_to,
        }
        if mode is PromptMode.AGENT:
            if capability_profile == "work_plan_consumer":
                return cls(
                    intent=intent,
                    mode=mode,
                    capability_profile=capability_profile,
                    allow_tools=True,
                    allowed_tool_names=frozenset({"work_plan"}),
                    **fixed_context,
                )
            return cls(
                intent=intent,
                mode=mode,
                capability_profile="agent_full",
                **fixed_context,
            )

        chat_profiles = {
            "private_chat": (cls._CHAT_TOOL_NAMES, cls._ALL_PLANNER_ACTIONS),
            "group_explicit": (cls._CHAT_TOOL_NAMES, cls._ALL_PLANNER_ACTIONS),
            "group_reply": (
                cls._CHAT_CONVERSATION_TOOL_NAMES,
                cls._LIMITED_PLANNER_ACTIONS,
            ),
            "group_ambient": (cls._AMBIENT_TOOL_NAMES, cls._LIMITED_PLANNER_ACTIONS),
        }
        try:
            allowed_tools, planner_actions = chat_profiles[capability_profile]
        except KeyError as exc:
            raise ValueError(
                f"unknown Chat capability profile: {capability_profile}"
            ) from exc
        is_ambient = capability_profile == "group_ambient"
        return cls(
            intent=intent,
            mode=mode,
            capability_profile=capability_profile,
            planner_actions=planner_actions,
            allow_tools=True,
            allowed_tool_names=allowed_tools,
            allow_automatic_reply=not is_ambient,
            allowed_media_uris=(
                (allowed_media_uris or frozenset()) if is_ambient else None
            ),
            allowed_delivery_kinds=(
                frozenset({"message", "emoji"}) if is_ambient else None
            ),
            **fixed_context,
        )

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
        """Require ambient media and Chat memory reads to keep runtime-owned scope."""
        if self.allowed_media_uris is None:
            return self._allows_chat_memory_args(name, args)
        if name not in {"image", "pdf", "read_file"}:
            return self._allows_chat_memory_args(name, args)
        media_uri = args.get("media_uri")
        if not isinstance(media_uri, str) or media_uri not in self.allowed_media_uris:
            return False
        # ``read_file`` also reads the workspace. Ambient may use only its
        # attachment branch, never the file_path branch.
        return name != "read_file" or not args.get("file_path")

    def _allows_chat_memory_args(self, name: str, args: dict) -> bool:
        """Keep Chat memory queries bound to the active sender and chat.

        The tool implementation independently derives Hindsight tags from the
        runtime context.  This guard prevents a model from widening that scope
        before it ever reaches the registry.
        """
        if self.mode is not PromptMode.CHAT or name != "memory":
            return True
        action = args.get("action", "search")
        if action == "search_shared":
            return self.capability_profile in {
                "group_explicit",
                "group_reply",
            } and not any(
                args.get(key)
                for key in ("person_name", "person_a", "person_b", "user_id", "chat_id")
            )
        return action == "search" and not any(
            args.get(key) for key in ("person_name", "person_a", "person_b", "user_id")
        )

    def _filter_chat_tool_schema(self, tool: dict) -> dict:
        """Remove unavailable memory actions from model-visible Chat schema."""
        name = tool.get("function", {}).get("name")
        if name not in {"memory", "work_plan"}:
            return tool
        import copy

        filtered = copy.deepcopy(tool)
        parameters = filtered.get("function", {}).get("parameters")
        if not isinstance(parameters, dict):
            return filtered
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return filtered
        if name == "work_plan":
            action = properties.get("action")
            if isinstance(action, dict):
                action["enum"] = ["list", "get"]
            return filtered
        allowed_actions = ["search"]
        if self.capability_profile in {"group_explicit", "group_reply"}:
            allowed_actions.append("search_shared")
        action = properties.get("action")
        if isinstance(action, dict):
            action["enum"] = allowed_actions
        for field in ("person_name", "person_a", "person_b"):
            properties.pop(field, None)
        return filtered

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

    def model_tool_schemas(self, tools: Optional[Sequence[dict]]) -> list[dict]:
        """Filter normal tools and append the one internal Chat control schema."""
        filtered = self.filter_tools(tools) or []
        if self.mode is PromptMode.CHAT:
            filtered = [self._filter_chat_tool_schema(tool) for tool in filtered]
        if self.mode is not PromptMode.CHAT or not self.planner_actions:
            return filtered
        from core.engine.planner_control import planner_control_tool

        return [
            tool
            for tool in filtered
            if tool.get("function", {}).get("name") != "planner_control"
        ] + [planner_control_tool(self.planner_actions)]
