import inspect
import logging
from typing import Any, Dict, List, Optional

from core.managers.command_manager import Command, PermissionLevel
from core.message import InputMessage

_log = logging.getLogger(__name__)

_HANDLER_REGISTRY: List[tuple] = []


def command(
    name: str,
    aliases: Optional[List[str]] = None,
    permission: str = "default",
    description: str = "",
):
    def wrapper(cls):
        _HANDLER_REGISTRY.append((cls, name, aliases or [], permission, description))
        return cls

    return wrapper


def register_all_commands(command_manager, **deps):
    deps["command_manager"] = command_manager
    for cls, name, aliases, perm, desc in _HANDLER_REGISTRY:
        try:
            sig = inspect.signature(cls.__init__)
            kwargs = {}
            for p_name, p in sig.parameters.items():
                if p_name == "self":
                    continue
                if p_name in deps:
                    kwargs[p_name] = deps[p_name]

            handler = cls(**kwargs)

            command_manager.register_command(
                Command(
                    name=name,
                    handler=handler.execute,
                    aliases=aliases,
                    permission=PermissionLevel(perm),
                    description=desc,
                )
            )
            _log.info(f"注册命令: {name}")
        except Exception as e:
            _log.error(f"注册命令 {name} 失败: {e}")


def make_reply(
    input_message: InputMessage,
    content: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "chat_id": (
                input_message.delivery_target.target_id
                if input_message.delivery_target is not None
                else input_message.chat_id
            ),
            "content": content,
            "message_id": input_message.id,
            "is_group": input_message.is_group,
        }
    ]


def session_key_for_message(input_message: InputMessage, agent_engine=None) -> str:
    """Return the internal key while keeping reply transport IDs raw."""
    if agent_engine is not None:
        resolver = getattr(agent_engine, "_session_key_for_message", None)
        if callable(resolver):
            return resolver(input_message)
    if input_message.chat_id.startswith(
        (
            "task:",
            "cron:",
            "heartbeat:",
            "work-plan:",
            "workplan:",
            "agent:",
            "system:",
            "exec:",
            "subagent:",
        )
    ):
        return input_message.session_key or input_message.chat_id
    return input_message.chat_id
