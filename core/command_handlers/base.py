import inspect
import logging
from typing import Any, Dict, List, Optional

from core.commands import Command, PermissionLevel
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
            "chat_id": input_message.chat_id,
            "content": content,
            "message_id": input_message.id,
            "is_group": input_message.is_group,
        }
    ]
