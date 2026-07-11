from core.command_handlers.base import command, register_all_commands, _HANDLER_REGISTRY

import core.command_handlers.help
import core.command_handlers.status
import core.command_handlers.emoji_list
import core.command_handlers.emoji_info
import core.command_handlers.emoji_edit
import core.command_handlers.emoji_reset
import core.command_handlers.history
import core.command_handlers.skills
import core.command_handlers.approval_test
import core.command_handlers.plugin_mgmt

__all__ = ["command", "register_all_commands", "_HANDLER_REGISTRY"]
