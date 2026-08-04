import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.managers.command_manager import CommandManager
from core.message import InputMessage

_log = logging.getLogger(__name__)


def _plugin_command_names() -> Dict[str, List[str]]:
    """返回 {插件名: [命令名, ...]}"""
    from core.plugins.manager import _current as _pm

    if not _pm:
        return {}
    return dict(_pm._plugin_commands)


@command(name="帮助", aliases=["help"], description="显示命令帮助")
class HelpCommand:
    def __init__(self, command_manager: CommandManager):
        self.command_manager = command_manager

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        try:
            user_id = input_message.sender_id
            all_commands = self.command_manager.get_all_commands()

            # 获取插件命令名集合
            plugin_cmd_names = set()
            plugin_map = _plugin_command_names()
            for names in plugin_map.values():
                plugin_cmd_names.update(names)

            # 分类内置命令和插件命令
            builtin_cmds = []
            plugin_cmds = {}  # 插件名 → [(name, aliases, desc), ...]
            for cmd in all_commands:
                if not self.command_manager.has_permission(cmd, user_id):
                    continue
                if cmd.name in plugin_cmd_names:
                    # 归属到对应插件
                    for pname, names in plugin_map.items():
                        if cmd.name in names:
                            plugin_cmds.setdefault(pname, []).append(cmd)
                            break
                else:
                    builtin_cmds.append(cmd)

            lines = ["**可用命令**\n"]
            if builtin_cmds:
                lines.append("**系统命令**")
                for cmd in builtin_cmds:
                    aliases_str = (
                        f" (`{', '.join(cmd.aliases)}`)" if cmd.aliases else ""
                    )
                    lines.append(f"- `{cmd.name}`{aliases_str} — {cmd.description}")

            for pname, cmds in plugin_cmds.items():
                lines.append("")
                lines.append(f"**插件: {pname}**")
                for cmd in cmds:
                    aliases_str = (
                        f" (`{', '.join(cmd.aliases)}`)" if cmd.aliases else ""
                    )
                    lines.append(f"- `{cmd.name}`{aliases_str} — {cmd.description}")

            return make_reply(input_message, "\n".join(lines))
        except Exception as e:
            _log.error(f"帮助命令处理失败: {e}")
            return []
