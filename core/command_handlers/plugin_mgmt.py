"""插件管理指令 — /plugin install / reload / unload / list"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+(?:\.git)?$")
_NAME_RE = re.compile(r"^[\w-]+$")


def _parse_plugin_url(url: str) -> Optional[str]:
    """从 GitHub URL 提取插件目录名"""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    match = _URL_RE.match(url)
    if not match:
        return None
    return url.rsplit("/", 1)[-1]


@command(
    name="plugin",
    aliases=["插件管理"],
    permission="admin",
    description="插件管理：/plugin install <url> | reload <name> | unload <name> | list",
)
class PluginManageCommand:
    async def execute(self, input_message: InputMessage, args: str) -> List[dict]:
        from core.plugin_manager import _current as _pm
        self.pm = _pm
        if not self.pm:
            return make_reply(input_message, "插件管理器未就绪")
        if not args:
            return make_reply(input_message, self._usage())

        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if subcmd == "install":
            return await self._install(input_message, rest)
        elif subcmd in ("reload", "重载"):
            return await self._reload(input_message, rest)
        elif subcmd in ("unload", "卸载"):
            return await self._unload(input_message, rest)
        elif subcmd in ("list", "列表"):
            return self._list(input_message)
        else:
            return make_reply(input_message, self._usage())

    def _usage(self) -> str:
        return (
            "插件管理命令：\n"
            "  /plugin install <仓库URL>  — 从 GitHub 安装\n"
            "  /plugin reload <插件名>     — 重载插件\n"
            "  /plugin unload <插件名>     — 卸载插件\n"
            "  /plugin list               — 列出已加载的插件"
        )

    async def _install(self, msg: InputMessage, url: str) -> List[dict]:
        url = url.strip()
        if not url:
            return make_reply(msg, "请提供 GitHub 仓库 URL")
        name = _parse_plugin_url(url)
        if not name:
            return make_reply(msg, "URL 格式不对，需要 GitHub 仓库地址")

        plugin_dir = os.path.join(self.pm._plugin_dir, name)
        if os.path.exists(plugin_dir):
            return make_reply(msg, f"插件 {name} 已存在，使用 /plugin reload {name} 重载")

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", url, plugin_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")[:300]
                return make_reply(msg, f"git clone 失败:\n{err}")

            _log.info(f"插件已克隆: {url} → {plugin_dir}")
        except asyncio.TimeoutError:
            return make_reply(msg, "git clone 超时（60秒），请检查网络")
        except FileNotFoundError:
            return make_reply(msg, "系统未安装 git，无法使用 install 功能")
        except Exception as e:
            return make_reply(msg, f"git clone 出错: {e}")

        # 热加载
        meta = self.pm.load_plugin(name, **self.pm._deps)
        if meta is None:
            return make_reply(msg, f"插件已安装到 {plugin_dir}，但加载失败，请检查日志")

        cmds = self.pm._plugin_commands.get(name, [])
        cmd_lines = "\n".join(f"- `/{c}`" for c in cmds) if cmds else "无命令"
        return make_reply(
            msg,
            f"**插件 `{name}` v{meta.version} 已安装并加载**\n\n{cmd_lines}",
        )

    async def _reload(self, msg: InputMessage, name: str) -> List[dict]:
        name = name.strip()
        if not name or not _NAME_RE.match(name):
            return make_reply(msg, "请提供有效的插件名")

        # 如果是 git 仓库，先 pull
        plugin_path = self.pm._plugin_dir / name
        git_dir = plugin_path / ".git"
        if git_dir.is_dir():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "-C", str(plugin_path), "pull",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0:
                    _log.info(f"插件 {name} git pull 成功")
            except Exception:
                _log.warning(f"插件 {name} git pull 失败，继续重载")

        meta = self.pm.reload_plugin(name)
        if meta is None:
            return make_reply(msg, f"**重载插件 `{name}` 失败**，请检查日志")
        cmds = self.pm._plugin_commands.get(name, [])
        cmd_lines = "\n".join(f"- `/{c}`" for c in cmds) if cmds else "无命令"
        return make_reply(msg, f"**插件 `{name}` v{meta.version} 已重载**\n\n{cmd_lines}")

    async def _unload(self, msg: InputMessage, name: str) -> List[dict]:
        name = name.strip()
        if not name or not _NAME_RE.match(name):
            return make_reply(msg, "请提供有效的插件名")
        if name not in self.pm._loaded:
            return make_reply(msg, f"插件 {name} 未加载")
        self.pm.unload_plugin(name)
        return make_reply(msg, f"插件 {name} 已卸载")

    def _list(self, msg: InputMessage) -> List[dict]:
        if self.pm.count == 0:
            return make_reply(msg, "未加载任何插件")
        lines = [f"**已加载插件** ({self.pm.count})"]
        for name, meta in self.pm.loaded_plugins.items():
            cmds = self.pm._plugin_commands.get(name, [])
            cmd_str = "`, `".join(c for c in cmds) if cmds else "—"
            lines.append(f"- **{name}** `v{meta.version}` — `{cmd_str}`")
        return make_reply(msg, "\n".join(lines))
