"""插件管理器 — 发现、加载、卸载、重载"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from core.managers.command_manager import Command, PermissionLevel
from core.plugins.base import _PLUGIN_REGISTRY, BasePlugin, PluginMeta

if TYPE_CHECKING:
    from core.managers.command_manager import CommandManager

_log = logging.getLogger(__name__)

_current: "PluginManager" = None


class PluginManager:
    """插件管理器

    扫描 plugins/ 目录下的独立插件包，加载并注册命令。
    支持热加载/卸载。
    每个插件是一个含 __init__.py 的子目录，作为独立 git 仓库维护。
    """

    def __init__(self, plugin_dir: str = "plugins"):
        global _current
        _current = self
        self._plugin_dir = Path(plugin_dir).resolve()
        self._loaded: Dict[str, PluginMeta] = {}
        self._plugin_commands: Dict[str, List[str]] = {}
        self._deps: Dict[str, Any] = {}
        self._plugin_registry_ranges: Dict[str, Tuple[int, int]] = {}

    @property
    def count(self) -> int:
        return len(self._loaded)

    @property
    def loaded_plugins(self) -> Dict[str, PluginMeta]:
        return dict(self._loaded)

    def discover(self) -> List[str]:
        if not self._plugin_dir.is_dir():
            _log.info(f"插件目录不存在: {self._plugin_dir}")
            return []

        plugins = []
        for entry in sorted(self._plugin_dir.iterdir()):
            if entry.is_dir() and (entry / "__init__.py").is_file():
                plugins.append(entry.name)
        return plugins

    def _ensure_sys_path(self):
        plugin_root = str(self._plugin_dir.parent)
        if plugin_root not in sys.path:
            sys.path.insert(0, plugin_root)

    def load_all(self, **deps: Any) -> Dict[str, PluginMeta]:
        self._deps = dict(deps)
        discovered = self.discover()
        if not discovered:
            _log.info("未发现任何插件")
            return {}

        _log.info(f"发现 {len(discovered)} 个插件: {', '.join(discovered)}")
        self._ensure_sys_path()

        for name in discovered:
            try:
                self.load_plugin(name, **deps)
            except Exception as e:
                _log.error(f"加载插件 {name} 失败: {e}", exc_info=True)

        global _current
        _current = self
        _log.info(f"插件加载完成: {self.count} 个成功")
        return dict(self._loaded)

    def load_plugin(self, name: str, **deps: Any) -> Optional[PluginMeta]:
        from core.command_handlers.base import _HANDLER_REGISTRY

        self._deps = dict(deps)
        self._ensure_sys_path()

        before = len(_HANDLER_REGISTRY)
        before_pm = len(_PLUGIN_REGISTRY)

        module_path = f"plugins.{name}"
        try:
            importlib.import_module(module_path)
        except Exception as e:
            _log.error(f"导入插件 {name} 失败: {e}", exc_info=True)
            return None

        after = len(_HANDLER_REGISTRY)
        new_handler_entries = _HANDLER_REGISTRY[before:]

        meta = None
        for m in list(_PLUGIN_REGISTRY.values()):
            mod = sys.modules.get(m.cls.__module__)
            if mod and mod.__name__ == module_path:
                meta = m
                break

        if meta is None:
            _log.info(f"插件 {name} 未使用 @plugin 装饰器，将根据目录名注册元信息")
            meta = PluginMeta(cls=type(name, (), {}), name=name)

        self._loaded[name] = meta
        meta.loaded = True
        self._plugin_registry_ranges[name] = (before, after)

        command_manager = deps.get("command_manager")
        registered_cmds = []
        if command_manager and new_handler_entries:
            for cls, cmd_name, aliases, perm, desc in new_handler_entries:
                try:
                    sig = inspect.signature(cls.__init__)
                    kwargs = {}
                    for p_name, p in sig.parameters.items():
                        if p_name == "self":
                            continue
                        if p_name in deps:
                            kwargs[p_name] = deps[p_name]
                    handler = cls(**kwargs)
                    cmd = Command(
                        name=cmd_name,
                        handler=handler.execute,
                        aliases=aliases,
                        permission=PermissionLevel(perm),
                        description=desc,
                    )
                    command_manager.register_command(cmd)
                    registered_cmds.append(cmd_name)
                    _log.info(f"[插件:{name}] 注册命令: {cmd_name}")
                except Exception as e:
                    _log.error(f"[插件:{name}] 注册命令 {cmd_name} 失败: {e}")
        self._plugin_commands[name] = registered_cmds

        if isinstance(meta.cls, type) and issubclass(meta.cls, BasePlugin):
            try:
                instance = meta.cls()
                instance.on_load(**deps)
            except Exception as e:
                _log.error(f"插件 {name} on_load 失败: {e}", exc_info=True)

        global _current
        _current = self
        _log.info(f"插件已加载: {meta.name} v{meta.version}")
        return meta

    def unload_plugin(self, name: str) -> bool:
        meta = self._loaded.get(name)
        if meta is None:
            _log.warning(f"插件 {name} 未加载，无法卸载")
            return False

        if isinstance(meta.cls, type) and issubclass(meta.cls, BasePlugin):
            try:
                instance = meta.cls()
                instance.on_unload()
            except Exception as e:
                _log.warning(f"插件 {name} on_unload 失败: {e}")

        command_manager = self._deps.get("command_manager")
        if command_manager:
            for cmd_name in self._plugin_commands.get(name, []):
                removed = command_manager.unregister_command(cmd_name)
                if removed:
                    _log.info(f"[插件:{name}] 注销命令: {cmd_name}")

        self._loaded.pop(name, None)
        self._plugin_commands.pop(name, None)
        self._plugin_registry_ranges.pop(name, None)

        _log.info(f"插件已卸载: {name}")
        return True

    def reload_plugin(self, name: str) -> Optional[PluginMeta]:
        if name in self._loaded:
            self.unload_plugin(name)

        module_name = f"plugins.{name}"
        for m in list(sys.modules.keys()):
            if m == module_name or m.startswith(f"{module_name}."):
                del sys.modules[m]

        from core.command_handlers.base import _HANDLER_REGISTRY
        from core.plugins.base import _PLUGIN_REGISTRY

        _HANDLER_REGISTRY[:] = [
            entry for entry in _HANDLER_REGISTRY
            if not self._is_entry_from_plugin(entry, name)
        ]

        old_keys = [k for k, v in _PLUGIN_REGISTRY.items() if v.name == name]
        for k in old_keys:
            del _PLUGIN_REGISTRY[k]

        return self.load_plugin(name, **self._deps)

    def _is_entry_from_plugin(self, entry: tuple, plugin_name: str) -> bool:
        cls = entry[0]
        module = sys.modules.get(cls.__module__)
        if module is None:
            return False
        expected_prefix = f"plugins.{plugin_name}"
        return module.__name__ == expected_prefix or module.__name__.startswith(f"{expected_prefix}.")

    def get_loaded_names(self) -> List[str]:
        return list(self._loaded.keys())
