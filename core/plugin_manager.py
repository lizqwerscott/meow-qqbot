"""插件管理器 — 发现、加载、统计"""

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.commands import Command, PermissionLevel
from core.plugin_base import _PLUGIN_REGISTRY, BasePlugin, PluginMeta

if TYPE_CHECKING:
    from core.command_manager import CommandManager

_log = logging.getLogger(__name__)


class PluginManager:
    """插件管理器

    扫描 plugins/ 目录下的独立插件包，加载并注册命令。
    每个插件是一个含 __init__.py 的子目录，作为独立 git 仓库维护。
    """

    def __init__(self, plugin_dir: str = "plugins"):
        self._plugin_dir = Path(plugin_dir).resolve()
        self._loaded: Dict[str, PluginMeta] = {}

    @property
    def count(self) -> int:
        return len(self._loaded)

    @property
    def loaded_plugins(self) -> Dict[str, PluginMeta]:
        return dict(self._loaded)

    def discover(self) -> List[str]:
        """扫描插件目录，返回发现的插件名列表"""
        if not self._plugin_dir.is_dir():
            _log.info(f"插件目录不存在: {self._plugin_dir}")
            return []

        plugins = []
        for entry in sorted(self._plugin_dir.iterdir()):
            if entry.is_dir() and (entry / "__init__.py").is_file():
                plugins.append(entry.name)
        return plugins

    def load_all(self, **deps: Any) -> Dict[str, PluginMeta]:
        """发现并加载所有插件"""
        discovered = self.discover()
        if not discovered:
            _log.info("未发现任何插件")
            return {}

        _log.info(f"发现 {len(discovered)} 个插件: {', '.join(discovered)}")

        # 确保插件目录在 sys.path 中（主要针对独立 git 仓库）
        plugin_root = str(self._plugin_dir.parent)
        if plugin_root not in sys.path:
            sys.path.insert(0, plugin_root)

        # 记录导入前的 _HANDLER_REGISTRY 长度，后续只注册新增的
        from core.command_handlers.base import _HANDLER_REGISTRY

        before = len(_HANDLER_REGISTRY)

        # 逐个导入插件包（触发 @plugin 和 @command 装饰器）
        for name in discovered:
            try:
                importlib.import_module(f"plugins.{name}")
            except Exception as e:
                _log.error(f"导入插件 {name} 失败: {e}", exc_info=True)

        # 从 _PLUGIN_REGISTRY 提取当前加载的插件元信息
        plugins_package = f"plugins."
        for meta in list(_PLUGIN_REGISTRY.values()):
            module = sys.modules.get(meta.cls.__module__)
            if module is None:
                continue
            if not module.__name__.startswith(plugins_package):
                continue

            self._loaded[meta.name] = meta
            meta.loaded = True

        # 仅注册插件新增的 @command 条目
        command_manager = deps.get("command_manager")
        if command_manager and len(_HANDLER_REGISTRY) > before:
            for cls, name, aliases, perm, desc in _HANDLER_REGISTRY[before:]:
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
                    _log.info(f"[插件] 注册命令: {name}")
                except Exception as e:
                    _log.error(f"[插件] 注册命令 {name} 失败: {e}")

        # 调用插件的 on_load 钩子
        for name, meta in self._loaded.items():
            try:
                instance = meta.cls()
                if isinstance(instance, BasePlugin):
                    instance.on_load(**deps)
                _log.info(f"插件已加载: {meta.name} v{meta.version}")
            except Exception as e:
                _log.error(f"调用插件 {name} on_load 失败: {e}", exc_info=True)

        _log.info(
            f"插件加载完成: {self.count} 个成功"
        )
        return dict(self._loaded)

    def get_loaded_names(self) -> List[str]:
        return list(self._loaded.keys())
