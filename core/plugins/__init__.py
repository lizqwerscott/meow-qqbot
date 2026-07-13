"""插件系统模块"""

from core.plugins.base import BasePlugin, PluginMeta, plugin, _PLUGIN_REGISTRY
from core.plugins.manager import PluginManager, _current

__all__ = ["BasePlugin", "PluginMeta", "plugin", "_PLUGIN_REGISTRY", "PluginManager", "_current"]
