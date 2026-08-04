"""插件基类与装饰器"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

_log = logging.getLogger(__name__)

_PLUGIN_REGISTRY: Dict[str, "PluginMeta"] = {}


@dataclass
class PluginMeta:
    cls: Type
    name: str
    version: str = "1.0.0"
    description: str = ""
    loaded: bool = False


def plugin(name: str, *, version: str = "1.0.0", description: str = ""):
    """装饰器：标记一个类为插件"""

    def wrapper(cls: Type) -> Type:
        if name in _PLUGIN_REGISTRY:
            _log.warning(f"插件 {name} 已注册，将被覆盖")
        _PLUGIN_REGISTRY[name] = PluginMeta(
            cls=cls, name=name, version=version, description=description
        )
        _log.debug(f"插件注册: {name} v{version}")
        return cls

    return wrapper


class BasePlugin:
    """插件基类（可选），提供生命周期钩子"""

    def on_load(self, **deps: Any) -> None:
        """插件加载时调用，可在此注册命令、初始化资源"""

    def on_unload(self) -> None:
        """插件卸载时调用，清理资源"""
