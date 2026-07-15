"""WorkspaceManager — 工作区路径管理与沙箱边界"""

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


class WorkspaceManager:
    """工作区管理器。

    管理 workspaces/ 下的私聊和群聊工作区，提供路径解析和沙箱校验。
    """

    def __init__(self, root: str = "workspaces"):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        _log.info(f"工作区根目录: {self._root.resolve()}")

    def _workspace_dir(self, is_group: bool, chat_id: str) -> Path:
        sub = "groups" if is_group else "private"
        path = self._root / sub / chat_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sandbox_dir(self, is_group: bool, chat_id: str) -> Path:
        """返回当前聊天的沙箱目录 paths/files/，自动创建。"""
        d = self._workspace_dir(is_group, chat_id) / "files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve_safe_path(self, is_group: bool, chat_id: str, relative_path: str) -> Path:
        """解析沙箱内相对路径，防止目录穿越。

        Raises ValueError 如果路径越界。
        """
        sandbox = self.sandbox_dir(is_group, chat_id)
        safe = relative_path.lstrip("/").lstrip("\\")
        target = (sandbox / safe).resolve()
        sandbox_str = str(sandbox.resolve()) + "/"
        if not str(target).startswith(sandbox_str):
            raise ValueError(f"路径越界: {relative_path}")
        return target

    def heartbeat_path(self) -> Path:
        """返回全局 HEARTBEAT.md 路径。"""
        return self._root / "HEARTBEAT.md"
