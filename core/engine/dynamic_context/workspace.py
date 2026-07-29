import logging
from typing import Optional

_log = logging.getLogger(__name__)


class WorkspaceBlockBuilder:
    """构建工作区上下文 + HEARTBEAT.md 动态块。"""

    def __init__(self, workspace_manager, perm, admin_ids) -> None:
        self._workspace_manager = workspace_manager
        self._perm = perm
        self._admin_ids = admin_ids

    async def build(
        self,
        *,
        chat_id: str,
        is_group: bool,
        sender_id: str,
    ) -> Optional[str]:
        if not self._workspace_manager:
            return None

        parts = []
        ws_type = "群聊" if is_group else "私聊"

        _admin_chat = False
        if not is_group:
            if self._perm:
                role = self._perm.get_user_role(sender_id)
                _admin_chat = self._perm.is_admin_role(role)
            else:
                _admin_chat = sender_id in self._admin_ids

        if _admin_chat:
            ws_root = str(self._workspace_manager.root_dir())
            parts.append(
                f"管理员工作区: {ws_root}/\n"
                "目录：HEARTBEAT.md（可选）| "
                "groups/{群聊ID}/files/ | private/{私聊ID}/files/\n"
                "\n"
                "你处于管理员模式，文件/搜索工具可访问整个 workspaces/ 目录。\n"
                "路径使用相对于工作区根目录的相对路径"
                "（如 'HEARTBEAT.md'、'groups/xxx/files/note.txt'）。\n"
                "访问 workspaces/ 外的文件用 .. 路径越界，"
                "系统会发送审批请求。\n"
                "使用 list_dir 可浏览 groups/ 和 private/ "
                "查看其他会话的工作区。"
            )

            hb_path = self._workspace_manager.heartbeat_path()
            if hb_path.exists():
                parts.append(
                    "【心跳配置 (HEARTBEAT.md)】\n"
                    "心跳配置文件存在于 workspaces/HEARTBEAT.md，"
                    "你可以使用 read_file 工具查看和 write_file 工具修改。"
                    "心跳执行时 AI 会自主读取此文件。"
                )
            else:
                parts.append(
                    "【心跳配置 (HEARTBEAT.md)】\n"
                    "你可以在 workspaces/ 根目录创建 HEARTBEAT.md "
                    "来定义心跳检查清单，"
                    "文件不存在时心跳自动跳过。"
                    "使用 write_file 工具写入 HEARTBEAT.md 即可。"
                )

            parts.append(
                "exec 工具默认工作目录与文件工具工作区根目录一致（workspaces/）。"
                "如需在项目根目录执行命令，设置 workdir='.'。"
            )
        else:
            sandbox = str(
                self._workspace_manager.sandbox_dir(is_group, chat_id)
            )
            parts.append(f"当前{ws_type}工作区: {sandbox}/")
            parts.append(
                "你的文件工作区位于上述 files/ 目录下。"
                "文件工具 (read_file / write_file / edit_file / apply_patch "
                "/ list_dir) 和搜索工具 (search_content / find_files) "
                "均限当前工作区内使用。"
                "文件路径请使用相对于工作区的相对路径（例如 'memo.txt'），"
                "不要使用绝对路径。"
            )
            parts.append(
                "exec 工具默认工作目录与此一致。"
            )

        return "\n\n".join(parts)
