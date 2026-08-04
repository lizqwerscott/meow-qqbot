import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage
from core.tools.skill_managers import SkillManagers

_log = logging.getLogger(__name__)


@command(
    name="技能列表",
    aliases=["skills"],
    permission="admin",
    description="查看所有已安装的技能（管理员专用）",
)
class SkillsListCommand:
    def __init__(self, skill_managers: SkillManagers):
        self.skill_managers = skill_managers

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if not self.skill_managers.has_skills:
            return make_reply(input_message, "当前没有安装任何技能。")

        skills = self.skill_managers.list_skill_names()
        lines = [f"当前已安装技能（共 {len(skills)} 个）："]
        for name in skills:
            detail = self.skill_managers.get_skill_detail(name)
            desc = ""
            for line in detail.splitlines():
                if line.startswith("description:"):
                    desc = line[len("description:") :].strip().strip('"').strip("'")
                    break
            if desc:
                lines.append(f"  • {name} — {desc}")
            else:
                lines.append(f"  • {name}")

        return make_reply(input_message, "\n".join(lines))


@command(
    name="技能重新扫描",
    aliases=["rescan_skills", "技能刷新"],
    permission="admin",
    description="重新扫描技能目录，发现新增或移除的技能（管理员专用）",
)
class SkillsRescanCommand:
    def __init__(self, skill_managers: SkillManagers):
        self.skill_managers = skill_managers

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        result = self.skill_managers.rescan()
        return make_reply(
            input_message,
            f"技能重新扫描完成: {result['before']} → {result['after']} 个技能",
        )
