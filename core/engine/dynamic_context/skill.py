import logging
from typing import Optional

_log = logging.getLogger(__name__)


class SkillBlockBuilder:
    """构建技能条目动态块。"""

    def __init__(self, skill_managers) -> None:
        self._skill_managers = skill_managers

    async def build(
        self, max_skills: int = 0, max_desc_chars: int = 0
    ) -> Optional[str]:
        if not self._skill_managers or not self._skill_managers.has_skills:
            return None
        entries = self._skill_managers.get_skill_entries_block(
            max_skills=max_skills, max_desc_chars=max_desc_chars,
        )
        return entries if entries else None
