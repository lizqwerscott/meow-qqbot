"""SkillManagers — skillkit 封装层

提供技能发现、查看详情、执行脚本、执行命令等功能。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from skillkit import SkillManager

_log = logging.getLogger(__name__)


class SkillManagers:
    """SkillManager 封装，提供技能发现、查看和执行能力。"""

    def __init__(self, project_skill_dir: str = "./.agents/skills/", permission_manager=None):
        skill_path = Path(project_skill_dir)
        if skill_path.exists() and skill_path.is_dir():
            self._manager = SkillManager(project_skill_dir=str(skill_path.resolve()))
        else:
            _log.warning(
                f"技能目录不存在: {skill_path}，使用默认发现路径"
            )
            self._manager = SkillManager(
                project_skill_dir="",
                anthropic_config_dir=None,
            )
        self._manager.discover()
        self._skills_loaded = len(self._manager.list_skills()) > 0
        self._perm = permission_manager
        _log.info(
            f"SkillManagers 已初始化: 发现 {len(self._manager.list_skills())} 个技能"
        )

    @property
    def has_skills(self) -> bool:
        return self._skills_loaded

    def get_skill_system_intro(self) -> str:
        return (
            "--- 技能系统 ---\n"
            "以下 <available_skills> 中列出的是技能（Skills）—— 各领域的专业知识包，\n"
            "包含完整的思考框架、方法论和操作指南。\n"
            "\n"
            "**使用方式：**\n"
            "- 涉及某个领域时，使用 view_skill 加载技能的指导说明\n"
            "- 将技能中的方法论融入思考，按框架处理问题\n"
            "- 附带可执行脚本时使用 execute_skill 运行\n"
            "\n"
            "多个技能可组合使用。多步骤或跨领域任务时，\n"
            "按顺序逐个加载技能，完成一个再加载下一个。\n"
            "\n"
            "技能定义了「怎么思考」（方法论的注入），工具定义了「怎么执行」（具体操作）。\n"
            "使用技能意味着吸收该领域的专业知识来指导行为。"
        )

    def get_skill_entries_block(self, max_skills: int = 0, max_desc_chars: int = 0) -> str:
        skills = self._manager.list_skills()
        if not skills:
            return ""

        if max_skills > 0:
            skills = skills[:max_skills]

        lines = ["<available_skills>"]
        for s in skills:
            name = s.name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            desc = (s.description or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if max_desc_chars > 0 and len(desc) > max_desc_chars:
                desc = desc[:max_desc_chars] + "..."
            lines.append("  <skill>")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_available_skills_block(self) -> str:
        intro = self.get_skill_system_intro()
        entries = self.get_skill_entries_block()
        if not entries:
            return ""
        return "\n\n" + intro + "\n\n" + entries

    def get_skill_detail(self, skill_name: str) -> str:
        try:
            skill = self._manager.load_skill(skill_name)
            return skill.content
        except Exception as e:
            _log.warning(f"加载技能详情失败 [{skill_name}]: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _default_timeout(self) -> int:
        return self._perm.get_default_timeout() if self._perm else 60

    def _max_timeout(self) -> int:
        return self._perm.get_max_timeout() if self._perm else 300

    def execute_skill_script(
        self,
        skill_name: str,
        script_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if timeout is None:
            timeout = self._default_timeout()
        try:
            result = self._manager.execute_skill_script(
                skill_name=skill_name,
                script_name=script_name,
                arguments=arguments or {},
                timeout=timeout,
            )
            return {
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "execution_time_ms": result.execution_time_ms,
            }
        except Exception as e:
            _log.warning(
                f"执行 skill 脚本失败 [{skill_name}/{script_name}]: {e}"
            )
            return {"success": False, "error": str(e)}

    def list_skill_names(self) -> list:
        return self._manager.list_skills(include_qualified=True)

    def rescan(self) -> Dict[str, Any]:
        before = len(self._manager.list_skills())
        self._manager.discover()
        after = len(self._manager.list_skills())
        self._skills_loaded = after > 0
        _log.info(f"技能重新扫描: {before} → {after} 个技能")
        return {
            "success": True,
            "before": before,
            "after": after,
            "available_skills": self.get_available_skills_block(),
        }
