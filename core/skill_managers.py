"""SkillManagers — skillkit 封装层

提供技能发现、查看详情、执行脚本、执行命令等功能。
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from skillkit import SkillManager

_log = logging.getLogger(__name__)


ALLOWED_COMMANDS = frozenset({
    "git", "python3", "python", "pip", "node", "npm", "npx", "uv", "make",
    "ls", "cat", "echo", "pwd", "head", "tail", "wc", "sort", "uniq",
    "grep", "rg", "find", "date", "which", "du", "df",
    "curl", "wget", "ping", "dig",
    "mkdir", "cp", "mv",
})

# 这些命令无论出现在哪个位置（含管道后）都拒绝
DENIED_COMMAND_PATTERNS = re.compile(
    r"\b(rm|chmod|chown|sudo|su|dd|mkfs|shutdown|reboot|"
    r"passwd|useradd|usermod|groupadd|"
    r"setuid|setgid|chattr|lsattr|tcpdump|nmap)\b"
)


class SkillManagers:
    """SkillManager 封装，提供技能发现、查看和执行能力。"""

    def __init__(self, project_skill_dir: str = "./.agents/skills/"):
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
        _log.info(
            f"SkillManagers 已初始化: 发现 {len(self._manager.list_skills())} 个技能"
        )

    @property
    def has_skills(self) -> bool:
        return self._skills_loaded

    def get_available_skills_block(self) -> str:
        skills = self._manager.list_skills()
        if not skills:
            return ""
        lines = ["\n\n<available_skills>"]
        for s in skills:
            name = s.name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            desc = (s.description or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines.append("  <skill>")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        lines.append(
            "\n你可以使用 view_skill 工具查看某个技能的详细内容，"
            "使用 execute_skill 运行技能自带的脚本，"
            "使用 execute_command 执行任意 bash 命令。"
        )
        return "\n".join(lines)

    def get_skill_detail(self, skill_name: str) -> str:
        try:
            skill = self._manager.load_skill(skill_name)
            return skill.content
        except Exception as e:
            _log.warning(f"加载技能详情失败 [{skill_name}]: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def execute_skill_script(
        self,
        skill_name: str,
        script_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
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

    def execute_command(
        self,
        command: str,
        timeout: int = 30,
        workdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        command = command.strip()
        if not command:
            return {"success": False, "error": "命令为空"}

        first_word = command.split()[0] if command.split() else ""
        if first_word not in ALLOWED_COMMANDS:
            _log.warning(f"execute_command 被拒绝: '{first_word}' 不在白名单")
            return {
                "success": False,
                "error": f"命令 '{first_word}' 不在允许执行的白名单中",
            }

        denied_match = DENIED_COMMAND_PATTERNS.search(command.lower())
        if denied_match:
            denied_word = denied_match.group(0)
            _log.warning(f"execute_command 被拒绝: 含危险命令 '{denied_word}'")
            return {
                "success": False,
                "error": f"命令包含不允许的危险命令: '{denied_word}'",
            }

        effective_timeout = min(timeout, 120)
        cwd = workdir or "."

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=cwd,
            )
            stdout = result.stdout[-100000:] if len(result.stdout) > 100000 else result.stdout
            stderr = result.stderr[-100000:] if len(result.stderr) > 100000 else result.stderr
            truncated = {
                "stdout": len(result.stdout) > 100000,
                "stderr": len(result.stderr) > 100000,
            }

            _log.info(
                f"execute_command: exit={result.returncode} "
                f"cmd={command[:80]!r}... "
                f"stdout={len(stdout)}b stderr={len(stderr)}b"
            )

            return {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "truncated": truncated,
            }
        except subprocess.TimeoutExpired:
            _log.warning(f"execute_command 超时 [{command[:80]}..]")
            return {
                "success": False,
                "error": f"命令执行超时 ({effective_timeout}秒)",
            }
        except Exception as e:
            _log.warning(f"execute_command 执行失败: {e}")
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
