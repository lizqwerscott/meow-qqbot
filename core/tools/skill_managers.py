"""SkillManagers — skillkit 封装层

提供技能发现、查看详情、执行脚本、执行命令等功能。
"""

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from skillkit import SkillManager

_log = logging.getLogger(__name__)


DENIED_COMMANDS: frozenset = frozenset({
    "rm", "chmod", "chown", "sudo", "su", "doas",
    "dd", "mkfs", "fdisk", "parted", "mkswap",
    "shutdown", "reboot", "poweroff", "halt", "init", "systemctl",
    "useradd", "usermod", "groupadd", "userdel", "groupdel",
    "setuid", "setgid", "chattr", "lsattr",
    "tcpdump", "nmap", "tshark",
    "pkill", "killall", "kill", "passwd",
    "service", "grub-install", "grub-mkconfig",
    "modprobe", "insmod", "rmmod",
    "iptables", "ufw",
})

_DANGEROUS_TARGET_PATTERNS = re.compile(r">/(?:dev|etc|boot|sys|proc)/")


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

    @staticmethod
    def _parse_command_safe(raw_command: str) -> Optional[List[str]]:
        """安全地将命令字符串解析为 args 列表。返回 None 表示解析失败。"""
        try:
            parts = shlex.split(raw_command)
        except ValueError:
            return None
        if not parts:
            return None
        return parts

    @staticmethod
    def _check_command_safe(parts: List[str]) -> Optional[str]:
        """检查命令是否安全。返回 None 表示通过，否则返回拒绝原因。"""
        cmd_name = os.path.basename(parts[0])
        if cmd_name in DENIED_COMMANDS:
            return f"命令 '{cmd_name}' 被禁止执行"
        for arg in parts[1:]:
            if _DANGEROUS_TARGET_PATTERNS.search(arg):
                return f"参数包含危险的重定向目标: {arg[:60]}"
        return None

    def get_skill_system_intro(self) -> str:
        return (
            "--- 技能系统 ---\n"
            "以下 <available_skills> 中列出的是我掌握的专业领域知识包（Skills）。\n"
            "每个技能包含该领域的完整思考框架、方法论和操作指南。\n"
            "\n"
            "当用户的问题涉及某个领域时，我应当：\n"
            "1. 使用 view_skill 查看该技能的完整指导说明\n"
            "2. 将技能中的方法论融入当前思考，按它的框架来处理问题\n"
            "3. 如果技能附带可执行脚本，使用 execute_skill 运行\n"
            "\n"
            "多个技能可以组合使用。如果任务涉及多个步骤或不同专业领域，\n"
            "先用 view_skill 加载第一个技能的指导说明执行相关步骤，\n"
            "再用 view_skill 加载下一个技能的指导说明继续处理。\n"
            "\n"
            "注意：技能 ≠ 工具。\n"
            "技能是「怎么思考」—— 一种专业方法的注入，\n"
            "工具是「怎么执行」—— 一个具体的操作。\n"
            "使用技能意味着我吸收了该领域的专业知识来指导行为。"
        )

    def get_skill_entries_block(self) -> str:
        skills = self._manager.list_skills()
        if not skills:
            return ""
        lines = ["<available_skills>"]
        for s in skills:
            name = s.name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            desc = (s.description or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
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

        # 安全解析命令字符串为 args 列表
        parts = self._parse_command_safe(command)
        if parts is None:
            _log.warning(f"execute_command 命令格式无效: {command[:80]}")
            return {
                "success": False,
                "error": f"命令格式无效（引号不匹配等）: {command[:80]}",
            }

        # 安全检查 — 黑名单 + 危险重定向
        reason = self._check_command_safe(parts)
        if reason:
            _log.warning(f"execute_command 被拒绝: {reason}")
            return {"success": False, "error": reason}

        effective_timeout = min(timeout, 120)
        cwd = workdir or "."

        try:
            result = subprocess.run(
                parts,
                shell=False,
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
