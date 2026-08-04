import json
import logging

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)


def create_skill_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _rescan_skills(args: dict, ctx: ToolContext) -> ToolResult:
        skill_managers = deps.skill_managers
        if not skill_managers:
            return ToolResult(
                content=json.dumps(
                    {"error": "技能系统未就绪"},
                    ensure_ascii=False,
                )
            )
        result = skill_managers.rescan()
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    async def _view_skill(args: dict, ctx: ToolContext) -> ToolResult:
        skill_managers = deps.skill_managers
        if not skill_managers:
            return ToolResult(
                content=json.dumps(
                    {"error": "技能系统未就绪"},
                    ensure_ascii=False,
                )
            )
        skill_name = (args.get("skill_name") or "").strip()
        if not skill_name:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供技能名称"},
                    ensure_ascii=False,
                )
            )
        content = skill_managers.get_skill_detail(skill_name)
        return ToolResult(content=content)

    async def _execute_skill(args: dict, ctx: ToolContext) -> ToolResult:
        skill_managers = deps.skill_managers
        if not skill_managers:
            return ToolResult(
                content=json.dumps(
                    {"error": "技能系统未就绪"},
                    ensure_ascii=False,
                )
            )
        skill_name = (args.get("skill_name") or "").strip()
        script_name = (args.get("script_name") or "").strip()
        if not skill_name or not script_name:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供技能名称和脚本名称"},
                    ensure_ascii=False,
                )
            )
        arguments = args.get("arguments") or {}
        perm = deps.permission_manager
        default_timeout = perm.get_default_timeout() if perm else 60
        timeout = args.get("timeout", default_timeout)

        result = skill_managers.execute_skill_script(
            skill_name=skill_name,
            script_name=script_name,
            arguments=arguments,
            timeout=timeout,
        )
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    RESCAN_SKILLS_PARAMS = {
        "type": "object",
        "properties": {},
    }

    VIEW_SKILL_PARAMS = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "要查看的技能名称，从 <available_skills> 中获取",
            },
        },
        "required": ["skill_name"],
    }

    EXECUTE_SKILL_PARAMS = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "技能名称",
            },
            "script_name": {
                "type": "string",
                "description": "脚本名称（无需后缀，如 'release'、'extract'）",
            },
            "arguments": {
                "type": "object",
                "description": "传递给脚本的参数（JSON 对象）",
                "additionalProperties": True,
            },
            "timeout": {
                "type": "integer",
                "description": "执行超时时间（秒），默认 30，最大 120",
            },
        },
        "required": ["skill_name", "script_name"],
    }

    return [
        ToolEntry(
            name="rescan_skills",
            section="skill",
            description="重新扫描 skills 目录，刷新可用技能列表。",
            parameters=RESCAN_SKILLS_PARAMS,
            handler=_rescan_skills,
        ),
        ToolEntry(
            name="view_skill",
            section="skill",
            description="查看并加载某个技能的完整指导说明。",
            parameters=VIEW_SKILL_PARAMS,
            handler=_view_skill,
        ),
        ToolEntry(
            name="execute_skill",
            section="skill",
            description="执行技能附带的脚本（如自动化分析、代码生成等）。参数以 JSON 形式传入。",
            parameters=EXECUTE_SKILL_PARAMS,
            handler=_execute_skill,
        ),
    ]
