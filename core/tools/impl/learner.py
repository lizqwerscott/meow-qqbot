import json
import logging

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)


def create_learner_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _define_jargon(args: dict, ctx: ToolContext) -> ToolResult:
        learners = deps.learning_orchestrator
        if not learners:
            return ToolResult(
                content=json.dumps(
                    {"error": "学习系统未就绪"},
                    ensure_ascii=False,
                )
            )

        term = (args.get("term") or "").strip()
        definition = (args.get("definition") or "").strip()
        example = (args.get("example") or "").strip()

        if not term or not definition:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供俚语词汇和含义"},
                    ensure_ascii=False,
                )
            )

        examples = [example] if example else []
        await learners.add_jargon(
            term=term,
            definition=definition,
            examples=examples,
            added_by="AI",
            chat_id=ctx.chat_id,
        )

        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "message": f"已学习俚语「{term}」: {definition}",
                },
                ensure_ascii=False,
            )
        )

    async def _report_behavior_effect(args: dict, ctx: ToolContext) -> ToolResult:
        learners = deps.learning_orchestrator
        if not learners:
            return ToolResult(
                content=json.dumps(
                    {"error": "学习系统未就绪"},
                    ensure_ascii=False,
                )
            )

        scene = (args.get("scene_summary") or "").strip()
        action = (args.get("action_taken") or "").strip()
        effect = (args.get("effect") or "neutral").strip()

        if not scene or not action:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供场景和行为描述"},
                    ensure_ascii=False,
                )
            )

        await learners.behavior.report_effect(
            scene_summary=scene,
            action_taken=action,
            effect=effect,
            chat_id=ctx.chat_id,
        )

        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "message": f"已记录行为效果「{effect}」: {scene[:40]}..",
                },
                ensure_ascii=False,
            )
        )

    DEFINE_JARGON_PARAMS = {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "俚语词汇本身，如 'YBB'、'暴龙'",
            },
            "definition": {
                "type": "string",
                "description": "俚语的含义解释",
            },
            "example": {
                "type": "string",
                "description": "一个使用该俚语的例句（可选）",
            },
        },
        "required": ["term", "definition"],
    }

    REPORT_BEHAVIOR_PARAMS = {
        "type": "object",
        "properties": {
            "scene_summary": {
                "type": "string",
                "description": "场景简要描述，如'用户抱怨工作压力大'",
            },
            "action_taken": {
                "type": "string",
                "description": "你采取的行为策略，如'先用幽默缓和气氛，再给实用建议'",
            },
            "effect": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "效果评估：positive=用户反应积极，negative=用户反应差，neutral=无明显变化",
            },
        },
        "required": ["scene_summary", "action_taken", "effect"],
    }

    return [
        ToolEntry(
            name="define_jargon",
            section="learner",
            description="学习社群俚语/黑话。当你听到某个生疏的词汇反复出现，或者用户询问某个俚语的含义时，主动调用此工具学习并记录该俚语。",
            parameters=DEFINE_JARGON_PARAMS,
            handler=_define_jargon,
        ),
        ToolEntry(
            name="report_behavior_effect",
            section="learner",
            description="报告你刚才的回复风格是否获得了良好效果，用于学习优化未来行为。当你发现某种语气或策略让用户积极性明显提高，或用户给了负面反馈时调用。",
            parameters=REPORT_BEHAVIOR_PARAMS,
            handler=_report_behavior_effect,
        ),
    ]
