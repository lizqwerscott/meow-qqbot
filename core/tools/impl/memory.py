import json
import logging
from typing import Optional, Tuple

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import _DEPS

_log = logging.getLogger(__name__)


async def _search_memory(args: dict, ctx: ToolContext) -> ToolResult:
    hindsight = _DEPS.get("hindsight")
    if hindsight is None:
        return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

    query = (args.get("query") or "").strip()
    if not query:
        return ToolResult(content=json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False))

    person_name = (args.get("person_name") or "").strip()
    method = args.get("method", "hybrid")

    user_id, resolved_name = _resolve_person(person_name, ctx)
    if user_id is None and person_name:
        return ToolResult(content=json.dumps(
            {"error": f"找不到「{person_name}」对应的群友"}, ensure_ascii=False,
        ))
    if user_id is None:
        user_id = ctx.sender_id
        resolved_name = "当前用户"

    top_k = _DEPS.get("search_top_k", 3)
    try:
        result = await hindsight.search(
            user_id=user_id, query=query, top_k=top_k,
            include_profile=True, method=method,
        )
    except Exception as e:
        _log.warning(f"search_memory 异常: {e}")
        return ToolResult(content=json.dumps(
            {"error": f"记忆搜索失败: {e}"}, ensure_ascii=False,
        ))

    episodes = result.get("episodes", [])
    profiles = result.get("profiles", [])

    lines = [f"关于「{resolved_name}」的检索结果："]
    if profiles:
        for p in profiles[:3]:
            pd = p.get("profile_data", {})
            if isinstance(pd, dict):
                for k, v in pd.items():
                    lines.append(f"- [{k}]: {str(v)[:200]}")
    if episodes:
        for e in episodes[:5]:
            content = e.get("summary", "") or e.get("episode", "")
            if content:
                lines.append(f"- {content[:200]}")
    if len(lines) == 1:
        lines.append("（未找到相关记忆）")

    return ToolResult(content="\n".join(lines))


async def _mark_important(args: dict, ctx: ToolContext) -> ToolResult:
    hindsight = _DEPS.get("hindsight")
    if hindsight is None:
        return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

    profile_data_str = (args.get("profile_data") or "").strip()
    summary = (args.get("summary") or "").strip()

    if not profile_data_str and not summary:
        return ToolResult(content=json.dumps(
            {"error": "请提供 profile_data 或 summary"}, ensure_ascii=False,
        ))

    stored = []
    if profile_data_str:
        try:
            profile_dict = json.loads(profile_data_str)
            if isinstance(profile_dict, dict) and profile_dict:
                facts = "；".join(f"{k}是{v}" for k, v in profile_dict.items())
                profile_msg = (
                    f"[这是关于我（{ctx.sender_id}）的自我介绍，请记住这些信息] {facts}"
                )
                await hindsight.add_message(
                    session_id=ctx.chat_id,
                    sender_id=ctx.sender_id,
                    content=profile_msg,
                    role="user",
                    context="用户自我描述",
                    timestamp=None,
                )
                stored.append("画像信息")
        except json.JSONDecodeError:
            _log.warning(f"mark_important profile_data JSON 解析失败: {profile_data_str[:100]}")

    if summary:
        await hindsight.add_message(
            session_id=ctx.chat_id,
            sender_id=ctx.sender_id,
            content=f"[重要事件] {summary}",
            role="user",
            context="重要事件记录",
            timestamp=None,
        )
        stored.append("事件摘要")

    msg = "已标记为重要记忆。"
    if stored:
        msg += f" 已记录：{'、'.join(stored)}。"

    return ToolResult(content=json.dumps(
        {"success": True, "message": msg}, ensure_ascii=False,
    ))


async def _search_relation(args: dict, ctx: ToolContext) -> ToolResult:
    hindsight = _DEPS.get("hindsight")
    if hindsight is None:
        return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

    person_a_raw = (args.get("person_a") or "").strip()
    person_b_raw = (args.get("person_b") or "").strip()
    query = (args.get("query") or "").strip()
    method = args.get("method", "hybrid")

    if not person_a_raw or not person_b_raw:
        return ToolResult(content=json.dumps(
            {"error": "请指定两个人名或昵称"}, ensure_ascii=False,
        ))

    a_id, a_name = _resolve_person(person_a_raw, ctx)
    b_id, b_name = _resolve_person(person_b_raw, ctx)

    if a_id is None:
        return ToolResult(content=json.dumps(
            {"error": f"找不到「{person_a_raw}」对应的用户"}, ensure_ascii=False,
        ))
    if b_id is None:
        return ToolResult(content=json.dumps(
            {"error": f"找不到「{person_b_raw}」对应的用户"}, ensure_ascii=False,
        ))

    async def search_for(uid, q):
        return await hindsight.search(
            user_id=uid, query=q, top_k=5, include_profile=True, method=method,
        )

    tasks = []
    task_meta = []

    tasks.append(search_for(a_id, f"{query} {b_name}" if query else b_name))
    task_meta.append(("a", a_id, a_name))

    if b_id != a_id:
        tasks.append(search_for(b_id, f"{query} {a_name}" if query else a_name))
        task_meta.append(("b", b_id, b_name))

    if ctx.sender_id not in (a_id, b_id):
        tasks.append(search_for(ctx.sender_id, f"{query} {a_name} {b_name}" if query else f"{a_name} {b_name}"))
        task_meta.append(("speaker", ctx.sender_id, "当前用户"))

    import asyncio
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    lines = [f"关于「{a_name}」和「{b_name}」的关系检索结果："]
    seen = set()

    for idx, r in enumerate(raw_results):
        role, uid, label = task_meta[idx]
        if isinstance(r, Exception):
            _log.warning(f"search_relation {role}({uid}) 失败: {r}")
            continue

        profiles = r.get("profiles", [])
        episodes = r.get("episodes", [])

        role_labels = {
            "a": f"{label} 的记忆中关于 {b_name} 的内容",
            "b": f"{label} 的记忆中关于 {a_name} 的内容",
            "speaker": "你对两人的相关记载",
        }

        if role == "a" or role == "b":
            if profiles:
                lines.append(f"【{label} 的人物画像】")
                for p in profiles[:3]:
                    pd = p.get("profile_data", {})
                    if isinstance(pd, dict):
                        for k, v in pd.items():
                            lines.append(f"- {k}: {v}")

        if episodes:
            lines.append(f"【{role_labels.get(role, '')}】")
            for e in episodes[:5]:
                content = e.get("summary", "") or e.get("episode", "")
                if content:
                    dedup_key = content[:100]
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        lines.append(f"- {content[:200]}")

    if len(lines) == 1:
        lines.append("（未找到两人相关的记忆记录）")

    return ToolResult(content="\n".join(lines))


async def _memory(args: dict, ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").strip()
    match action:
        case "search":   return await _search_memory(args, ctx)
        case "relation": return await _search_relation(args, ctx)
        case _:
            return ToolResult(content=json.dumps(
                {"error": f"未知 action: {action}，可用: search, relation"},
                ensure_ascii=False,
            ))


def _resolve_person(raw: str, ctx: ToolContext) -> Tuple[Optional[str], str]:
    nm = _DEPS.get("nickname_manager")
    if nm is None:
        return None, ""

    raw_lower = raw.strip().lower()
    if not raw_lower:
        return None, ""

    sender_aliases = nm.get_aliases(ctx.sender_id)
    sender_display = sender_aliases[0] if sender_aliases else ctx.sender_id

    if raw_lower in ("我", "自己", "myself"):
        return ctx.sender_id, "当前用户"
    if raw_lower == ctx.sender_id.lower():
        return ctx.sender_id, sender_display
    if any(raw_lower == a.lower() for a in sender_aliases):
        return ctx.sender_id, sender_display

    if not ctx.is_group:
        return None, ""

    best_score = 0
    best_candidates = []

    for uid, aliases in nm.iter_users():
        score = 0
        if raw_lower == uid.lower():
            score = 10
        elif any(raw_lower == a.lower() for a in aliases):
            score = 8
        elif any(raw_lower in a.lower() for a in aliases):
            score = 3
        elif raw_lower in uid.lower():
            score = 1

        if score > best_score:
            best_score = score
            best_candidates = [(uid, aliases[-1] if aliases else uid)]
        elif score == best_score and score > 0:
            best_candidates.append((uid, aliases[-1] if aliases else uid))

    if not best_candidates:
        return None, ""
    if len(best_candidates) == 1:
        return best_candidates[0]

    names = "、".join(f"「{n}」({u[:12]}..)" for u, n in best_candidates)
    return None, f"找到多个匹配「{raw}」的用户: {names}"


_SEARCH_FIELDS = {
    "query": {
        "type": "string",
        "description": "搜索关键词或问题，例如 '他喜欢什么'、'上次提到的新显卡'、'生日是什么时候'",
    },
    "person_name": {
        "type": "string",
        "description": "要搜索的人名或昵称（可选）。支持模糊搜索，输入昵称的一部分（如'小'）也能匹配。不填则搜索当前对话用户。私聊中不可用。",
    },
    "method": {
        "type": "string",
        "enum": ["hybrid", "agentic"],
        "description": "检索方法。hybrid（默认）适合大多数情况；agentic 适合需要深度挖掘的复杂查询。",
    },
}

_RELATION_FIELDS = {
    "person_a": {
        "type": "string",
        "description": "第一个人的人名或昵称，支持模糊搜索（部分匹配即可）。如果搜'我'或'自己'则代表当前说话者。",
    },
    "person_b": {
        "type": "string",
        "description": "第二个人的人名或昵称，支持模糊搜索（部分匹配即可）。如果搜'我'或'自己'则代表当前说话者。",
    },
}

MEMORY_PARAMS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string", "enum": ["search", "relation"],
            "description": "操作类型：search 搜索记忆 | relation 查询两人关系",
        },
        **_SEARCH_FIELDS,
        **_RELATION_FIELDS,
    },
    "required": ["action"],
}

MARK_IMPORTANT_PARAMS = {
    "type": "object",
    "properties": {
        "profile_data": {
            "type": "string",
            "description": (
                "需要记住的关于用户的结构化信息，JSON 对象格式。"
                "例如 {\"name\": \"小明\", \"likes\": \"打篮球\", \"job\": \"程序员\"}。"
                "这些信息会写入长期记忆，下次查询时将作为该用户画像返回。"
                "如果不需要记录画像则不传。"
            ),
        },
        "summary": {
            "type": "string",
            "description": (
                "需要记住的重要事件或事实的一句话摘要，"
                "将作为该用户的一条经历存入长期记忆。"
                "如果不需要记录则不传。"
            ),
        },
    },
}


def _register_all(register):
    register(ToolEntry(
        name="memory",
        section="memory",
        description=(
            "记忆搜索和关系查询工具。action=search 搜索人物画像/经历/事实（指定 person_name 可查群友）；"
            "action=relation 查询两人关系（指定 person_a + person_b）。"
            "person_name/person_a/person_b 支持模糊搜索，输入昵称的一部分也能匹配到。"
            "当需要了解某人的背景、确认某件事、查找说过的话时使用。"
        ),
        parameters=MEMORY_PARAMS,
        handler=_memory,
    ))
    register(ToolEntry(
        name="mark_important",
        section="memory",
        description=(
            "记录重要信息至长期记忆。主动判断，不需要用户每次都说'记好了'。"
            "在以下场景应主动调用：\n"
            "1. 用户明确要求'记住这个'、'记好了'\n"
            "2. 用户在解释自己的背景、喜好、习惯、个人信息\n"
            "3. 用户在描述关于自己或他人的重要事实或关系\n"
            "4. 用户分享值得长期记住的知识或信息\n"
            "5. 当前讨论出现对理解用户有重要帮助的上下文"
        ),
        parameters=MARK_IMPORTANT_PARAMS,
        handler=_mark_important,
    ))
