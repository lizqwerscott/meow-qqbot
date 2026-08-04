import json
import logging

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)


def create_user_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _search_user(args: dict, ctx: ToolContext) -> ToolResult:
        nm = deps.nickname_manager
        if nm is None:
            return ToolResult(
                content=json.dumps({"error": "昵称管理器未就绪"}, ensure_ascii=False)
            )

        query = (args.get("query") or "").strip().lower()
        if not query:
            return ToolResult(
                content=json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False)
            )

        matches = []
        for uid, aliases in nm.iter_users():
            if uid == deps.bot_id:
                continue
            score = 0
            if query == uid.lower():
                score = 10
            elif any(query == a.lower() for a in aliases):
                score = 9
            elif any(query in a.lower() for a in aliases):
                score = 5
            if score > 0:
                matches.append((score, uid, aliases[-1] if aliases else uid))

        if not matches:
            return ToolResult(
                content=json.dumps(
                    {"error": f"未找到匹配的用户: {query}"},
                    ensure_ascii=False,
                )
            )

        matches.sort(key=lambda x: -x[0])
        result = [{"user_id": uid, "nickname": name} for _, uid, name in matches[:10]]
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    SEARCH_USER_PARAMS = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如用户名、昵称或ID的一部分",
            }
        },
        "required": ["query"],
    }

    return [
        ToolEntry(
            name="search_user",
            section="user",
            description="根据昵称或昵称的一部分模糊搜索群里的用户。输入昵称的一部分（如'小'）即可找到所有匹配的人。返回用户的ID和昵称，获取到用户ID后你可以在回复中使用 <qqbot-at-user id=\"xxx\" /> 来@该用户。",
            parameters=SEARCH_USER_PARAMS,
            handler=_search_user,
        ),
    ]
