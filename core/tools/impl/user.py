import json
import logging

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)


def create_user_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _search_user(args: dict, ctx: ToolContext) -> ToolResult:
        query = (args.get("query") or "").strip().lower()
        if not query:
            return ToolResult(
                content=json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False)
            )
        identity_manager = getattr(deps, "identity_manager", None)
        target = ctx.delivery_target
        if identity_manager is None or target is None:
            return ToolResult(
                content=json.dumps({"error": "身份管理器未就绪"}, ensure_ascii=False)
            )
        if target.chat_type != "group":
            return ToolResult(
                content=json.dumps(
                    {"error": "用户搜索仅限群聊"},
                    ensure_ascii=False,
                )
            )
        matches = identity_manager.search(target, query, limit=10)
        if not matches:
            return ToolResult(
                content=json.dumps(
                    {"error": f"未找到匹配的用户: {query}"},
                    ensure_ascii=False,
                )
            )
        result = [
            {
                "identity_ref": ref.identity_ref,
                "person_ref": ref.person_ref or None,
                "current_name": ref.chat_name or ref.current_name,
                "historical_names": list(ref.historical_names),
                "scope": ref.scope,
            }
            for ref in matches
        ]
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
            description="在当前群范围内根据当前或历史昵称模糊搜索用户。返回匿名 identity_ref、统一人物引用和名字；同名时返回全部候选，不返回真实平台 ID。",
            parameters=SEARCH_USER_PARAMS,
            handler=_search_user,
        ),
    ]
