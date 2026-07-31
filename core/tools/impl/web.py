"""web_search / web_fetch 工具条目 — 薄封装，逻辑在 core/web_search/service.py。"""

import json
import logging

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)

WEB_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词（必填），中文/英文均可",
        },
        "count": {
            "type": "integer",
            "description": "返回结果数，1-10，默认按配置（通常 5）",
            "minimum": 1,
            "maximum": 10,
        },
        "region": {
            "type": "string",
            "description": "区域/语言（仅 DuckDuckGo 生效），如 us-en、zh-cn",
        },
        "freshness": {
            "type": "string",
            "description": "时间过滤（尽力支持）：day / week / month / year",
        },
        "safe_search": {
            "type": "string",
            "description": "安全搜索级别（仅 DuckDuckGo 生效）：strict / moderate / off",
        },
    },
    "required": ["query"],
}

WEB_FETCH_PARAMS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "要抓取的网页 URL（仅 http/https；自动拦截内网地址）",
        },
        "max_chars": {
            "type": "integer",
            "description": "返回正文的最大字符数，默认按配置（通常 20000）",
            "minimum": 1000,
            "maximum": 200000,
        },
    },
    "required": ["url"],
}


def create_web_entries(deps: ToolDeps) -> list[ToolEntry]:
    service = getattr(deps, "web", None)

    async def _web_search(args: dict, ctx: ToolContext) -> ToolResult:
        if service is None or not service.search_cfg.enabled:
            return ToolResult(
                content=json.dumps(
                    {"error": "网页搜索未启用（[web_search].enabled=false）"},
                    ensure_ascii=False,
                )
            )
        try:
            query = str(args.get("query") or "").strip()
            if not query:
                return ToolResult(
                    content=json.dumps({"error": "请提供 query"}, ensure_ascii=False)
                )
            result = await service.search(
                query=query,
                count=args.get("count"),
                region=str(args.get("region") or ""),
                freshness=str(args.get("freshness") or ""),
                safe_search=str(args.get("safe_search") or ""),
            )
            return ToolResult(content=json.dumps(result, ensure_ascii=False))
        except Exception as e:
            _log.exception("web_search 工具异常")
            return ToolResult(
                content=json.dumps({"error": f"搜索失败: {e}"}, ensure_ascii=False)
            )

    async def _web_fetch(args: dict, ctx: ToolContext) -> ToolResult:
        if service is None or not service.fetch_cfg.enabled:
            return ToolResult(
                content=json.dumps(
                    {"error": "网页抓取未启用（[web_fetch].enabled=false）"},
                    ensure_ascii=False,
                )
            )
        try:
            url = str(args.get("url") or "").strip()
            if not url:
                return ToolResult(
                    content=json.dumps({"error": "请提供 url"}, ensure_ascii=False)
                )
            result = await service.fetch(
                url=url,
                max_chars=args.get("max_chars"),
            )
            return ToolResult(content=json.dumps(result, ensure_ascii=False))
        except Exception as e:
            _log.exception("web_fetch 工具异常")
            return ToolResult(
                content=json.dumps({"error": f"抓取失败: {e}"}, ensure_ascii=False)
            )

    return [
        ToolEntry(
            name="web_search",
            section="web",
            description=(
                "网页搜索（联网）。返回 title/url/内容摘要列表，自动多 provider 回退"
                "（Ollama → Tavily → DuckDuckGo）。"
                "适合查询最新新闻、文档、实时信息；与本地文件搜索 search_content 不同，"
                "本工具搜互联网。count 限制 1-10。"
            ),
            parameters=WEB_SEARCH_PARAMS,
            handler=_web_search,
        ),
        ToolEntry(
            name="web_fetch",
            section="web",
            description=(
                "抓取单个网页内容（联网）。返回标题、正文（截断）和页面链接列表。"
                "自动回退：本地抓取 → Ollama → Tavily。"
                "内置 SSRF 防护（仅 http/https，拒绝内网地址）。"
            ),
            parameters=WEB_FETCH_PARAMS,
            handler=_web_fetch,
        ),
    ]
