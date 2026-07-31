"""web_search / web_fetch 服务层与工具接线测试。

用 httpx.MockTransport 拦截所有 HTTP，不触网。
"""

import json

import httpx

from core.tools._types import ToolContext
from core.tools.policy import ChatContext, build_tools
from core.web_search.config import WebFetchConfig, WebSearchConfig
from core.web_search.service import WebService

DDG_HTML = """<html><body>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F1&amp;rut=x">First Result</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=...">Snippet one.</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://example.com/2">Second Result</a>
  <a class="result__snippet" href="https://example.com/2">Snippet two.</a>
</div>
</body></html>"""


def make_service(
    search_cfg: dict | None = None, fetch_cfg: dict | None = None, handler=None
):
    """构造 WebService + MockTransport 的 AsyncClient。"""
    search = WebSearchConfig.from_dict(
        search_cfg
        or {"enabled": True, "providers": ["tavily"], "tavily_api_key": "tvly-x"}
    )
    fetch = WebFetchConfig.from_dict(
        fetch_cfg or {"enabled": True, "providers": ["local"]}
    )
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, timeout=10)
    service = WebService(search, fetch, client, env={})
    return service, client


def json_response(payload: dict, status: int = 200):
    return httpx.Response(
        status, json=payload, request=httpx.Request("POST", "http://test")
    )


# ── 链路与回退 ─────────────────────────────────────────────────────


async def test_search_primary_fails_then_fallback():
    """ollama 401 → 自动切 tavily，provider 标签正确。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "ollama.com" in str(request.url):
            return json_response({"error": "unauthorized"}, status=401)
        if "tavily.com" in str(request.url):
            return json_response(
                {
                    "results": [
                        {
                            "title": "T",
                            "url": "https://tavily.example",
                            "content": "c1",
                        },
                        {
                            "title": "T2",
                            "url": "https://tavily.example/2",
                            "content": "c2",
                        },
                    ]
                }
            )
        return json_response({"results": []})

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["ollama", "tavily"],
            "ollama_base_url": "https://ollama.com",
            "ollama_api_key": "k",
            "tavily_api_key": "tvly-x",
        },
        handler=handler,
    )
    try:
        out = await service.search("hello")
        assert out["success"] is True
        assert out["provider"] == "tavily"
        assert len(out["results"]) == 2
        assert calls  # ollama 被调用过
    finally:
        await client.aclose()


async def test_search_all_providers_fail():
    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily"],
            "tavily_api_key": "tvly-x",
        },
        handler=lambda req: json_response({"error": "boom"}, status=500),
    )
    try:
        out = await service.search("hello")
        assert out["success"] is False
        assert "tavily" in out["error"]
    finally:
        await client.aclose()


async def test_search_skips_uncredentialed_provider():
    """strict_credential_skip=true：tavily 无 key 直接跳过，不发起请求。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "duckduckgo.com" in str(request.url):
            return httpx.Response(200, text=DDG_HTML, request=request)
        return json_response({"results": []})

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily", "duckduckgo"],
            "tavily_api_key": "",
        },
        handler=handler,
    )
    try:
        out = await service.search("hello")
        assert out["success"] is True
        assert out["provider"] == "duckduckgo"
        assert all("tavily.com" not in c for c in calls)
    finally:
        await client.aclose()


async def test_search_empty_no_fallback_by_default():
    """fallback_on_empty=false：空结果不触发回退。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "tavily.com" in str(request.url):
            return json_response({"results": []})
        if "duckduckgo.com" in str(request.url):
            return httpx.Response(200, text=DDG_HTML, request=request)
        return json_response({"results": []})

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily", "duckduckgo"],
            "tavily_api_key": "tvly-x",
            "fallback_on_empty": False,
        },
        handler=handler,
    )
    try:
        out = await service.search("hello")
        assert out["success"] is True
        assert out["provider"] == "tavily"
        assert out["results"] == []
        assert len(calls) == 1  # 只调用了一次
    finally:
        await client.aclose()


async def test_search_empty_fallback_when_enabled():
    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily", "duckduckgo"],
            "tavily_api_key": "tvly-x",
            "fallback_on_empty": True,
        },
        handler=lambda req: (
            json_response({"results": []})
            if "tavily.com" in str(req.url)
            else httpx.Response(200, text=DDG_HTML, request=req)
        ),
    )
    try:
        out = await service.search("hello")
        assert out["provider"] == "duckduckgo"
        assert len(out["results"]) == 2
    finally:
        await client.aclose()


async def test_search_cache_single_request():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return json_response(
            {"results": [{"title": "T", "url": "https://x.example", "content": "c"}]}
        )

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily"],
            "tavily_api_key": "tvly-x",
        },
        handler=handler,
    )
    try:
        o1 = await service.search("same")
        o2 = await service.search("same")
        assert o1 == o2
        assert len(calls) == 1
    finally:
        await client.aclose()


async def test_search_duckduckgo_parse_via_service():
    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["duckduckgo"],
        },
        handler=lambda req: httpx.Response(200, text=DDG_HTML, request=req),
    )
    try:
        out = await service.search("hello")
        assert out["provider"] == "duckduckgo"
        assert out["results"][0]["url"] == "https://example.com/1"
        assert out["results"][0]["title"] == "First Result"
        assert "Snippet one" in out["results"][0]["content"]
    finally:
        await client.aclose()


async def test_search_result_content_truncation():
    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily"],
            "tavily_api_key": "tvly-x",
            "result_content_cap": 100,
        },
        handler=lambda req: json_response(
            {
                "results": [
                    {"title": "T", "url": "https://x.example", "content": "x" * 1000}
                ]
            }
        ),
    )
    try:
        out = await service.search("hello")
        assert len(out["results"][0]["content"]) <= 101
    finally:
        await client.aclose()


async def test_search_max_results_clamped():
    """count 超出上限(10)时钳制到 10，不抛错。"""
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "tavily.com" in str(request.url):
            body = json.loads(request.content)
            seen.append(body.get("max_results"))
            return json_response({"results": []})
        return json_response({"results": []})

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["tavily"],
            "tavily_api_key": "tvly-x",
        },
        handler=handler,
    )
    try:
        out = await service.search("hello", count=999)
        assert out["success"] is True
        assert seen == [10]
    finally:
        await client.aclose()


async def test_search_ollama_hosted_fallback_targets_ollama_com():
    """本地 ollama 配了 key 时，托管 fallback 必须指向 https://ollama.com 且本地端点不携带 key。"""
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization", "")))
        if "ollama.com" in str(request.url):
            return json_response(
                {
                    "results": [
                        {"title": "T", "url": "https://h.example", "content": "c"}
                    ]
                }
            )
        return json_response({"error": "not signed in"}, status=401)

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["ollama"],
            "ollama_api_key": "sk-secret",
            # 默认本地 host 127.0.0.1:11434
        },
        handler=handler,
    )
    try:
        out = await service.search("hello")
        assert out["success"] is True
        assert out["provider"] == "ollama"
        # 本地端点被调用且无 Authorization；托管端点带 key
        assert any("127.0.0.1:11434" in u and not auth for u, auth in seen)
        assert any(
            "https://ollama.com" in u and auth == "Bearer sk-secret" for u, auth in seen
        )
    finally:
        await client.aclose()


async def test_search_ollama_hosted_without_key_skipped():
    """strict_credential_skip：托管 ollama 无 key → 直接跳过。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text=DDG_HTML, request=request)

    service, client = make_service(
        search_cfg={
            "enabled": True,
            "providers": ["ollama", "duckduckgo"],
            "ollama_base_url": "https://ollama.com",
            "ollama_api_key": "",
        },
        handler=handler,
    )
    try:
        out = await service.search("hello")
        assert out["provider"] == "duckduckgo"
        assert all("ollama.com" not in c for c in calls)
    finally:
        await client.aclose()


async def test_search_ollama_hosted_fallback_targets_ollama_com_fetch():
    """ollama_fetch 本地失败后回退托管（URL 正确 + key 只发给托管）。"""
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization", "")))
        if "ollama.com" in str(request.url):
            return json_response(
                {"title": "Page", "content": "hosted content", "links": []}
            )
        return json_response({"error": "nope"}, status=404)

    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["ollama"]},
        search_cfg={
            "enabled": True,
            "providers": ["ollama"],
            "ollama_api_key": "sk-secret",
        },
        handler=handler,
    )
    try:
        out = await service.fetch("https://example.com/page")
        assert out["success"] is True
        assert out["provider"] == "ollama"
        assert any("127.0.0.1:11434" in u and not auth for u, auth in seen)
        assert any(
            "https://ollama.com/api/web_fetch" in u and auth == "Bearer sk-secret"
            for u, auth in seen
        )
    finally:
        await client.aclose()


# ── web_fetch ──────────────────────────────────────────────────────


async def test_fetch_local_ok():
    page = (
        "<html><head><title>Page</title></head><body>"
        "<p>Hello <b>world</b></p><a href='https://ext.example/x'>x</a></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page, request=request)

    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["local"]}, handler=handler
    )
    try:
        out = await service.fetch("https://example.com/index.html")
        assert out["success"] is True
        assert out["provider"] == "local"
        assert "Hello world" in out["content"]
        assert "https://ext.example/x" in out["links"]
    finally:
        await client.aclose()


async def test_fetch_ssrf_literal_ips():
    service, client = make_service(fetch_cfg={"enabled": True, "providers": ["local"]})
    try:
        for bad in [
            "http://127.0.0.1/x",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://10.0.0.5/",
            "http://localhost/x",
            "file:///etc/passwd",
            "ftp://example.com/x",
        ]:
            out = await service.fetch(bad)
            assert out["success"] is False, f"should block {bad}"
            assert (
                "SSRF" in out["error"]
                or "http" in out["error"]
                or "https" in out["error"]
            )
    finally:
        await client.aclose()


async def test_fetch_local_fails_then_provider():
    """local 失败(500) → tavily extract 兜底。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if "tavily.com" in str(request.url):
            return json_response(
                {
                    "results": [
                        {"url": "https://example.com", "raw_content": "extracted body"}
                    ]
                }
            )
        return httpx.Response(500, text="boom", request=request)

    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["local", "tavily"]},
        search_cfg={
            "enabled": True,
            "providers": ["tavily"],
            "tavily_api_key": "tvly-x",
        },
        handler=handler,
    )
    try:
        out = await service.fetch("https://example.com/")
        assert out["success"] is True
        assert out["provider"] == "tavily"
        assert "extracted body" in out["content"]
    finally:
        await client.aclose()


async def test_fetch_redirect_to_private_ip_blocked():
    """重定向逐跳 SSRF 校验：跳转目标为私网地址时拦截。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(
                302, headers={"location": "http://127.0.0.1/steal"}, request=request
            )
        return httpx.Response(200, text="<html>ok</html>", request=request)

    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["local"]},
        handler=handler,
    )
    try:
        out = await service.fetch("https://example.com/")
        assert out["success"] is False
        assert "SSRF" in out["error"]
    finally:
        await client.aclose()


async def test_fetch_redirect_loop_limited():
    """重定向次数超过 max_redirects 上限时报错。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": str(request.url)}, request=request
        )

    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["local"], "max_redirects": 2},
        handler=handler,
    )
    try:
        out = await service.fetch("https://example.com/")
        assert out["success"] is False
        assert "重定向过多" in out["error"]
    finally:
        await client.aclose()


async def test_fetch_skips_uncredentialed_provider():
    """strict_credential_skip：fetch 链中无 key 的 tavily 被跳过，不发起请求。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "example.com":
            return httpx.Response(
                200, text="<html><title>T</title><p>body</p></html>", request=request
            )
        return json_response({})

    service, client = make_service(
        fetch_cfg={
            "enabled": True,
            "providers": ["tavily", "local"],
            "tavily_api_key": "",
        },
        search_cfg={"enabled": True, "providers": ["tavily"], "tavily_api_key": ""},
        handler=handler,
    )
    try:
        out = await service.fetch("https://example.com/")
        assert out["success"] is True
        assert out["provider"] == "local"
        assert all("tavily.com" not in c for c in calls)
    finally:
        await client.aclose()


async def test_fetch_block_private_ip_false_allows_private():
    """block_private_ip=false 时允许抓取私网地址。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html><title>private</title><p>x</p></html>", request=request
        )

    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["local"], "block_private_ip": False},
        handler=handler,
    )
    try:
        out = await service.fetch("http://127.0.0.1:8080/health")
        assert out["success"] is True
        assert out["provider"] == "local"
    finally:
        await client.aclose()


async def test_fetch_truncation():
    service, client = make_service(
        fetch_cfg={"enabled": True, "providers": ["local"]},
        handler=lambda req: httpx.Response(
            200, text="<html><body>" + "x" * 50000 + "</body></html>", request=req
        ),
    )
    try:
        out = await service.fetch("https://example.com/", max_chars=1000)
        assert out["success"] is True
        assert len(out["content"]) <= 1001
        assert out["truncated"] is True
    finally:
        await client.aclose()


# ── 工具接线 / 政策 ────────────────────────────────────────────────


def test_build_tools_web_section_flag():
    """has_web=False 时 web 工具被过滤，True 时出现。"""
    from core.tools.impl import registry
    from core.tools.impl.web import create_web_entries

    class _Deps:
        web = object()

    # 临时注册并快照恢复，避免污染全局 registry
    saved = dict(registry._tools)
    for entry in create_web_entries(_Deps()):
        if registry.get(entry.name) is None:
            registry.register(entry)
    try:
        off = build_tools("normal", ChatContext(has_web=False))
        names_off = {t["function"]["name"] for t in off}
        assert "web_search" not in names_off
        assert "web_fetch" not in names_off

        on = build_tools("normal", ChatContext(has_web=True))
        names_on = {t["function"]["name"] for t in on}
        assert "web_search" in names_on
        assert "web_fetch" in names_on
    finally:
        registry._tools.clear()
        registry._tools.update(saved)


def test_web_entries_no_service_returns_error():
    from core.tools.impl.web import create_web_entries

    class _Deps:
        web = None

    entries = {e.name: e for e in create_web_entries(_Deps())}
    ctx = ToolContext(
        chat_id="c",
        is_group=False,
        reply_to="",
        sender_id="s",
        reply_callback=lambda *a, **k: None,
    )

    import asyncio

    result = asyncio.run(entries["web_search"].handler({"query": "x"}, ctx))
    payload = json.loads(result.content)
    assert "未启用" in payload["error"]


def test_web_entries_respect_enabled_flags():
    """只开 [web_fetch] 时 web_search 应返回未启用（独立开关）。"""
    import asyncio

    from core.tools.impl.web import create_web_entries
    from core.web_search.config import WebFetchConfig, WebSearchConfig

    class _Deps:
        web = WebService(
            WebSearchConfig.from_dict({"enabled": False}),
            WebFetchConfig.from_dict({"enabled": True, "providers": ["local"]}),
            None,
            env={},
        )

    entries = {e.name: e for e in create_web_entries(_Deps())}
    ctx = ToolContext(
        chat_id="c",
        is_group=False,
        reply_to="",
        sender_id="s",
        reply_callback=lambda *a, **k: None,
    )

    r1 = asyncio.run(entries["web_search"].handler({"query": "x"}, ctx))
    assert "未启用" in json.loads(r1.content)["error"]

    r2 = asyncio.run(entries["web_fetch"].handler({"url": "https://example.com/"}, ctx))
    assert "未启用" not in json.loads(r2.content).get("error", "")
