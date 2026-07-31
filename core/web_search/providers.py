"""三个网页 provider 的实现（Ollama / Tavily / DuckDuckGo）。

每个函数返回归一化后的纯数据，失败统一抛 ProviderError；
凭证 / 链路 / 回退逻辑在 service.py 中处理。
"""

import html as html_lib
import logging
import re
from urllib.parse import parse_qs, urljoin, urlparse

_log = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────

OLLAMA_HOSTED = "https://ollama.com"
OLLAMA_SEARCH_LOCAL_EXPERIMENTAL = "/api/experimental/web_search"
OLLAMA_SEARCH_LOCAL = "/api/web_search"
OLLAMA_SEARCH_HOSTED = "/api/web_search"
OLLAMA_FETCH_PATH = "/api/web_fetch"

TAVILY_BASE = "https://api.tavily.com"
TAVILY_SEARCH_PATH = "/search"
TAVILY_EXTRACT_PATH = "/extract"

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html"
DDG_SAFE_SEARCH_PARAM = {"strict": "1", "moderate": "-1", "off": "-2"}

# DDG 人机验证特征
_BOT_CHALLENGE_RE = re.compile(
    r"g-recaptcha|are you a human|id=[\"']challenge-form[\"']|name=[\"']challenge[\"']",
    re.I,
)
_RESULT_A_RE = re.compile(
    r'class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.I | re.S
)
_SNIPPET_RE = re.compile(
    r'class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>', re.I | re.S
)


class ProviderError(Exception):
    """provider 调用失败（网络 / HTTP / 解析 / 人机验证）。"""

    def __init__(self, provider: str, message: str):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


# ── 通用工具 ──────────────────────────────────────────────────────


def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def _ollama_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# ── Ollama ────────────────────────────────────────────────────────


async def ollama_search(
    client,
    query: str,
    max_results: int,
    api_key: str,
    base_url: str,
    timeout: float,
) -> list[dict]:
    """Ollama 网页搜索。

    host 解析（service 完成）：本地 host 先试 /api/experimental/web_search，
    再试 /api/web_search；host 为 ollama.com（托管）或配了 key 时直连托管 API。
    """
    host = base_url.rstrip("/")
    hosted = host == OLLAMA_HOSTED

    # (endpoint, api_key) 候选：本地端点绝不携带 key（spec: 不把 key 发给本地 host），
    # 托管 fallback 仅在持有 key 时加入且必须指向 ollama.com
    if hosted:
        candidates = [(f"{host}{OLLAMA_SEARCH_HOSTED}", api_key)]
    else:
        candidates = [
            (f"{host}{OLLAMA_SEARCH_LOCAL_EXPERIMENTAL}", ""),
            (f"{host}{OLLAMA_SEARCH_LOCAL}", ""),
        ]
        if api_key:
            candidates.append((f"{OLLAMA_HOSTED}{OLLAMA_SEARCH_HOSTED}", api_key))

    last_err: Exception | None = None
    for endpoint, key in candidates:
        try:
            resp = await client.post(
                endpoint,
                json={"query": query, "max_results": max_results},
                headers=_ollama_headers(key),
                timeout=timeout,
            )
            if resp.status_code >= 400:
                _log.warning(
                    "ollama web_search %s -> HTTP %d", endpoint, resp.status_code
                )
                last_err = ProviderError(
                    "ollama", f"HTTP {resp.status_code} from {endpoint}"
                )
                continue
            data = resp.json()
            results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(results, list):
                raise ProviderError("ollama", f"意外响应结构: {str(data)[:200]}")
            return [
                {
                    "title": _strip_html(str(item.get("title", ""))),
                    "url": str(item.get("url", "")),
                    "content": _strip_html(str(item.get("content", ""))),
                }
                for item in results
                if item and item.get("url")
            ]
        except ProviderError:
            raise
        except Exception as e:  # 网络/超时/JSON 错误 → 试下一个端点
            _log.warning("ollama web_search %s failed: %s", endpoint, e)
            last_err = e
    raise ProviderError("ollama", f"所有端点失败: {last_err}")


async def ollama_fetch(
    client,
    url: str,
    api_key: str,
    base_url: str,
    timeout: float,
) -> dict:
    """Ollama 网页抓取。本地 host 无实验端点，失败后回退托管。"""
    host = base_url.rstrip("/")
    candidates: list[tuple[str, str]] = []
    if host == OLLAMA_HOSTED:
        candidates.append((f"{host}{OLLAMA_FETCH_PATH}", api_key))
    else:
        candidates.append((f"{host}{OLLAMA_FETCH_PATH}", ""))  # 本地不携 key
        if api_key:
            candidates.append((f"{OLLAMA_HOSTED}{OLLAMA_FETCH_PATH}", api_key))

    last_err: Exception | None = None
    for endpoint, key in candidates:
        try:
            resp = await client.post(
                endpoint,
                json={"url": url},
                headers=_ollama_headers(key),
                timeout=timeout,
            )
            if resp.status_code >= 400:
                last_err = ProviderError(
                    "ollama", f"HTTP {resp.status_code} from {endpoint}"
                )
                continue
            data = resp.json()
            if not isinstance(data, dict):
                raise ProviderError("ollama", f"意外响应结构: {str(data)[:200]}")
            return {
                "title": _strip_html(str(data.get("title", ""))),
                "content": str(data.get("content", "")),
                "links": [
                    str(x) for x in (data.get("links") or []) if isinstance(x, str)
                ],
            }
        except ProviderError:
            raise
        except Exception as e:
            _log.warning("ollama web_fetch %s failed: %s", endpoint, e)
            last_err = e
    raise ProviderError("ollama", f"所有端点失败: {last_err}")


# ── Tavily ────────────────────────────────────────────────────────


async def tavily_search(
    client,
    query: str,
    max_results: int,
    api_key: str,
    timeout: float,
    time_range: str = "",
) -> list[dict]:
    body: dict = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "topic": "general",
    }
    if time_range:
        body["time_range"] = time_range
    resp = await client.post(
        f"{TAVILY_BASE}{TAVILY_SEARCH_PATH}",
        json=body,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise ProviderError("tavily", f"HTTP {resp.status_code}")
    data = resp.json()
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        raise ProviderError("tavily", f"意外响应结构: {str(data)[:200]}")
    return [
        {
            "title": _strip_html(str(item.get("title", ""))),
            "url": str(item.get("url", "")),
            "content": _strip_html(str(item.get("content", ""))),
        }
        for item in results
        if item and item.get("url")
    ]


async def tavily_fetch(client, url: str, api_key: str, timeout: float) -> dict:
    resp = await client.post(
        f"{TAVILY_BASE}{TAVILY_EXTRACT_PATH}",
        json={"api_key": api_key, "urls": [url], "extract_depth": "basic"},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise ProviderError("tavily", f"HTTP {resp.status_code}")
    data = resp.json()
    if not isinstance(data, dict):
        raise ProviderError("tavily", f"意外响应结构: {str(data)[:200]}")
    results = data.get("results") or []
    failed = data.get("failed_results") or []
    if isinstance(failed, list) and failed:
        first = failed[0] if isinstance(failed[0], dict) else {}
        raise ProviderError("tavily", f"extract 失败: {first.get('error', 'unknown')}")
    if isinstance(results, list) and results:
        first = results[0] if isinstance(results[0], dict) else {}
        return {
            "title": "",
            "content": str(first.get("raw_content", "") or ""),
            "links": [
                str(x) for x in (first.get("images") or []) if isinstance(x, str)
            ],
        }
    raise ProviderError("tavily", "extract 返回空结果")


# ── DuckDuckGo（免 key，HTML 解析）────────────────────────────────


def _decode_ddg_url(raw: str) -> str:
    normalized = f"https:{raw}" if raw.startswith("//") else raw
    try:
        parsed = urlparse(normalized)
        query = parse_qs(parsed.query)
        if "uddg" in query:
            return query["uddg"][0]
    except Exception:
        pass
    return raw


def parse_duckduckgo_html(html_text: str) -> list[dict]:
    """解析 DDG HTML 结果页 → [{title, url, content}]。

    与 OpenClaw 相同的策略：定位 result__a 链接，snippet 按位置就近关联。
    """
    if not _RESULT_A_RE.search(html_text):
        if _BOT_CHALLENGE_RE.search(html_text):
            raise ProviderError(
                "duckduckgo", "DuckDuckGo 返回人机验证页（challenge），请稍后重试"
            )
        return []

    results: list[dict] = []
    for m in _RESULT_A_RE.finditer(html_text):
        raw_url = html_lib.unescape(m.group(1))
        title = _strip_html(html_lib.unescape(m.group(2)))
        if not title:
            continue
        # 就近 snippet：本 result 结束到下一个 result__a 之间
        snippet = ""
        snip = _SNIPPET_RE.search(html_text, m.end())
        if snip:
            nxt = _RESULT_A_RE.search(html_text, m.end())
            if nxt is None or snip.end() <= nxt.start():
                snippet = _strip_html(html_lib.unescape(snip.group(1)))
        results.append(
            {
                "title": title,
                "url": _decode_ddg_url(raw_url),
                "content": snippet,
            }
        )
    return results


async def duckduckgo_search(
    client,
    query: str,
    count: int,
    timeout: float,
    region: str = "",
    safe_search: str = "",
) -> list[dict]:
    params = {"q": query}
    if region:
        params["kl"] = region
    if safe_search in DDG_SAFE_SEARCH_PARAM:
        params["kp"] = DDG_SAFE_SEARCH_PARAM[safe_search]
    try:
        resp = await client.get(
            DDG_HTML_ENDPOINT,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout,
        )
    except Exception as e:
        raise ProviderError("duckduckgo", f"请求失败: {e}") from e
    if resp.status_code >= 400:
        raise ProviderError("duckduckgo", f"HTTP {resp.status_code}")
    results = parse_duckduckgo_html(resp.text)
    return results[:count]


# ── 本地抓取（web_fetch 链路首位）──────────────────────────────────


def extract_local_html(html_text: str, base_url: str) -> dict:
    """轻量本地提取：title / meta description / 正文文本 / 链接。"""
    title = ""
    tm = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
    if tm:
        title = _strip_html(html_lib.unescape(tm.group(1)))

    description = ""
    dm = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        html_text,
        re.I | re.S,
    )
    if not dm:
        dm = re.search(
            r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',
            html_text,
            re.I | re.S,
        )
    if dm:
        description = _strip_html(html_lib.unescape(dm.group(1)))

    body = re.sub(r"<script.*?</script>", " ", html_text, flags=re.I | re.S)
    body = re.sub(r"<style.*?</style>", " ", body, flags=re.I | re.S)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    text = _strip_html(html_lib.unescape(body))

    links: list[str] = []
    for hm in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\']', html_text, re.I):
        href = hm.group(1).strip()
        if href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        try:
            resolved = urljoin(base_url, href)
        except Exception:
            continue
        if resolved.startswith(("http://", "https://")) and resolved not in links:
            links.append(resolved)
        if len(links) >= 50:
            break

    content = (
        description if text and description and len(description) > len(text) else ""
    )
    content = content or (text[:2000] if text else "")
    return {"title": title, "content": content, "links": links}


async def local_fetch(
    client,
    url: str,
    timeout: float,
    user_agent: str,
    max_redirects: int = 3,
    max_response_bytes: int = 750_000,
    hop_validator=None,
) -> dict:
    """本地抓取单个 URL 并轻量提取。

    手动跟随重定向（上限 max_redirects），每跳经 hop_validator 做 SSRF 校验；
    响应体超 max_response_bytes 视为失败（交给 provider 兜底）。
    """
    current = url
    seen_redirects = 0
    while True:
        if hop_validator is not None:
            try:
                await hop_validator(current)
            except ValueError as e:
                raise ProviderError("local", f"SSRF: {e}") from e
        resp = await client.get(
            current,
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            if not location:
                raise ProviderError("local", "重定向缺少 Location")
            current = urljoin(current, location)
            seen_redirects += 1
            if seen_redirects > max_redirects:
                raise ProviderError("local", f"重定向过多（>{max_redirects} 次）")
            continue
        if resp.status_code >= 400:
            raise ProviderError("local", f"HTTP {resp.status_code}")
        text = resp.text or ""
        if len(text) > max_response_bytes:
            raise ProviderError("local", f"响应过大（>{max_response_bytes} 字节）")
        if not text.strip():
            raise ProviderError("local", "空响应体")
        extracted = extract_local_html(text, str(resp.url))
        if not extracted["title"] and not extracted["content"]:
            raise ProviderError("local", "无法提取有效内容")
        return extracted
