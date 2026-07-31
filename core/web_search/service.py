"""WebService — 网页搜索 / 抓取的统一入口。

职责：
- 按 providers 链路顺序遍历，跳过无凭证的 provider（strict_credential_skip）
- 仅在「失败 / 不可用」时回退到下一个（fallback_on_empty=false 时空结果不回退）
- 15 分钟内存缓存
- web_fetch 的 SSRF 防护（scheme + DNS 解析后私网拦截）
- 结果归一化 + 截断 + provider 标签
"""

import asyncio
import ipaddress
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

from core.web_search import providers as P
from core.web_search.config import OLLAMA_HOSTED_URL, WebFetchConfig, WebSearchConfig

_log = logging.getLogger(__name__)

_MAX_SEARCH_RESULTS = 10
_MAX_CACHE_ENTRIES = 500


class WebService:
    def __init__(
        self,
        search_cfg: WebSearchConfig,
        fetch_cfg: WebFetchConfig,
        http_client,
        env: dict | None = None,
    ):
        self.search_cfg = search_cfg
        self.fetch_cfg = fetch_cfg
        self.http_client = http_client
        self._env = env if env is not None else os.environ
        self._cache: dict[str, tuple[float, Any]] = {}

    # ── 内部工具 ──────────────────────────────────────────────────

    def _cache_get(self, key: str, ttl_seconds: float):
        entry = self._cache.get(key)
        if entry is None:
            return None
        inserted, value = entry
        if time.monotonic() - inserted > ttl_seconds:
            self._cache.pop(key, None)
            return None
        return value

    def _cache_set(self, key: str, value: Any):
        # 有界缓存：超过上限时淘汰最早插入的条目
        if key not in self._cache and len(self._cache) >= _MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (time.monotonic(), value)

    def _cache_ttl(self, minutes: int) -> float:
        return max(0.0, minutes * 60.0)

    def _ollama_key(self) -> str:
        return self.search_cfg.effective_ollama_key(self._env)

    def _ollama_base_url(self) -> str:
        return self.search_cfg.effective_ollama_base_url(self._env)

    def _tavily_key(self) -> str:
        return self.search_cfg.effective_tavily_key(self._env)

    def _provider_usable(self, pid: str) -> bool:
        """strict_credential_skip：无凭证的 provider 直接跳过（不浪费一次失败请求）。"""
        if not self.search_cfg.strict_credential_skip:
            return True
        if pid == "tavily":
            return bool(self._tavily_key())
        if pid == "ollama":
            # 托管 ollama.com 必须有 key；本地 host 免 key 恒可尝试
            if self._ollama_base_url() == OLLAMA_HOSTED_URL:
                return bool(self._ollama_key())
        return True  # duckduckgo（免 key）恒可尝试

    def _fetch_provider_usable(self, pid: str) -> bool:
        """web_fetch 链的凭证跳过（local 恒可用；tavily/托管 ollama 需 key）。"""
        if not self.fetch_cfg.strict_credential_skip:
            return True
        if pid == "tavily":
            return bool(self._tavily_key())
        if pid == "ollama":
            if self._ollama_base_url() == OLLAMA_HOSTED_URL:
                return bool(self._ollama_key())
        return True  # local / 本地 ollama 免 key

    # ── web_search ────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        count: int | None = None,
        region: str = "",
        freshness: str = "",
        safe_search: str = "",
    ) -> dict:
        if not query or not query.strip():
            return {"success": False, "error": "请提供 query"}
        try:
            count = min(
                _MAX_SEARCH_RESULTS, max(1, int(count or self.search_cfg.max_results))
            )
        except (TypeError, ValueError):
            count = min(_MAX_SEARCH_RESULTS, max(1, self.search_cfg.max_results))

        cache_key = f"search:{query.strip()}:{count}:{region}:{freshness}:{safe_search}"
        ttl = self._cache_ttl(self.search_cfg.cache_ttl_minutes)
        cached = self._cache_get(cache_key, ttl) if ttl > 0 else None
        if cached is not None:
            return cached

        providers = self.search_cfg.providers
        errors: list[str] = []
        attempted: list[str] = []
        last_error: str = ""
        for pid in providers:
            if not self._provider_usable(pid):
                _log.info("web_search: skip provider %s (无凭证)", pid)
                continue
            attempted.append(pid)
            try:
                results = await self._run_search_provider(
                    pid, query, count, region, freshness, safe_search
                )
                if results:
                    out = {
                        "success": True,
                        "provider": pid,
                        "query": query,
                        "results": results,
                    }
                elif self.search_cfg.fallback_on_empty:
                    _log.info("web_search: %s 空结果，继续回退", pid)
                    last_error = f"{pid}: 空结果"
                    continue
                else:
                    out = {
                        "success": True,
                        "provider": pid,
                        "query": query,
                        "results": [],
                    }
                if ttl > 0:
                    self._cache_set(cache_key, out)
                return out
            except P.ProviderError as e:
                _log.warning("web_search provider %s 失败: %s", pid, e.message)
                last_error = str(e)
                errors.append(str(e))
                continue
            except Exception as e:  # 未预期异常也回退
                _log.exception("web_search provider %s 未预期异常", pid)
                last_error = str(e)
                errors.append(f"{pid}: {e}")
                continue

        out = {
            "success": False,
            "error": f"所有搜索 provider 均失败: {'; '.join(errors) or last_error or '无可用的 provider'}",
            "providers_tried": attempted,
        }
        if ttl > 0:
            self._cache_set(cache_key, out)
        return out

    async def _run_search_provider(
        self, pid, query, count, region, freshness, safe_search
    ) -> list[dict]:
        timeout = float(self.search_cfg.timeout_seconds)
        cap = self.search_cfg.result_content_cap
        if pid == "ollama":
            raw = await P.ollama_search(
                self.http_client,
                query,
                count,
                self._ollama_key(),
                self._ollama_base_url(),
                timeout,
            )
        elif pid == "tavily":
            raw = await P.tavily_search(
                self.http_client,
                query,
                count,
                self._tavily_key(),
                timeout,
                time_range=(
                    freshness if freshness in ("day", "week", "month", "year") else ""
                ),
            )
        elif pid == "duckduckgo":
            raw = await P.duckduckgo_search(
                self.http_client,
                query,
                count,
                timeout,
                region=region,
                safe_search=safe_search,
            )
        else:
            raise P.ProviderError(pid, f"未知 provider")
        return [self._normalize_result_item(item, cap) for item in raw]

    def _normalize_result_item(self, item: dict, cap: int) -> dict:
        content, _ = _truncate(item.get("content", ""), cap)
        return {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": content,
        }

    # ── web_fetch ─────────────────────────────────────────────────

    def _validate_fetch_url(self, url: str) -> str:
        """scheme + 字面 IP 拦截。返回规范化后的 URL（完整 DNS 校验见 _assert_public_host_async）。"""
        url = url.strip()
        if not url:
            raise ValueError("请提供 url")
        if not url.startswith(("http://", "https://")):
            raise ValueError("仅支持 http/https 链接")
        if self.fetch_cfg.block_private_ip:
            parsed = urlparse(url)
            host = parsed.hostname
            if host and host.lower() == "localhost":
                raise ValueError(f"SSRF 拦截: 禁止访问 localhost ({url})")
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                ip = None
            if ip is not None and _is_blocked_ip(
                ip, self.fetch_cfg.allow_fake_ip_range
            ):
                raise ValueError(f"SSRF 拦截: 禁止访问私网地址 {url}")
        return url

    async def _assert_public_host_async(self, url: str):
        if not self.fetch_cfg.block_private_ip:
            return  # block_private_ip=false：显式允许抓取私网/内网地址
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            raise ValueError(f"无法解析主机名: {url}")
        if host.lower() == "localhost":
            raise ValueError(f"SSRF 拦截: 禁止访问 localhost ({url})")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None:
            if _is_blocked_ip(ip, self.fetch_cfg.allow_fake_ip_range):
                raise ValueError(f"SSRF 拦截: 禁止访问私网地址 {url}")
            return
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, None, proto=0)
        except Exception as e:
            raise ValueError(f"DNS 解析失败: {host} ({e})") from e
        for info in infos:
            addr = info[4][0]
            try:
                ip_obj = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if _is_blocked_ip(ip_obj, self.fetch_cfg.allow_fake_ip_range):
                raise ValueError(f"SSRF 拦截: {host} 解析到私网地址 {addr}")

    async def fetch(self, url: str, max_chars: int | None = None) -> dict:
        try:
            safe_url = self._validate_fetch_url(url)
        except ValueError as e:
            return {"success": False, "error": str(e)}

        try:
            max_chars = max(
                1000, min(200_000, int(max_chars or self.fetch_cfg.max_chars))
            )
        except (TypeError, ValueError):
            max_chars = max(1000, min(200_000, self.fetch_cfg.max_chars))
        cache_key = f"fetch:{safe_url}:{max_chars}"
        ttl = self._cache_ttl(self.fetch_cfg.cache_ttl_minutes)
        cached = self._cache_get(cache_key, ttl) if ttl > 0 else None
        if cached is not None:
            return cached

        await self._assert_public_host_async(safe_url)

        errors: list[str] = []
        attempted: list[str] = []
        for pid in self.fetch_cfg.providers:
            if not self._fetch_provider_usable(pid):
                _log.info("web_fetch: skip provider %s (无凭证)", pid)
                continue
            attempted.append(pid)
            try:
                payload = await self._run_fetch_provider(pid, safe_url, max_chars)
                content, truncated = _truncate(payload.get("content", ""), max_chars)
                out = {
                    "success": True,
                    "provider": pid,
                    "url": safe_url,
                    "title": payload.get("title", ""),
                    "content": content,
                    "links": payload.get("links", [])[:50],
                    "truncated": truncated,
                }
                if ttl > 0:
                    self._cache_set(cache_key, out)
                return out
            except P.ProviderError as e:
                _log.info("web_fetch provider %s 失败: %s", pid, e.message)
                errors.append(str(e))
                continue
            except Exception as e:
                _log.warning("web_fetch provider %s 未预期异常: %s", pid, e)
                errors.append(f"{pid}: {e}")
                continue

        out = {
            "success": False,
            "error": f"所有抓取通道均失败: {'; '.join(errors)}",
            "providers_tried": attempted,
        }
        if ttl > 0:
            self._cache_set(cache_key, out)
        return out

    async def _run_fetch_provider(self, pid: str, url: str, max_chars: int) -> dict:
        timeout = float(self.fetch_cfg.timeout_seconds)
        if pid == "local":
            return await P.local_fetch(
                self.http_client,
                url,
                timeout,
                self.fetch_cfg.user_agent,
                max_redirects=self.fetch_cfg.max_redirects,
                max_response_bytes=self.fetch_cfg.max_response_bytes,
                hop_validator=self._assert_public_host_async,
            )
        if pid == "ollama":
            return await P.ollama_fetch(
                self.http_client,
                url,
                self._ollama_key(),
                self._ollama_base_url(),
                timeout,
            )
        if pid == "tavily":
            return await P.tavily_fetch(
                self.http_client, url, self._tavily_key(), timeout
            )
        raise P.ProviderError(pid, f"未知 provider")


# ── 模块级工具 ────────────────────────────────────────────────────

# 198.18.0.0/15（v4）与 fc00::/7（v6）是 Surge/Clash/sing-box 等代理的 fake-ip 段，
# 很多机器解析所有域名都会落到这里；默认放行，严格环境可设 allow_fake_ip_range=false
_FAKE_IP_NETWORKS = [
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_blocked_ip(ip, allow_fake_ip_range: bool = True) -> bool:
    if allow_fake_ip_range:
        for net in _FAKE_IP_NETWORKS:
            if ip in net:
                return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _truncate(text: str, cap: int) -> tuple[str, bool]:
    text = text.strip()
    if len(text) <= cap:
        return text, False
    return text[:cap].rstrip() + "…", True
