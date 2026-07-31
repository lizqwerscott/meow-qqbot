"""Web 搜索 / 抓取配置 — 从 config.toml 的 [web_search] / [web_fetch] 节解析。

凭证解析统一为「环境变量优先，config 兜底」（在 service.py 中应用）。
"""

from dataclasses import dataclass, field

DEFAULT_SEARCH_PROVIDERS = ["ollama", "tavily", "duckduckgo"]
DEFAULT_FETCH_PROVIDERS = ["local", "ollama", "tavily"]

DEFAULT_OLLAMA_LOCAL_HOST = "http://127.0.0.1:11434"
OLLAMA_HOSTED_URL = "https://ollama.com"


def _normalize_providers(raw: list | tuple | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    seen: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        pid = item.strip().lower()
        if pid and pid not in seen:
            seen.append(pid)
    return seen or list(default)


def _clamp_int(value, lo: int, hi: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, parsed))


def _strip(value) -> str:
    return str(value or "").strip()


@dataclass
class WebSearchConfig:
    """[web_search] 节配置。"""

    enabled: bool = False
    providers: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_PROVIDERS))
    strict_credential_skip: bool = True
    fallback_on_empty: bool = False
    ollama_api_key: str = ""
    ollama_base_url: str = ""
    tavily_api_key: str = ""
    max_results: int = 5
    timeout_seconds: int = 20
    cache_ttl_minutes: int = 15
    result_content_cap: int = 800

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "WebSearchConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            providers=_normalize_providers(
                cfg.get("providers"), DEFAULT_SEARCH_PROVIDERS
            ),
            strict_credential_skip=bool(cfg.get("strict_credential_skip", True)),
            fallback_on_empty=bool(cfg.get("fallback_on_empty", False)),
            ollama_api_key=_strip(cfg.get("ollama_api_key")),
            ollama_base_url=_strip(cfg.get("ollama_base_url")),
            tavily_api_key=_strip(cfg.get("tavily_api_key")),
            max_results=_clamp_int(cfg.get("max_results"), 1, 10, 5),
            timeout_seconds=_clamp_int(cfg.get("timeout_seconds"), 1, 120, 20),
            cache_ttl_minutes=_clamp_int(cfg.get("cache_ttl_minutes"), 0, 1440, 15),
            result_content_cap=_clamp_int(
                cfg.get("result_content_cap"), 100, 10000, 800
            ),
        )

    # ── 运行时解析（env 优先，config 兜底）──
    def effective_ollama_key(self, env=None) -> str:
        env = env if env is not None else __import__("os").environ
        return _strip(env.get("OLLAMA_API_KEY")) or self.ollama_api_key

    def effective_ollama_base_url(self, env=None) -> str:
        env = env if env is not None else __import__("os").environ
        raw = (
            self.ollama_base_url
            or _strip(env.get("OLLAMA_BASE_URL"))
            or DEFAULT_OLLAMA_LOCAL_HOST
        )
        return raw.rstrip("/")

    def effective_tavily_key(self, env=None) -> str:
        env = env if env is not None else __import__("os").environ
        return _strip(env.get("TAVILY_API_KEY")) or self.tavily_api_key


@dataclass
class WebFetchConfig:
    """[web_fetch] 节配置。"""

    enabled: bool = False
    providers: list[str] = field(default_factory=lambda: list(DEFAULT_FETCH_PROVIDERS))
    strict_credential_skip: bool = True
    timeout_seconds: int = 25
    cache_ttl_minutes: int = 15
    max_chars: int = 20000
    max_redirects: int = 3
    block_private_ip: bool = True
    allow_fake_ip_range: bool = True
    max_response_bytes: int = 750_000
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "WebFetchConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            providers=_normalize_providers(
                cfg.get("providers"), DEFAULT_FETCH_PROVIDERS
            ),
            strict_credential_skip=bool(cfg.get("strict_credential_skip", True)),
            timeout_seconds=_clamp_int(cfg.get("timeout_seconds"), 1, 120, 25),
            cache_ttl_minutes=_clamp_int(cfg.get("cache_ttl_minutes"), 0, 1440, 15),
            max_chars=_clamp_int(cfg.get("max_chars"), 1000, 200_000, 20000),
            max_redirects=_clamp_int(cfg.get("max_redirects"), 0, 10, 3),
            block_private_ip=bool(cfg.get("block_private_ip", True)),
            allow_fake_ip_range=bool(cfg.get("allow_fake_ip_range", True)),
            max_response_bytes=_clamp_int(
                cfg.get("max_response_bytes"), 32000, 10_000_000, 750_000
            ),
            user_agent=_strip(cfg.get("user_agent"))
            or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
