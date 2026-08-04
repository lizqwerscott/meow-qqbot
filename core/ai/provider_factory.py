"""Provider factory 注册表 — 新 provider 自注册，注册表不感知具体类型。

对应 Pi 架构文档的「Provider factory」模式：ModelRegistry 不再用 if/elif
硬编码构造服务，而是按 provider type 查注册表；新增 provider 只需
`@register_provider("xxx")` 写一个 factory + 在 config/models.toml 配置，
ModelRegistry 零改动。

factory 签名：由 provider 配置段 + 单模型配置构造一个服务实例。
两个参数：
    pcfg: provider 段配置（api_key / base_url / host / type 等）
    mcfg: 单模型配置（name / model / timeout / temperature / max_tokens 等）
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.ai.protocol import LLMService

_log = logging.getLogger(__name__)

# factory 签名：provider 配置段 + 单模型配置 → 服务实例（须满足 LLMService 协议）
ProviderFactory = Callable[[dict[str, Any], dict[str, Any]], LLMService]

_FACTORIES: dict[str, ProviderFactory] = {}


def register_provider(type_name: str) -> Callable[[ProviderFactory], ProviderFactory]:
    """装饰器：注册 provider 构造器。"""

    def deco(fn: ProviderFactory) -> ProviderFactory:
        if type_name in _FACTORIES:
            _log.warning(f"Provider factory [{type_name}] 重复注册，覆盖旧实现")
        _FACTORIES[type_name] = fn
        return fn

    return deco


def get_provider_factory(type_name: str) -> ProviderFactory | None:
    return _FACTORIES.get(type_name)


# ── 内置 provider 注册 ──
# 注意：factory 内部延迟 import，避免注册表模块与 service 模块循环依赖。
# 各默认参数逐字保留 ModelRegistry 原 if/elif 分支语义（重构不重构行为）：
#   openai: timeout=30, max_retries=0, temperature=0.7, max_tokens=8192
#   ollama: host 默认 http://localhost:11434, base_url 回退 {host}/v1,
#           timeout=120, max_retries=0, temperature=0.7, max_tokens=4096


@register_provider("openai")
def _build_openai(pcfg: dict[str, Any], mcfg: dict[str, Any]) -> LLMService:
    from core.ai.service import AIService

    return AIService(
        api_key=pcfg.get("api_key", ""),
        base_url=pcfg.get("base_url"),
        model=mcfg.get("model", "gpt-3.5-turbo"),
        timeout=mcfg.get("timeout", 30),
        max_retries=mcfg.get("max_retries", 0),
        temperature=mcfg.get("temperature", 0.7),
        max_tokens=mcfg.get("max_tokens", 8192),
        reasoning_effort=mcfg.get("reasoning_effort"),
    )


@register_provider("ollama")
def _build_ollama(pcfg: dict[str, Any], mcfg: dict[str, Any]) -> LLMService:
    from core.ai.service import AIService

    host = pcfg.get("host", "http://localhost:11434").rstrip("/")
    return AIService(
        api_key=pcfg.get("api_key") or "not-needed",
        base_url=pcfg.get("base_url") or f"{host}/v1",
        model=mcfg.get("model", "llama3.2"),
        timeout=mcfg.get("timeout", 120),
        max_retries=mcfg.get("max_retries", 0),
        temperature=mcfg.get("temperature", 0.7),
        max_tokens=mcfg.get("max_tokens", 4096),
        reasoning_effort=mcfg.get("reasoning_effort"),
    )


@register_provider("modelscope")
def _build_modelscope(pcfg: dict[str, Any], mcfg: dict[str, Any]) -> LLMService:
    from core.ai.modelscope_service import ModelScopeService

    return ModelScopeService(
        api_key=pcfg.get("api_key", ""),
        base_url=pcfg.get("base_url"),
        model=mcfg.get("model", "gpt-3.5-turbo"),
        timeout=mcfg.get("timeout", 30),
        max_retries=mcfg.get("max_retries", 0),
        temperature=mcfg.get("temperature", 0.7),
        max_tokens=mcfg.get("max_tokens", 8192),
        reasoning_effort=mcfg.get("reasoning_effort"),
    )


@register_provider("deepseek_responses")
def _build_deepseek_responses(pcfg: dict[str, Any], mcfg: dict[str, Any]) -> LLMService:
    """DeepSeek Responses API（OpenAI responses 端点，非 chat-completions）。

    base_url 默认 https://api.deepseek.com（Responses API 无 /v1 前缀）。
    目前仅支持 deepseek-v4-flash 模型。
    """
    from core.ai.deepseek_service import DeepSeekResponsesService

    model = mcfg.get("model", "deepseek-v4-flash")
    if model != "deepseek-v4-flash":
        _log.warning(
            f"deepseek_responses 目前仅支持 deepseek-v4-flash，配置了 [{model}]，"
            "Responses API 调用可能失败"
        )
    return DeepSeekResponsesService(
        api_key=pcfg.get("api_key", ""),
        base_url=pcfg.get("base_url"),
        model=model,
        timeout=mcfg.get("timeout", 30),
        max_retries=mcfg.get("max_retries", 0),
        temperature=mcfg.get("temperature", 0.7),
        max_tokens=mcfg.get("max_tokens", 8192),
        reasoning_effort=mcfg.get("reasoning_effort"),
    )
