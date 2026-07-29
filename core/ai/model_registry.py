"""ModelRegistry — 多模型注册表 + 组 fallback 链。

通过 Provider 定义 API key/base_url，通过 Group 定义逻辑模型链。
支持按组名或全限定模型名查找，失败自动 fallback。
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from core.ai.cooldown import ModelCooldownManager
from core.ai.service import AIService
from core.ai.fallback_runner import FallbackRunner

_log = logging.getLogger(__name__)


@dataclass
class _SessionConfig:
    budget: int = 30
    ttl: float = 600.0


class ModelRegistry:
    """模型注册表。

    用法:
        registry = ModelRegistry(providers_config, groups_config)
        result, model_name = await registry.chat_with_fallback(
            registry.get_chain("cheap"), messages, tools,
        )
    """

    def __init__(
        self,
        providers_config: dict,
        groups_config: dict,
        cooldown_config: Optional[Dict] = None,
    ):
        self._services: Dict[str, Any] = {}
        self._groups: Dict[str, List[str]] = {}
        self._tier_map: Dict[str, str] = {}
        self._default_service: Optional[Any] = None
        self._cooldown = ModelCooldownManager(cooldown_config or {})
        self._session_configs: Dict[str, _SessionConfig] = {}
        self._default_session_config = _SessionConfig()

        for provider_name, pcfg in providers_config.items():
            provider_budget = pcfg.get("session_budget")
            provider_ttl = pcfg.get("session_ttl")
            provider_type = pcfg.get("type", "openai")
            api_key = pcfg.get("api_key", "")
            base_url = pcfg.get("base_url")

            for model_cfg in pcfg.get("models", []):
                model_name = model_cfg.get("name")
                if not model_name:
                    _log.warning(
                        f"Provider [{provider_name}] 中存在无 name 的模型配置，跳过"
                    )
                    continue
                qualified_name = f"{provider_name}/{model_name}"

                if provider_type == "modelscope":
                    from core.ai.modelscope_service import ModelScopeService
                    svc = ModelScopeService(
                        api_key=api_key,
                        base_url=base_url,
                        model=model_cfg.get("model", "gpt-3.5-turbo"),
                        timeout=model_cfg.get("timeout", 30),
                        max_retries=model_cfg.get("max_retries", 0),
                        temperature=model_cfg.get("temperature", 0.7),
                        max_tokens=model_cfg.get("max_tokens", 8192),
                        reasoning_effort=model_cfg.get("reasoning_effort"),
                    )
                elif provider_type == "ollama":
                    from core.ai.service import AIService
                    host = pcfg.get("host", "http://localhost:11434").rstrip("/")
                    base_url = pcfg.get("base_url") or f"{host}/v1"
                    svc = AIService(
                        api_key=api_key or "not-needed",
                        base_url=base_url,
                        model=model_cfg.get("model", "llama3.2"),
                        timeout=model_cfg.get("timeout", 120),
                        max_retries=model_cfg.get("max_retries", 0),
                        temperature=model_cfg.get("temperature", 0.7),
                        max_tokens=model_cfg.get("max_tokens", 4096),
                        reasoning_effort=model_cfg.get("reasoning_effort"),
                    )
                else:
                    svc = AIService(
                        api_key=api_key,
                        base_url=base_url,
                        model=model_cfg.get("model", "gpt-3.5-turbo"),
                        timeout=model_cfg.get("timeout", 30),
                        max_retries=model_cfg.get("max_retries", 0),
                        temperature=model_cfg.get("temperature", 0.7),
                        max_tokens=model_cfg.get("max_tokens", 8192),
                        reasoning_effort=model_cfg.get("reasoning_effort"),
                    )
                self._services[qualified_name] = svc

                m_budget = model_cfg.get("session_budget")
                m_ttl = model_cfg.get("session_ttl")
                budget = (m_budget if m_budget is not None
                          else provider_budget if provider_budget is not None
                          else self._default_session_config.budget)
                ttl = (m_ttl if m_ttl is not None
                       else provider_ttl if provider_ttl is not None
                       else self._default_session_config.ttl)
                self._session_configs[qualified_name] = _SessionConfig(budget=budget, ttl=ttl)

                _log.info(
                    f"模型 [{qualified_name}]({provider_type}): "
                    f"{model_cfg.get('model')} @ {base_url}"
                )

        for group_name, gcfg in groups_config.items():
            raw_chain = gcfg.get("models", [])
            valid_chain = [m for m in raw_chain if m in self._services]
            if not valid_chain:
                _log.warning(
                    f"组 [{group_name}]: 无可用的模型 (配置: {raw_chain})"
                )
                continue
            self._groups[group_name] = valid_chain
            _log.info(f"  组 [{group_name}]: 模型链 {valid_chain}")
            if self._default_service is None:
                self._default_service = self._services.get(valid_chain[0])

    @property
    def default_service(self) -> Optional[Any]:
        return self._default_service

    def get(self, name: str) -> Optional[Any]:
        return self._services.get(name)

    def get_group(self, group_name: str) -> List[str]:
        """获取组的模型链。"""
        return list(self._groups.get(group_name, []))

    def configure_tiers(self, tier_config: dict):
        """配置分档 → 组名映射。

        tier_config: {
            "simple": "cheap",
            "medium": "primary",
            ...
        }
        """
        self._tier_map = {}
        for tier, group_name in tier_config.items():
            if group_name in self._groups:
                self._tier_map[tier] = group_name
                _log.info(
                    f"  分档 [{tier}] → 组 [{group_name}]: "
                    f"模型链 {self._groups[group_name]}"
                )
            else:
                _log.warning(
                    f"  分档 [{tier}]: 组 [{group_name}] 不存在"
                )

    def get_chain(self, tier: str, fallback: Optional[List[str]] = None) -> List[str]:
        """获取指定分档的模型链。

        先尝试按 tier 查分档，再当组名查，最后 fallback。
        """
        group_name = self._tier_map.get(tier)
        if group_name:
            return list(self._groups.get(group_name, []))
        if tier in self._groups:
            return list(self._groups[tier])
        if fallback:
            return fallback
        if self._default_service:
            for name in self._services:
                return [name]
        return []

    def get_session_config(self, qualified_name: str) -> tuple[int, float]:
        """获取模型的 session 绑定配置 (budget, ttl)。

        三层级：model > provider > default（在 __init__ 中计算并存储）。
        """
        cfg = self._session_configs.get(qualified_name)
        if cfg:
            return cfg.budget, cfg.ttl
        d = self._default_session_config
        return d.budget, d.ttl

    async def resolve_model_chain(
        self, model_chain: List[str]
    ) -> Optional[tuple[str, Any]]:
        """从模型链中解析第一个可用模型（不做 API 调用）。

        只做冷却检查和服务存在性检查。
        返回 (qualified_name, AIService) 或 None。
        """
        for qualified_name in model_chain:
            if await self._cooldown.is_cooled_down(qualified_name):
                _log.info(f"模型链解析: [{qualified_name}] 冷却中，跳过")
                continue
            svc = self._services.get(qualified_name)
            if svc is not None:
                _log.info(f"模型链解析: [{qualified_name}] 可用")
                return qualified_name, svc
        _log.warning(f"模型链中无可用模型: {model_chain}")
        return None

    async def chat_with_fallback(
        self,
        model_chain: Optional[List[str]],
        messages: Iterable[ChatCompletionMessageParam],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[Any], Optional[Dict], Optional[str]]:
        """按模型链顺序调用，失败自动 fallback。

        Returns:
            (ChatCompletionMessage | None, usage_dict | None, model_name_used | None)
        """
        if not model_chain:
            _log.error("chat_with_fallback 收到空的模型链")
            return None, None, None

        # 预检并记录所有模型的额度状态
        _log.warning(f"[fallback] 模型链额度状态: chain={model_chain}")
        for qualified_name in model_chain:
            svc = self._services.get(qualified_name)
            if svc is None:
                continue
            if hasattr(svc, "quota_info"):
                qi = svc.quota_info
                if qi["exhausted"]:
                    reasons = []
                    if qi["user_remaining"] <= 0:
                        reasons.append("用户额度耗尽")
                    if qi["model_remaining"] <= 0:
                        reasons.append("模型额度耗尽")
                    _log.warning(
                        f"  [{qualified_name}]: {' + '.join(reasons)} "
                        f"(用户{qi['user_remaining']}/{qi['user_limit']}, "
                        f"模型{qi['model_remaining']}/{qi['model_limit']})"
                    )
                else:
                    _log.warning(
                        f"  [{qualified_name}]: 正常 "
                        f"(用户剩余{qi['user_remaining']}/{qi['user_limit']}, "
                        f"模型剩余{qi['model_remaining']}/{qi['model_limit']})"
                    )

        runner = FallbackRunner(self, model_chain)
        result = await runner.run(
            lambda svc, name: svc.chat_completion_with_tools(
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
            ),
            record_cooldown=True,
        )
        if result.ok:
            return result.message, result.usage, result.model_name
        _log.error(f"所有模型 fallback 失败: chain={model_chain}")
        return None, None, None

    @property
    def cooldown_manager(self):
        """暴露冷却管理器，供 MultimodalService 等外部使用。"""
        return self._cooldown

    async def get_cooldown_states(self) -> dict:
        """获取所有模型的冷却状态。"""
        return await self._cooldown.get_all_states()

    async def simple_chat(
        self,
        model_name: str,
        messages: Iterable[ChatCompletionMessageParam],
        max_tokens: int = 500,
    ) -> Optional[str]:
        """不带工具的单轮对话（给心跳和简单任务用）。

        model_name 可以是全限定名 (deepseek/primary) 或组名。
        如果模型在冷却期，直接返回 None 让调用方走 fallback。
        """
        # 解析为全限定名
        qualified_name = model_name
        svc = self._services.get(qualified_name)
        if svc is None:
            chain = self._groups.get(model_name)
            if chain:
                qualified_name = chain[0]
                svc = self._services.get(qualified_name)
        if svc is None:
            _log.warning(f"模型 [{model_name}] 未注册")
            return None

        # 冷却检查
        if await self._cooldown.is_cooled_down(qualified_name):
            _log.info(f"模型 [{qualified_name}] 处于冷却期，simple_chat 跳过")
            return None

        try:
            result, _ = await svc.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
            )
            if result is not None:
                await self._cooldown.record_success(qualified_name)
            return result
        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            await self._cooldown.record_failure(qualified_name)
            _log.warning(f"模型 [{qualified_name}] simple_chat 失败: {e}")
            return None

    async def close(self):
        results = await asyncio.gather(
            *[svc.close() for svc in self._services.values()],
            return_exceptions=True,
        )
        for name, result in zip(self._services, results):
            if isinstance(result, Exception):
                _log.warning("关闭模型服务失败 [%s]: %s", name, result)
        _log.info("ModelRegistry 已关闭")
