"""ModelRegistry — 多模型注册表 + 组 fallback 链。

通过 Provider 定义 API key/base_url，通过 Group 定义逻辑模型链。
支持按组名或全限定模型名查找，失败自动 fallback。
"""

import logging
from typing import Any, Dict, Iterable, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from core.ai.service import AIService

_log = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表。

    用法:
        registry = ModelRegistry(providers_config, groups_config)
        result, model_name = await registry.chat_with_fallback(
            registry.get_chain("cheap"), messages, tools,
        )
    """

    def __init__(self, providers_config: dict, groups_config: dict):
        self._services: Dict[str, AIService] = {}
        self._groups: Dict[str, List[str]] = {}
        self._tier_map: Dict[str, str] = {}
        self._default_service: Optional[AIService] = None

        for provider_name, pcfg in providers_config.items():
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
    def default_service(self) -> Optional[AIService]:
        return self._default_service

    def get(self, name: str) -> Optional[AIService]:
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

    async def chat_with_fallback(
        self,
        model_chain: List[str],
        messages: Iterable[ChatCompletionMessageParam],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[Any], Optional[Dict], Optional[str]]:
        """按模型链顺序调用，失败自动 fallback。

        Returns:
            (ChatCompletionMessage | None, usage_dict | None, model_name_used | None)
        """
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

        last_error = None
        for qualified_name in model_chain:
            svc = self._services.get(qualified_name)
            if svc is None:
                _log.warning(f"模型 [{qualified_name}] 未注册，跳过")
                continue

            try:
                result, usage = await svc.chat_completion_with_tools(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
                if result is not None:
                    _log.debug(f"模型 [{qualified_name}] 调用成功")
                    return result, usage, qualified_name
                last_error = "返回空结果"
                if hasattr(svc, "quota_info") and svc.quota_info["exhausted"]:
                    _log.warning(
                        f"模型 [{qualified_name}] 额度已耗尽，尝试 fallback..."
                    )
                else:
                    _log.warning(
                        f"模型 [{qualified_name}] 返回空结果，尝试 fallback..."
                    )
            except Exception as e:
                last_error = str(e)
                _log.warning(
                    f"模型 [{qualified_name}] 调用失败: {e}，尝试 fallback..."
                )

        _log.error(
            f"所有模型 fallback 失败: chain={model_chain} last_error={last_error}"
        )
        return None, None, None

    async def simple_chat(
        self,
        model_name: str,
        messages: Iterable[ChatCompletionMessageParam],
        max_tokens: int = 500,
    ) -> Optional[str]:
        """不带工具的单轮对话（给心跳和简单任务用）。

        model_name 可以是全限定名 (deepseek/primary) 或组名。
        """
        svc = self._services.get(model_name)
        if svc is None:
            chain = self._groups.get(model_name)
            if chain:
                svc = self._services.get(chain[0])
        if svc is None:
            _log.warning(f"模型 [{model_name}] 未注册")
            return None
        try:
            result, _ = await svc.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
            )
            return result
        except Exception as e:
            _log.warning(f"模型 [{model_name}] simple_chat 失败: {e}")
            return None

    async def close(self):
        for name, svc in self._services.items():
            await svc.close()
        _log.info("ModelRegistry 已关闭")
