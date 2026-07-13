"""ModelRegistry — 多模型注册表 + fallback 链。

管理多个 AIService 实例（每个模型一个），
支持按模型链顺序调用，失败自动 fallback。
"""

import logging
from typing import Any, Dict, Iterable, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from core.ai.service import AIService

_log = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表。

    用法:
        registry = ModelRegistry(config["models"])
        result, model_name = await registry.chat_with_fallback(
            ["cheap", "primary"], messages, tools,
        )
    """

    def __init__(self, models_config: dict):
        self._services: Dict[str, AIService] = {}
        self._tier_map: Dict[str, List[str]] = {}
        self._default_service: Optional[AIService] = None

        for name, cfg in models_config.items():
            svc = AIService(
                api_key=cfg.get("api_key", ""),
                base_url=cfg.get("base_url"),
                model=cfg.get("model", "gpt-3.5-turbo"),
                timeout=cfg.get("timeout", 30),
                max_retries=cfg.get("max_retries", 0),
                temperature=cfg.get("temperature", 0.7),
                max_tokens=cfg.get("max_tokens", 8192),
                reasoning_effort=cfg.get("reasoning_effort"),
            )
            self._services[name] = svc
            if self._default_service is None:
                self._default_service = svc

            _log.info(
                f"模型 [{name}]: {cfg.get('model')} @ {cfg.get('base_url')}"
            )

    @property
    def default_service(self) -> Optional[AIService]:
        return self._default_service

    def get(self, name: str) -> Optional[AIService]:
        return self._services.get(name)

    def configure_tiers(self, tier_config: dict):
        """配置分档 → 模型链映射。

        tier_config: {
            "simple": ["cheap", "primary"],
            "medium": ["primary"],
            ...
        }
        """
        self._tier_map = {}
        for tier, chain in tier_config.items():
            valid = [m for m in chain if m in self._services]
            if valid:
                self._tier_map[tier] = valid
                _log.info(f"  分档 [{tier}]: 模型链 {valid}")
            else:
                _log.warning(f"  分档 [{tier}]: 无可用的模型")

    def get_chain(self, tier: str, fallback: Optional[List[str]] = None) -> List[str]:
        """获取指定分档的模型链。"""
        chain = self._tier_map.get(tier)
        if chain:
            return chain
        if fallback:
            return fallback
        if self._default_service:
            # 回退到默认模型的 key
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
        last_error = None
        for model_name in model_chain:
            svc = self._services.get(model_name)
            if svc is None:
                _log.warning(f"模型 [{model_name}] 未注册，跳过")
                continue

            try:
                result, usage = await svc.chat_completion_with_tools(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
                if result is not None:
                    _log.debug(
                        f"模型 [{model_name}] 调用成功"
                    )
                    return result, usage, model_name
                last_error = "返回空结果"
            except Exception as e:
                last_error = str(e)
                _log.warning(
                    f"模型 [{model_name}] 调用失败: {e}，尝试 fallback..."
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
        """不带工具的单轮对话（给心跳和简单任务用）。"""
        svc = self._services.get(model_name)
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
