"""FallbackRunner — 统一回退编排器。

按模型链顺序尝试调用，失败自动 fallback。
消除 ModelRegistry.chat_with_fallback 与 ToolLoop.run 中重复的回退编排逻辑。
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

_log = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """回退执行结果。

    .ok → model_name 不为 None 即成功。
    """
    message: Any = None
    usage: Optional[Dict] = None
    model_name: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.model_name is not None


class FallbackRunner:
    """统一回退编排器。

    封装模型链的可用性解析、失败回退和冷却管理。
    两种使用模式：

    模式1 — 迭代模式（给 ToolLoop，每轮工具循环可独立重试）：
        runner = FallbackRunner(registry, chain)
        if not await runner.acquire():
            # 全部不可用
            return
        while True:
            try:
                result = await runner.service().chat(...)
                runner.mark_success()
            except:
                runner.mark_failure(record_cooldown=True)
                if not await runner.acquire():
                    result = await runner.last_resort(...)
                    break

    模式2 — 一次性模式（给 chat_with_fallback）：
        result = await runner.run(
            lambda svc, name: svc.chat_completion_with_tools(...),
        )
    """

    def __init__(self, registry, chain: List[str]):
        self._registry = registry
        self._chain = list(chain)
        self._failed: Set[str] = set()
        self._current_name: Optional[str] = None
        self._current_svc: Optional[Any] = None

    @property
    def current(self) -> Optional[str]:
        return self._current_name

    @property
    def remaining(self) -> List[str]:
        return [m for m in self._chain if m not in self._failed]

    def service(self) -> Optional[Any]:
        return self._current_svc

    async def acquire(self) -> bool:
        """从剩余链中获取第一个可用模型。

        跳过冷却中的和已失败的模型。
        Returns:
            True 表示可用 (可通过 .service() 和 .current 获取)
            False 表示全部不可用
        """
        resolved = await self._registry.resolve_model_chain(self.remaining)
        if resolved:
            self._current_name, self._current_svc = resolved
            return True
        self._current_name = None
        self._current_svc = None
        return False

    async def try_acquire_with_binding(self, mgr, chat_id, tier) -> bool:
        """尝试 session 绑定 -> 命中且有效则直接使用；否则回退 acquire 并绑定。

        Returns:
            True 表示可用 (service() / current 就绪)
        """
        if mgr and tier:
            # get_if_valid 原子读取绑定，暂用 inf 跳过过期检查（需要 binding.model_name 查配置）
            binding = await mgr.get_if_valid(chat_id, tier,
                                             budget=float('inf'), ttl=float('inf'))
            if binding:
                cfg = (self._registry.get_session_config(binding.model_name)
                       if self._registry else None)
                budget, ttl = cfg if cfg else (30, 600)
                if binding.is_expired(budget, ttl):
                    await mgr.unbind(chat_id, tier)
                else:
                    svc = self._registry.get(binding.model_name) if self._registry else None
                    if svc and not await self._registry.cooldown_manager.is_cooled_down(
                        binding.model_name
                    ):
                        self._current_name = binding.model_name
                        self._current_svc = svc
                        return True
                    await mgr.unbind(chat_id, tier)

        ok = await self.acquire()
        if ok and mgr and tier:
            await mgr.bind(chat_id, tier, self._current_name)
        return ok

    async def mark_success(self, *, mgr=None, chat_id=None, tier=None):
        """标记当前模型调用成功。

        Args:
            mgr: SessionBindingManager 实例（传入则自动计数 +1）
            chat_id, tier: 绑定键，与 mgr 搭配使用
        """
        if self._current_name and self._registry:
            await self._registry.cooldown_manager.record_success(self._current_name)
        if mgr and chat_id and tier:
            await mgr.tick(chat_id, tier)

    async def mark_failure(self, *, record_cooldown: bool = True):
        """标记当前模型失败。

        Args:
            record_cooldown: 是否写入全局冷却。
              异常场景=True（模型可能出问题了），空结果=False（可能是安全过滤等偶发现象）。
        """
        if self._current_name:
            self._failed.add(self._current_name)
            if record_cooldown and self._registry:
                await self._registry.cooldown_manager.record_failure(
                    self._current_name
                )
            _log.warning(f"模型 [{self._current_name}] 失败，尝试 fallback...")
            self._current_name = None
            self._current_svc = None

    def reset_failures(self):
        """重置失败记录（工具轮次间调用，清除对回退失败的记忆）。"""
        self._failed.clear()

    async def run(
        self,
        invoke,
        *,
        record_cooldown: bool = False,
    ) -> FallbackResult:
        """逐个尝试模型链，失败自动 fallback。

        Args:
            invoke: async (service, model_name) -> (message, usage)
            record_cooldown: 空结果时是否写入冷却（异常时总是写入）。

        Returns:
            FallbackResult
        """
        while True:
            ok = await self.acquire()
            if not ok:
                return FallbackResult()

            try:
                msg, usage = await invoke(self._current_svc, self._current_name)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _log.warning(f"模型 [{self._current_name}] 调用异常: {e}")
                await self.mark_failure(record_cooldown=True)
                continue

            if msg is not None:
                await self.mark_success()
                return FallbackResult(
                    message=msg, usage=usage, model_name=self._current_name,
                )

            await self.mark_failure(record_cooldown=record_cooldown)

    async def last_resort(
        self,
        messages: Iterable[Any],
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
    ) -> FallbackResult:
        """兜底：剩余链全在冷却时，强制尝试一遍（忽略冷却）。

        跳过 quota 预检，直接逐个调用——兜底场景下死马当活马医。
        """
        remaining = self.remaining
        if not remaining:
            return FallbackResult()
        for qualified_name in remaining:
            svc = self._registry.get(qualified_name) if self._registry else None
            if svc is None:
                continue
            try:
                msg, usage = await svc.chat_completion_with_tools(
                    messages=messages, tools=tools, max_tokens=max_tokens,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _log.warning(f"兜底: 模型 [{qualified_name}] 调用失败: {e}")
                continue
            if msg is not None:
                _log.warning(f"兜底成功: 模型 [{qualified_name}]")
                self._current_name = qualified_name
                self._current_svc = svc
                return FallbackResult(message=msg, usage=usage, model_name=qualified_name)
        _log.error(f"兜底失败: remaining={remaining}")
        return FallbackResult()
