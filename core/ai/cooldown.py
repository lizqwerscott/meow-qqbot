"""ModelCooldownManager — 模型冷却管理器。

追踪模型的连续失败次数，在冷却期内直接跳过该模型，
避免每次请求都重复尝试已失效的模型，加速 fallback。
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

_log = logging.getLogger(__name__)


@dataclass
class CooldownState:
    failure_count: int = 0
    last_failure_time: float = 0.0
    cooldown_until: float = 0.0


class ModelCooldownManager:
    """模型冷却管理器。

    用法:
        cm = ModelCooldownManager({"base_cooldown": 60, "max_cooldown": 3600})
        if cm.is_cooled_down("modelscope/ds-flash"):
            # 跳过此模型
            ...
        # 调用后
        if success:
            cm.record_success("modelscope/ds-flash")
        else:
            cm.record_failure("modelscope/ds-flash")
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._base_cooldown = cfg.get("base_cooldown", 60)
        self._max_cooldown = cfg.get("max_cooldown", 3600)
        self._failure_threshold = cfg.get("failure_threshold", 1)
        self._state: Dict[str, CooldownState] = {}

    def is_cooled_down(self, name: str) -> bool:
        if not self._enabled:
            return False
        state = self._state.get(name)
        if state is None:
            return False
        if state.failure_count < self._failure_threshold:
            return False
        now = time.time()
        if now >= state.cooldown_until:
            return False
        remaining = state.cooldown_until - now
        _log.info(
            f"模型 [{name}] 冷却中 (剩余 {remaining:.0f}s, "
            f"连续失败 {state.failure_count} 次)"
        )
        return True

    def record_failure(self, name: str):
        if not self._enabled:
            return
        now = time.time()
        state = self._state.setdefault(name, CooldownState())
        state.failure_count += 1
        state.last_failure_time = now
        delay = self._compute_delay(state.failure_count)
        state.cooldown_until = now + delay
        _log.warning(
            f"模型 [{name}] 记录失败 (连续 {state.failure_count} 次, "
            f"冷却 {delay:.0f}s)"
        )

    def record_success(self, name: str):
        if not self._enabled:
            return
        state = self._state.get(name)
        if state is None or state.failure_count == 0:
            return
        old_count = state.failure_count
        state.failure_count = 0
        state.cooldown_until = 0.0
        _log.info(
            f"模型 [{name}] 调用成功，重置冷却 "
            f"(此前连续失败 {old_count} 次)"
        )

    def reset(self, name: Optional[str] = None):
        if name:
            self._state.pop(name, None)
            _log.info(f"已清除模型 [{name}] 的冷却状态")
        else:
            self._state.clear()
            _log.info("已清除所有模型冷却状态")

    def get_all_states(self) -> Dict[str, dict]:
        """返回所有模型冷却状态的快照（dict 副本，不会被后续变更影响）。"""
        return {
            name: {
                "failure_count": s.failure_count,
                "last_failure_time": s.last_failure_time,
                "cooldown_until": s.cooldown_until,
            }
            for name, s in self._state.items()
        }

    def _compute_delay(self, failure_count: int) -> float:
        if failure_count < self._failure_threshold:
            return 0.0
        # 指数退避: base * 2^(count - threshold)
        exponent = failure_count - self._failure_threshold
        delay = self._base_cooldown * (2 ** exponent)
        return min(delay, self._max_cooldown)
