"""Real-time cost tracking — per-turn and session-level token usage & cost estimates."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

_log = logging.getLogger(__name__)

DEFAULT_PRICING = {
    "deepseek-v4-flash": {
        "input_per_million": 1.0,
        "output_per_million": 2.0,
        "cache_hit_per_million": 0.02,
    },
    "deepseek-v4-pro": {
        "input_per_million": 3.0,
        "output_per_million": 6.0,
        "cache_hit_per_million": 0.025,
    },
}


@dataclass
class SessionCostStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cost: float = 0.0
    turn_count: int = 0
    cache_observation_count: int = 0
    cache_usage_missing_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        if total == 0:
            return 0.0
        return self.cache_hit_tokens / total


class CostTracker:
    def __init__(self, pricing: Optional[Dict] = None):
        self._pricing = DEFAULT_PRICING.copy()
        if pricing:
            self._pricing.update(pricing)
        self._session_stats: Dict[str, SessionCostStats] = {}
        self._global_stats = SessionCostStats()
        self._cache_observations = deque(maxlen=1000)

    def record_turn(
        self,
        chat_id: str,
        model: str,
        usage: Optional[Dict],
        metadata: Optional[Dict] = None,
    ) -> None:
        usage = usage or {}
        observation = dict(metadata or {})
        usage_kind = observation.get("usage_kind", "completion")
        is_cache_observation = usage_kind == "completion"

        cache_keys_present = (
            "prompt_cache_hit_tokens" in usage and "prompt_cache_miss_tokens" in usage
        )
        hit = usage.get("prompt_cache_hit_tokens", 0) or 0
        miss = usage.get("prompt_cache_miss_tokens", 0) or 0
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0

        price = self._pricing.get(model, self._pricing.get("deepseek-v4-flash", {}))
        cost = (
            miss * price.get("input_per_million", 1.0) / 1_000_000
            + hit * price.get("cache_hit_per_million", 0.02) / 1_000_000
            + completion * price.get("output_per_million", 2.0) / 1_000_000
        )

        if chat_id not in self._session_stats:
            self._session_stats[chat_id] = SessionCostStats()
        s = self._session_stats[chat_id]
        s.prompt_tokens += prompt
        s.completion_tokens += completion
        s.cache_hit_tokens += hit
        s.cache_miss_tokens += miss
        s.cost += cost
        s.turn_count += 1
        s.cache_observation_count += int(is_cache_observation)
        s.cache_usage_missing_count += int(
            is_cache_observation and not cache_keys_present
        )

        g = self._global_stats
        g.prompt_tokens += prompt
        g.completion_tokens += completion
        g.cache_hit_tokens += hit
        g.cache_miss_tokens += miss
        g.cost += cost
        g.turn_count += 1
        g.cache_observation_count += int(is_cache_observation)
        g.cache_usage_missing_count += int(
            is_cache_observation and not cache_keys_present
        )

        observation.update(
            {
                "model": model,
                "usage_kind": usage_kind,
                "prompt_tokens": prompt,
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "cache_usage_present": cache_keys_present,
            }
        )
        if is_cache_observation:
            self._cache_observations.append(observation)

    def cache_observations(self) -> list[dict]:
        return list(self._cache_observations)

    def get_session_stats(self, chat_id: str) -> Optional[SessionCostStats]:
        return self._session_stats.get(chat_id)

    def get_all_sessions(self) -> Dict[str, SessionCostStats]:
        return dict(self._session_stats)

    def get_global_stats(self) -> SessionCostStats:
        return self._global_stats
