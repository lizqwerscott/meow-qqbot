"""ModelScopeRateLimit — ModelScope 限流信息追踪。

从 HTTP 响应头解析 4 个限流字段：
- modelscope-ratelimit-requests-limit      用户当天总限额
- modelscope-ratelimit-requests-remaining  用户当天剩余额度
- modelscope-ratelimit-model-requests-limit      模型当天总限额
- modelscope-ratelimit-model-requests-remaining  模型当天剩余额度
"""

import time
from dataclasses import dataclass


@dataclass
class ModelScopeRateLimit:
    user_limit: int = 0
    user_remaining: int = -1
    model_limit: int = 0
    model_remaining: int = -1
    last_updated: float = 0.0

    @property
    def can_call(self) -> bool:
        """用户和模型都有剩余额度时才能调用。
        负值表示尚未初始化（未知），允许首次调用以获取真实额度。"""
        if self.user_remaining < 0 or self.model_remaining < 0:
            return True
        return self.user_remaining > 0 and self.model_remaining > 0

    @property
    def exhausted(self) -> bool:
        return not self.can_call

    @classmethod
    def from_headers(cls, headers) -> "ModelScopeRateLimit":
        return cls(
            user_limit=int(headers.get("modelscope-ratelimit-requests-limit", 0)),
            user_remaining=int(headers.get("modelscope-ratelimit-requests-remaining", -1)),
            model_limit=int(headers.get("modelscope-ratelimit-model-requests-limit", 0)),
            model_remaining=int(headers.get("modelscope-ratelimit-model-requests-remaining", -1)),
            last_updated=time.time(),
        )
