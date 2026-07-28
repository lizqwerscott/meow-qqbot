import time

from core.ai.rate_limit import ModelScopeRateLimit


def test_can_call_both_unknown():
    rl = ModelScopeRateLimit(user_remaining=-1, model_remaining=-1)
    assert rl.can_call is True


def test_can_call_both_positive():
    rl = ModelScopeRateLimit(user_remaining=5, model_remaining=10)
    assert rl.can_call is True


def test_can_call_user_exhausted():
    rl = ModelScopeRateLimit(user_remaining=0, model_remaining=10)
    assert rl.can_call is False


def test_can_call_model_exhausted():
    rl = ModelScopeRateLimit(user_remaining=5, model_remaining=0)
    assert rl.can_call is False


def test_can_call_both_exhausted():
    rl = ModelScopeRateLimit(user_remaining=0, model_remaining=0)
    assert rl.can_call is False


# ── bug #2 回归：其中一个标头不存在时 should still allow ──

def test_can_call_user_known_model_unknown():
    rl = ModelScopeRateLimit(user_remaining=5, model_remaining=-1)
    assert rl.can_call is True, "一个标头未知时不应拒绝调用"


def test_can_call_model_known_user_unknown():
    rl = ModelScopeRateLimit(user_remaining=-1, model_remaining=10)
    assert rl.can_call is True, "一个标头未知时不应拒绝调用"


# ── exhausted ──

def test_exhausted():
    rl = ModelScopeRateLimit(user_remaining=0, model_remaining=0)
    assert rl.exhausted is True


def test_not_exhausted():
    rl = ModelScopeRateLimit(user_remaining=5, model_remaining=10)
    assert rl.exhausted is False


# ── from_headers ──

def test_from_headers_full():
    headers = {
        "modelscope-ratelimit-requests-limit": "100",
        "modelscope-ratelimit-requests-remaining": "42",
        "modelscope-ratelimit-model-requests-limit": "200",
        "modelscope-ratelimit-model-requests-remaining": "88",
    }
    rl = ModelScopeRateLimit.from_headers(headers)
    assert rl.user_limit == 100
    assert rl.user_remaining == 42
    assert rl.model_limit == 200
    assert rl.model_remaining == 88
    assert rl.last_updated > 0


def test_from_headers_missing():
    headers = {
        "modelscope-ratelimit-requests-limit": "100",
        "modelscope-ratelimit-requests-remaining": "42",
    }
    rl = ModelScopeRateLimit.from_headers(headers)
    assert rl.user_remaining == 42
    assert rl.model_remaining == -1
    assert rl.model_limit == 0


def test_from_headers_empty():
    rl = ModelScopeRateLimit.from_headers({})
    assert rl.user_limit == 0
    assert rl.user_remaining == -1
    assert rl.model_remaining == -1


def test_defaults():
    rl = ModelScopeRateLimit()
    assert rl.user_limit == 0
    assert rl.user_remaining == -1
    assert rl.model_limit == 0
    assert rl.model_remaining == -1
    assert rl.last_updated == 0.0
