"""ModelRegistry.get_session_config 三层级配置解析测试。"""

import pytest

from core.ai.model_registry import ModelRegistry


def _make_registry(**overrides):
    """创建 ModelRegistry，仅 session_budget / session_ttl 相关字段。"""
    providers = {
        "provider_a": {
            "api_key": "test",
            "models": [
                {"name": "base", "model": "base-model"},
            ],
        },
        "provider_budget": {
            "api_key": "test",
            "session_budget": 100,
            "models": [
                {"name": "from_provider", "model": "fm"},
            ],
        },
        "provider_ttl": {
            "api_key": "test",
            "session_ttl": 300.0,
            "models": [
                {"name": "ttl_from_provider", "model": "tm"},
            ],
        },
        "provider_both": {
            "api_key": "test",
            "session_budget": 50,
            "session_ttl": 200.0,
            "models": [
                {"name": "both_from_provider", "model": "bm"},
                {"name": "override", "model": "om", "session_budget": 10, "session_ttl": 30.0},
                {"name": "partial_override", "model": "pm", "session_budget": 99},
            ],
        },
    }
    return ModelRegistry(providers, {}, **overrides)


class TestSessionConfigDefault:
    """兜底默认值。"""

    def test_default_when_nothing_set(self):
        reg = _make_registry()
        budget, ttl = reg.get_session_config("provider_a/base")
        assert budget == 30
        assert ttl == 600.0


class TestSessionConfigProvider:
    """Provider 级继承。"""

    def test_budget_from_provider(self):
        reg = _make_registry()
        budget, _ = reg.get_session_config("provider_budget/from_provider")
        assert budget == 100

    def test_ttl_from_provider(self):
        reg = _make_registry()
        _, ttl = reg.get_session_config("provider_ttl/ttl_from_provider")
        assert ttl == 300.0

    def test_both_from_provider(self):
        reg = _make_registry()
        budget, ttl = reg.get_session_config("provider_both/both_from_provider")
        assert budget == 50
        assert ttl == 200.0


class TestSessionConfigModel:
    """Model 级覆盖 Provider。"""

    def test_override_both(self):
        reg = _make_registry()
        budget, ttl = reg.get_session_config("provider_both/override")
        assert budget == 10
        assert ttl == 30.0

    def test_partial_override(self):
        """只覆盖 budget，ttl 继承 provider。"""
        reg = _make_registry()
        budget, ttl = reg.get_session_config("provider_both/partial_override")
        assert budget == 99
        assert ttl == 200.0

    def test_model_zero_budget_preserved(self):
        """regression: budget=0 不应被 or 吞成 provider 的 50。"""
        providers = {
            "p": {
                "api_key": "test",
                "session_budget": 50,
                "models": [
                    {"name": "z", "model": "z", "session_budget": 0},
                ],
            },
        }
        reg = ModelRegistry(providers, {})
        budget, _ = reg.get_session_config("p/z")
        assert budget == 0

    def test_model_zero_ttl_preserved(self):
        providers = {
            "p": {
                "api_key": "test",
                "session_ttl": 300.0,
                "models": [
                    {"name": "z", "model": "z", "session_ttl": 0},
                ],
            },
        }
        reg = ModelRegistry(providers, {})
        _, ttl = reg.get_session_config("p/z")
        assert ttl == 0

    def test_unknown_model_returns_default(self):
        reg = _make_registry()
        budget, ttl = reg.get_session_config("nonexistent/model")
        assert budget == 30
        assert ttl == 600.0
