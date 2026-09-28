from types import SimpleNamespace

import pytest

from core.ai import tokenizers
from core.ai.model_registry import ModelRegistry
from core.ai.tokenizers import (
    TOKEN_COUNTERS,
    TokenCounterRegistry,
    deepseek_token_counter,
    default_token_counter,
)


def test_default_token_counter_is_quarter_length():
    assert default_token_counter("") == 0
    assert default_token_counter("abcdefghij") == 2


def test_registry_resolves_deepseek_family():
    assert TOKEN_COUNTERS.resolve("deepseek-v4-flash") is deepseek_token_counter
    assert (
        TOKEN_COUNTERS.resolve("deepseek-ai/DeepSeek-V4-Pro") is deepseek_token_counter
    )


def test_registry_falls_back_for_other_families():
    assert TOKEN_COUNTERS.resolve("gpt-4o") is default_token_counter
    assert TOKEN_COUNTERS.resolve("Qwen/Qwen3-235B-A22B-Instruct-2507") is (
        default_token_counter
    )
    assert TOKEN_COUNTERS.resolve("") is default_token_counter


def test_deepseek_counter_is_not_the_heuristic():
    text = "你好，世界 hello world 这是一段用于对比分词结果的中文与英文混合文本。"
    assert deepseek_token_counter(text) != default_token_counter(text)
    assert deepseek_token_counter(text) > 0


def test_deepseek_counter_falls_back_when_tokenizer_unavailable(monkeypatch):
    from core.ai import tokenizers

    def _boom(_text):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(tokenizers, "_deepseek_count", _boom)
    assert deepseek_token_counter("hello world") == default_token_counter("hello world")


def test_model_registry_count_tokens_uses_provider_when_available():
    registry = ModelRegistry({}, {})
    registry._services["p/deep"] = SimpleNamespace(count_tokens=lambda text: 7)

    assert registry.count_tokens("p/deep", "whatever") == 7
    assert registry.count_tokens("p/missing", "abcdefgh") == 2


def test_model_registry_count_tokens_falls_back_on_error():
    registry = ModelRegistry({}, {})

    def _boom(_text):
        raise RuntimeError("bad")

    registry._services["p/bad"] = SimpleNamespace(count_tokens=_boom)
    assert registry.count_tokens("p/bad", "abcdefgh") == 2


def test_model_registry_token_counter_adapter():
    registry = ModelRegistry({}, {})
    registry._services["p/deep"] = SimpleNamespace(count_tokens=lambda text: 9)
    resolver = registry.token_counter()

    assert resolver("anything", "p/deep") == 9
    assert resolver("abcdefgh", "") == 2


def test_verify_available_passes_for_installed_tokenizer():
    # deepseek-tokenizer is a hard dependency, so this must not raise.
    TOKEN_COUNTERS.verify_available(["deepseek-v4-flash", "gpt-4o"])


def test_verify_available_raises_for_missing_needed_tokenizer():
    registry = TokenCounterRegistry()

    def _boom() -> None:
        raise ImportError("no module named 'deepseek_tokenizer'")

    registry.register(lambda m: "deepseek" in m, deepseek_token_counter, verify=_boom)

    with pytest.raises(RuntimeError, match="tokenizer"):
        registry.verify_available(["deepseek-v4-flash"])
    # A model whose family declares no verify hook never blocks startup.
    registry.verify_available(["gpt-4o"])


def test_deepseek_fallback_warns_only_once(monkeypatch, caplog):
    def _boom(_text):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(tokenizers, "_deepseek_count", _boom)
    monkeypatch.setattr(tokenizers, "_warned_missing_tokenizer", False)

    with caplog.at_level("WARNING", logger="core.ai.tokenizers"):
        deepseek_token_counter("one")
        deepseek_token_counter("two")

    warnings = [r for r in caplog.records if "deepseek tokenizer" in r.message]
    assert len(warnings) == 1
