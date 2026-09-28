"""Provider-family token counting seam.

Real token accounting (cost, cache hit/miss) comes from the provider ``usage``
payload.  This module is only for *estimating* token counts where no usage is
available — retention/compression watermarks and prompt budgets.

Each model family can register its own counter; the default is a cheap
``len(text) // 4`` heuristic.  Callers that know the model name should resolve a
counter through :data:`TOKEN_COUNTERS` (or ``ModelRegistry.count_tokens``).
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Iterable

_log = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]
VerifyHook = Callable[[], None]

_warned_missing_tokenizer = False


def _warn_missing_tokenizer(exc: Exception) -> None:
    """Warn once when a family tokenizer degrades to the heuristic."""
    global _warned_missing_tokenizer
    if not _warned_missing_tokenizer:
        _warned_missing_tokenizer = True
        _log.warning("deepseek tokenizer 不可用，回落 len//4: %s", exc)


def default_token_counter(text: str) -> int:
    """Cheap provider-agnostic estimate (roughly 4 chars per token)."""
    return max(0, len(text or "") // 4)


@functools.lru_cache(maxsize=4096)
def _deepseek_count(text: str) -> int:
    from deepseek_tokenizer import ds_token

    return len(ds_token.encode(text))


def verify_deepseek_tokenizer() -> None:
    """Raise when the DeepSeek tokenizer cannot be imported or used."""
    from deepseek_tokenizer import ds_token

    ds_token.encode("ok")


def deepseek_token_counter(text: str) -> int:
    """DeepSeek-family counter; falls back to the heuristic if unavailable."""
    if not text:
        return 0
    try:
        return _deepseek_count(text)
    except Exception as exc:  # pragma: no cover - import/runtime failure
        _warn_missing_tokenizer(exc)
        return default_token_counter(text)


class TokenCounterRegistry:
    """Map a model name to a family token counter (first matcher wins)."""

    def __init__(self) -> None:
        self._matchers: list[
            tuple[Callable[[str], bool], TokenCounter, VerifyHook | None]
        ] = []

    def register(
        self,
        matches: Callable[[str], bool],
        counter: TokenCounter,
        *,
        verify: VerifyHook | None = None,
    ) -> None:
        self._matchers.append((matches, counter, verify))

    def resolve(self, model: str) -> TokenCounter:
        name = (model or "").lower()
        for matches, counter, _verify in self._matchers:
            try:
                if matches(name):
                    return counter
            except Exception:  # pragma: no cover - defensive
                continue
        return default_token_counter

    def verify_available(self, models: Iterable[str]) -> None:
        """Fail fast when a tokenizer needed by a configured model is missing.

        Only families that declare a ``verify`` hook are checked, so an
        unrelated missing optional tokenizer never blocks startup.
        """
        checked: set[int] = set()
        missing: list[str] = []
        for model in models:
            name = (model or "").lower()
            for matches, _counter, verify in self._matchers:
                if verify is None or id(verify) in checked:
                    continue
                try:
                    matched = matches(name)
                except Exception:  # pragma: no cover - defensive
                    matched = False
                if not matched:
                    continue
                checked.add(id(verify))
                try:
                    verify()
                except Exception as exc:
                    missing.append(f"{model}: {exc}")
        if missing:
            raise RuntimeError(
                "tokenizer 依赖不可用，拒绝启动（请确认已 `uv sync`）："
                + "; ".join(sorted(set(missing)))
            )


def _is_deepseek(model: str) -> bool:
    return "deepseek" in model


TOKEN_COUNTERS = TokenCounterRegistry()
TOKEN_COUNTERS.register(
    _is_deepseek, deepseek_token_counter, verify=verify_deepseek_tokenizer
)


def count_tokens_for(model: str, text: str) -> int:
    """Resolve the family counter for ``model`` and count ``text``."""
    return TOKEN_COUNTERS.resolve(model)(text)
