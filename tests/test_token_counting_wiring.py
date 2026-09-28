from types import SimpleNamespace

import pytest

from core.ai.model_registry import ModelRegistry
from core.ai.tokenizers import count_tokens_for, deepseek_token_counter
from core.engine.conversation_event_log import ConversationEventLog
from core.engine.model_context_transcript import (
    ModelContextScope,
    ModelContextTranscript,
)
from core.engine.turn_protocol_history import TurnProtocolHistory
from core.engine.turn_summary import TurnSummaryStore
from core.managers.session_manager import InboundIntent

_MIXED = "你好，世界 hello world 这是一段用于对比分词结果的中文与英文混合文本。"


def _scope(task_correlation_id: str = "") -> ModelContextScope:
    return ModelContextScope.for_intent(
        chat_id="chat-1",
        principal_id="user-1",
        intent=(
            InboundIntent.DIRECT_TASK
            if task_correlation_id
            else InboundIntent.PRIVATE_CONVERSATION
        ),
        task_correlation_id=task_correlation_id,
    )


@pytest.mark.asyncio
async def test_event_log_uses_real_deepseek_counter(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    log.set_token_counter(lambda text, model: count_tokens_for(model, text))

    await log.append_user_message(
        chat_id="c",
        turn_id="t",
        message_id="m",
        content=_MIXED,
        model="deepseek-v4-flash",
    )

    snapshot = await log.snapshot_events("c", include_internal=True)
    assert snapshot.events[0].token_count == deepseek_token_counter(_MIXED)
    assert snapshot.events[0].token_count != len(_MIXED) // 4
    await log.close()


@pytest.mark.asyncio
async def test_event_log_counts_through_model_registry_provider(tmp_path):
    registry = ModelRegistry({}, {})
    registry._services["deepseek/deepseek-v4-flash"] = SimpleNamespace(
        count_tokens=deepseek_token_counter
    )
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    log.set_token_counter(registry.token_counter())

    await log.append_user_message(
        chat_id="c",
        turn_id="t",
        message_id="m",
        content=_MIXED,
        model="deepseek/deepseek-v4-flash",
    )

    snapshot = await log.snapshot_events("c", include_internal=True)
    assert snapshot.events[0].token_count == deepseek_token_counter(_MIXED)
    await log.close()


@pytest.mark.asyncio
async def test_event_log_defaults_to_heuristic_without_counter(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    await log.append_user_message(
        chat_id="c", turn_id="t", message_id="m", content="abcdefghij"
    )

    snapshot = await log.snapshot_events("c", include_internal=True)
    assert snapshot.events[0].token_count == 2
    await log.close()


@pytest.mark.asyncio
async def test_protocol_history_uses_injected_provider_counter(tmp_path):
    history = TurnProtocolHistory(
        str(tmp_path / "protocol.sqlite3"),
        token_counter=lambda text, model: 7,
    )

    event = await history.append_assistant(
        turn_id="t", event_id="e", content="hi", model="deepseek-v4-flash"
    )

    assert event.token_count == 7
    await history.close()


@pytest.mark.asyncio
async def test_protocol_history_defaults_to_heuristic_without_counter(tmp_path):
    history = TurnProtocolHistory(str(tmp_path / "protocol.sqlite3"))

    event = await history.append_assistant(
        turn_id="t", event_id="e", content="abcdefgh"
    )

    assert event.token_count == 2
    await history.close()


@pytest.mark.asyncio
async def test_transcript_scope_model_resolves_from_db_after_restart(tmp_path):
    path = str(tmp_path / "model_context.sqlite3")
    scope = _scope()

    first = ModelContextTranscript(path)
    await first.record_provider_usage(
        scope,
        {"prompt_tokens": 10},
        provider="deepseek",
        model="deepseek-v4-flash",
        turn_id="t1",
    )
    await first.close()

    restarted = ModelContextTranscript(path)
    await restarted._ensure_open()
    assert restarted._model_for(scope) == "deepseek-v4-flash"
    await restarted.close()


@pytest.mark.asyncio
async def test_turn_summary_uses_injected_counter(tmp_path):
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    store = TurnSummaryStore(log)
    store.set_token_counter(lambda text, model: 123)

    assert store._count_tokens("anything", "deepseek-v4-flash") == 123
    await store.close()
    await log.close()


def test_registry_resolves_bare_model_id_to_provider_counter():
    # Production shape: registry key is "provider/name" while the service's
    # ``model`` (and therefore AgentEngine._active_model_name) is the bare id.
    registry = ModelRegistry({}, {})
    registry._services["deepseek/primary"] = SimpleNamespace(
        model="deepseek-v4-flash", count_tokens=deepseek_token_counter
    )

    assert registry.count_tokens("deepseek-v4-flash", _MIXED) == (
        deepseek_token_counter(_MIXED)
    )
    assert registry.count_tokens("deepseek/primary", _MIXED) == (
        deepseek_token_counter(_MIXED)
    )


def test_registry_falls_back_to_family_dispatch_without_service():
    registry = ModelRegistry({}, {})

    assert registry.count_tokens("deepseek-v4-flash", _MIXED) != len(_MIXED) // 4
    assert registry.count_tokens("gpt-4o", "abcdefgh") == 2


@pytest.mark.asyncio
async def test_event_log_counts_deepseek_with_bare_model_id(tmp_path):
    registry = ModelRegistry({}, {})
    registry._services["deepseek/primary"] = SimpleNamespace(
        model="deepseek-v4-flash", count_tokens=deepseek_token_counter
    )
    log = ConversationEventLog(str(tmp_path / "events.sqlite3"))
    log.set_token_counter(registry.token_counter())

    await log.append_user_message(
        chat_id="c",
        turn_id="t",
        message_id="m",
        content=_MIXED,
        model="deepseek-v4-flash",
    )

    snapshot = await log.snapshot_events("c", include_internal=True)
    assert snapshot.events[0].token_count == deepseek_token_counter(_MIXED)
    assert snapshot.events[0].token_count != len(_MIXED) // 4
    await log.close()
