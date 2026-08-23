import sqlite3
from types import SimpleNamespace

import pytest

from core.engine.model_context_transcript import (
    ModelContextInvariantError,
    ModelContextScope,
    ModelContextTranscript,
)
from core.engine.turn_protocol_history import ProtocolEvent
from core.managers.cost_tracker import CostTracker
from core.managers.session_manager import InboundIntent


def _scope(principal_id="user-1", task_correlation_id=""):
    return ModelContextScope.for_intent(
        chat_id="chat-1",
        principal_id=principal_id,
        intent=(
            InboundIntent.DIRECT_TASK
            if task_correlation_id
            else InboundIntent.PRIVATE_CONVERSATION
        ),
        task_correlation_id=task_correlation_id,
    )


def _protocol(turn_id="turn-1", content="answer"):
    return (
        ProtocolEvent(
            turn_id=turn_id,
            seq=1,
            event_id=f"assistant:{turn_id}",
            role="assistant",
            content=content,
            timestamp=10,
        ),
    )


@pytest.mark.asyncio
async def test_transcript_is_idempotent_and_scope_isolated(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    user_event = SimpleNamespace(
        role="user",
        event_id="user:1",
        content="hello",
        sender_id="user-1",
        timestamp=10,
    )

    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(user_event,),
        protocol_events=_protocol(),
    )
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(user_event,),
        protocol_events=_protocol(),
    )

    snapshot = await transcript.snapshot(scope)
    assert [event.role for event in snapshot.events] == ["user", "assistant"]
    assert snapshot.to_wire()[-1] == {"role": "assistant", "content": "answer"}
    assert "source_turn_id" not in snapshot.to_wire()[-1]
    assert (await transcript.snapshot(_scope("user-2"))).events == ()
    await transcript.close()


@pytest.mark.asyncio
async def test_transcript_rejects_incomplete_tool_protocol(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    assistant = ProtocolEvent(
        turn_id="turn-1",
        seq=1,
        event_id="assistant:1",
        role="assistant",
        content="",
        tool_calls=({"id": "call-1", "type": "function"},),
    )

    with pytest.raises(ModelContextInvariantError, match="incomplete"):
        await transcript.append_turn(
            _scope(),
            turn_id="turn-1",
            user_events=(),
            protocol_events=(assistant,),
        )

    orphan = ProtocolEvent(
        turn_id="turn-2",
        seq=1,
        event_id="tool:1",
        role="tool",
        content="result",
        tool_call_id="missing",
    )
    with pytest.raises(ModelContextInvariantError, match="orphan"):
        await transcript.append_turn(
            _scope(),
            turn_id="turn-2",
            user_events=(),
            protocol_events=(orphan,),
        )
    await transcript.close()


@pytest.mark.asyncio
async def test_snapshot_token_limit_counts_tool_wire_fields(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"), max_tokens=2000
    )
    protocol = (
        ProtocolEvent(
            turn_id="turn-1",
            seq=1,
            event_id="assistant:1",
            role="assistant",
            content="",
            tool_calls=(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "x" * 100},
                },
            ),
        ),
        ProtocolEvent(
            turn_id="turn-1",
            seq=2,
            event_id="tool:1",
            role="tool",
            content="result",
            tool_call_id="call-1",
        ),
    )
    await transcript.append_turn(
        _scope(), turn_id="turn-1", user_events=(), protocol_events=protocol
    )

    with pytest.raises(ModelContextInvariantError, match="token limit"):
        await transcript.snapshot(_scope(), max_tokens=5)
    await transcript.close()


@pytest.mark.asyncio
async def test_append_turn_rolls_back_when_idempotency_collision_follows_insert(
    tmp_path,
):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol("turn-1", "original"),
    )

    with pytest.raises(ModelContextInvariantError, match="idempotency key collision"):
        await transcript.append_turn(
            scope,
            turn_id="turn-1",
            user_events=(),
            protocol_events=(
                ProtocolEvent(
                    turn_id="turn-1",
                    seq=1,
                    event_id="assistant:new",
                    role="assistant",
                    content="must rollback",
                ),
                ProtocolEvent(
                    turn_id="turn-1",
                    seq=2,
                    event_id="assistant:turn-1",
                    role="assistant",
                    content="changed",
                ),
            ),
        )

    snapshot = await transcript.snapshot(scope)
    assert [event.content for event in snapshot.events] == ["original"]
    await transcript.close()


@pytest.mark.asyncio
async def test_generation_bump_hides_previous_prefix(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope(task_correlation_id="task-1")
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol(),
    )

    next_scope = await transcript.bump_generation(scope, fingerprint="new-tools")
    assert next_scope.generation == 2
    assert (await transcript.snapshot(next_scope)).events == ()
    assert (await transcript.snapshot(scope)).events == ()
    await transcript.close()


@pytest.mark.asyncio
async def test_provider_binding_keeps_generation_stable_and_changes_on_provider_switch(
    tmp_path,
):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()

    first = await transcript.ensure_generation(
        scope, "base", provider_identity="deepseek:flash"
    )
    same = await transcript.ensure_generation(
        first, "base", provider_identity="deepseek:flash"
    )
    switched = await transcript.ensure_generation(
        same, "base", provider_identity="openai:gpt"
    )

    assert same.generation == first.generation
    assert switched.generation == first.generation + 1
    await transcript.close()


@pytest.mark.asyncio
async def test_compaction_replaces_complete_protocol_turn_with_summary(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    protocol = (
        ProtocolEvent(
            turn_id="turn-1",
            seq=1,
            event_id="assistant:1",
            role="assistant",
            content="",
            tool_calls=({"id": "call-1", "type": "function"},),
        ),
        ProtocolEvent(
            turn_id="turn-1",
            seq=2,
            event_id="tool:1",
            role="tool",
            content="private result",
            tool_call_id="call-1",
        ),
    )
    await transcript.append_turn(
        scope, turn_id="turn-1", user_events=(), protocol_events=protocol
    )

    snapshot = await transcript.compact(
        scope,
        summary="The tool lookup completed successfully.",
        source_turn_ids=("turn-1",),
        source_event_ids=("assistant:1", "tool:1"),
        replacement_event_id="turn-1-summary",
    )

    assert len(snapshot.events) == 1
    assert snapshot.events[0].compacted is True
    assert snapshot.events[0].content == "The tool lookup completed successfully."
    assert snapshot.events[0].source_event_ids == ("assistant:1", "tool:1")
    assert "private result" not in str(snapshot.to_wire())
    await transcript.close()


@pytest.mark.asyncio
async def test_compaction_rolls_back_when_replacement_insert_fails(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol("turn-1", "original"),
    )
    conn = await transcript._ensure_open()
    conn.execute("""
        CREATE TRIGGER reject_compaction_summary
        BEFORE INSERT ON model_context_events
        WHEN NEW.event_id LIKE 'compaction:%'
        BEGIN
            SELECT RAISE(ABORT, 'test compaction failure');
        END
        """)
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="test compaction failure"):
        await transcript.compact(
            scope,
            summary="must not replace",
            source_turn_ids=("turn-1",),
        )

    snapshot = await transcript.snapshot(scope)
    assert [event.content for event in snapshot.events] == ["original"]
    await transcript.close()


@pytest.mark.asyncio
async def test_close_deletes_task_scope_instead_of_leaving_empty_metadata(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope(task_correlation_id="task-1")
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol(),
    )

    await transcript.close_scope(scope)

    status = await transcript.status()
    assert status["scope_count"] == 0
    assert status["event_count"] == 0
    await transcript.close()


@pytest.mark.asyncio
async def test_transcript_prunes_complete_old_turns_to_configured_bound(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"), max_events=1
    )
    scope = _scope()

    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol("turn-1", "first"),
    )
    await transcript.append_turn(
        scope,
        turn_id="turn-2",
        user_events=(),
        protocol_events=_protocol("turn-2", "second"),
    )

    snapshot = await transcript.snapshot(scope)
    assert [event.content for event in snapshot.events] == ["second"]
    await transcript.close()


@pytest.mark.asyncio
async def test_transcript_rejects_unsupported_schema_version(tmp_path):
    path = tmp_path / "model-context.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE model_context_schema (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO model_context_schema(version) VALUES (99)")
    conn.commit()
    conn.close()

    transcript = ModelContextTranscript(str(path))
    with pytest.raises(ModelContextInvariantError, match="schema version"):
        await transcript.snapshot(_scope())


def test_cost_tracker_records_missing_cache_usage_as_an_observation():
    tracker = CostTracker()

    tracker.record_turn("chat-1", "deepseek-v4-flash", {"prompt_tokens": 12})

    observation = tracker.cache_observations()[0]
    assert observation["cache_usage_present"] is False
    assert observation["prompt_cache_hit_tokens"] == 0
    assert observation["prompt_cache_miss_tokens"] == 0
    assert tracker.get_global_stats().cache_usage_missing_count == 1


def test_cost_tracker_excludes_compaction_from_cache_observations():
    tracker = CostTracker()

    tracker.record_turn(
        "chat-1",
        "deepseek-v4-flash",
        {"prompt_tokens": 10},
        metadata={"usage_kind": "compaction"},
    )

    assert tracker.cache_observations() == []
    assert tracker.get_global_stats().turn_count == 1
