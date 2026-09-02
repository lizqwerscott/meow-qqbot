import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from core.engine.model_context_transcript import (
    ModelContextCompactionInProgressError,
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
async def test_archive_rotation_removes_hidden_source_turn_atomically(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    user_event = SimpleNamespace(
        role="user",
        event_id="user:1",
        content="old",
        sender_id="user-1",
        timestamp=10,
    )
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(user_event,),
        protocol_events=_protocol("turn-1", "old answer"),
    )
    current = await transcript.rotate_for_hidden_sources(
        scope,
        ["user:1"],
        summary_texts=("old turn summary",),
        operation_id="archive-op:scope",
    )

    assert current.scope.generation == 2
    assert [event.role for event in current.events] == ["assistant"]
    assert current.events[0].content == "old turn summary"
    assert "old answer" not in str(current.to_wire())
    await transcript.close()


@pytest.mark.asyncio
async def test_archive_rotation_replay_does_not_advance_generation_again(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    user_event = SimpleNamespace(
        role="user",
        event_id="user:1",
        content="old",
        sender_id="user-1",
        timestamp=10,
    )
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(user_event,),
        protocol_events=_protocol("turn-1", "old answer"),
    )

    first = await transcript.rotate_for_hidden_sources(
        scope,
        ["user:1"],
        summary_texts=("old turn summary",),
        summary_source_event_ids=(("user:1",),),
        operation_id="archive-op:scope",
    )
    second = await transcript.rotate_for_hidden_sources(
        scope,
        ["user:1"],
        summary_texts=("old turn summary",),
        summary_source_event_ids=(("user:1",),),
        operation_id="archive-op:scope",
    )

    assert first.scope.generation == second.scope.generation == 2
    assert [event.content for event in second.events] == ["old turn summary"]
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
async def test_local_waterline_compaction_snips_and_prunes_without_breaking_pairs(
    tmp_path,
):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=160,
        compaction_enabled=True,
        compaction_keep_recent_tokens=1,
        compaction_snip_max_chars=64,
    )
    scope = _scope()
    for turn_id in ("turn-1", "turn-2", "turn-3"):
        protocol = (
            ProtocolEvent(
                turn_id=turn_id,
                seq=1,
                event_id=f"assistant:{turn_id}",
                role="assistant",
                content="",
                tool_calls=(
                    {
                        "id": f"call-{turn_id}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    },
                ),
            ),
            ProtocolEvent(
                turn_id=turn_id,
                seq=2,
                event_id=f"tool:{turn_id}",
                role="tool",
                content="diagnostic output " * 300,
                tool_call_id=f"call-{turn_id}",
            ),
        )
        await transcript.append_turn(
            scope, turn_id=turn_id, user_events=(), protocol_events=protocol
        )
        scope = await transcript.current_scope(scope)

    result = await transcript.compact_if_needed(scope)

    assert result.changed is True
    assert result.tier in {1, 2, 3}
    assert result.scope.generation > 1
    assert any(event.operation in {"snip", "prune"} for event in result.snapshot.events)
    protocol_events = [
        event for event in result.snapshot.events if event.role in {"assistant", "tool"}
    ]
    transcript._validate_protocol(protocol_events)
    status = await transcript.status()
    assert status["compaction_committed_count"] == 1
    await transcript.close()


@pytest.mark.asyncio
async def test_incremental_summary_replaces_only_safe_old_turns(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=120,
        compaction_enabled=True,
        compaction_keep_recent_tokens=1,
    )
    scope = _scope()
    for turn_id, content in (
        ("turn-1", "old progress " * 300),
        ("turn-2", "more old progress " * 300),
        ("turn-3", "recent answer"),
    ):
        await transcript.append_turn(
            scope,
            turn_id=turn_id,
            user_events=(),
            protocol_events=_protocol(turn_id, content),
        )
        scope = await transcript.current_scope(scope)

    calls = []

    async def summarize(messages, max_tokens):
        calls.append((messages, max_tokens))
        return (
            "进展：已完成旧任务\n文件：无\n待办：继续当前任务\n上下文：保留旧结论",
            {"prompt_tokens": 20, "completion_tokens": 8},
            42.5,
        )

    result = await transcript.compact_if_needed(scope, summary_factory=summarize)

    assert result.tier == 3
    assert result.operation == "compact_replace"
    assert result.snapshot.scope.generation > scope.generation
    assert len(calls) == 1
    assert calls[0][1] == 500
    assert result.snapshot.events[-1].compacted is True
    assert result.snapshot.events[-1].operation == "compact_replace"
    assert result.snapshot.events[-1].content.startswith("进展：")
    assert result.elapsed_ms == 42.5
    assert result.snapshot.events[-2].content == "recent answer"
    status = await transcript.status()
    assert status["compaction_committed_count"] == 1
    assert status["summary_prompt_tokens"] == 20
    assert status["summary_completion_tokens"] == 8
    assert status["summary_elapsed_ms"] == 42.5
    await transcript.close()


@pytest.mark.asyncio
async def test_later_summary_keeps_frozen_checkpoint_and_summarizes_delta(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=120,
        compaction_enabled=True,
        compaction_keep_recent_tokens=1,
    )
    scope = _scope()
    for turn_id, content in (
        ("turn-1", "first old progress " * 300),
        ("turn-2", "second old progress " * 300),
        ("turn-3", "recent answer"),
    ):
        await transcript.append_turn(
            scope,
            turn_id=turn_id,
            user_events=(),
            protocol_events=_protocol(turn_id, content),
        )
        scope = await transcript.current_scope(scope)

    calls = []

    async def summarize(messages, max_tokens):
        calls.append(messages)
        return "进展：新的增量\n文件：无\n待办：继续\n上下文：新的结论"

    first = await transcript.compact_if_needed(scope, summary_factory=summarize)
    assert first.snapshot.events[-1].operation == "compact_replace"
    scope = first.scope

    await transcript.append_turn(
        scope,
        turn_id="turn-4",
        user_events=(),
        protocol_events=_protocol("turn-4", "third old progress " * 300),
    )
    scope = await transcript.current_scope(scope)
    await transcript.append_turn(
        scope,
        turn_id="turn-5",
        user_events=(),
        protocol_events=_protocol("turn-5", "latest answer"),
    )
    scope = await transcript.current_scope(scope)

    second = await transcript.compact_if_needed(scope, summary_factory=summarize)
    compacted_events = [
        event
        for event in second.snapshot.events
        if event.operation == "compact_replace"
    ]

    assert len(calls) == 2
    assert calls[1][0]["content"] == first.snapshot.events[-1].content
    assert len(compacted_events) == 2
    assert "latest answer" in [event.content for event in second.snapshot.events]
    assert not any(
        "third old progress" in event.content for event in second.snapshot.events
    )
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


@pytest.mark.asyncio
async def test_provider_usage_is_persisted_and_used_for_waterline(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=100,
        compaction_enabled=True,
        compaction_keep_recent_tokens=4096,
    )
    scope = _scope()
    await transcript.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol("turn-1", "short"),
    )
    scope = await transcript.current_scope(scope)
    await transcript.record_provider_usage(
        scope,
        {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        },
        provider="deepseek:flash",
        model="deepseek-v4-flash",
        turn_id="turn-1",
    )

    usage = await transcript.latest_provider_usage(scope, provider="deepseek:flash")
    result = await transcript.compact_if_needed(scope)

    assert usage is not None
    assert usage.prompt_tokens == 100
    assert usage.usage_present is True
    assert usage.elapsed_ms == 0
    assert result.before_tokens == 100
    assert "provider usage" in result.reason
    await transcript.close()


@pytest.mark.asyncio
async def test_force_compaction_bypasses_normal_waterline(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=10_000,
        compaction_enabled=False,
        compaction_keep_recent_tokens=1,
    )
    scope = _scope()
    for turn_id, content in (("turn-1", "old context"), ("turn-2", "recent")):
        await transcript.append_turn(
            scope,
            turn_id=turn_id,
            user_events=(),
            protocol_events=_protocol(turn_id, content),
        )
        scope = await transcript.current_scope(scope)

    calls = []

    async def summarize(messages, max_tokens):
        calls.append(messages)
        return "进展：完成\n文件：无\n待办：继续\n上下文：保留"

    result = await transcript.compact_if_needed(
        scope, summary_factory=summarize, force=True
    )

    assert len(calls) == 1
    assert result.operation == "compact_replace"
    assert result.snapshot.events[-1].compacted is True
    await transcript.close()


@pytest.mark.asyncio
async def test_missing_provider_usage_is_explicit_fallback(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    observation = await transcript.record_provider_usage(
        scope, {"prompt_tokens": 12}, provider="openai:gpt", turn_id="turn-1"
    )

    status = await transcript.status()

    assert observation.usage_present is False
    assert status["usage_observation_count"] == 1
    assert status["usage_missing_count"] == 1
    await transcript.close()


@pytest.mark.asyncio
async def test_status_reports_cache_totals_and_close_scope_cleans_observations(
    tmp_path,
):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    await transcript.record_provider_usage(
        scope,
        {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 75,
            "prompt_cache_miss_tokens": 25,
        },
        elapsed_ms=12.5,
    )
    await transcript.record_incident(
        scope,
        "context_overflow",
        recovered=True,
        elapsed_ms=3.0,
    )

    status = await transcript.status()
    assert status["prompt_tokens"] == 100
    assert status["cache_hit_tokens"] == 75
    assert status["cache_miss_tokens"] == 25
    assert status["cache_hit_rate"] == 75.0
    assert status["usage_elapsed_ms"] == 12.5
    assert status["overflow_recovery_count"] == 1

    await transcript.close_scope(scope)
    status = await transcript.status()
    assert status["usage_observation_count"] == 0
    assert status["overflow_count"] == 0
    await transcript.close()


@pytest.mark.asyncio
async def test_started_compaction_is_abandoned_after_restart(tmp_path):
    path = str(tmp_path / "model-context.sqlite3")
    transcript = ModelContextTranscript(path)
    operation_id = await transcript.begin_compaction(
        _scope(), operation="compact_replace", reason="test"
    )
    await transcript.close()

    restarted = ModelContextTranscript(path)
    status = await restarted.status()

    assert operation_id
    assert status["compaction_abandoned_count"] == 1
    assert status["abandoned_compaction_count"] == 1
    await restarted.close()


@pytest.mark.asyncio
async def test_v1_schema_migrates_operation_column(tmp_path):
    path = str(tmp_path / "model-context.sqlite3")
    initial = ModelContextTranscript(path)
    scope = _scope()
    await initial.append_turn(
        scope,
        turn_id="turn-1",
        user_events=(),
        protocol_events=_protocol(),
    )
    await initial.close()

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE model_context_events DROP COLUMN operation")
    conn.execute("UPDATE model_context_schema SET version = 1")
    conn.commit()
    conn.close()

    migrated = ModelContextTranscript(path)
    snapshot = await migrated.snapshot(scope)

    assert snapshot.events[0].operation == "append"
    assert (await migrated.status())["schema_version"] == 7
    await migrated.close()


@pytest.mark.asyncio
async def test_compaction_does_not_overwrite_append_during_summary(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=120,
        compaction_enabled=True,
        compaction_keep_recent_tokens=1,
    )
    scope = _scope()
    for turn_id, content in (
        ("turn-1", "old context " * 300),
        ("turn-2", "recent context " * 300),
    ):
        await transcript.append_turn(
            scope,
            turn_id=turn_id,
            user_events=(),
            protocol_events=_protocol(turn_id, content),
        )
        scope = await transcript.current_scope(scope)

    summary_started = asyncio.Event()
    release_summary = asyncio.Event()

    async def summarize(messages, max_tokens):
        summary_started.set()
        await release_summary.wait()
        return "进展：旧上下文\n文件：无\n待办：继续\n上下文：旧结论"

    compaction_task = asyncio.create_task(
        transcript.compact_if_needed(scope, summary_factory=summarize)
    )
    await summary_started.wait()
    await transcript.append_turn(
        scope,
        turn_id="turn-3",
        user_events=(),
        protocol_events=_protocol("turn-3", "new append"),
    )
    release_summary.set()
    result = await compaction_task

    contents = [event.content for event in result.snapshot.events]
    assert "new append" in contents
    assert not any(content.startswith("进展：旧上下文") for content in contents)
    assert (await transcript.status())["compaction_failed_count"] == 1
    await transcript.close()


@pytest.mark.asyncio
async def test_scope_allows_only_one_active_compaction(tmp_path):
    transcript = ModelContextTranscript(str(tmp_path / "model-context.sqlite3"))
    scope = _scope()
    first = await transcript.begin_compaction(
        scope, operation="compact_replace", reason="first"
    )

    with pytest.raises(ModelContextCompactionInProgressError):
        await transcript.begin_compaction(
            scope, operation="compact_replace", reason="second"
        )

    await transcript.fail_compaction(first, "test release")
    second = await transcript.begin_compaction(
        scope, operation="compact_replace", reason="retry"
    )
    assert second != first
    await transcript.close()


@pytest.mark.asyncio
async def test_compaction_conflict_returns_fresh_snapshot(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"),
        max_tokens=1,
        compaction_enabled=True,
        compaction_keep_recent_tokens=1,
    )
    scope = _scope()
    for turn_id, content in (("turn-1", "old"), ("turn-2", "recent")):
        await transcript.append_turn(
            scope,
            turn_id=turn_id,
            user_events=(),
            protocol_events=_protocol(turn_id, content),
        )
        scope = await transcript.current_scope(scope)

    operation_id = await transcript.begin_compaction(
        scope, operation="compact_replace", reason="test lock"
    )
    result = await transcript.compact_if_needed(
        scope,
        summary_factory=lambda messages, max_tokens: asyncio.sleep(0),
        force=True,
    )

    assert result.changed is False
    assert "already in progress" in result.reason
    assert [event.content for event in result.snapshot.events] == ["old", "recent"]
    await transcript.fail_compaction(operation_id, "test release")
    await transcript.close()


@pytest.mark.asyncio
async def test_event_bound_pruning_records_provenance(tmp_path):
    transcript = ModelContextTranscript(
        str(tmp_path / "model-context.sqlite3"), max_events=2
    )
    scope = _scope()
    for turn_id in ("turn-1", "turn-2", "turn-3"):
        await transcript.append_turn(
            scope,
            turn_id=turn_id,
            user_events=(),
            protocol_events=_protocol(turn_id, turn_id),
        )
        scope = await transcript.current_scope(scope)

    status = await transcript.status()
    assert status["compaction_event_prune_count"] == 1
    assert status["compaction_committed_count"] == 1
    await transcript.close()


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
