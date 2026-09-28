import pytest

from core.engine.token_usage_log import (
    SOURCE_AUX_COMPACTION,
    SOURCE_TURN_AMBIENT,
    SOURCE_TURN_REPLY,
    TokenUsageLog,
)
from core.managers.cost_tracker import CostTracker


def test_record_turn_persists_usage_with_source(tmp_path):
    log = TokenUsageLog(str(tmp_path / "usage.sqlite3"))
    tracker = CostTracker(usage_log=log)

    tracker.record_turn(
        "chat",
        "deepseek-v4-flash",
        {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
        },
        metadata={"turn_kind": "ai", "turn_id": "t1", "provider": "deepseek"},
    )

    rows = log.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == SOURCE_TURN_REPLY
    assert row["turn_id"] == "t1"
    assert row["prompt_tokens"] == 1000
    assert row["completion_tokens"] == 200
    assert row["cache_usage_present"] == 1
    assert row["cost"] > 0
    log.close()


def test_record_turn_without_usage_log_does_not_persist():
    tracker = CostTracker()
    tracker.record_turn("chat", "m", {"prompt_tokens": 1}, metadata={})
    assert tracker.get_global_stats().prompt_tokens == 1


def test_source_mapping_for_ambient_and_compaction(tmp_path):
    log = TokenUsageLog(str(tmp_path / "usage.sqlite3"))
    tracker = CostTracker(usage_log=log)

    tracker.record_turn(
        "c", "m", {"prompt_tokens": 5}, metadata={"turn_kind": "ambient"}
    )
    tracker.record_turn(
        "c",
        "m",
        {"prompt_tokens": 5},
        metadata={"usage_kind": "model_context_compaction"},
    )

    assert {row["source"] for row in log.recent()} == {
        SOURCE_TURN_AMBIENT,
        SOURCE_AUX_COMPACTION,
    }
    log.close()


def test_cost_bills_prompt_when_cache_split_absent():
    tracker = CostTracker(
        pricing={"m": {"input_per_million": 10.0, "output_per_million": 20.0}}
    )

    tracker.record_turn("c", "m", {"prompt_tokens": 1_000_000, "completion_tokens": 0})

    assert tracker.get_global_stats().cost == pytest.approx(10.0)


def test_usage_log_failure_does_not_break_record_turn():
    class Boom:
        def record(self, **kwargs):
            raise RuntimeError("disk full")

    tracker = CostTracker(usage_log=Boom())
    tracker.record_turn("c", "m", {"prompt_tokens": 7})

    assert tracker.get_global_stats().prompt_tokens == 7
