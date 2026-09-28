import sqlite3
import time

import pytest

from core.engine.token_usage_log import (
    SOURCE_TURN_AMBIENT,
    SOURCE_TURN_REPLY,
    TokenUsageLog,
)


def test_record_and_summary():
    log = TokenUsageLog(":memory:")
    log.record(
        chat_id="c",
        turn_id="t",
        model="deepseek-v4-flash",
        provider="deepseek",
        source=SOURCE_TURN_REPLY,
        prompt_tokens=100,
        completion_tokens=20,
        cache_hit_tokens=60,
        cache_miss_tokens=40,
        usage_present=True,
        cache_usage_present=True,
        cost=0.001,
    )
    log.record(
        chat_id="c2",
        model="gpt-4o",
        source=SOURCE_TURN_AMBIENT,
        prompt_tokens=10,
        completion_tokens=5,
        usage_present=True,
        cost=0.0001,
    )

    summary = log.summary()

    assert summary["record_count"] == 2
    assert summary["prompt_tokens"] == 110
    assert summary["completion_tokens"] == 25
    assert summary["cache_hit_tokens"] == 60
    assert summary["cache_miss_tokens"] == 40
    assert summary["cost"] == pytest.approx(0.0011)
    assert summary["usage_present_count"] == 2
    assert summary["cache_usage_present_count"] == 1
    log.close()


def test_breakdown_dimensions_and_filters():
    log = TokenUsageLog(":memory:")
    now = time.time()
    log.record(
        chat_id="c",
        model="m1",
        source=SOURCE_TURN_REPLY,
        prompt_tokens=10,
        recorded_at=now,
    )
    log.record(
        chat_id="c2",
        model="m2",
        source=SOURCE_TURN_AMBIENT,
        prompt_tokens=20,
        recorded_at=now,
    )

    by_model = {row["key"]: row for row in log.breakdown("model")}
    assert set(by_model) == {"m1", "m2"}
    assert by_model["m2"]["prompt_tokens"] == 20
    assert {row["key"] for row in log.breakdown("source")} == {
        SOURCE_TURN_REPLY,
        SOURCE_TURN_AMBIENT,
    }
    assert log.breakdown("day")[0]["record_count"] == 2

    assert log.summary(since=now + 10)["record_count"] == 0
    assert log.count(until=now - 10) == 0
    assert log.count(model="m1") == 1
    assert log.count(source=SOURCE_TURN_AMBIENT) == 1
    log.close()


def test_recent_pagination_and_status():
    log = TokenUsageLog(":memory:")
    for index in range(5):
        log.record(model="m", prompt_tokens=index)

    assert len(log.recent(limit=2)) == 2
    assert len(log.recent(limit=2, offset=4)) == 1
    status = log.status()
    assert status["record_count"] == 5
    assert status["usage_present_count"] == 0
    log.close()


def test_unsupported_breakdown_dimension_raises():
    log = TokenUsageLog(":memory:")
    with pytest.raises(ValueError):
        log.breakdown("nope")
    log.close()


def test_chat_filter_scopes_summary_and_breakdown():
    log = TokenUsageLog(":memory:")
    log.record(chat_id="chat-A", model="m", source=SOURCE_TURN_REPLY, cost=0.5)
    log.record(chat_id="chat-B", model="m", source=SOURCE_TURN_REPLY, cost=0.25)

    scoped = log.summary(chat_id="chat-A")
    assert scoped["record_count"] == 1
    assert scoped["cost"] == pytest.approx(0.5)
    assert log.count(chat_id="chat-B") == 1
    assert [row["key"] for row in log.breakdown("chat")] == ["chat-A", "chat-B"]

    by_source = log.breakdown("source", chat_id="chat-B")
    assert len(by_source) == 1
    assert by_source[0]["key"] == SOURCE_TURN_REPLY
    assert by_source[0]["cost"] == pytest.approx(0.25)
    log.close()


def test_timeseries_is_chronological_and_bucketed():
    log = TokenUsageLog(":memory:")
    now = time.time()
    log.record(model="m", prompt_tokens=10, cost=0.1, recorded_at=now)
    log.record(model="m", prompt_tokens=20, cost=0.2, recorded_at=now + 86400)

    daily = log.timeseries("day")
    assert len(daily) == 2
    assert daily[0]["key"] < daily[1]["key"]  # ascending time order
    assert daily[0]["prompt_tokens"] == 10

    assert log.timeseries("hour")[0]["prompt_tokens"] == 10

    with pytest.raises(ValueError):
        log.timeseries("minute")
    log.close()


def test_invalid_schema_version_rejected(tmp_path):
    path = tmp_path / "usage.sqlite3"
    log = TokenUsageLog(str(path))
    log.status()  # open + create schema
    log.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE token_usage_log_schema SET version = 99")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError):
        TokenUsageLog(str(path)).status()
