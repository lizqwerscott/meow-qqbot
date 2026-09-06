import pytest

from core.engine.prompt_context_report import PromptContextReportStore


@pytest.mark.asyncio
async def test_prompt_context_report_records_attempt_diagnostics(tmp_path):
    store = PromptContextReportStore(str(tmp_path / "reports.sqlite3"))

    await store.record(
        chat_id="chat",
        turn_id="turn-1",
        source="model_context",
        generation=3,
        attempt_id="turn-1:2",
        projection_version=4,
        prompt_hash="hash",
        summary_dates=("2025-01-01",),
        summary_count=2,
        degraded_reason="budget",
        truncated_event_ids=("event-1",),
    )

    reports = await store.list_for_webui("chat")

    assert reports[0].attempt_id == "turn-1:2"
    assert reports[0].projection_version == 4
    assert reports[0].prompt_hash == "hash"
    assert reports[0].summary_dates == ("2025-01-01",)
    assert reports[0].summary_count == 2
    assert reports[0].degraded_reason == "budget"
    assert reports[0].truncated_event_ids == ("event-1",)
    await store.close()


@pytest.mark.asyncio
async def test_prompt_context_report_webui_listing_supports_count_and_offset(tmp_path):
    store = PromptContextReportStore(str(tmp_path / "reports.sqlite3"))
    for index in range(3):
        await store.record(
            chat_id="chat",
            turn_id=f"turn-{index}",
            source="timeline",
            generation=index,
        )

    reports = await store.list_for_webui("chat", limit=1, offset=1)

    assert await store.count_for_webui("chat") == 3
    assert len(reports) == 1
    assert reports[0].turn_id == "turn-1"
    await store.close()


@pytest.mark.asyncio
async def test_prompt_context_report_status_aggregates_fallbacks(tmp_path):
    store = PromptContextReportStore(str(tmp_path / "reports.sqlite3"))
    await store.record(
        chat_id="chat",
        turn_id="turn-1",
        source="prompt_projection",
        estimated_tokens=12,
    )
    await store.record(
        chat_id="chat",
        turn_id="turn-2",
        source="bounded_fallback",
        estimated_tokens=8,
        fallback_reason="projection unavailable",
        degraded_reason="bounded",
    )

    assert await store.status() == {
        "report_count": 2,
        "fallback_count": 1,
        "degraded_count": 1,
        "estimated_tokens": 20,
    }
    await store.close()
