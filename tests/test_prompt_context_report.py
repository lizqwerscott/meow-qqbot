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
