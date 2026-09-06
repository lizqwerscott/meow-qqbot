from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.bootstrap import ServiceGraph


@pytest.mark.asyncio
async def test_bootstrap_skips_legacy_scan_after_watermark():
    event_log = SimpleNamespace(
        legacy_migration_is_complete=AsyncMock(return_value=True),
    )
    get_ids = AsyncMock()
    graph = ServiceGraph.__new__(ServiceGraph)
    graph.agent_engine = SimpleNamespace(event_log=event_log)
    graph.context_manager = SimpleNamespace(get_legacy_chat_ids_async=get_ids)

    await graph._migrate_legacy_history()

    get_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_marks_legacy_scan_complete_only_without_failures():
    event_log = SimpleNamespace(
        legacy_migration_is_complete=AsyncMock(return_value=False),
        mark_legacy_migration_complete=AsyncMock(),
    )
    get_ids = AsyncMock(return_value=[])
    graph = ServiceGraph.__new__(ServiceGraph)
    graph.agent_engine = SimpleNamespace(
        event_log=event_log,
        migrate_legacy_history_async=AsyncMock(),
    )
    graph.context_manager = SimpleNamespace(get_legacy_chat_ids_async=get_ids)
    graph.archive_manager = None

    await graph._migrate_legacy_history()

    event_log.mark_legacy_migration_complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_bootstrap_keeps_watermark_pending_when_legacy_scan_fails():
    event_log = SimpleNamespace(
        legacy_migration_is_complete=AsyncMock(return_value=False),
        mark_legacy_migration_complete=AsyncMock(),
    )
    get_ids = AsyncMock(side_effect=OSError("legacy store unavailable"))
    graph = ServiceGraph.__new__(ServiceGraph)
    graph.agent_engine = SimpleNamespace(event_log=event_log)
    graph.context_manager = SimpleNamespace(get_legacy_chat_ids_async=get_ids)

    await graph._migrate_legacy_history()

    event_log.mark_legacy_migration_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_keeps_watermark_pending_on_degraded_archive_import():
    event_log = SimpleNamespace(
        legacy_migration_is_complete=AsyncMock(return_value=False),
        legacy_chat_migration_is_complete=AsyncMock(return_value=False),
        mark_legacy_chat_migration_complete=AsyncMock(),
        mark_legacy_migration_complete=AsyncMock(),
    )
    graph = ServiceGraph.__new__(ServiceGraph)
    graph.agent_engine = SimpleNamespace(
        event_log=event_log,
        migrate_legacy_history_async=AsyncMock(),
    )
    graph.context_manager = SimpleNamespace(
        get_legacy_chat_ids_async=AsyncMock(return_value=["chat"])
    )
    graph.archive_manager = SimpleNamespace(
        import_legacy_archives_async=AsyncMock(
            return_value={"status": "degraded", "error_count": 1}
        )
    )

    await graph._migrate_legacy_history()

    event_log.mark_legacy_chat_migration_complete.assert_not_awaited()
    event_log.mark_legacy_migration_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_keeps_chat_checkpoint_pending_on_legacy_identity_conflict():
    event_log = SimpleNamespace(
        legacy_migration_is_complete=AsyncMock(return_value=False),
        legacy_chat_migration_is_complete=AsyncMock(return_value=False),
        legacy_conflict_event_ids=AsyncMock(return_value=("conflict-event",)),
        mark_legacy_chat_migration_complete=AsyncMock(),
        mark_legacy_migration_complete=AsyncMock(),
    )
    graph = ServiceGraph.__new__(ServiceGraph)
    graph.agent_engine = SimpleNamespace(
        event_log=event_log,
        migrate_legacy_history_async=AsyncMock(),
    )
    graph.context_manager = SimpleNamespace(
        get_legacy_chat_ids_async=AsyncMock(return_value=["chat"])
    )
    graph.archive_manager = None

    await graph._migrate_legacy_history()

    event_log.mark_legacy_chat_migration_complete.assert_not_awaited()
    event_log.mark_legacy_migration_complete.assert_not_awaited()


@pytest.mark.asyncio
async def test_bootstrap_resumes_only_uncheckpointed_legacy_chats():
    event_log = SimpleNamespace(
        legacy_migration_is_complete=AsyncMock(return_value=False),
        legacy_chat_migration_is_complete=AsyncMock(
            side_effect=lambda chat_id: chat_id == "done"
        ),
        mark_legacy_chat_migration_complete=AsyncMock(),
        mark_legacy_migration_complete=AsyncMock(),
    )
    graph = ServiceGraph.__new__(ServiceGraph)
    graph.agent_engine = SimpleNamespace(
        event_log=event_log,
        migrate_legacy_history_async=AsyncMock(),
    )
    graph.context_manager = SimpleNamespace(
        get_legacy_chat_ids_async=AsyncMock(return_value=["done", "pending"])
    )
    graph.archive_manager = None

    await graph._migrate_legacy_history()

    graph.agent_engine.migrate_legacy_history_async.assert_awaited_once_with("pending")
    event_log.mark_legacy_chat_migration_complete.assert_awaited_once_with("pending")
    event_log.mark_legacy_migration_complete.assert_awaited_once()
