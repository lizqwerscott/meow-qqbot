import json
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


def test_canonical_startup_requires_verified_hindsight_tags(tmp_path):
    (tmp_path / "business.sqlite3").touch()
    migration_dir = tmp_path / "migrations" / "run-1"
    migration_dir.mkdir(parents=True)
    (migration_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "canonical_cutover",
                "mappings": [{"canonical_key": "agent:main:chat:qq:default:direct:1"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Hindsight"):
        ServiceGraph._assert_canonical_cutover_ready(tmp_path)


def test_canonical_startup_accepts_matching_verified_hindsight_tags(tmp_path):
    (tmp_path / "business.sqlite3").touch()
    migration_dir = tmp_path / "migrations" / "run-1"
    migration_dir.mkdir(parents=True)
    (migration_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "canonical_cutover",
                "mappings": [{"canonical_key": "agent:main:chat:qq:default:direct:1"}],
            }
        ),
        encoding="utf-8",
    )
    (migration_dir / "hindsight-tag-plan.json").write_text(
        json.dumps(
            {
                "source_run_id": "run-1",
                "bank_id": "qq_bot",
                "state": "verified",
            }
        ),
        encoding="utf-8",
    )

    ServiceGraph._assert_canonical_cutover_ready(tmp_path)


def test_canonical_startup_rejects_newer_unverified_migration(tmp_path):
    (tmp_path / "business.sqlite3").touch()
    old_dir = tmp_path / "migrations" / "run-1"
    old_dir.mkdir(parents=True)
    (old_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-1", "state": "verified", "created_at": 1}),
        encoding="utf-8",
    )
    (old_dir / "hindsight-tag-plan.json").write_text(
        json.dumps(
            {"source_run_id": "run-1", "bank_id": "qq_bot", "state": "verified"}
        ),
        encoding="utf-8",
    )
    new_dir = tmp_path / "migrations" / "run-2"
    new_dir.mkdir(parents=True)
    (new_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-2", "state": "applying", "created_at": 2}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="latest cold migration"):
        ServiceGraph._assert_canonical_cutover_ready(tmp_path)
