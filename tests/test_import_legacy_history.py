import importlib.util
import json
from pathlib import Path

import pytest

from core.engine.conversation_event_log import ConversationEventLog

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "import_legacy_history.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("import_legacy_history", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_sources_groups_active_and_archived(tmp_path):
    module = _load_module()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "chat1.jsonl").write_text(
        json.dumps({"role": "user", "content": "hi", "message_id": "m1"}) + "\n",
        encoding="utf-8",
    )
    (sessions / "chat1.jsonl.archived.2025-01-01.batch1").write_text(
        json.dumps({"role": "assistant", "content": "old"}) + "\n",
        encoding="utf-8",
    )
    (sessions / "chat2.jsonl").write_text("not-json\n", encoding="utf-8")

    sources = module._collect_sources(sessions)

    assert set(sources) == {"chat1"}
    assert [sid for sid, _ in sources["chat1"]] == ["legacy-active", "legacy-archives"]


@pytest.mark.asyncio
async def test_import_legacy_history_is_idempotent(tmp_path):
    module = _load_module()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "chat1.jsonl").write_text(
        json.dumps({"role": "user", "content": "hi", "message_id": "m1"}) + "\n",
        encoding="utf-8",
    )
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    assert await module.import_legacy_history(sessions, event_log) == 0
    history = await event_log.history("chat1")
    assert any(item.get("role") == "user" for item in history)
    assert await event_log.legacy_migration_is_complete() is True

    # watermark present → second run is a no-op
    assert await module.import_legacy_history(sessions, event_log) == 0
    await event_log.close()


@pytest.mark.asyncio
async def test_import_legacy_history_empty_dir(tmp_path):
    module = _load_module()
    event_log = ConversationEventLog(str(tmp_path / "events.sqlite3"))

    assert await module.import_legacy_history(tmp_path / "missing", event_log) == 0
    await event_log.close()
