import json

from core.managers.context_store import (
    JSONLContextStore,
    SQLiteContextStore,
    deduplicate_history,
)


def test_jsonl_repair_duplicates_uses_message_identity(tmp_path):
    store = JSONLContextStore(str(tmp_path / "sessions"))
    path = tmp_path / "sessions" / "chat.jsonl"
    path.parent.mkdir()
    duplicate = {
        "role": "user",
        "content": "hello",
        "message_id": "m1",
    }
    path.write_text(
        "\n".join(
            [
                json.dumps(duplicate),
                json.dumps(duplicate),
                json.dumps({"role": "user", "content": "hello"}),
                json.dumps({"role": "user", "content": "hello", "timestamp": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert store.repair_duplicates("chat") == 1
    assert len(store.load("chat")) == 3


def test_sqlite_context_store_migrates_deduplicated_legacy_history(tmp_path):
    archive_dir = tmp_path / "sessions"
    legacy = JSONLContextStore(str(archive_dir))
    legacy.flush(
        "chat",
        [
            {"role": "user", "content": "hello", "message_id": "m1"},
            {"role": "user", "content": "hello", "message_id": "m1"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    store = SQLiteContextStore(str(tmp_path / "context.sqlite3"), str(archive_dir))
    assert store.load("chat") == [
        {"role": "user", "content": "hello", "message_id": "m1"},
        {"role": "assistant", "content": "answer"},
    ]

    store.flush(
        "chat",
        [
            {"role": "user", "content": "hello", "message_id": "m1"},
            {"role": "user", "content": "hello", "timestamp": 1},
            {"role": "user", "content": "hello", "timestamp": 2},
        ],
    )
    assert len(store.load("chat")) == 3
    store.close()


def test_deduplication_preserves_records_without_stable_identity():
    messages = [
        {"role": "assistant", "content": "好的", "timestamp": 1},
        {"role": "assistant", "content": "好的", "timestamp": 1},
        {"role": "tool", "content": "same", "tool_name": "read_file"},
        {"role": "tool", "content": "same", "tool_name": "read_file"},
    ]

    assert deduplicate_history(messages) == messages


def test_sqlite_context_store_preserves_identical_assistant_records(tmp_path):
    store = SQLiteContextStore(
        str(tmp_path / "context.sqlite3"), str(tmp_path / "sessions")
    )
    messages = [
        {"role": "assistant", "content": "好的", "timestamp": 1},
        {"role": "assistant", "content": "好的", "timestamp": 1},
    ]

    store.flush("chat", messages)

    assert store.load("chat") == messages
    store.close()
