import asyncio
import sqlite3
from pathlib import Path

from core.media.store import MediaStore


def test_media_store_deduplicates_and_isolates(tmp_path):
    async def run():
        store = MediaStore(tmp_path)
        await store.open()
        first = await store.save(
            chat_id="group-a",
            message_id="m1",
            sender_id="u1",
            resource_type="image",
            source_url="https://example.test/a.jpg",
            mime_type="image/jpeg",
            filename="a.jpg",
            data=b"same",
        )
        second = await store.save(
            chat_id="group-b",
            message_id="m2",
            sender_id="u2",
            resource_type="image",
            source_url="https://example.test/b.jpg",
            mime_type="image/jpeg",
            filename="b.jpg",
            data=b"same",
        )
        assert first.media_id == second.media_id
        assert await store.authorize("group-a", first.media_uri, image_only=True)
        assert await store.authorize("group-b", first.media_uri, image_only=True)
        assert (
            await store.authorize("group-c", first.media_uri, image_only=True) is None
        )
        await store.close()

    asyncio.run(run())


def test_media_store_detects_qq_silk_voice(tmp_path):
    async def run():
        store = MediaStore(tmp_path)
        await store.open()
        record = await store.save(
            chat_id="group-a",
            message_id="m1",
            sender_id="u1",
            resource_type="voice",
            source_url="https://example.test/voice.amr",
            mime_type="audio/mp3",
            filename="voice.amr",
            data=b"\x02#!SILK_V3\x11\x00payload",
        )

        assert record.mime_type == "audio/silk"
        assert record.local_path.suffix == ".silk"
        row = store._conn.execute(
            "SELECT mime_type, local_path FROM media_objects WHERE media_id=?",
            (record.media_id,),
        ).fetchone()
        assert row["mime_type"] == "audio/silk"
        assert row["local_path"].endswith(".silk")
        await store.close()

    asyncio.run(run())


def test_media_store_migrates_existing_qq_silk_voice(tmp_path):
    async def run():
        store = MediaStore(tmp_path)
        await store.open()
        record = await store.save(
            chat_id="group-a",
            message_id="m1",
            sender_id="u1",
            resource_type="voice",
            source_url="https://example.test/voice.bin",
            mime_type="application/octet-stream",
            filename="voice.bin",
            data=b"placeholder",
        )
        old_path = record.local_path.with_suffix(".amr")
        record.local_path.replace(old_path)
        old_path.write_bytes(b"\x02#!SILK_V3\x11\x00payload")
        relative_path = (
            Path("data/media/inbound") / old_path.parent.name / old_path.name
        )
        store._conn.execute(
            "UPDATE media_objects SET local_path=?, mime_type=? WHERE media_id=?",
            (str(relative_path), "audio/mp3", record.media_id),
        )
        store._conn.commit()
        await store.close()

        migrated_store = MediaStore(tmp_path)
        await migrated_store.open()
        row = migrated_store._conn.execute(
            "SELECT mime_type, local_path FROM media_objects WHERE media_id=?",
            (record.media_id,),
        ).fetchone()
        assert row["mime_type"] == "audio/silk"
        assert row["local_path"].endswith(".silk")
        assert not old_path.exists()
        assert (tmp_path / row["local_path"]).is_file()
        await migrated_store.close()

    asyncio.run(run())


def test_media_store_migrates_legacy_silk_voice_with_empty_resource_type(tmp_path):
    async def run():
        store = MediaStore(tmp_path)
        await store.open()
        record = await store.save(
            chat_id="group-a",
            message_id="m1",
            sender_id="u1",
            resource_type="voice",
            source_url="https://example.test/voice.bin",
            mime_type="application/octet-stream",
            filename="voice.bin",
            data=b"placeholder",
        )
        old_path = record.local_path.with_suffix(".amr")
        record.local_path.replace(old_path)
        old_path.write_bytes(b"\x02#!SILK_V3\x11\x00payload")
        relative_path = (
            Path("data/media/inbound") / old_path.parent.name / old_path.name
        )
        store._conn.execute(
            "UPDATE media_objects SET local_path=?, mime_type=? WHERE media_id=?",
            (str(relative_path), "audio/mp3", record.media_id),
        )
        store._conn.execute(
            "UPDATE media_messages SET resource_type='' WHERE media_id=?",
            (record.media_id,),
        )
        store._conn.commit()
        await store.close()

        migrated_store = MediaStore(tmp_path)
        await migrated_store.open()
        row = migrated_store._conn.execute(
            "SELECT mime_type, local_path FROM media_objects WHERE media_id=?",
            (record.media_id,),
        ).fetchone()
        assert row["mime_type"] == "audio/silk"
        assert row["local_path"].endswith(".silk")
        assert (tmp_path / row["local_path"]).is_file()
        await migrated_store.close()

    asyncio.run(run())


def test_media_store_cleanup_removes_source_sidecar(tmp_path):
    async def run():
        store = MediaStore(tmp_path)
        await store.open()
        record = await store.save(
            chat_id="group-a",
            message_id="m1",
            sender_id="u1",
            resource_type="voice",
            source_url="https://example.test/voice.amr",
            mime_type="audio/mp3",
            filename="voice.amr",
            data=b"\x02#!SILK_V3\x11\x00payload",
            original_data=b"raw-silk-bytes",
        )
        source_path = store._source_path(record.local_path)
        assert source_path.is_file()

        removed = await store.clear_all()

        assert removed == 1
        assert not record.local_path.exists()
        assert not source_path.exists()
        await store.close()

    asyncio.run(run())


def test_media_store_rejects_invalid_uri(tmp_path):
    async def run():
        store = MediaStore(tmp_path)
        await store.open()
        assert (
            await store.authorize("group-a", "file:///tmp/a", image_only=True) is None
        )
        assert (
            await store.authorize("group-a", "media://inbound/../x", image_only=True)
            is None
        )
        await store.close()

    asyncio.run(run())


def test_media_store_migrates_summary_version(tmp_path):
    async def run():
        index_path = tmp_path / "index.sqlite3"
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                "CREATE TABLE media_objects ("
                "media_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, "
                "local_path TEXT NOT NULL, mime_type TEXT NOT NULL, "
                "size INTEGER NOT NULL, created_at REAL NOT NULL, "
                "expires_at REAL NOT NULL DEFAULT 0, filename TEXT NOT NULL DEFAULT '', "
                "summary TEXT NOT NULL DEFAULT '', summary_model TEXT NOT NULL DEFAULT ''"
                ")"
            )

        store = MediaStore(tmp_path)
        await store.open()
        columns = {
            row["name"]
            for row in store._conn.execute(
                "PRAGMA table_info(media_objects)"
            ).fetchall()
        }

        assert "summary_version" in columns
        await store.close()

    asyncio.run(run())


def test_media_store_migrates_file_summary_columns(tmp_path):
    async def run():
        index_path = tmp_path / "index.sqlite3"
        with sqlite3.connect(index_path) as connection:
            connection.execute(
                "CREATE TABLE media_objects ("
                "media_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE, "
                "local_path TEXT NOT NULL, mime_type TEXT NOT NULL, "
                "size INTEGER NOT NULL, created_at REAL NOT NULL, "
                "expires_at REAL NOT NULL DEFAULT 0, filename TEXT NOT NULL DEFAULT '', "
                "summary TEXT NOT NULL DEFAULT '', summary_model TEXT NOT NULL DEFAULT '', "
                "summary_version TEXT NOT NULL DEFAULT ''"
                ")"
            )

        store = MediaStore(tmp_path)
        await store.open()
        columns = {
            row["name"]
            for row in store._conn.execute(
                "PRAGMA table_info(media_objects)"
            ).fetchall()
        }

        assert {
            "file_summary",
            "file_summary_model",
            "file_summary_version",
        } <= columns
        await store.close()

    asyncio.run(run())
