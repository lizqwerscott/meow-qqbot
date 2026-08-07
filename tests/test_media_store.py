import asyncio

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
