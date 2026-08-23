import asyncio

import pytest

from core.engine.batch_media_context import BatchMediaContextBuilder, BatchMediaLimits
from core.managers.session_manager import AdmissionOrigin, InboundIntent, PendingInbound
from core.media.models import MediaTurnContext
from core.message import InputMessage, ResourceMeta


def _pending(message: InputMessage) -> PendingInbound:
    return PendingInbound(
        message,
        message.content,
        InboundIntent.PRIVATE_CONVERSATION,
        AdmissionOrigin.USER_MESSAGE,
    )


class FakeMediaService:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def prepare_for_ai(self, message: InputMessage) -> MediaTurnContext:
        self.calls.append(message)
        self.started.set()
        await self.release.wait()
        return MediaTurnContext(current_blocks=("[当前图片]",))


@pytest.mark.asyncio
async def test_batch_media_context_coalesces_inflight_build_and_resources():
    service = FakeMediaService()
    builder = BatchMediaContextBuilder(service)
    first = _pending(
        InputMessage(
            "first",
            "user",
            "chat",
            "first",
            False,
            resources=[
                ResourceMeta(resource_type="image", media_uri="media://inbound/a")
            ],
        )
    )
    second = _pending(
        InputMessage(
            "second",
            "user",
            "chat",
            "second",
            False,
            resources=[
                ResourceMeta(resource_type="image", media_uri="media://inbound/a"),
                ResourceMeta(resource_type="voice", media_uri="media://inbound/b"),
            ],
            replied_resources=[
                ResourceMeta(resource_type="file", media_uri="media://inbound/c")
            ],
        )
    )

    first_build = asyncio.create_task(
        builder.build(turn_id="turn", items=(first, second))
    )
    await service.started.wait()
    second_build = asyncio.create_task(
        builder.build(turn_id="turn", items=(first, second))
    )
    await asyncio.sleep(0)
    service.release.set()
    first_context, second_context = await asyncio.gather(first_build, second_build)

    assert len(service.calls) == 1
    prepared = service.calls[0]
    assert [resource.media_uri for resource in prepared.resources] == [
        "media://inbound/a",
        "media://inbound/b",
    ]
    assert [resource.media_uri for resource in prepared.replied_resources] == [
        "media://inbound/c"
    ]
    assert first_context.context == second_context.context
    assert first_context.receipt.cache_state == "miss"
    assert second_context.receipt.cache_state == "inflight"
    assert first_context.resource_uris == (
        "media://inbound/a",
        "media://inbound/b",
        "media://inbound/c",
    )
    assert first_context.as_text() == "[当前图片]"


@pytest.mark.asyncio
async def test_batch_media_context_without_service_is_empty():
    context = await BatchMediaContextBuilder(None).build(
        turn_id="turn",
        items=(_pending(InputMessage("message", "user", "chat", "hello", False)),),
    )

    assert context.resource_uris == ()
    assert context.as_text() == ""


@pytest.mark.asyncio
async def test_batch_media_context_applies_resource_and_output_budgets():
    service = FakeMediaService()
    service.release.set()
    builder = BatchMediaContextBuilder(
        service,
        limits=BatchMediaLimits(
            max_resources=1,
            max_chars=4,
            max_download_bytes=5,
            capability_timeout_seconds=1,
        ),
    )
    first = _pending(
        InputMessage(
            "first",
            "user",
            "chat",
            "hello",
            False,
            resources=[
                ResourceMeta(
                    resource_type="image", media_uri="media://inbound/a", size=4
                ),
                ResourceMeta(
                    resource_type="image", media_uri="media://inbound/b", size=4
                ),
            ],
        )
    )

    context = await builder.build(turn_id="turn", items=(first,))

    assert len(service.calls) == 1
    assert [resource.media_uri for resource in service.calls[0].resources] == [
        "media://inbound/a"
    ]
    assert context.resource_uris == ("media://inbound/a",)
    assert context.as_text() == "[当前图"
    assert context.receipt.status == "truncated"
    assert context.receipt.resource_count == 1
    assert context.receipt.skipped_resource_count == 1
    assert context.receipt.download_bytes == 4
    assert context.receipt.output_chars == 4


@pytest.mark.asyncio
async def test_batch_media_context_reports_timeout_and_reuses_completed_task():
    service = FakeMediaService()
    builder = BatchMediaContextBuilder(
        service,
        limits=BatchMediaLimits(capability_timeout_seconds=0.01),
    )
    pending = _pending(
        InputMessage(
            "message",
            "user",
            "chat",
            "hello",
            False,
            resources=[
                ResourceMeta(resource_type="image", media_uri="media://inbound/a")
            ],
        )
    )

    timed_out = await builder.build(turn_id="turn", items=(pending,))
    reused = await builder.build(turn_id="turn", items=(pending,))

    assert timed_out.receipt.status == "timeout"
    assert timed_out.receipt.error_code == "CAPABILITY_TIMEOUT"
    assert reused.receipt.cache_state == "hit"
    assert len(service.calls) == 1
