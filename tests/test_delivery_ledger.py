import time

import pytest

from core.engine.delivery_ledger import (
    DeliveryController,
    DeliveryLedger,
    DeliveryReceipt,
)


@pytest.mark.asyncio
async def test_delivery_ledger_prepare_is_idempotent_and_settles_once(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    first = await ledger.prepare(
        key="ambient:chat:turn",
        chat_id="chat",
        turn_id="turn",
        reason="final_reply",
        reply_anchor_id="message-1",
        content_hash=ledger.content_hash("answer"),
    )
    duplicate = await ledger.prepare(
        key="ambient:chat:turn",
        chat_id="chat",
        turn_id="turn",
        reason="different_reason",
        reply_anchor_id="message-2",
        content_hash=ledger.content_hash("other"),
    )

    assert first == duplicate
    sent = await ledger.settle(first.key, status="sent", transport_id="qq-message-1")
    assert sent is not None
    assert sent.status == "sent"
    assert sent.transport_id == "qq-message-1"

    second_settle = await ledger.settle(first.key, status="failed")
    assert second_settle is not None
    assert second_settle.status == "sent"
    counts = await ledger.status_counts()
    assert counts == {"sent": 1}
    assert await ledger.stale_prepared(older_than=10**20) == []
    await ledger.close()


@pytest.mark.asyncio
async def test_delivery_ledger_exposes_stale_prepared_and_status_counts(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    await ledger.prepare(
        key="ambient:chat:stale",
        chat_id="chat",
        turn_id="stale",
        reason="final_reply",
        reply_anchor_id="m1",
        content_hash=ledger.content_hash("answer"),
    )
    conn = await ledger._ensure_open()
    conn.execute("UPDATE delivery_ledger SET updated_at = 0")
    conn.commit()
    stale = await ledger.stale_prepared(older_than=10**20)
    assert [record.turn_id for record in stale] == ["stale"]
    assert await ledger.status_counts() == {"prepared": 1}


@pytest.mark.asyncio
async def test_stale_prepared_can_be_limited_to_delivery_key_prefix(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    await ledger.prepare(
        key="ambient:chat:turn",
        chat_id="chat",
        turn_id="turn",
        reason="ambient",
        reply_anchor_id="anchor",
        content_hash="hash",
    )
    await ledger.prepare(
        key="external:chat:turn",
        chat_id="chat",
        turn_id="turn",
        reason="external",
        reply_anchor_id="anchor",
        content_hash="hash",
    )

    records = await ledger.stale_prepared(
        older_than=time.time() + 1,
        key_prefix="ambient:",
        now=time.time() + 31,
    )

    assert [record.key for record in records] == ["ambient:chat:turn"]
    await ledger.close()


@pytest.mark.asyncio
async def test_accepted_delivery_is_projected_to_timeline_only_once(tmp_path):
    from core.engine.conversation_timeline import ConversationTimeline

    timeline = ConversationTimeline(str(tmp_path / "timeline.sqlite3"))
    controller = DeliveryController(
        DeliveryLedger(str(tmp_path / "delivery.sqlite3")),
        timeline=timeline,
    )

    record = await controller.prepare_reply_delivery(
        chat_id="chat",
        turn_id="turn",
        sequence=1,
        content="visible answer",
        reply_anchor_id="question",
    )
    receipt = DeliveryReceipt(
        status="accepted",
        logical_delivery_id=record.logical_delivery_id,
        platform_message_id="qq-1",
    )
    await controller.settle_receipt(record, receipt, content="visible answer")
    await controller.settle_receipt(record, receipt, content="visible answer")

    events = await timeline.snapshot("chat")
    assert len(events) == 1
    assert events[0].content == "visible answer"
    assert events[0].event_kind == "delivery"
    assert events[0].delivery_kind == "response"
    await timeline.close()
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))

    silent, silent_record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="silent",
        content="NO_REPLY",
        delivery_mode="automatic",
    )
    assert silent.should_deliver is False
    assert silent_record is not None
    assert silent_record.status == "suppressed"

    tool, tool_record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="tool",
        content="answer",
        delivery_mode="automatic",
        tool_delivered=True,
    )
    assert tool.should_deliver is False
    assert tool_record is not None
    assert tool_record.status == "suppressed"


@pytest.mark.asyncio
async def test_delivery_controller_recovers_prepared_records(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    controller = DeliveryController(ledger)
    for turn_id, content_hash in (
        ("sent", ledger.content_hash("answer")),
        ("mismatch", ledger.content_hash("different")),
        ("retry", ledger.content_hash("answer")),
    ):
        await ledger.prepare(
            key=f"ambient:chat:{turn_id}",
            chat_id="chat",
            turn_id=turn_id,
            reason="final_reply",
            reply_anchor_id="",
            content_hash=content_hash,
        )

    async def resolve(record):
        return "answer"

    async def transport(record, content):
        if record.turn_id == "retry":
            raise RuntimeError("temporary transport failure")
        return f"transport-{record.turn_id}"

    conn = await ledger._ensure_open()
    conn.execute("UPDATE delivery_ledger SET updated_at = 0")
    conn.commit()
    result = await controller.recover_prepared(
        older_than=10**20,
        content_resolver=resolve,
        transport=transport,
        allow_transport_retry=True,
    )

    assert result.scanned == 3
    assert result.sent == 1
    assert result.retryable == 0
    assert result.failed == 1
    assert result.unknown == 1
    assert (await ledger.get("ambient:chat:sent")).status == "sent"
    assert (
        await ledger.get("ambient:chat:mismatch")
    ).reason == "recovery_content_hash_mismatch"
    retry = await ledger.get("ambient:chat:retry")
    assert retry.status == "unknown"
    assert retry.receipt_status == "unknown"
    again = await controller.recover_prepared(
        older_than=10**20,
        content_resolver=resolve,
        transport=transport,
        allow_transport_retry=True,
    )
    assert again.scanned == 0
    await ledger.close()


@pytest.mark.asyncio
async def test_recovery_keeps_retryable_transport_failure_prepared(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    controller = DeliveryController(ledger, retry_base_seconds=1)
    record = await ledger.prepare(
        key="ambient:chat:retryable",
        chat_id="chat",
        turn_id="retryable",
        reason="final_reply",
        reply_anchor_id="anchor",
        content_hash=ledger.content_hash("answer"),
    )
    conn = await ledger._ensure_open()
    conn.execute(
        "UPDATE delivery_ledger SET updated_at = 0 WHERE delivery_key = ?",
        (record.key,),
    )
    conn.commit()

    async def resolve(_record):
        return "answer"

    async def transport(_record, _content):
        return DeliveryReceipt(
            status="failed",
            error_code="timeout",
            retryable=True,
        )

    result = await controller.recover_prepared(
        older_than=10**20,
        content_resolver=resolve,
        transport=transport,
        allow_transport_retry=True,
    )

    assert result.scanned == 1
    assert result.retryable == 1
    assert result.failed == 0
    recovered = await ledger.get(record.key)
    assert recovered is not None
    assert recovered.status == "prepared"
    assert recovered.attempts == 2
    assert recovered.logical_delivery_id == record.logical_delivery_id
    assert recovered.receipt_status == "failed"
    assert recovered.error_code == "timeout"
    await ledger.close()


@pytest.mark.asyncio
async def test_delivery_controller_does_not_reprepare_settled_turn(tmp_path):
    controller = DeliveryController(DeliveryLedger(str(tmp_path / "delivery.sqlite3")))
    decision, record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="turn",
        content="answer",
        delivery_mode="automatic",
    )
    assert decision.should_deliver is True
    assert record is not None
    await controller.mark_sent(record, transport_id="t1")

    duplicate, duplicate_record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="turn",
        content="answer",
        delivery_mode="automatic",
    )
    assert duplicate.should_deliver is False
    assert duplicate.reason == "already_settled"
    assert duplicate_record is not None
    assert duplicate_record.status == "sent"


@pytest.mark.asyncio
async def test_delivery_receipt_preserves_transport_state_and_chunk_metadata(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    controller = DeliveryController(ledger)
    _, record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="receipt",
        content="answer",
        delivery_mode="automatic",
    )
    assert record is not None

    settled = await controller.settle_receipt(
        record,
        DeliveryReceipt(
            status="accepted",
            logical_delivery_id="delivery-1",
            transport_id="request-1",
            platform_message_id="qq-1",
            chunk_index=1,
            chunk_count=2,
        ),
    )
    assert settled is not None
    assert settled.status == "sent"
    assert settled.receipt_status == "accepted"
    assert settled.logical_delivery_id == "delivery-1"
    assert settled.platform_message_id == "qq-1"
    assert settled.chunk_index == 1
    assert settled.chunk_count == 2
    await ledger.close()


@pytest.mark.asyncio
async def test_empty_receipt_logical_id_keeps_prepared_delivery_id(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    controller = DeliveryController(ledger)
    _, record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="stable",
        content="answer",
        delivery_mode="automatic",
    )
    assert record is not None

    settled = await controller.settle_receipt(
        record,
        DeliveryReceipt(status="accepted"),
    )

    assert settled is not None
    assert settled.logical_delivery_id == record.logical_delivery_id
    await ledger.close()


@pytest.mark.asyncio
async def test_unknown_receipt_is_terminal_and_not_retried(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    controller = DeliveryController(ledger)
    _, record = await controller.prepare_ambient(
        chat_id="chat",
        turn_id="unknown",
        content="answer",
        delivery_mode="automatic",
    )
    assert record is not None

    settled = await controller.settle_receipt(
        record,
        DeliveryReceipt(
            status="unknown",
            logical_delivery_id="delivery-unknown",
            error_code="receipt_lost",
        ),
    )
    assert settled is not None
    assert settled.status == "unknown"
    assert await ledger.stale_prepared(older_than=10**20) == []
    await ledger.close()


@pytest.mark.asyncio
async def test_reply_delivery_uses_a_stable_turn_sequence_key(tmp_path):
    ledger = DeliveryLedger(str(tmp_path / "delivery.sqlite3"))
    controller = DeliveryController(ledger)

    first = await controller.prepare_reply_delivery(
        chat_id="chat",
        turn_id="turn",
        sequence=1,
        content="first chunk",
        reply_anchor_id="message-1",
    )
    duplicate = await controller.prepare_reply_delivery(
        chat_id="chat",
        turn_id="turn",
        sequence=1,
        content="changed content",
        reply_anchor_id="other-message",
    )

    assert first == duplicate
    assert first.key == "reply:chat:turn:1"
    assert first.logical_delivery_id == first.key
    assert first.content_hash == ledger.content_hash("first chunk")
    await ledger.close()


@pytest.mark.asyncio
async def test_external_delivery_records_prepare_and_transport_failure(tmp_path):
    audited = []

    async def audit_delivery(turn_id, delivery_id, status):
        audited.append((turn_id, delivery_id, status))

    controller = DeliveryController(
        DeliveryLedger(str(tmp_path / "delivery.sqlite3")),
        audit_delivery=audit_delivery,
    )

    async def callback(**kwargs):
        raise TimeoutError("transport down")

    receipt = await controller.deliver_text(
        delivery_id="consumer-error:chat:message",
        chat_id="chat",
        content="fallback",
        callback=callback,
        message_id="message",
        is_group=True,
    )

    record = await controller.ledger.get("external:consumer-error:chat:message")
    assert receipt.status == "failed"
    assert receipt.retryable is True
    assert record is not None
    assert record.status == "failed"
    assert [status for _, _, status in audited] == ["prepared", "failed"]
