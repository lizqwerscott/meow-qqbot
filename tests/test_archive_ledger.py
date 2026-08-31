from core.managers.archive_ledger import ArchiveLedger


def test_archive_ledger_commits_membership_idempotently(tmp_path):
    ledger = ArchiveLedger(str(tmp_path / "archive-ledger.sqlite3"))

    ledger.commit_membership("batch-1", "chat-1", ["timeline:event-1"])
    ledger.commit_membership(
        "batch-1", "chat-1", ["timeline:event-1", "timeline:event-2"]
    )

    assert ledger.is_archived("chat-1", "timeline:event-1")
    assert ledger.is_archived("chat-1", "timeline:event-2")
    assert ledger.batch_ids_for_identities("chat-1", ["timeline:event-1"]) == {
        "timeline:event-1": "batch-1"
    }
    assert not ledger.is_archived("chat-2", "timeline:event-1")

    ledger.close()


def test_archive_ledger_rejects_cross_batch_identity_conflict(tmp_path):
    ledger = ArchiveLedger(str(tmp_path / "archive-ledger.sqlite3"))
    ledger.commit_membership("batch-1", "chat-1", ["timeline:event-1"])

    try:
        ledger.commit_membership("batch-2", "chat-1", ["timeline:event-1"])
    except RuntimeError as exc:
        assert "already committed" in str(exc)
    else:
        raise AssertionError("cross-batch identity conflict was accepted")

    assert ledger.committed_batch_count("chat-1") == 1
    ledger.close()


def test_archive_ledger_rejects_batch_id_reuse_across_chats(tmp_path):
    ledger = ArchiveLedger(str(tmp_path / "archive-ledger.sqlite3"))
    ledger.prepare_batch("batch-1", "chat-1")

    try:
        ledger.prepare_batch("batch-1", "chat-2")
    except RuntimeError as exc:
        assert "another chat" in str(exc)
    else:
        raise AssertionError("batch ID was reused across chats")

    ledger.close()


def test_archive_ledger_validates_batch_hash_during_prepare_and_recovery(tmp_path):
    ledger = ArchiveLedger(str(tmp_path / "archive-ledger.sqlite3"))

    ledger.prepare_batch("batch-1", "chat-1", records_hash="hash-a")
    assert ledger.recover_batch("batch-1", "chat-1", records_hash="hash-a") == (
        "prepared"
    )

    try:
        ledger.recover_batch("batch-1", "chat-1", records_hash="hash-b")
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("batch hash mismatch was accepted")

    ledger.close()
