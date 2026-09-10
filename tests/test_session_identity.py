import json
import sqlite3
from pathlib import Path

import pytest

from core.session_identity import (
    ChannelRegistry,
    DeliveryTarget,
    DeliveryTargetCatalog,
    DeliveryValidation,
    QQAdapter,
    SessionIdentityRegistry,
    SessionIdentityResolver,
    TargetFilters,
    TargetPrincipal,
    build_chat_session_key,
    build_internal_session_key,
    build_workspace_slug,
    parse_chat_session_key,
)


def test_chat_key_round_trip_encodes_structural_delimiters():
    target = DeliveryTarget("qq", "default", "group", "abc:123/456")
    key = build_chat_session_key(target)

    assert key == "agent:main:qq:default:group:abc%3A123%2F456"
    assert parse_chat_session_key(key) == target


def test_group_and_direct_same_raw_id_are_distinct():
    group = DeliveryTarget("qq", "default", "group", "123")
    direct = DeliveryTarget("qq", "default", "direct", "123")

    assert build_chat_session_key(group) != build_chat_session_key(direct)


def test_internal_keys_do_not_have_delivery_target():
    assert build_internal_session_key("cron", "job", "run") == (
        "agent:main:cron:job:run"
    )
    with pytest.raises(ValueError):
        build_internal_session_key("qq", "bad")


def test_workspace_slug_uses_chat_type_and_encoded_target():
    target = DeliveryTarget("qq", "default", "group", "abc:123/456")
    assert build_workspace_slug(target) == "groups/v1-abc%3A123%2F456"


def test_workspace_manager_uses_migrated_slug_for_canonical_chat_key(tmp_path: Path):
    from core.managers.workspace_manager import WorkspaceManager

    manager = WorkspaceManager(root=str(tmp_path))
    path = manager.sandbox_dir(True, "agent:main:qq:default:group:abc%3A123%2F456")

    assert path == tmp_path / "groups" / "v1-abc%3A123%2F456" / "files"


def test_registry_reuses_document_id_and_preserves_raw_target(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "session_identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "abc:123/456")

    first = registry.register_chat(
        target, legacy_key="abc123", workspace_slug="v1-abc%3A123%2F456"
    )
    second = registry.register_chat(target)

    assert first.session_key == "agent:main:qq:default:group:abc%3A123%2F456"
    assert first.hindsight_document_id.startswith("hdoc_")
    assert second.hindsight_document_id == first.hindsight_document_id
    assert registry.find_by_target(target).target == target
    assert registry.find_by_legacy("abc123")[0].session_key == first.session_key
    registry.close()


def test_registry_migration_can_fill_existing_legacy_facts(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "session_identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "123")
    first = registry.register_chat(target)
    updated = registry.register_chat(
        target,
        legacy_key="123",
        hindsight_document_id="session-123",
        migration_run_id="run-1",
    )

    assert updated.hindsight_document_id == "session-123"
    assert updated.legacy_session_keys == ("123",)
    registry.close()


def test_registry_keeps_group_direct_mappings_separate(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "session_identity.sqlite3")
    group = registry.register_chat(DeliveryTarget("qq", "default", "group", "123"))
    direct = registry.register_chat(DeliveryTarget("qq", "default", "direct", "123"))

    assert group.session_key != direct.session_key
    assert len(registry.list_all()) == 2
    registry.close()


def test_registry_allows_same_target_for_different_agents(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "session_identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "123")

    main = registry.register_chat(target, agent_id="main")
    support = registry.register_chat(target, agent_id="support")

    assert main.session_key != support.session_key
    assert (
        registry.find_by_target(target, agent_id="support").session_key
        == support.session_key
    )
    registry.close()


def test_catalog_marks_inactive_without_deleting(tmp_path: Path):
    catalog = DeliveryTargetCatalog(tmp_path / "delivery_targets.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "123")

    catalog.observe(target, observed_at=10.0)
    catalog.mark_inactive(target)
    assert catalog.list_visible() == []
    assert catalog.list_visible(include_inactive=True)[0].status == "inactive"
    assert catalog.validate(target) == DeliveryValidation(
        False, "target_inactive", target
    )
    catalog.close()


def test_catalog_cron_resolution_requires_admin_and_filters(tmp_path: Path):
    catalog = DeliveryTargetCatalog(tmp_path / "delivery.sqlite3")
    current = DeliveryTarget("qq", "default", "group", "current")
    target = DeliveryTarget("qq", "default", "group", "target")
    catalog.observe(current)
    catalog.observe(target)

    assert catalog.list_visible(TargetPrincipal("user")) == []
    assert catalog.resolve_for_cron("target", TargetPrincipal("user"), current) is None
    assert (
        catalog.resolve_for_cron("current", TargetPrincipal("user"), current) == current
    )
    assert catalog.list_visible(
        TargetPrincipal("admin", is_admin=True),
        TargetFilters(chat_type="group"),
    )
    assert (
        catalog.resolve_for_cron(
            "target", TargetPrincipal("admin", is_admin=True), current
        )
        == target
    )
    catalog.close()


def test_resolver_uses_legacy_until_cutover(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "identity.sqlite3")
    resolver = SessionIdentityResolver(registry)
    target = DeliveryTarget("qq", "default", "group", "123")

    ref = resolver.resolve_inbound(target)

    assert ref.session_key == "123"
    assert ref.legacy_session_keys == ()
    assert ref.hindsight_document_id == ""
    assert resolver.recall_aliases(ref) == ("chat:123",)
    assert registry.list_all() == []
    registry.close()


def test_resolver_switches_to_canonical_and_resolves_internal(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "identity.sqlite3")
    resolver = SessionIdentityResolver(registry, canonical_enabled=True)
    target = DeliveryTarget("qq", "default", "direct", "abc")

    chat = resolver.resolve_inbound(target, legacy_key="abc")
    cron = resolver.resolve_internal("cron", job_id="job", run_id="run")
    heartbeat = resolver.resolve_internal("heartbeat")

    assert chat.session_key == "agent:main:qq:default:direct:abc"
    assert chat.hindsight_document_id.startswith("hdoc_")
    assert cron.session_key == "agent:main:cron:job:job:run:run"
    assert cron.target is None
    assert heartbeat.session_key == "agent:main:heartbeat:events"
    assert (
        resolver.resolve_legacy("abc", is_group=False).session_key == chat.session_key
    )
    registry.close()


def test_legacy_resolver_reuses_migrated_document_without_creating_new_identity(
    tmp_path: Path,
):
    registry = SessionIdentityRegistry(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "123")
    registry.register_chat(
        target,
        legacy_key="123",
        hindsight_document_id="session-123",
        state="verified",
    )
    resolver = SessionIdentityResolver(registry)

    ref = resolver.resolve_inbound(target)

    assert ref.session_key == "123"
    assert ref.hindsight_document_id == "session-123"
    assert ref.legacy_session_keys == ("agent:main:qq:default:group:123",)
    registry.close()


def test_resolver_rejects_ambiguous_legacy_key(tmp_path: Path):
    registry = SessionIdentityRegistry(tmp_path / "identity.sqlite3")
    resolver = SessionIdentityResolver(registry, canonical_enabled=True)
    resolver.resolve_inbound(DeliveryTarget("qq", "default", "group", "same"))
    resolver.resolve_inbound(DeliveryTarget("qq", "default", "direct", "same"))

    with pytest.raises(ValueError, match="ambiguous"):
        resolver.resolve_legacy("same")
    registry.close()


def test_local_migration_apply_verify_and_rollback(tmp_path: Path):
    from scripts.migrate_session_identity import (
        apply_manifest,
        preflight,
        rollback_manifest,
        verify_manifest,
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_db = data_dir / "context.sqlite3"
    connection = sqlite3.connect(source_db)
    connection.execute("CREATE TABLE messages (chat_id TEXT NOT NULL, body TEXT)")
    connection.execute(
        "CREATE TABLE engagement_targets (chat_id TEXT PRIMARY KEY, status TEXT)"
    )
    connection.execute("INSERT INTO messages VALUES ('123', 'hello')")
    connection.execute("INSERT INTO engagement_targets VALUES ('123', 'verified')")
    connection.commit()
    connection.close()
    orchestration_db = data_dir / "orchestration.sqlite"
    connection = sqlite3.connect(orchestration_db)
    connection.execute("CREATE TABLE plans (chat_id TEXT NOT NULL, plan_id TEXT)")
    connection.execute("INSERT INTO plans VALUES ('123', 'plan-1')")
    connection.commit()
    connection.close()
    (data_dir / "tasks.json").write_text(
        json.dumps([{"session_id": "123", "delivery_channel": "123"}])
    )
    manifest_path = data_dir / "migrations" / "run-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "data_dir": str(data_dir),
                "state": "planned",
                "mappings": [
                    {
                        "kind": "chat",
                        "legacy_key": "123",
                        "canonical_key": "agent:main:qq:default:group:123",
                        "chat_type": "group",
                        "channel": "qq",
                        "account_id": "default",
                        "workspace_slug": "groups/v1-123",
                        "legacy_document_id": "session-123",
                    }
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="confirm-stopped"):
        apply_manifest(manifest_path)
    apply_manifest(manifest_path, confirm_stopped=True)
    assert verify_manifest(manifest_path)["state"] == "verified"
    assert (
        sqlite3.connect(source_db).execute("SELECT chat_id FROM messages").fetchone()[0]
        == "agent:main:qq:default:group:123"
    )
    assert (
        sqlite3.connect(source_db)
        .execute("SELECT chat_id FROM engagement_targets")
        .fetchone()[0]
        == "123"
    )
    assert (
        sqlite3.connect(orchestration_db)
        .execute("SELECT chat_id FROM plans")
        .fetchone()[0]
        == "agent:main:qq:default:group:123"
    )
    migrated_task = json.loads((data_dir / "tasks.json").read_text())[0]
    assert migrated_task["session_id"] == "agent:main:qq:default:group:123"
    assert migrated_task["delivery_channel"] == "123"
    rollback_manifest(manifest_path, confirm_stopped=True)
    assert (
        sqlite3.connect(source_db).execute("SELECT chat_id FROM messages").fetchone()[0]
        == "123"
    )
    assert (
        sqlite3.connect(orchestration_db)
        .execute("SELECT chat_id FROM plans")
        .fetchone()[0]
        == "123"
    )
    rolled_back_task = json.loads((data_dir / "tasks.json").read_text())[0]
    assert rolled_back_task["session_id"] == "123"


def test_local_migration_preflight_blocks_active_work_and_receipts(tmp_path: Path):
    from scripts.migrate_session_identity import preflight

    data_dir = tmp_path / "data"
    (data_dir / "tasks").mkdir(parents=True)
    (data_dir / "tasks" / "tasks.json").write_text(
        json.dumps([{"id": "task-1", "status": "running"}])
    )
    (data_dir / "tasks" / "turn_states.json").write_text(
        json.dumps([{"turn_id": "turn-1", "phase": "awaiting_approval"}])
    )
    ledger = sqlite3.connect(data_dir / "delivery_ledger.sqlite3")
    ledger.execute("CREATE TABLE delivery_ledger (status TEXT, receipt_status TEXT)")
    ledger.execute("INSERT INTO delivery_ledger VALUES ('prepared', '')")
    ledger.commit()
    ledger.close()

    result = preflight(data_dir)

    assert result["ok"] is False
    assert {item["kind"] for item in result["blockers"]} == {
        "active_tasks",
        "active_turns",
        "blocking_delivery_receipts",
    }


def test_local_migration_verify_detects_post_copy_row_mutation(tmp_path: Path):
    from scripts.migrate_session_identity import apply_manifest, verify_manifest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = data_dir / "events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE events (chat_id TEXT, event_id TEXT, seq INTEGER)")
    connection.execute("INSERT INTO events VALUES ('123', 'event-1', 1)")
    connection.commit()
    connection.close()
    manifest_path = data_dir / "migrations" / "run-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "data_dir": str(data_dir),
                "state": "planned",
                "mappings": [
                    {
                        "kind": "chat",
                        "legacy_key": "123",
                        "canonical_key": "agent:main:qq:default:group:123",
                        "chat_type": "group",
                        "workspace_slug": "groups/v1-123",
                    }
                ],
            }
        )
    )

    apply_manifest(manifest_path, confirm_stopped=True)
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE events SET event_id = 'tampered' WHERE chat_id LIKE 'agent:%'"
    )
    connection.commit()
    connection.close()

    result = verify_manifest(manifest_path)

    assert result["state"] == "verify_failed"
    assert any(item.startswith("sqlite:") for item in result["failures"])


def test_migration_audit_emits_group_and_direct_candidates_for_one_raw_id(
    tmp_path: Path,
):
    from scripts.migrate_session_identity import audit

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = data_dir / "events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE messages (chat_id TEXT, chat_type TEXT, body TEXT)"
    )
    connection.executemany(
        "INSERT INTO messages VALUES (?, ?, ?)",
        [("same", "group", "g"), ("same", "direct", "d")],
    )
    connection.commit()
    connection.close()

    report = audit(data_dir)

    candidates = [item for item in report["sessions"] if item["legacy_key"] == "same"]
    assert {item["chat_type"] for item in candidates} == {"group", "direct"}
    assert all(item["manual_hindsight_binding_required"] for item in candidates)
    assert all(item["manual_store_binding_required"] for item in candidates)
    assert report["ambiguous_count"] == 2


def test_migration_audit_discovers_jsonl_only_session(tmp_path: Path):
    from scripts.migrate_session_identity import audit

    data_dir = tmp_path / "data" / "sessions"
    data_dir.mkdir(parents=True)
    (data_dir / "jsonl-only.jsonl").write_text(
        json.dumps({"session_id": "jsonl-only", "body": "hello"}) + "\n"
    )

    report = audit(data_dir.parent)

    assert any(item["legacy_key"] == "jsonl-only" for item in report["sessions"])


def test_migration_plan_matches_cron_runtime_keys(tmp_path: Path):
    from scripts.migrate_session_identity import audit, plan

    data_dir = tmp_path / "data"
    (data_dir / "sessions").mkdir(parents=True)
    (data_dir / "tasks").mkdir()
    (data_dir / "sessions" / "task:task-1.jsonl").write_text(
        json.dumps({"session_id": "task:task-1"}) + "\n"
    )
    (data_dir / "tasks" / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "task-1",
                    "job_id": "job-1",
                    "session_id": "task:task-1",
                }
            ]
        )
    )
    audit_path = data_dir / "audit.json"
    audit(data_dir, audit_path)

    manifest = plan(data_dir, "run-1", audit_path, data_dir / "manifest.json")

    mapping = next(
        item for item in manifest["mappings"] if item["legacy_key"] == "task:task-1"
    )
    assert mapping["canonical_key"] == "agent:main:cron:job:job-1:run:task-1"
    assert not manifest["blockers"]


def test_cutover_requires_verified_hindsight_plan(tmp_path: Path):
    from scripts.migrate_session_identity import mark_canonical_cutover

    manifest_path = tmp_path / "manifest.json"
    hindsight_path = tmp_path / "hindsight.json"
    manifest_path.write_text(
        json.dumps({"run_id": "run-1", "state": "verified", "mappings": []})
    )
    hindsight_path.write_text(
        json.dumps({"source_run_id": "run-1", "state": "applied"})
    )

    with pytest.raises(RuntimeError, match="Hindsight"):
        mark_canonical_cutover(manifest_path, hindsight_path)


def test_migration_audit_ignores_archive_and_migration_artifacts(tmp_path: Path):
    from scripts.migrate_session_identity import audit

    data_dir = tmp_path / "data"
    (data_dir / "sessions").mkdir(parents=True)
    (data_dir / "archives" / "archive_audit").mkdir(parents=True)
    (data_dir / "archives" / "manifests").mkdir(parents=True)
    (data_dir / "migrations" / "run-1").mkdir(parents=True)
    (data_dir / "sessions" / "task:real.jsonl").write_text(
        json.dumps({"session_id": "task:real"}) + "\n"
    )
    for directory, filename, value in (
        ("archives/archive_audit", "audit.json", "archive-op_fake"),
        ("archives/manifests", "manifest.json", "cron_fake"),
        ("migrations/run-1", "report.json", "cron_migration"),
    ):
        path = data_dir / directory / filename
        path.write_text(json.dumps({"chat_id": value}))

    report = audit(data_dir)
    discovered = {item["legacy_key"] for item in report["sessions"]}

    assert discovered == {"task:real"}


def test_migration_plan_blocks_unresolved_hindsight_binding(
    tmp_path: Path, monkeypatch
):
    from scripts.migrate_session_identity import audit, plan

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = data_dir / "events.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE messages (chat_id TEXT, chat_type TEXT)")
    connection.executemany(
        "INSERT INTO messages VALUES (?, ?)",
        [("same", "group"), ("same", "direct")],
    )
    connection.commit()
    connection.close()
    audit_path = data_dir / "audit.json"
    audit(data_dir, audit_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    manifest = plan(data_dir, "run-ambiguous", audit_path, None)

    assert manifest["state"] == "blocked"
    assert {item["kind"] for item in manifest["blockers"]} == {
        "manual_hindsight_binding_required",
        "manual_store_binding_required",
    }
    assert len(manifest["mappings"]) == 2


def test_local_migration_moves_jsonl_archives_and_verifies_workspace_tree(
    tmp_path: Path,
):
    from scripts.migrate_session_identity import (
        apply_manifest,
        rollback_manifest,
        verify_manifest,
    )

    data_dir = tmp_path / "data"
    sessions = data_dir / "sessions"
    archives = data_dir / "archives" / "memory"
    sessions.mkdir(parents=True)
    archives.mkdir(parents=True)
    (sessions / "123.jsonl").write_text(
        json.dumps({"session_id": "123", "body": "hello"}) + "\n"
    )
    (archives / "123.jsonl.archived.2026-01-01").write_text(
        json.dumps({"chat_id": "123", "body": "old"}) + "\n"
    )
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "groups" / "123"
    workspace.mkdir(parents=True)
    (workspace / "note.txt").write_text("keep")
    manifest_path = data_dir / "migrations" / "run-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "data_dir": str(data_dir),
                "state": "planned",
                "mappings": [
                    {
                        "kind": "chat",
                        "legacy_key": "123",
                        "canonical_key": "agent:main:qq:default:group:123",
                        "chat_type": "group",
                        "workspace_slug": "groups/v1-123",
                        "legacy_document_id": "session-123",
                    }
                ],
            }
        )
    )

    apply_manifest(
        manifest_path,
        confirm_stopped=True,
        workspace_root=workspace_root,
    )

    canonical = "agent:main:qq:default:group:123"
    assert (sessions / f"{canonical}.jsonl").exists()
    assert not (sessions / "123.jsonl").exists()
    assert canonical in (sessions / f"{canonical}.jsonl").read_text()
    assert (archives / f"{canonical}.jsonl.archived.2026-01-01").exists()
    assert (workspace_root / "groups" / "v1-123" / "note.txt").read_text() == "keep"
    assert verify_manifest(manifest_path)["state"] == "verified"

    (workspace_root / "groups" / "v1-123" / "note.txt").write_text("tampered")
    assert verify_manifest(manifest_path)["state"] == "verify_failed"

    (workspace_root / "groups" / "v1-123" / "note.txt").write_text("keep")
    rollback_manifest(manifest_path, confirm_stopped=True)
    assert (sessions / "123.jsonl").exists()
    assert (workspace_root / "groups" / "123" / "note.txt").read_text() == "keep"


@pytest.mark.asyncio
async def test_qq_adapter_rejects_other_channel_target():
    sent = []

    async def send(target, content, *, reply_to=""):
        sent.append((target, content, reply_to))

    adapter = QQAdapter(send_callback=send)
    target = DeliveryTarget("telegram", "default", "direct", "123")

    with pytest.raises(ValueError):
        await adapter.send_message(target, "hello")
    assert sent == []


def test_channel_registry_binds_accounts():
    qq_default = QQAdapter(account_id="default")
    qq_work = QQAdapter(account_id="work")
    registry = ChannelRegistry()

    registry.register(qq_default)
    registry.register(qq_work)

    assert registry.list_accounts("qq") == [("qq", "default"), ("qq", "work")]
    assert registry.for_target(DeliveryTarget("qq", "work", "direct", "1")) is qq_work
    with pytest.raises(ValueError):
        registry.register(qq_default)
