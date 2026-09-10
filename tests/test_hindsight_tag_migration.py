import asyncio
from types import SimpleNamespace

from scripts.migrate_hindsight_session_tags import HindsightTagMigrator, build_plan


def test_build_plan_only_contains_chat_mappings():
    plan = build_plan(
        {
            "run_id": "run-1",
            "mappings": [
                {
                    "kind": "internal",
                    "legacy_key": "heartbeat:events",
                    "canonical_key": "heartbeat:events",
                },
                {
                    "kind": "chat",
                    "legacy_key": "123",
                    "canonical_key": "agent:main:qq:default:group:123",
                },
            ],
        }
    )
    assert len(plan["operations"]) == 1
    assert plan["operations"][0]["legacy_tag"] == "chat:123"


def test_migrator_preserves_existing_tags_and_is_idempotent():
    class Documents:
        def __init__(self):
            self.items = [{"id": "session-123", "tags": ["user:u1", "chat:123"]}]
            self.updates = []

        def list_documents(self, **kwargs):
            return SimpleNamespace(items=self.items, total=len(self.items))

        def update_document(self, **kwargs):
            self.updates.append(kwargs)
            self.items[0]["tags"] = kwargs["update_document_request"].tags

        def get_document(self, **kwargs):
            return SimpleNamespace(tags=self.items[0]["tags"])

    documents = Documents()
    migrator = HindsightTagMigrator(
        SimpleNamespace(documents=documents), bank_id="qq_bot"
    )
    operation = {
        "legacy_tag": "chat:123",
        "canonical_tag": "chat:canonical",
        "document_ids": [],
        "updated": 0,
    }

    assert asyncio.run(migrator.migrate_operation(operation)) == 1
    assert documents.items[0]["tags"] == ["user:u1", "chat:123", "chat:canonical"]
    assert asyncio.run(migrator.migrate_operation(operation)) == 0
    assert len(documents.updates) == 1


def test_migrator_checkpoints_each_document():
    class Documents:
        def __init__(self):
            self.items = [
                {"id": "session-1", "tags": ["chat:123"]},
                {"id": "session-2", "tags": ["chat:123"]},
            ]

        def list_documents(self, **kwargs):
            return SimpleNamespace(items=self.items, total=len(self.items))

        def update_document(self, **kwargs):
            request = kwargs["update_document_request"]
            document_id = kwargs["document_id"]
            for item in self.items:
                if item["id"] == document_id:
                    item["tags"] = request.tags

    checkpoints = []

    async def checkpoint(operation):
        checkpoints.append((operation["status"], tuple(operation["document_ids"])))

    operation = {
        "legacy_tag": "chat:123",
        "canonical_tag": "chat:canonical",
        "document_ids": [],
        "updated": 0,
    }
    migrator = HindsightTagMigrator(
        SimpleNamespace(documents=Documents()), bank_id="qq_bot"
    )

    assert asyncio.run(migrator.migrate_operation(operation, checkpoint)) == 2
    assert checkpoints[0] == ("applying", ())
    assert checkpoints[-1] == ("applied", ("session-1", "session-2"))


def test_verify_rejects_empty_operation():
    migrator = HindsightTagMigrator(SimpleNamespace(), bank_id="qq_bot")

    assert asyncio.run(
        migrator.verify_operation(
            {
                "legacy_tag": "chat:missing",
                "canonical_tag": "chat:canonical",
                "document_ids": [],
            }
        )
    ) == (False, 1)
