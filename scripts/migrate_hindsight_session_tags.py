#!/usr/bin/env python3
"""Migrate Hindsight chat tags without renaming or copying documents.

The script is intentionally independent from the local SQLite migration.  It
only adds ``chat:<canonical-session-key>`` to documents found by their legacy
chat tag and preserves every existing tag (including ``user:...`` and the old
chat tag).  Checkpoints make retries safe when the remote service is briefly
unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hindsight_client import Hindsight
from hindsight_client_api.models.update_document_request import UpdateDocumentRequest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("mappings"), list):
        raise ValueError("invalid session migration manifest")
    return payload


def build_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    operations = []
    for mapping in manifest["mappings"]:
        if mapping.get("kind") != "chat":
            continue
        legacy = str(mapping.get("legacy_key") or "")
        canonical = str(mapping.get("canonical_key") or "")
        if not legacy or not canonical:
            raise ValueError("chat mapping requires legacy_key and canonical_key")
        operations.append(
            {
                "legacy_tag": f"chat:{legacy}",
                "canonical_tag": f"chat:{canonical}",
                "legacy_document_id": mapping.get("legacy_document_id")
                or f"session-{legacy}",
                "status": "planned",
                "document_ids": [],
                "updated": 0,
            }
        )
    return {
        "schema_version": 1,
        "created_at": time.time(),
        "source_run_id": manifest.get("run_id", ""),
        "bank_id": "qq_bot",
        "operations": operations,
        "state": "planned",
    }


class HindsightTagMigrator:
    def __init__(self, client: Any, *, bank_id: str, page_size: int = 100) -> None:
        self.client = client
        self.bank_id = bank_id
        self.page_size = max(1, min(int(page_size), 1000))

    async def _call(self, method: Any, **kwargs: Any) -> Any:
        result = method(**kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result

    async def migrate_operation(
        self,
        operation: dict[str, Any],
        checkpoint: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> int:
        legacy_tag = operation["legacy_tag"]
        canonical_tag = operation["canonical_tag"]
        offset = 0
        changed = 0
        initial_updated = int(operation.get("updated", 0))
        found = set(str(item) for item in operation.get("document_ids", []))
        operation["status"] = "applying"
        if checkpoint is not None:
            await checkpoint(operation)
        while True:
            response = await self._call(
                self.client.documents.list_documents,
                bank_id=self.bank_id,
                tags=[legacy_tag],
                tags_match="all_strict",
                limit=self.page_size,
                offset=offset,
            )
            items = list(getattr(response, "items", None) or [])
            if not items:
                break
            for item in items:
                document_id = str(
                    item.get("id")
                    if isinstance(item, dict)
                    else getattr(item, "id", "")
                )
                if not document_id:
                    continue
                found.add(document_id)
                tags = (
                    item.get("tags")
                    if isinstance(item, dict)
                    else getattr(item, "tags", None)
                )
                if tags is None:
                    document = await self._call(
                        self.client.documents.get_document,
                        bank_id=self.bank_id,
                        document_id=document_id,
                    )
                    tags = getattr(document, "tags", None) or []
                merged = list(
                    dict.fromkeys(
                        [*(str(tag) for tag in tags), canonical_tag, legacy_tag]
                    )
                )
                if set(merged) != set(str(tag) for tag in tags):
                    await self._call(
                        self.client.documents.update_document,
                        bank_id=self.bank_id,
                        document_id=document_id,
                        update_document_request=UpdateDocumentRequest(tags=merged),
                    )
                    changed += 1
                operation["document_ids"] = sorted(found)
                operation["updated"] = initial_updated + changed
                if checkpoint is not None:
                    await checkpoint(operation)
            offset += len(items)
            total = int(getattr(response, "total", 0) or 0)
            if offset >= total or len(items) < self.page_size:
                break
        operation["document_ids"] = sorted(found)
        operation["status"] = "applied"
        operation["fingerprint"] = hashlib.sha256(
            "\n".join(operation["document_ids"]).encode()
        ).hexdigest()
        if checkpoint is not None:
            await checkpoint(operation)
        return changed

    async def verify_operation(self, operation: dict[str, Any]) -> tuple[bool, int]:
        if not operation.get("document_ids"):
            return False, 1
        missing = 0
        for document_id in operation.get("document_ids", []):
            document = await self._call(
                self.client.documents.get_document,
                bank_id=self.bank_id,
                document_id=document_id,
            )
            tags = set(getattr(document, "tags", None) or [])
            if (
                operation["canonical_tag"] not in tags
                or operation["legacy_tag"] not in tags
            ):
                missing += 1
        return missing == 0, missing


async def apply_plan(path: Path, *, base_url: str, bank_id: str) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    client = Hindsight(base_url=base_url, timeout=30.0)
    migrator = HindsightTagMigrator(client, bank_id=bank_id)
    plan["bank_id"] = bank_id
    plan["state"] = "applying"
    _write_json(path, plan)
    try:

        async def checkpoint(_operation: dict[str, Any]) -> None:
            plan["updated_at"] = time.time()
            _write_json(path, plan)

        for operation in plan.get("operations", []):
            if operation.get("status") == "applied":
                continue
            await migrator.migrate_operation(operation, checkpoint=checkpoint)
        plan["bank_id"] = bank_id
        plan["state"] = "applied"
        plan["updated_at"] = time.time()
        _write_json(path, plan)
        return plan
    finally:
        await client.aclose()


async def verify_plan(path: Path, *, base_url: str, bank_id: str) -> bool:
    plan = json.loads(path.read_text(encoding="utf-8"))
    client = Hindsight(base_url=base_url, timeout=30.0)
    migrator = HindsightTagMigrator(client, bank_id=bank_id)
    try:
        failures = 0
        for operation in plan.get("operations", []):
            ok, missing = await migrator.verify_operation(operation)
            operation["verify_missing"] = missing
            if not ok:
                failures += 1
            operation["status"] = "verified" if ok else "verify_failed"
        plan["state"] = "verified" if failures == 0 else "verify_failed"
        plan["verified_at"] = time.time()
        _write_json(path, plan)
        return failures == 0
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify"))
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="local session migration manifest or Hindsight plan",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HINDSIGHT_BASE_URL", "http://127.0.0.1:8888"),
    )
    parser.add_argument(
        "--bank-id", default=os.environ.get("HINDSIGHT_BANK_ID", "qq_bot")
    )
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_plan(_load_manifest(args.manifest))
        plan["bank_id"] = args.bank_id
        output = args.output or args.manifest.with_name("hindsight-tag-plan.json")
        _write_json(output, plan)
        print(output)
        return 0
    if args.command == "apply":
        asyncio.run(
            apply_plan(args.manifest, base_url=args.base_url, bank_id=args.bank_id)
        )
        return 0
    return (
        0
        if asyncio.run(
            verify_plan(args.manifest, base_url=args.base_url, bank_id=args.bank_id)
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
