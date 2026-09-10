#!/usr/bin/env python3
"""Audit, plan, and execute the offline session identity migration.

Apply/rollback require an explicit stopped-process confirmation and always keep
0600 backups plus a resumable journal beside the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.session_identity import (
    DeliveryTarget,
    SessionIdentityRegistry,
    build_chat_session_key,
    build_internal_session_key,
    build_workspace_slug,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows is not a supported runtime
    fcntl = None

INTERNAL_PREFIXES = ("agent:", "cron:", "task:", "heartbeat:", "system:", "exec:")
_ACTIVE_TURN_PHASES = {"active", "waiting", "awaiting_approval", "finalizing"}
_ACTIVE_TASK_STATUSES = {"pending", "running"}
# Runtime engagement settings are a delivery-target directory, not a session
# store.  Their raw QQ id must remain usable by the channel adapter.
_RAW_TARGET_TABLES = {"engagement_targets"}
_JSON_SESSION_FIELDS = {
    "session_id",
    "execution_session_key",
    "chat_id",
    "media_source_chat_id",
    "parent_chat_id",
    "source_chat_id",
}
_JSON_ROOT_SOURCES = {"chat_types.json", "tasks.json"}


def _is_json_identity_source(data_dir: Path, path: Path) -> bool:
    """Return whether a JSON file belongs to a migratable business store."""
    try:
        relative_parts = path.relative_to(data_dir).parts
    except ValueError:
        return False
    if not relative_parts or "migrations" in relative_parts:
        return False
    if "archive_audit" in relative_parts or "manifests" in relative_parts:
        return False
    if path.suffix != ".json" and ".jsonl" not in path.name:
        return False
    if len(relative_parts) == 1:
        return path.name in _JSON_ROOT_SOURCES
    if relative_parts[0] in {"sessions", "tasks"}:
        return True
    return relative_parts[:2] == ("archives", "memory")


def _sqlite_paths(data_dir: Path) -> list[Path]:
    """Find SQLite stores regardless of the historical suffix in use."""
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".sqlite", ".sqlite3"}
        and "migrations" not in path.relative_to(data_dir).parts
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_tables(path: Path) -> list[tuple[str, list[str]]]:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        tables: list[tuple[str, list[str]]] = []
        names = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (name,) in names:
            columns = [
                row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')
            ]
            tables.append((name, columns))
        return tables
    finally:
        connection.close()


def _read_chat_ids(path: Path, table: str) -> tuple[set[str], dict[str, set[str]]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f'SELECT DISTINCT chat_id FROM "{table}" WHERE chat_id IS NOT NULL AND chat_id != ""'
        ).fetchall()
        values = {str(row[0]) for row in rows}
        return values, {value: set() for value in values}
    except sqlite3.Error:
        return set(), {}
    finally:
        connection.close()


def _load_chat_type_hints(data_dir: Path) -> dict[str, set[str]]:
    candidates = (
        data_dir / "chat_types.json",
        data_dir / "sessions" / "chat_types.json",
        data_dir / "sessions" / "meta" / "chat_types.json",
        data_dir / "meta" / "chat_types.json",
    )
    hints: dict[str, set[str]] = {}

    def add_hint(chat_id: object, value: object) -> None:
        normalized = str(value).strip().lower()
        if normalized in {"group", "direct"}:
            hints.setdefault(str(chat_id), set()).add(normalized)

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for chat_id, value in payload.items():
            if isinstance(value, bool):
                add_hint(chat_id, "group" if value else "direct")
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    add_hint(chat_id, item)
            else:
                add_hint(chat_id, value)
    for path in _sqlite_paths(data_dir):
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'context_chat_types'"
            ).fetchone()
            if table is None:
                connection.close()
                continue
            for chat_id, is_group in connection.execute(
                "SELECT chat_id, is_group FROM context_chat_types"
            ):
                add_hint(chat_id, "group" if bool(is_group) else "direct")
            connection.close()
        except sqlite3.Error:
            continue

    for path in _sqlite_paths(data_dir):
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            for table, columns in _sqlite_tables(path):
                if "chat_id" not in columns:
                    continue
                type_column = next(
                    (
                        column
                        for column in ("chat_type", "conversation_type")
                        if column in columns
                    ),
                    None,
                )
                if type_column is not None:
                    for chat_id, value in connection.execute(
                        f'SELECT chat_id, "{type_column}" FROM "{table}" '
                        "WHERE chat_id IS NOT NULL"
                    ):
                        add_hint(chat_id, value)
                if "is_group" in columns:
                    for chat_id, value in connection.execute(
                        f'SELECT chat_id, is_group FROM "{table}" '
                        "WHERE chat_id IS NOT NULL"
                    ):
                        add_hint(chat_id, "group" if bool(value) else "direct")
            connection.close()
        except sqlite3.Error:
            continue
    return hints


def _load_task_job_hints(data_dir: Path) -> dict[str, set[str]]:
    """Map legacy task session IDs to their owning cron job IDs."""
    hints: dict[str, set[str]] = {}
    paths = (data_dir / "tasks" / "tasks.json", data_dir / "tasks.json")
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            task_id = str(record.get("id") or "")
            job_id = str(record.get("job_id") or "")
            if not task_id or not job_id:
                continue
            legacy_keys = {
                str(record.get(field) or "")
                for field in ("session_id", "execution_session_key")
            }
            if any(key == f"task:{task_id}" for key in legacy_keys):
                hints.setdefault(task_id, set()).add(job_id)
    return hints


def _discover_sources(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    sources: list[dict[str, Any]] = []
    chat_sources: dict[str, set[str]] = {}
    for path in _sqlite_paths(data_dir):
        for table, columns in _sqlite_tables(path):
            if "chat_id" not in columns or table in _RAW_TARGET_TABLES:
                continue
            chat_ids, _ = _read_chat_ids(path, table)
            source = {
                "path": str(path),
                "table": table,
                "chat_id_count": len(chat_ids),
                "chat_id_hash": hashlib.sha256(
                    "\n".join(sorted(chat_ids)).encode("utf-8")
                ).hexdigest(),
            }
            sources.append(source)
            for chat_id in chat_ids:
                chat_sources.setdefault(chat_id, set()).add(f"{path}:{table}")
    for path in data_dir.rglob("*"):
        if not path.is_file() or not _is_json_identity_source(data_dir, path):
            continue
        discovered: set[str] = set()
        name = path.name
        marker = next(
            (
                suffix
                for suffix in (".jsonl.archived.", ".jsonl", ".json")
                if suffix in name
            ),
            None,
        )
        relative_parts = path.relative_to(data_dir).parts
        path_looks_like_session_store = ".jsonl" in name or (
            relative_parts[0] in {"sessions", "archives"}
        )
        if marker is not None and path_looks_like_session_store:
            prefix = name.split(marker, 1)[0]
            if prefix and prefix not in {"tasks", "turn_states", "chat_types"}:
                discovered.add(prefix)
        try:
            if ".jsonl" in name:
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        discovered.update(_json_identity_values(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
            else:
                discovered.update(
                    _json_identity_values(json.loads(path.read_text(encoding="utf-8")))
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        discovered = {
            value for value in discovered if value and not value.startswith("agent:")
        }
        if not discovered:
            continue
        sources.append(
            {
                "path": str(path),
                "format": "jsonl" if ".jsonl" in name else "json",
                "chat_id_count": len(discovered),
                "chat_id_hash": hashlib.sha256(
                    "\n".join(sorted(discovered)).encode("utf-8")
                ).hexdigest(),
            }
        )
        for chat_id in discovered:
            chat_sources.setdefault(chat_id, set()).add(str(path))
    return sources, chat_sources, _load_chat_type_hints(data_dir)


def _infer_chat_types(
    chat_id: str, sources: set[str], hints: dict[str, set[str]]
) -> set[str]:
    if chat_id.startswith(INTERNAL_PREFIXES):
        return set()
    return set(hints.get(chat_id, set()))


def audit(data_dir: Path, output: Path | None = None) -> dict[str, Any]:
    sources, chat_sources, type_hints = _discover_sources(data_dir)
    sessions = []
    for chat_id in sorted(chat_sources):
        if chat_id.startswith(INTERNAL_PREFIXES):
            sessions.append(
                {
                    "legacy_key": chat_id,
                    "kind": "internal",
                    "sources": sorted(chat_sources[chat_id]),
                }
            )
            continue
        chat_types = sorted(
            _infer_chat_types(chat_id, chat_sources[chat_id], type_hints)
        )
        if not chat_types:
            sessions.append(
                {
                    "legacy_key": chat_id,
                    "kind": "chat",
                    "chat_type": None,
                    "manual_chat_type_required": True,
                    "sources": sorted(chat_sources[chat_id]),
                }
            )
            continue
        for chat_type in chat_types:
            collision = len(chat_types) > 1
            sessions.append(
                {
                    "legacy_key": chat_id,
                    "kind": "chat",
                    "chat_type": chat_type,
                    "sources": sorted(chat_sources[chat_id]),
                    "candidate_types": chat_types,
                    "ambiguous": collision,
                    "manual_hindsight_binding_required": collision,
                    "manual_store_binding_required": collision,
                    "ambiguity_reason": (
                        "same_raw_id_has_group_and_direct_candidates"
                        if collision
                        else ""
                    ),
                }
            )
    report = {
        "schema_version": 1,
        "created_at": time.time(),
        "data_dir": str(data_dir),
        "sources": sources,
        "sessions": sessions,
        "ambiguous_count": sum(
            1
            for session in sessions
            if session.get("kind") == "chat"
            and (
                not session.get("chat_type")
                or session.get("manual_hindsight_binding_required")
                or session.get("manual_store_binding_required")
            )
        ),
    }
    if output is not None:
        _write_json(output, report)
    return report


def _prompt_chat_type(session: dict[str, Any]) -> str:
    legacy_key = session["legacy_key"]
    while True:
        answer = (
            input(f"旧 chat_id {legacy_key!r} 是 group 还是 direct？[g/d，q 退出] ")
            .strip()
            .lower()
        )
        if answer in {"g", "group"}:
            return "group"
        if answer in {"d", "direct", "private", "c2c"}:
            return "direct"
        if answer in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        print("请输入 group 或 direct。")


def _prompt_hindsight_document(session: dict[str, Any]) -> str | None:
    """Ask for the legacy document when one raw id has two chat types.

    Hindsight documents are append identities.  ``session-<raw-id>`` is not
    enough to distinguish a group and a direct conversation, so an empty
    answer deliberately leaves the candidate blocked instead of guessing.
    """
    legacy_key = session["legacy_key"]
    chat_type = session["chat_type"]
    while True:
        answer = input(
            f"旧 chat_id {legacy_key!r} ({chat_type}) 对应哪个 Hindsight document_id？"
            "[输入 ID，留空标记人工绑定，q 退出] "
        ).strip()
        if answer.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if answer:
            return answer
        return None


def plan(
    data_dir: Path,
    run_id: str | None,
    audit_path: Path | None,
    output: Path | None,
    *,
    allow_orphaned_task_sessions: bool = False,
) -> dict[str, Any]:
    report = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path
        else audit(data_dir)
    )
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    mappings = []
    blockers: list[dict[str, Any]] = []
    orphaned_legacy_sessions: list[dict[str, Any]] = []
    task_job_hints = _load_task_job_hints(data_dir)
    for session in report["sessions"]:
        if session["kind"] == "internal":
            legacy_key = session["legacy_key"]
            if legacy_key.startswith("heartbeat:"):
                canonical_key = build_internal_session_key(
                    "heartbeat", legacy_key.removeprefix("heartbeat:") or "events"
                )
                session_kind = "heartbeat"
            elif legacy_key.startswith("work-plan:"):
                canonical_key = build_internal_session_key(
                    "work-plan", legacy_key.removeprefix("work-plan:") or "unknown"
                )
                session_kind = "work_plan"
            elif legacy_key.startswith("cron:"):
                name = legacy_key.removeprefix("cron:") or "main"
                canonical_key = build_internal_session_key(
                    "cron", "main" if name == "main" else "custom", name
                )
                session_kind = "cron"
            elif legacy_key.startswith("task:"):
                task_id = legacy_key.removeprefix("task:") or "unknown"
                job_ids = task_job_hints.get(task_id, set())
                if len(job_ids) != 1:
                    blocker = {
                        "kind": "task_job_binding_required",
                        "legacy_key": legacy_key,
                        "job_ids": sorted(job_ids),
                        "reason": "task session requires exactly one owning cron job",
                    }
                    if allow_orphaned_task_sessions and not job_ids:
                        orphaned_legacy_sessions.append(
                            {
                                **session,
                                "retention": "legacy_read_only",
                                "reason": "owning cron job record is unavailable",
                            }
                        )
                        continue
                    blockers.append(blocker)
                    canonical_key = build_internal_session_key(
                        "cron", "unbound", task_id
                    )
                else:
                    canonical_key = build_internal_session_key(
                        "cron", "job", next(iter(job_ids)), "run", task_id
                    )
                session_kind = "cron"
            else:
                mappings.append({**session, "canonical_key": legacy_key})
                continue
            mappings.append(
                {
                    **session,
                    "canonical_key": canonical_key,
                    "session_kind": session_kind,
                    "confirmed_by": os.environ.get("USER", "unknown"),
                    "confirmed_at": time.time(),
                }
            )
            continue
        chat_type = session.get("chat_type") or _prompt_chat_type(session)
        session = {**session, "chat_type": chat_type}
        hindsight_document_id = f"session-{session['legacy_key']}"
        if session.get("manual_hindsight_binding_required"):
            hindsight_document_id = _prompt_hindsight_document(session) or ""
            if not hindsight_document_id:
                blockers.append(
                    {
                        "kind": "manual_hindsight_binding_required",
                        "legacy_key": session["legacy_key"],
                        "chat_type": chat_type,
                    }
                )
        if session.get("manual_store_binding_required"):
            blockers.append(
                {
                    "kind": "manual_store_binding_required",
                    "legacy_key": session["legacy_key"],
                    "chat_type": chat_type,
                    "reason": session.get("ambiguity_reason", ""),
                }
            )
        target = DeliveryTarget("qq", "default", chat_type, session["legacy_key"])
        mappings.append(
            {
                **session,
                "chat_type": chat_type,
                "channel": target.channel,
                "account_id": target.account_id,
                "canonical_key": build_chat_session_key(target),
                "workspace_slug": build_workspace_slug(target),
                "legacy_document_id": hindsight_document_id,
                "confirmed_by": os.environ.get("USER", "unknown"),
                "confirmed_at": time.time(),
            }
        )
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": time.time(),
        "data_dir": str(data_dir),
        "source_report": str(audit_path) if audit_path else None,
        "mappings": mappings,
        "blockers": blockers,
        "orphaned_legacy_sessions": orphaned_legacy_sessions,
        "allow_orphaned_task_sessions": allow_orphaned_task_sessions,
        "state": "blocked" if blockers else "planned",
    }
    if output is None:
        output = data_dir / "migrations" / run_id / "manifest.json"
    _write_json(output, manifest)
    return manifest


def mark_canonical_cutover(
    manifest_path: Path, hindsight_plan_path: Path
) -> dict[str, Any]:
    """Mark one verified local and Hindsight migration run as cut over."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hindsight_plan = json.loads(hindsight_plan_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "verified":
        raise RuntimeError("local migration must be verified before cutover")
    if hindsight_plan.get("state") != "verified":
        raise RuntimeError("Hindsight tag migration must be verified before cutover")
    if hindsight_plan.get("source_run_id") != manifest.get("run_id"):
        raise RuntimeError("Hindsight plan belongs to a different migration run")
    manifest["state"] = "canonical_cutover"
    manifest["cutover_at"] = time.time()
    _write_json(manifest_path, manifest)
    return manifest


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)


def _running_bot_processes() -> list[dict[str, Any]]:
    """Return likely bot processes, excluding this migration process.

    The migration is intentionally conservative: a process whose command line
    mentions ``main.py`` is treated as the bot.  On non-Linux hosts ``/proc``
    is absent and the explicit ``--confirm-stopped`` flag remains the guard.
    """
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        if not command or "main.py" not in command:
            continue
        found.append({"pid": int(entry.name), "command": command[:240]})
    return found


def _scan_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def preflight(data_dir: Path) -> dict[str, Any]:
    """Inspect persisted runtime state before an offline identity rewrite.

    This check is deliberately read-only and fail-closed.  It reports only
    identifiers/counts, never message or model content, so callers can show it
    to an operator before taking a maintenance window.
    """
    blockers: list[dict[str, Any]] = []
    observations: dict[str, Any] = {}

    task_records: list[dict[str, Any]] = []
    for path in (data_dir / "tasks" / "tasks.json", data_dir / "tasks.json"):
        if path.exists():
            task_records.extend(_scan_json_records(path))
    active_tasks = [
        str(item.get("id") or "unknown")
        for item in task_records
        if str(item.get("status") or "").lower() in _ACTIVE_TASK_STATUSES
    ]
    observations["active_tasks"] = len(active_tasks)
    if active_tasks:
        blockers.append({"kind": "active_tasks", "count": len(active_tasks)})

    turn_states = _scan_json_records(data_dir / "tasks" / "turn_states.json")
    active_turns = [
        str(item.get("turn_id") or "unknown")
        for item in turn_states
        if str(item.get("phase") or "").lower() in _ACTIVE_TURN_PHASES
    ]
    observations["active_turns"] = len(active_turns)
    if active_turns:
        blockers.append({"kind": "active_turns", "count": len(active_turns)})

    pending_archives: list[str] = []
    for path in sorted(data_dir.rglob("manifests/*.json")):
        if "migrations" in path.relative_to(data_dir).parts:
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pending_archives.append(str(path))
            continue
        if isinstance(manifest, dict) and manifest.get("state") != "committed":
            pending_archives.append(str(path))
    observations["pending_archive_manifests"] = len(pending_archives)
    if pending_archives:
        blockers.append(
            {"kind": "pending_archive_manifests", "count": len(pending_archives)}
        )

    delivery_rows = 0
    for path in _sqlite_paths(data_dir):
        if "migrations" in path.relative_to(data_dir).parts:
            continue
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='delivery_ledger'"
            ).fetchone()
            if table is None:
                connection.close()
                continue
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info('delivery_ledger')"
                ).fetchall()
            }
            if "status" not in columns:
                connection.close()
                continue
            predicate = "status IN ('prepared', 'unknown', 'partial')"
            if "receipt_status" in columns:
                predicate += " OR receipt_status IN ('unknown', 'partial')"
            rows = connection.execute(
                f"SELECT COUNT(*) FROM delivery_ledger WHERE {predicate}"
            ).fetchone()
            connection.close()
        except sqlite3.Error:
            continue
        count = int(rows[0]) if rows else 0
        delivery_rows += count
    observations["blocking_delivery_receipts"] = delivery_rows
    if delivery_rows:
        blockers.append({"kind": "blocking_delivery_receipts", "count": delivery_rows})

    bot_processes = _running_bot_processes()
    observations["bot_processes"] = len(bot_processes)
    if bot_processes:
        blockers.append({"kind": "bot_processes", "count": len(bot_processes)})

    return {"ok": not blockers, "blockers": blockers, "observations": observations}


def _require_preflight_clear(data_dir: Path) -> dict[str, Any]:
    result = preflight(data_dir)
    if not result["ok"]:
        details = ", ".join(
            f"{item['kind']}={item['count']}" for item in result["blockers"]
        )
        raise RuntimeError(f"offline migration preflight failed: {details}")
    return result


@contextmanager
def _maintenance_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"maintenance lock is already held: {path}") from exc
        yield handle
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _manifest_mappings(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("state") == "blocked" or manifest.get("blockers"):
        blockers = ", ".join(
            str(item.get("kind", "unknown"))
            for item in manifest.get("blockers", [])
            if isinstance(item, dict)
        )
        raise RuntimeError(
            "migration manifest has unresolved manual bindings"
            + (f": {blockers}" if blockers else "")
        )
    mappings = [
        item
        for item in manifest.get("mappings", [])
        if item.get("kind") in {"chat", "internal"}
    ]
    by_legacy: dict[str, dict[str, Any]] = {}
    for mapping in mappings:
        legacy = str(mapping.get("legacy_key") or "")
        canonical = str(mapping.get("canonical_key") or "")
        if not legacy or not canonical:
            raise ValueError("incomplete chat mapping")
        if (
            mapping.get("kind") == "chat"
            and mapping.get("manual_hindsight_binding_required")
            and not mapping.get("legacy_document_id")
        ):
            raise ValueError(
                f"missing Hindsight document binding: {legacy}/{mapping.get('chat_type')}"
            )
        if legacy in by_legacy and by_legacy[legacy]["canonical_key"] != canonical:
            raise ValueError(
                f"ambiguous legacy mapping requires manual store binding: {legacy}"
            )
        by_legacy[legacy] = mapping
    return by_legacy


def _table_identity_hash(
    connection: sqlite3.Connection,
    table: str,
    columns: list[str],
    chat_id: str,
    replacement: str,
) -> str:
    """Hash complete rows while normalizing only the partition key.

    The resulting digest makes verify sensitive to event/message identity,
    sequence, archive metadata, and ledger fields without storing message
    content in the migration journal.
    """
    try:
        rows = connection.execute(
            f'SELECT * FROM "{table}" WHERE chat_id = ? ORDER BY rowid',
            (chat_id,),
        ).fetchall()
    except sqlite3.Error:
        rows = connection.execute(
            f'SELECT * FROM "{table}" WHERE chat_id = ?',
            (chat_id,),
        ).fetchall()
    chat_index = columns.index("chat_id")
    normalized = []
    for row in rows:
        values = list(row)
        values[chat_index] = replacement
        normalized.append(values)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sqlite_rewrite(
    path: Path, mapping: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        tables = _sqlite_tables(path)
        with connection:
            for table, columns in tables:
                if "chat_id" not in columns or table in _RAW_TARGET_TABLES:
                    continue
                for legacy, item in mapping.items():
                    canonical = str(item["canonical_key"])
                    if legacy == canonical:
                        continue
                    count = int(
                        connection.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE chat_id = ?',
                            (legacy,),
                        ).fetchone()[0]
                    )
                    if not count:
                        continue
                    conflict = int(
                        connection.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE chat_id = ?',
                            (canonical,),
                        ).fetchone()[0]
                    )
                    if conflict:
                        raise RuntimeError(
                            f"canonical key already exists in {path}:{table}: {canonical}"
                        )
                    identity_hash = _table_identity_hash(
                        connection, table, columns, legacy, canonical
                    )
                    connection.execute(
                        f'UPDATE "{table}" SET chat_id = ? WHERE chat_id = ?',
                        (canonical, legacy),
                    )
                    changes.append(
                        {
                            "path": str(path),
                            "table": table,
                            "legacy_key": legacy,
                            "canonical_key": canonical,
                            "rows": count,
                            "identity_hash": identity_hash,
                        }
                    )
    finally:
        connection.close()
    return changes


def _rename_workspaces(
    workspace_root: Path, mapping: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    renames: list[dict[str, Any]] = []
    for legacy, item in mapping.items():
        if item.get("kind") != "chat":
            continue
        chat_type = item["chat_type"]
        source = (
            workspace_root / ("groups" if chat_type == "group" else "private") / legacy
        )
        destination = workspace_root / str(item["workspace_slug"])
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"workspace source is not a directory: {source}")
        if destination.exists():
            raise RuntimeError(f"workspace destination already exists: {destination}")
        source_tree = _tree_fingerprint(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        destination_tree = _tree_fingerprint(destination)
        renames.append(
            {
                "source": str(source),
                "destination": str(destination),
                "source_tree": source_tree,
                "destination_tree": destination_tree,
            }
        )
    return renames


def _tree_fingerprint(path: Path) -> dict[str, Any]:
    """Return a stable recursive fingerprint for a workspace directory."""
    if not path.exists() or not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"workspace path is not a regular directory: {path}")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise RuntimeError(f"workspace contains symlink: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        files.append({"path": relative, "size": size, "sha256": _hash_file(child)})
        total_bytes += size
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "file_count": len(files),
        "byte_count": total_bytes,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def _rewrite_json_identity(value: Any, mapping: dict[str, dict[str, Any]]) -> int:
    changed = 0
    if isinstance(value, list):
        for item in value:
            changed += _rewrite_json_identity(item, mapping)
        return changed
    if not isinstance(value, dict):
        return changed
    for field, item in value.items():
        if field in _JSON_SESSION_FIELDS and isinstance(item, str):
            target = mapping.get(item)
            if target is not None and target["canonical_key"] != item:
                value[field] = target["canonical_key"]
                changed += 1
            continue
        changed += _rewrite_json_identity(item, mapping)
    return changed


def _json_rewrite(path: Path, mapping: dict[str, dict[str, Any]]) -> int:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    changed = _rewrite_json_file_value(path, value, mapping)
    if not changed:
        return 0
    tmp = path.with_suffix(path.suffix + ".migration.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)
    return changed


def _rewrite_json_file_value(
    path: Path, value: Any, mapping: dict[str, dict[str, Any]]
) -> int:
    changed = _rewrite_json_identity(value, mapping)
    if path.name != "chat_types.json" or not isinstance(value, dict):
        return changed
    for legacy, item in mapping.items():
        canonical = str(item["canonical_key"])
        if legacy not in value or canonical == legacy:
            continue
        if canonical in value:
            raise RuntimeError(f"chat type destination already exists: {canonical}")
        value[canonical] = value.pop(legacy)
        changed += 1
    return changed


def _jsonl_rewrite(
    path: Path, mapping: dict[str, dict[str, Any]], *, write: bool = True
) -> tuple[int, int]:
    """Rewrite identity fields in JSONL while preserving malformed lines."""
    try:
        original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeDecodeError):
        return 0, 0
    changed = 0
    parsed_lines = 0
    rewritten: list[str] = []
    for line in original_lines:
        stripped = line.strip()
        if not stripped:
            rewritten.append(line)
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            rewritten.append(line)
            continue
        parsed_lines += 1
        fields = _rewrite_json_identity(value, mapping)
        if fields:
            changed += fields
            rewritten.append(json.dumps(value, ensure_ascii=False) + "\n")
        else:
            rewritten.append(line)
    if changed and write:
        tmp = path.with_suffix(path.suffix + ".migration.tmp")
        tmp.write_text("".join(rewritten), encoding="utf-8")
        os.replace(tmp, path)
    return changed, parsed_lines


def _safe_path_component(value: str) -> str:
    if value and all(char.isalnum() or char in "-_." for char in value):
        return value
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _migrated_file_path(
    path: Path,
    mapping: dict[str, dict[str, Any]],
    identity_values: set[str] | None = None,
) -> Path:
    """Return a path with an exact legacy session component replaced."""
    identity_values = identity_values or set()
    for legacy, item in mapping.items():
        canonical = str(item["canonical_key"])
        if legacy == canonical:
            continue
        destination = path
        parts = list(path.parts)
        replaced = False
        for index, part in enumerate(parts):
            if part == legacy:
                parts[index] = canonical
                replaced = True
        name = path.name
        for suffix in (".jsonl.archived.", ".jsonl", ".json"):
            prefix = f"{legacy}{suffix}"
            if name.startswith(prefix):
                parts[-1] = f"{canonical}{name[len(legacy):]}"
                replaced = True
                break
        if path.name == _safe_path_component(legacy):
            parts[-1] = _safe_path_component(canonical)
            replaced = True
        elif path.parent.name == _safe_path_component(legacy):
            parts[-2] = _safe_path_component(canonical)
            replaced = True
        if not replaced and legacy not in identity_values:
            continue
        destination = Path(*parts)
        if destination != path:
            return destination
    return path


def _json_identity_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, list):
        for item in value:
            values.extend(_json_identity_values(item))
    elif isinstance(value, dict):
        for field, item in value.items():
            if field in _JSON_SESSION_FIELDS and isinstance(item, str):
                values.append(item)
            elif isinstance(item, (dict, list)):
                values.extend(_json_identity_values(item))
    return values


def apply_manifest(
    manifest_path: Path,
    *,
    confirm_stopped: bool = False,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    if not confirm_stopped:
        raise RuntimeError("apply requires --confirm-stopped")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") not in {"planned", "registry_applied"}:
        raise ValueError(
            f"manifest is not applicable in state={manifest.get('state')!r}"
        )
    mapping = _manifest_mappings(manifest)
    data_dir = Path(manifest["data_dir"])
    run_dir = manifest_path.parent
    backup_dir = run_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    journal: dict[str, Any] = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "preflight": None,
        "backups": [],
        "sqlite_changes": [],
        "json_changes": [],
        "file_changes": [],
        "workspace_renames": [],
    }
    with _maintenance_lock(run_dir / "maintenance.lock"):
        journal["preflight"] = _require_preflight_clear(data_dir)
        journal_path = run_dir / "journal.json"
        _write_json(journal_path, journal)
        manifest["journal"] = str(journal_path)
        manifest["state"] = "applying"
        _write_json(manifest_path, manifest)
        for path in _sqlite_paths(data_dir):
            if "migrations" in path.relative_to(data_dir).parts:
                continue
            if path.resolve() == (data_dir / "session_identity.sqlite3").resolve():
                continue
            relative = path.relative_to(data_dir)
            backup = backup_dir / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            journal["backups"].append({"source": str(path), "backup": str(backup)})
            journal["sqlite_changes"].extend(_sqlite_rewrite(path, mapping))
            _write_json(journal_path, journal)
        file_paths = sorted(
            path
            for path in data_dir.rglob("*")
            if path.is_file() and _is_json_identity_source(data_dir, path)
        )
        for path in file_paths:
            if "migrations" in path.relative_to(data_dir).parts:
                continue
            try:
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            identity_values: set[str] = set()
            fields = 0
            lines = 0
            is_jsonl = ".jsonl" in path.name
            if is_jsonl:
                for line in original.splitlines():
                    try:
                        identity_values.update(_json_identity_values(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
            else:
                try:
                    probe = json.loads(original)
                except json.JSONDecodeError:
                    continue
                identity_values.update(_json_identity_values(probe))
            destination = _migrated_file_path(path, mapping, identity_values)
            if is_jsonl:
                fields, lines = _jsonl_rewrite(path, mapping, write=False)
            else:
                try:
                    probe = json.loads(original)
                except json.JSONDecodeError:
                    continue
                fields = _rewrite_json_file_value(path, probe, mapping)
            if not fields and destination == path:
                continue
            relative = path.relative_to(data_dir)
            backup = backup_dir / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            if is_jsonl and fields:
                _jsonl_rewrite(path, mapping)
            elif not is_jsonl and path.suffix != ".json" and fields:
                raise AssertionError("unsupported migration file format")
            elif path.suffix == ".json":
                _json_rewrite(path, mapping)
            if destination != path:
                if destination.exists():
                    raise RuntimeError(
                        f"file destination already exists: {destination}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
            change = {
                "source": str(path),
                "destination": str(destination),
                "format": "jsonl" if is_jsonl else "json",
                "fields": fields,
                "lines": lines,
                "source_sha256": _hash_file(backup),
                "destination_sha256": _hash_file(destination),
            }
            journal["backups"].append(
                {
                    "source": str(path),
                    "destination": str(destination),
                    "backup": str(backup),
                    "existed": True,
                    "renamed": destination != path,
                }
            )
            journal["file_changes"].append(change)
            if not is_jsonl and path.suffix == ".json":
                journal["json_changes"].append(
                    {
                        "path": str(destination),
                        "fields": fields,
                        "sha256": change["destination_sha256"],
                    }
                )
            _write_json(journal_path, journal)
        registry_path = data_dir / "session_identity.sqlite3"
        if registry_path.exists():
            backup = backup_dir / "session_identity.sqlite3"
            shutil.copy2(registry_path, backup)
            journal["backups"].append(
                {"source": str(registry_path), "backup": str(backup), "existed": True}
            )
        else:
            journal["backups"].append(
                {"source": str(registry_path), "backup": "", "existed": False}
            )
        _write_json(journal_path, journal)
        registry = SessionIdentityRegistry(registry_path)
        try:
            for item in manifest.get("mappings", []):
                if item.get("kind") == "chat":
                    target = DeliveryTarget(
                        item.get("channel", "qq"),
                        item.get("account_id", "default"),
                        item["chat_type"],
                        item["legacy_key"],
                    )
                    registry.register_chat(
                        target,
                        legacy_key=item["legacy_key"],
                        workspace_slug=item.get("workspace_slug"),
                        hindsight_document_id=item.get("legacy_document_id")
                        or f"session-{item['legacy_key']}",
                        state="copied",
                        migration_run_id=manifest["run_id"],
                    )
                elif item.get("kind") == "internal":
                    registry.register_internal(
                        item["canonical_key"],
                        session_kind=item.get("session_kind", "cron"),
                        legacy_key=item.get("legacy_key"),
                        state="copied",
                        migration_run_id=manifest["run_id"],
                    )
        finally:
            registry.close()
        if workspace_root is not None:
            journal["workspace_renames"] = _rename_workspaces(workspace_root, mapping)
            manifest["workspace_root"] = str(workspace_root)
        _write_json(journal_path, journal)
        manifest["state"] = "copied"
        _write_json(manifest_path, manifest)
    return manifest


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    journal_path = Path(
        manifest.get("journal") or manifest_path.parent / "journal.json"
    )
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    data_dir = Path(manifest["data_dir"])
    with _maintenance_lock(manifest_path.parent / "maintenance.lock"):
        for change in journal.get("sqlite_changes", []):
            connection = sqlite3.connect(change["path"])
            try:
                columns = [
                    row[1]
                    for row in connection.execute(
                        f'PRAGMA table_info("{change["table"]}")'
                    ).fetchall()
                ]
                old_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{change["table"]}" WHERE chat_id = ?',
                    (change["legacy_key"],),
                ).fetchone()[0]
                new_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{change["table"]}" WHERE chat_id = ?',
                    (change["canonical_key"],),
                ).fetchone()[0]
                identity_hash = ""
                if columns and "identity_hash" in change:
                    identity_hash = _table_identity_hash(
                        connection,
                        change["table"],
                        columns,
                        change["canonical_key"],
                        change["canonical_key"],
                    )
                if (
                    old_count
                    or new_count != change["rows"]
                    or (identity_hash and identity_hash != change["identity_hash"])
                ):
                    failures.append(f"sqlite:{change['path']}:{change['table']}")
            finally:
                connection.close()
        for rename in journal.get("workspace_renames", []):
            destination = Path(rename["destination"])
            if not destination.exists():
                failures.append(f"workspace:{rename['destination']}")
                continue
            try:
                current_tree = _tree_fingerprint(destination)
            except RuntimeError:
                failures.append(f"workspace:{rename['destination']}")
                continue
            if current_tree != rename.get("destination_tree"):
                failures.append(f"workspace_hash:{rename['destination']}")
        for change in journal.get("file_changes", []):
            destination = Path(change["destination"])
            if not destination.exists():
                failures.append(f"file:{destination}")
                continue
            if (
                change.get("destination_sha256")
                and _hash_file(destination) != change["destination_sha256"]
            ):
                failures.append(f"file_hash:{destination}")
            if change.get("format") == "jsonl":
                try:
                    for line in destination.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            json.loads(line)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    failures.append(f"jsonl:{destination}")
        mapping = _manifest_mappings(manifest)
        for change in journal.get("json_changes", []):
            path = Path(change["path"])
            try:
                values = _json_identity_values(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                failures.append(f"json:{path}")
                continue
            if any(value in values for value in mapping):
                failures.append(f"json:{path}")
            if change.get("sha256") and _hash_file(path) != change["sha256"]:
                failures.append(f"json_hash:{path}")
        registry_path = data_dir / "session_identity.sqlite3"
        if not registry_path.exists():
            failures.append(f"registry:{registry_path}")
        else:
            registry = SessionIdentityRegistry(registry_path)
            try:
                for item in manifest.get("mappings", []):
                    if item.get("kind") in {"chat", "internal"}:
                        try:
                            registry.set_state(item["canonical_key"], "verified")
                        except KeyError:
                            failures.append(f"registry:{item['canonical_key']}")
            finally:
                registry.close()
        result = {
            "state": "verified" if not failures else "verify_failed",
            "failures": failures,
        }
        if not failures:
            manifest["state"] = "verified"
            _write_json(manifest_path, manifest)
        return result


def rollback_manifest(
    manifest_path: Path, *, confirm_stopped: bool = False
) -> dict[str, Any]:
    if not confirm_stopped:
        raise RuntimeError("rollback requires --confirm-stopped")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    journal_path = Path(
        manifest.get("journal") or manifest_path.parent / "journal.json"
    )
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    with _maintenance_lock(manifest_path.parent / "maintenance.lock"):
        for rename in reversed(journal.get("workspace_renames", [])):
            source = Path(rename["source"])
            destination = Path(rename["destination"])
            if destination.exists() and not source.exists():
                expected = rename.get("destination_tree")
                if expected and _tree_fingerprint(destination) != expected:
                    raise RuntimeError(
                        f"workspace changed after migration; refusing rollback: {destination}"
                    )
                destination.rename(source)
        for backup in reversed(journal.get("backups", [])):
            destination = Path(backup.get("destination") or backup["source"])
            source = Path(backup["source"])
            if backup.get("renamed") and destination.exists() and destination != source:
                expected = next(
                    (
                        item.get("destination_sha256")
                        for item in journal.get("file_changes", [])
                        if item.get("destination") == str(destination)
                    ),
                    None,
                )
                if expected and _hash_file(destination) != expected:
                    raise RuntimeError(
                        f"file changed after migration; refusing rollback: {destination}"
                    )
                destination.unlink()
            if backup.get("existed", True):
                shutil.copy2(backup["backup"], backup["source"])
            elif Path(backup["source"]).exists():
                Path(backup["source"]).unlink()
        manifest["state"] = "rolled_back"
        _write_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "plan", "preflight"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, default=Path("data"))
        subparser.add_argument("--output", type=Path)
    plan_parser = subparsers.choices["plan"]
    plan_parser.add_argument("--run-id")
    plan_parser.add_argument("--audit", type=Path)
    plan_parser.add_argument(
        "--allow-orphaned-task-sessions",
        action="store_true",
        help="保留无法绑定 job_id 的 task session 为 legacy，只迁移可确认记录",
    )
    for command in ("apply", "verify", "rollback"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--manifest", type=Path, required=True)
        subparser.add_argument("--confirm-stopped", action="store_true")
        if command == "apply":
            subparser.add_argument("--workspace-root", type=Path)
    cutover_parser = subparsers.add_parser("cutover")
    cutover_parser.add_argument("--manifest", type=Path, required=True)
    cutover_parser.add_argument("--hindsight-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        report = audit(args.data_dir, args.output)
        print(
            json.dumps(
                {
                    "sessions": len(report["sessions"]),
                    "ambiguous": report["ambiguous_count"],
                }
            )
        )
        return 0
    if args.command == "preflight":
        result = preflight(args.data_dir)
        if args.output is not None:
            _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ok"] else 2
    if args.command == "plan":
        try:
            manifest = plan(
                args.data_dir,
                args.run_id,
                args.audit,
                args.output,
                allow_orphaned_task_sessions=args.allow_orphaned_task_sessions,
            )
        except KeyboardInterrupt:
            print(
                "plan cancelled; no executable manifest was produced",
                file=sys.stderr,
            )
            return 2
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "mappings": len(manifest["mappings"]),
                    "orphaned_legacy_sessions": len(
                        manifest.get("orphaned_legacy_sessions", [])
                    ),
                    "state": manifest["state"],
                    "blockers": len(manifest.get("blockers", [])),
                },
                ensure_ascii=False,
            )
        )
        return 0 if manifest["state"] == "planned" else 2
    try:
        if args.command == "cutover":
            result = mark_canonical_cutover(args.manifest, args.hindsight_plan)
            print(json.dumps({"run_id": result["run_id"], "state": result["state"]}))
            return 0
        if args.command == "apply":
            result = apply_manifest(
                args.manifest,
                confirm_stopped=args.confirm_stopped,
                workspace_root=args.workspace_root,
            )
            print(json.dumps({"run_id": result["run_id"], "state": result["state"]}))
            return 0
        if args.command == "verify":
            result = verify_manifest(args.manifest)
            print(json.dumps(result))
            return 0 if result["state"] == "verified" else 2
        result = rollback_manifest(
            args.manifest,
            confirm_stopped=args.confirm_stopped,
        )
        print(json.dumps({"run_id": result["run_id"], "state": result["state"]}))
        return 0
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
