"""Create a content-free audit for legacy archive identity collisions."""

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_content(record: dict[str, Any]) -> str:
    content = str(record.get("raw_content", record.get("content", "")) or "")
    if record.get("role") != "user" or "raw_content" in record:
        return content
    while content.startswith("[") and "]:" in content[:80]:
        prefix, separator, remaining = content.partition("]: ")
        if not separator:
            break
        content = remaining
    return content


def _legacy_candidate_id(record: dict[str, Any]) -> str:
    role = str(record.get("role") or "")
    event_id = str(record.get("event_id") or "")
    if event_id:
        return event_id
    record_id = str(record.get("record_id") or "")
    if record_id:
        return f"legacy:record:{record_id}:{role}"
    message_id = str(record.get("message_id") or "")
    if message_id:
        return f"{role}:{message_id}"
    tool_call_id = str(record.get("tool_call_id") or "")
    if role == "tool" and tool_call_id:
        return f"tool:{tool_call_id}"
    return ""


def _legacy_kind(record: dict[str, Any]) -> str:
    role = str(record.get("role") or "")
    if role == "user":
        return "user_message"
    if role == "tool":
        return "tool_result"
    if role == "assistant" and record.get("tool_calls"):
        return "assistant_tool_call"
    if role == "assistant":
        return "accepted_delivery"
    return "unknown"


def _record_metadata(
    record: dict[str, Any], *, source_file: str, record_index: int
) -> dict[str, Any]:
    content = _normalized_content(record)
    tool_calls = record.get("tool_calls") or []
    reasoning = str(record.get("reasoning_content") or "")
    return {
        "source_file": source_file,
        "record_index": record_index,
        "role": str(record.get("role") or ""),
        "kind": _legacy_kind(record),
        "timestamp": record.get("timestamp"),
        "message_id": str(record.get("message_id") or ""),
        "tool_call_id": str(record.get("tool_call_id") or ""),
        "tool_name": str(record.get("tool_name") or ""),
        "content_length": len(content),
        "content_sha256": _sha256(content),
        "tool_calls_sha256": _sha256(tool_calls),
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "reasoning_length": len(reasoning),
        "reasoning_sha256": _sha256(reasoning),
    }


def _event_metadata(row: sqlite3.Row) -> dict[str, Any]:
    content = str(row["content"] or "")
    tool_calls = str(row["tool_calls"] or "[]")
    reasoning = str(row["reasoning_content"] or "")
    try:
        tool_call_count = len(json.loads(tool_calls))
    except (TypeError, json.JSONDecodeError):
        tool_call_count = 0
    return {
        "event_id": str(row["event_id"]),
        "event_seq": int(row["event_seq"]),
        "turn_id": str(row["turn_id"]),
        "role": str(row["role"]),
        "kind": str(row["kind"]),
        "timestamp": float(row["timestamp"]),
        "message_id": str(row["message_id"] or ""),
        "tool_call_id": str(row["tool_call_id"] or ""),
        "tool_name": str(row["tool_name"] or ""),
        "content_length": len(content),
        "content_sha256": _sha256(content),
        "tool_calls_sha256": _sha256(tool_calls),
        "tool_call_count": tool_call_count,
        "reasoning_length": len(reasoning),
        "reasoning_sha256": _sha256(reasoning),
        "_content": content,
    }


def _collision_base(event_id: str) -> str:
    return event_id.partition(":legacy-conflict:")[0]


def _is_legacy_inbound_mirror(contents: list[str]) -> bool:
    mirror_prefix = re.compile(r"^\[来自 .+? 的新消息\]:\s*")
    mirrored = [mirror_prefix.sub("", content, count=1) for content in contents]
    if not any(mirror_prefix.match(content) for content in contents):
        return False
    originals = [
        content for content in contents if not mirror_prefix.match(content) and content
    ]
    if not originals:
        return False
    if all(
        not value
        or any(value == original or value in original for original in originals)
        for value in mirrored
    ):
        return True
    return any("[媒体引用:" in content for content in originals) and all(
        not value or value in originals for value in mirrored
    )


def _classify(events: list[dict[str, Any]], *, chat_id: str) -> str:
    kinds = {event["kind"] for event in events}
    if kinds == {"assistant_tool_call", "accepted_delivery"}:
        return "assistant_id_reused_for_tool_call_and_delivery"
    if len(kinds) > 1:
        return "id_reused_across_event_kinds"
    fingerprints = {
        (
            event["kind"],
            event["content_sha256"],
            event["tool_calls_sha256"],
            event["tool_call_id"],
        )
        for event in events
    }
    if len(fingerprints) == 1:
        return "same_payload_replayed_with_different_metadata"
    if kinds == {"accepted_delivery"}:
        contents = [event["_content"] for event in events if event["_content"]]
        if any(
            left != right and (left.startswith(right) or right.startswith(left))
            for index, left in enumerate(contents)
            for right in contents[index + 1 :]
        ):
            return "progressive_or_revised_delivery_snapshot"
        return "distinct_assistant_delivery_payloads"
    if kinds == {"assistant_tool_call"}:
        if chat_id.startswith("task:"):
            return "task_multi_round_tool_calls"
        return "distinct_assistant_tool_call_payloads"
    if kinds == {"tool_result"}:
        return "distinct_tool_result_payloads"
    if kinds == {"user_message"}:
        contents = [event["_content"] for event in events]
        if _is_legacy_inbound_mirror(contents):
            return "legacy_inbound_mirror_snapshot"
        return "distinct_user_message_payloads"
    return "distinct_payloads"


def _without_content(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "_content"}


def _load_report_sources(report_path: Path) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [str(item) for item in report.get("source_files", [])]


def _read_source_records(
    sessions_dir: Path, source_files: list[str], collision_bases: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], int, list[str]]:
    records_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned_records = 0
    missing_files: list[str] = []
    for source_file in source_files:
        path = sessions_dir / source_file
        if not path.is_file():
            missing_files.append(source_file)
            continue
        with path.open(encoding="utf-8") as stream:
            for record_index, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                scanned_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                candidate_id = _legacy_candidate_id(record)
                if candidate_id in collision_bases:
                    records_by_base[candidate_id].append(
                        _record_metadata(
                            record,
                            source_file=source_file,
                            record_index=record_index,
                        )
                    )
    return records_by_base, scanned_records, missing_files


def _audit_chat(
    connection: sqlite3.Connection,
    *,
    chat_id: str,
    sessions_dir: Path,
    migration_report: Path,
) -> dict[str, Any]:
    source_files = _load_report_sources(migration_report)
    rows = connection.execute(
        "SELECT * FROM conversation_events "
        "WHERE chat_id = ? AND event_id LIKE ? ORDER BY event_seq",
        (chat_id, "%:legacy-conflict:%"),
    ).fetchall()
    conflicts_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conflicts_by_base[_collision_base(str(row["event_id"]))].append(
            _event_metadata(row)
        )
    canonical_by_base: dict[str, dict[str, Any] | None] = {}
    for base in conflicts_by_base:
        row = connection.execute(
            "SELECT * FROM conversation_events WHERE chat_id = ? AND event_id = ?",
            (chat_id, base),
        ).fetchone()
        canonical_by_base[base] = _event_metadata(row) if row is not None else None
    source_records, scanned_records, missing_files = _read_source_records(
        sessions_dir, source_files, set(conflicts_by_base)
    )
    classifications: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    for base in sorted(conflicts_by_base):
        canonical = canonical_by_base[base]
        events = list(conflicts_by_base[base])
        if canonical is not None:
            events.insert(0, canonical)
        classification = _classify(events, chat_id=chat_id)
        classifications[classification] += 1
        entries.append(
            {
                "canonical_event_id": base,
                "classification": classification,
                "canonical": _without_content(canonical) if canonical else None,
                "conflicts": [
                    _without_content(event) for event in conflicts_by_base[base]
                ],
                "legacy_records": source_records.get(base, []),
            }
        )
    return {
        "chat_id": chat_id,
        "content_included": False,
        "migration_report": str(migration_report),
        "source_file_count": len(source_files),
        "source_records_scanned": scanned_records,
        "missing_source_files": missing_files,
        "ledger_conflict_event_count": len(rows),
        "collision_group_count": len(entries),
        "classification_counts": dict(sorted(classifications.items())),
        "groups_without_source_record": sum(
            1 for entry in entries if not entry["legacy_records"]
        ),
        "collisions": entries,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-id")
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("--sessions-dir", required=True, type=Path)
    report_scope = parser.add_mutually_exclusive_group(required=True)
    report_scope.add_argument("--migration-report", type=Path)
    report_scope.add_argument("--migration-report-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.migration_report and not args.chat_id:
        parser.error("--chat-id is required with --migration-report")
    if args.migration_report_dir and args.chat_id:
        parser.error("--chat-id cannot be used with --migration-report-dir")

    connection = sqlite3.connect(f"file:{args.event_log.absolute()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if args.migration_report:
            payload = _audit_chat(
                connection,
                chat_id=args.chat_id,
                sessions_dir=args.sessions_dir,
                migration_report=args.migration_report,
            )
            payload["event_log"] = str(args.event_log)
            _write_json(args.output, payload)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "collisions"
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        reports: list[dict[str, Any]] = []
        classification_counts: Counter[str] = Counter()
        for report_path in sorted(args.migration_report_dir.glob("*.migration.json")):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                chat_id = str(report.get("chat_id") or "")
                if not chat_id:
                    raise ValueError("missing chat_id")
                audit = _audit_chat(
                    connection,
                    chat_id=chat_id,
                    sessions_dir=args.sessions_dir,
                    migration_report=report_path,
                )
                audit.pop("collisions")
                classification_counts.update(audit["classification_counts"])
                reports.append(audit)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                reports.append(
                    {"migration_report": str(report_path), "error": str(exc)}
                )
        payload = {
            "content_included": False,
            "event_log": str(args.event_log),
            "report_count": len(reports),
            "report_error_count": sum(1 for report in reports if "error" in report),
            "ledger_conflict_event_count": sum(
                int(report.get("ledger_conflict_event_count", 0)) for report in reports
            ),
            "collision_group_count": sum(
                int(report.get("collision_group_count", 0)) for report in reports
            ),
            "classification_counts": dict(sorted(classification_counts.items())),
            "reports": reports,
        }
        _write_json(args.output, payload)
        print(
            json.dumps(
                {key: value for key, value in payload.items() if key != "reports"},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
