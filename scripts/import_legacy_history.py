"""Offline importer: legacy JSONL sessions -> ConversationEventLog.

The runtime cold-start migration was retired.  Run this against a *stopped* bot
to import pre-ledger data once:

    uv run python scripts/import_legacy_history.py

It reads ``data/sessions/*.jsonl`` (active history) and
``*.jsonl.archived.*`` (legacy archive files), feeds them through
``ConversationEventLog.repair_from_legacy_history`` (idempotent), and records the
``legacy-active-v1`` watermark when every chat succeeds.

Limitations versus the retired runtime path:

- Legacy identity/nickname observation is not replayed here; nickname migration
  still runs at normal bootstrap (``identity_manager.migrate_legacy_nicknames``).
- Imported messages enter the ledger as events; archival/retention is then owned
  by the normal event-log archive path (``猫猫归档 修复 <date>`` is available for
  static repair).
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from core.config_loader import ConfigLoader
from core.engine.conversation_event_log import ConversationEventLog

_log = logging.getLogger("import_legacy_history")


def _load_jsonl(path: Path) -> list[dict]:
    messages: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    messages.append(record)
    except OSError as exc:
        _log.warning("读取失败 %s: %s", path, exc)
    return messages


def _collect_sources(sessions_dir: Path) -> dict[str, list[tuple[str, list[dict]]]]:
    """Return ``{chat_id: [(source_id, messages), ...]}``."""
    sources: dict[str, list[tuple[str, list[dict]]]] = {}
    if not sessions_dir.is_dir():
        return sources
    for path in sorted(sessions_dir.glob("*.jsonl")):
        if ".archived." in path.name:
            continue
        messages = _load_jsonl(path)
        if messages:
            sources.setdefault(path.stem, []).append(("legacy-active", messages))
    for path in sorted(sessions_dir.glob("*.jsonl.archived.*")):
        chat_id = path.name.split(".jsonl.archived.")[0]
        messages = _load_jsonl(path)
        if messages:
            sources.setdefault(chat_id, []).append(("legacy-archives", messages))
    return sources


async def import_legacy_history(
    sessions_dir: Path, event_log: ConversationEventLog, *, force: bool = False
) -> int:
    """Import every legacy JSONL session into ``event_log``. Idempotent."""
    if not force and await event_log.legacy_migration_is_complete():
        print("旧历史迁移水位已存在；如需重新导入请加 --force")
        return 0

    sources = _collect_sources(sessions_dir)
    if not sources:
        print(f"未发现 legacy JSONL: {sessions_dir}")
        return 0

    total_events = 0
    total_conflicts = 0
    failed = 0
    for index, chat_id in enumerate(sorted(sources), start=1):
        imported = 0
        try:
            for source_id, messages in sources[chat_id]:
                imported += await event_log.repair_from_legacy_history(
                    chat_id, messages, source_id=source_id
                )
        except Exception as exc:
            failed += 1
            _log.warning("导入失败 [%s..]: %s", chat_id[:12], exc)
            continue
        total_events += imported
        conflicts = 0
        conflict_reader = getattr(event_log, "legacy_conflict_event_ids", None)
        if callable(conflict_reader):
            try:
                conflicts = len(await conflict_reader(chat_id, max_ids=100))
            except Exception as exc:
                _log.warning("读取 identity 冲突失败 [%s..]: %s", chat_id[:12], exc)
        total_conflicts += conflicts
        suffix = f"，identity 冲突 {conflicts}" if conflicts else ""
        print(f"[{index}/{len(sources)}] {chat_id[:16]}..: +{imported} 事件{suffix}")

    if total_conflicts:
        print(f"警告：发现 {total_conflicts} 条 identity 冲突，需人工核对")
    if failed:
        print(f"部分失败：{failed} 会话未完成，未写入迁移水位")
        return 1
    await event_log.mark_legacy_migration_complete()
    print(
        f"完成：{len(sources)} 会话，{total_events} 事件，"
        f"{total_conflicts} 冲突；已写入迁移水位"
    )
    return 0


async def _run(args: argparse.Namespace) -> int:
    cfg = ConfigLoader(args.config)
    archive_cfg = getattr(cfg, "archive", {}) or {}
    timezone_name = str(archive_cfg.get("timezone", "Asia/Shanghai"))
    event_log = ConversationEventLog(args.event_log, timezone_name=timezone_name)
    return await import_legacy_history(
        Path(args.sessions_dir), event_log, force=args.force
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.toml")
    parser.add_argument("--sessions-dir", default="data/sessions")
    parser.add_argument("--event-log", default="data/conversation_event_log.sqlite3")
    parser.add_argument("--force", action="store_true", help="已有水位时仍重新导入")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
