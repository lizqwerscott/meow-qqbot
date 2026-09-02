import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER, HTTP_400_BAD_REQUEST

from core.engine.history_projection import visible_legacy_history
from core.managers.chat_message import strip_content_prefix

_log = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])

# chat_id 必须仅含字母数字、下划线、冒号、横线、点（避免路径遍历）
_CHAT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_:\-\.]+$")
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|cookie|authorization)"
)
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|cookie|authorization)\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)([^\s,;]+)")
_PAGE_SIZE_DEFAULT = 20
_PAGE_SIZE_MAX = 100


def _validate_chat_id(chat_id: str) -> None:
    if not _CHAT_ID_PATTERN.match(chat_id):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="无效的 chat_id")


def _validate_date(date: str) -> None:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="无效的 date")


def _redact_text(value: Any, *, limit: int = 4000) -> str:
    text = str(value or "")
    text = _BEARER_PATTERN.sub(r"\1[已脱敏]", text)
    text = _SENSITIVE_TEXT_PATTERN.sub(_replace_sensitive_text, text)
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _replace_sensitive_text(match: re.Match[str]) -> str:
    value = match.group(2)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[0] + "[已脱敏]" + value[-1]
    else:
        value = "[已脱敏]"
    return match.group(1) + value


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[已脱敏]"
                if _SENSITIVE_KEY_PATTERN.search(str(key))
                else _redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_message(message: dict[str, Any]) -> dict[str, Any]:
    safe = _redact_value(dict(message))
    if not isinstance(safe, dict):
        return {}
    safe["content"] = _redact_text(safe.get("content"), limit=4000)
    if safe.get("role") == "user":
        safe["content"] = strip_content_prefix(safe["content"])
    if safe.get("reasoning_content"):
        safe["reasoning_content"] = _redact_text(safe["reasoning_content"], limit=1200)
    tool_calls = safe.get("tool_calls")
    if tool_calls:
        safe["tool_calls"] = _redact_value(tool_calls)
        for call in safe["tool_calls"]:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = _redact_text(arguments, limit=1000)
                else:
                    arguments = json.dumps(
                        _redact_value(arguments), ensure_ascii=False, sort_keys=True
                    )
                function["arguments"] = _redact_text(arguments, limit=1000)
    return safe


def _paginate_cards(
    cards: list[dict[str, Any]], page: int, page_size: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    page = max(1, int(page))
    page_size = min(_PAGE_SIZE_MAX, max(1, int(page_size)))
    total = len(cards)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    return cards[start : start + page_size], {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }


def _pagination_query(pagination: dict[str, int]) -> str:
    return f"page={pagination['page']}&page_size={pagination['page_size']}"


def _session_kind(chat_id: str, context_manager) -> str:
    if chat_id.startswith("heartbeat:"):
        return "heartbeat"
    if chat_id.startswith(("cron:", "task:", "exec:", "system:")):
        return "system"
    get_chat_type = getattr(context_manager, "get_chat_type", None)
    chat_type = get_chat_type(chat_id) if callable(get_chat_type) else None
    if chat_type is True:
        return "group"
    if chat_type is False:
        return "private"
    return "unknown"


def _make_flash_redirect(url: str, category: str, message: str):
    separator = "&" if "?" in url else "?"
    return RedirectResponse(
        url=f"{url}{separator}flash_{category}={message}",
        status_code=HTTP_303_SEE_OTHER,
    )


async def _repair_timeline_from_legacy(context_manager, timeline, chat_id: str):
    if timeline is None:
        return
    repair = getattr(timeline, "repair_from_legacy_history", None)
    get_history = getattr(context_manager, "get_chat_history_async", None)
    if not callable(repair) or not callable(get_history):
        return
    legacy_history = await get_history(chat_id)
    await repair(chat_id, legacy_history)


async def _legacy_protocol_history(context_manager, chat_id: str) -> list[dict]:
    """Expose protocol records persisted before the protocol projection existed."""
    get_history = getattr(context_manager, "get_chat_history_async", None)
    if not callable(get_history):
        return []
    history = await get_history(chat_id)
    return [
        message for message in history if message.get("role") in {"assistant", "tool"}
    ]


async def _claim_legacy_protocol_history(
    protocol_history, context_manager, chat_id: str
) -> None:
    claim = getattr(protocol_history, "claim_orphan_turns", None)
    get_history = getattr(context_manager, "get_chat_history_async", None)
    if not callable(claim) or not callable(get_history):
        return
    history = await get_history(chat_id)
    turn_ids = [str(message.get("message_id") or "") for message in history]
    await claim(chat_id, turn_ids)


@router.get("/sessions", response_class=HTMLResponse)
async def session_list(
    request: Request,
    q: Optional[str] = Query(None),
    kind: str = Query("all"),
):
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")
    protocol_history = managers.get("protocol_history")
    event_log = managers.get("conversation_event_log")

    if event_log is not None:
        all_chat_ids = []
        try:
            all_chat_ids.extend(await event_log.chat_ids())
        except Exception:
            pass
        archive_index = managers.get("archive_index")
        try:
            if archive_index is not None:
                all_chat_ids.extend(await archive_index.chat_ids())
        except Exception:
            pass
        all_chat_ids = list(dict.fromkeys(all_chat_ids))
    else:
        all_chat_ids = await context_manager.get_all_disk_chat_ids_async()
        if timeline := managers.get("conversation_timeline"):
            try:
                timeline_chat_ids = await timeline.chat_ids()
                all_chat_ids = list(dict.fromkeys(all_chat_ids + timeline_chat_ids))
            except Exception:
                pass
        if protocol_history is not None:
            try:
                all_chat_ids = list(
                    dict.fromkeys(all_chat_ids + await protocol_history.chat_ids())
                )
            except Exception:
                pass
    if q:
        all_chat_ids = [cid for cid in all_chat_ids if q.lower() in cid.lower()]

    archive_index = managers.get("archive_index")
    if archive_index is not None:
        archived_counts = {}
        for cid in all_chat_ids:
            try:
                archived_counts[cid] = sum(
                    1
                    for batch in await archive_index.list_for_webui(cid)
                    if batch.get("state") == "committed"
                )
            except Exception:
                archived_counts[cid] = 0
    else:
        archived_counts = await context_manager.get_archived_sessions_summary_async()

    timeline = managers.get("conversation_timeline")
    if event_log is None and timeline is not None:
        summaries = await asyncio.gather(
            *[timeline.session_summary(cid) for cid in all_chat_ids],
            return_exceptions=True,
        )
    else:
        summaries = [None] * len(all_chat_ids)

    sessions = []
    for cid, timeline_summary in zip(all_chat_ids, summaries):
        summary = timeline_summary
        event_log = managers.get("conversation_event_log")
        ledger_summary = None
        if event_log is not None:
            try:
                ledger_summary = await event_log.session_summary(cid)
                if ledger_summary.get("event_count", 0):
                    summary = ledger_summary
            except Exception:
                pass
        protocol_summary = None
        if ledger_summary is not None:
            try:
                protocol_summary = {
                    "message_count": ledger_summary.get(
                        "wire_count", ledger_summary.get("protocol_count", 0)
                    ),
                    "last_activity": ledger_summary.get("last_activity", 0),
                    "estimated_tokens": ledger_summary.get("estimated_tokens", 0),
                }
            except Exception:
                protocol_summary = None
        elif protocol_history is not None:
            try:
                protocol_summary = await protocol_history.session_summary(cid)
                if not protocol_summary.get("message_count", 0):
                    await _claim_legacy_protocol_history(
                        protocol_history, context_manager, cid
                    )
                    protocol_summary = await protocol_history.session_summary(cid)
            except Exception:
                protocol_summary = None
        if event_log is None and not (protocol_summary or {}).get("message_count", 0):
            try:
                legacy_protocol = await _legacy_protocol_history(context_manager, cid)
                if legacy_protocol:
                    protocol_summary = {
                        "message_count": len(legacy_protocol),
                        "last_activity": max(
                            (
                                float(message.get("timestamp") or 0)
                                for message in legacy_protocol
                            ),
                            default=0,
                        ),
                        "estimated_tokens": sum(
                            max(0, len(str(message.get("content") or "")) // 4)
                            for message in legacy_protocol
                        ),
                    }
            except Exception:
                pass
        try:
            if isinstance(summary, Exception):
                summary = None
            if event_log is None and timeline is not None:
                await _repair_timeline_from_legacy(context_manager, timeline, cid)
                repaired = await timeline.session_summary(cid)
                if repaired.get("message_count", 0):
                    summary = repaired
            if not summary or not summary.get("message_count", 0):
                summary = await context_manager.get_session_summary_async(cid)
            if (
                (not summary or not summary.get("message_count", 0))
                and protocol_summary
                and protocol_summary.get("message_count", 0)
            ):
                summary = protocol_summary
        except Exception:
            summary = None

        session_kind = _session_kind(cid, context_manager)
        if kind != "all" and session_kind != kind:
            continue
        if summary is None:
            sessions.append(
                {
                    "chat_id": cid,
                    "message_count": 0,
                    "last_activity": "-",
                    "estimated_tokens": 0,
                    "session_kind": session_kind,
                    "protocol_count": int(
                        (protocol_summary or {}).get("message_count", 0)
                    ),
                    "archived_count": archived_counts.get(cid, 0),
                }
            )
        else:
            sessions.append(
                {
                    "chat_id": cid,
                    "message_count": summary["message_count"],
                    "last_activity": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(summary["last_activity"]),
                    ),
                    "estimated_tokens": summary["estimated_tokens"],
                    "session_kind": session_kind,
                    "protocol_count": int(
                        (protocol_summary or {}).get("message_count", 0)
                    ),
                    "archived_count": archived_counts.get(cid, 0),
                }
            )

    sessions.sort(key=lambda s: s["last_activity"], reverse=True)

    return templates.TemplateResponse(
        request,
        "sessions/list.html",
        {
            "request": request,
            "sessions": sessions,
            "query": q or "",
            "kind": kind,
            "total_archived": len(archived_counts),
        },
    )


@router.get("/sessions/archived", response_class=HTMLResponse)
async def archived_list(
    request: Request,
    q: Optional[str] = Query(None),
):
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")
    archive_index = managers.get("archive_index")

    if archive_index is not None:
        all_archived_ids = []
        for cid in await archive_index.chat_ids():
            batches = await archive_index.list_for_webui(cid)
            if any(batch.get("state") == "committed" for batch in batches):
                all_archived_ids.append(cid)
        all_archived_ids.sort()
    else:
        archived_counts = await context_manager.get_archived_sessions_summary_async()
        all_archived_ids = sorted(archived_counts.keys())
    if q:
        all_archived_ids = [cid for cid in all_archived_ids if q.lower() in cid.lower()]

    sessions = []
    for cid in all_archived_ids:
        if archive_index is not None:
            batches = [
                batch
                for batch in await archive_index.list_for_webui(cid)
                if batch.get("state") == "committed"
            ]
            latest = max((batch.get("committed_at", 0) for batch in batches), default=0)
            sessions.append(
                {
                    "chat_id": cid,
                    "archive_count": len(batches),
                    "latest_archive": (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest))
                        if latest
                        else "-"
                    ),
                    "total_size": sum(
                        int(batch.get("event_count", 0)) for batch in batches
                    ),
                    "ledger_archive": True,
                }
            )
            continue
        files = await context_manager.get_archived_files_async(cid)
        sessions.append(
            {
                "chat_id": cid,
                "archive_count": len(files),
                "latest_archive": (
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(files[0]["mtime"])
                    )
                    if files
                    else "-"
                ),
                "total_size": sum(f["size"] for f in files),
                "ledger_archive": False,
            }
        )

    return templates.TemplateResponse(
        request,
        "sessions/archived_list.html",
        {
            "request": request,
            "sessions": sessions,
            "query": q or "",
        },
    )


@router.get("/sessions/archived/{chat_id}", response_class=HTMLResponse)
async def archived_detail(
    request: Request,
    chat_id: str,
    tab: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")
    archive_manager = managers.get("archive_manager")
    archive_index = managers.get("archive_index")

    if archive_index is not None:
        batches = [
            batch
            for batch in await archive_index.list_for_webui(chat_id)
            if batch.get("state") == "committed"
        ]
        summary_store = managers.get("turn_summary_store")
        summary_total = (
            await summary_store.count_for_webui(chat_id)
            if summary_store is not None and hasattr(summary_store, "count_for_webui")
            else 0
        )
        summary_rows = (
            [
                summary.to_dict()
                for summary in await summary_store.list_for_webui(
                    chat_id, limit=page_size, offset=(page - 1) * page_size
                )
            ]
            if summary_store is not None
            else []
        )
        visible_batches, pagination = _paginate_cards(batches, page, page_size)
        messages_by_batch = {}
        if tab == "messages":
            event_log = managers.get("conversation_event_log")
            if event_log is not None:
                for batch in visible_batches:
                    event_ids = await archive_index.event_ids(batch["batch_id"])
                    snapshot = await event_log.snapshot_events(
                        chat_id, include_internal=False, event_ids=tuple(event_ids)
                    )
                    messages_by_batch[batch["batch_id"]] = _history_turn_cards(
                        [event.to_history_dict() for event in snapshot.events]
                    )
        return templates.TemplateResponse(
            request,
            "sessions/archived_detail.html",
            {
                "request": request,
                "chat_id": chat_id,
                "archived_files": [],
                "archive_batches": visible_batches,
                "summaries": [],
                "summary_rows": summary_rows,
                "summary_total": summary_total,
                "messages_by_file": {},
                "messages_by_batch": messages_by_batch,
                "tab": tab,
                "ledger_archive": True,
                "pagination": pagination,
                "pagination_query": _pagination_query(pagination),
            },
        )

    files = await context_manager.get_archived_files_async(chat_id)
    memory_dir = (
        Path(getattr(archive_manager, "_memory_dir", "data/archives/memory"))
        if archive_manager
        else Path("data/archives/memory")
    )
    summaries = await asyncio.to_thread(_list_memory_summaries, memory_dir, chat_id)

    messages_by_file = {}
    pagination = {
        "page": 1,
        "page_size": page_size,
        "total": len(files),
        "total_pages": 1,
    }
    if tab == "messages":
        visible_files, pagination = _paginate_cards(
            [dict(item, _archive_file=True) for item in files], page, page_size
        )
        for f in visible_files:
            msgs = await context_manager.read_archived_messages_async(
                f["path"], max_messages=100
            )
            msgs = [
                message
                for message in msgs
                if message.get("role") == "user"
                or (
                    message.get("role") == "assistant" and not message.get("tool_calls")
                )
            ]
            messages_by_file[f["timestamp_str"]] = _history_turn_cards(msgs)

    return templates.TemplateResponse(
        request,
        "sessions/archived_detail.html",
        {
            "request": request,
            "chat_id": chat_id,
            "archived_files": files,
            "summaries": summaries,
            "messages_by_file": messages_by_file,
            "archive_batches": [],
            "summary_rows": [],
            "summary_total": 0,
            "messages_by_batch": {},
            "tab": tab,
            "ledger_archive": False,
            "pagination": pagination,
            "pagination_query": _pagination_query(pagination),
        },
    )


@router.get(
    "/sessions/archived/{chat_id}/messages/{timestamp}", response_class=HTMLResponse
)
async def archived_messages_full(
    request: Request,
    chat_id: str,
    timestamp: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
    protocol: bool = Query(False),
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")
    archive_index = managers.get("archive_index")
    if archive_index is not None:
        batch = await archive_index.get(timestamp)
        if batch is None or batch.chat_id != chat_id or batch.state != "committed":
            return _make_flash_redirect(
                f"/sessions/archived/{chat_id}", "error", "未找到该归档 batch"
            )
        event_log = managers.get("conversation_event_log")
        event_ids = await archive_index.event_ids(timestamp)
        snapshot = (
            await event_log.snapshot_events(
                chat_id, include_internal=protocol, event_ids=tuple(event_ids)
            )
            if event_log is not None
            else None
        )
        messages = [
            event.to_history_dict()
            for event in (snapshot.events if snapshot is not None else ())
            if event.event_id in event_ids
        ]
        turns, pagination = _paginate_cards(
            _history_turn_cards(messages), page, page_size
        )
        return templates.TemplateResponse(
            request,
            "sessions/archived_messages.html",
            {
                "request": request,
                "chat_id": chat_id,
                "timestamp": timestamp,
                "messages": [],
                "turns": turns,
                "pagination": pagination,
                "pagination_query": _pagination_query(pagination),
                "protocol": protocol,
                "ledger_archive": True,
                "batch": batch,
            },
        )

    files = await context_manager.get_archived_files_async(chat_id)
    target = None
    for f in files:
        if f["timestamp_str"] == timestamp:
            target = f
            break

    if not target:
        return _make_flash_redirect(
            f"/sessions/archived/{chat_id}", "error", "未找到该归档文件"
        )

    messages = await context_manager.read_archived_messages_async(
        target["path"], max_messages=500
    )
    if not protocol:
        messages = [
            message
            for message in messages
            if message.get("role") == "user"
            or (message.get("role") == "assistant" and not message.get("tool_calls"))
        ]
    turns, pagination = _paginate_cards(_history_turn_cards(messages), page, page_size)

    return templates.TemplateResponse(
        request,
        "sessions/archived_messages.html",
        {
            "request": request,
            "chat_id": chat_id,
            "timestamp": timestamp,
            "messages": [],
            "turns": turns,
            "pagination": pagination,
            "pagination_query": _pagination_query(pagination),
            "protocol": protocol,
            "ledger_archive": False,
            "batch": None,
        },
    )


@router.get("/sessions/archived/{chat_id}/summary/{date}", response_class=HTMLResponse)
async def archived_summary_view(request: Request, chat_id: str, date: str):
    _validate_chat_id(chat_id)
    _validate_date(date)
    managers = request.app.state.managers
    templates = request.app.state.templates
    archive_manager = managers.get("archive_manager")
    summary_store = managers.get("turn_summary_store")
    archive_index = managers.get("archive_index")

    if archive_index is not None:
        summaries = (
            await summary_store.list_for_webui(chat_id)
            if summary_store is not None
            else []
        )
        content = "\n\n---\n\n".join(
            summary.text for summary in summaries if summary.source_date == date
        )
        return templates.TemplateResponse(
            request,
            "sessions/archived_summary.html",
            {
                "request": request,
                "chat_id": chat_id,
                "date": date,
                "content": content,
                "ledger_summary": True,
            },
        )

    memory_dir = (
        Path(getattr(archive_manager, "_memory_dir", "data/archives/memory"))
        if archive_manager
        else Path("data/archives/memory")
    )
    summary_path = memory_dir / chat_id / f"{date}.md"
    content = ""
    if summary_path.exists():
        content = await asyncio.to_thread(summary_path.read_text, encoding="utf-8")

    return templates.TemplateResponse(
        request,
        "sessions/archived_summary.html",
        {
            "request": request,
            "chat_id": chat_id,
            "date": date,
            "content": content,
            "ledger_summary": False,
        },
    )


@router.post("/sessions/archived/{chat_id}/delete/{timestamp}")
async def archived_delete(request: Request, chat_id: str, timestamp: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    context_manager = managers.get("context_manager")
    archive_manager = managers.get("archive_manager")
    archive_index = managers.get("archive_index")

    if archive_index is not None:
        batch = await archive_index.get(timestamp)
        if batch is None or batch.chat_id != chat_id:
            return _make_flash_redirect(
                f"/sessions/archived/{chat_id}", "error", "未找到该归档 batch"
            )
        await archive_index.mark_state(timestamp, "soft_deleted")
        return _make_flash_redirect(
            f"/sessions/archived/{chat_id}",
            "success",
            f"已隐藏归档 batch {timestamp}（核心事件仍保留）",
        )

    files = await context_manager.get_archived_files_async(chat_id)
    target = None
    for f in files:
        if f["timestamp_str"] == timestamp:
            target = f
            break

    if not target:
        return _make_flash_redirect(
            f"/sessions/archived/{chat_id}", "error", "未找到该归档文件"
        )

    try:
        await asyncio.to_thread(Path(target["path"]).unlink, missing_ok=True)
        date_str = timestamp[:10]
        if archive_manager:
            memory_dir = Path(
                getattr(archive_manager, "_memory_dir", "data/archives/memory")
            )
            summary_path = memory_dir / chat_id / f"{date_str}.md"
            await asyncio.to_thread(summary_path.unlink, missing_ok=True)
        return _make_flash_redirect(
            f"/sessions/archived/{chat_id}", "success", f"已删除归档 {timestamp}"
        )
    except Exception as e:
        return _make_flash_redirect(
            f"/sessions/archived/{chat_id}", "error", f"删除失败: {e}"
        )


@router.get("/sessions/{chat_id}", response_class=HTMLResponse)
async def session_detail(request: Request, chat_id: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")
    timeline = managers.get("conversation_timeline")
    protocol_history = managers.get("protocol_history")
    archive_index = managers.get("archive_index")

    event_log = managers.get("conversation_event_log")
    history = await event_log.history(chat_id) if event_log is not None else []
    if event_log is None and not history:
        history = await timeline.history(chat_id) if timeline else []
    if event_log is None and timeline is not None and not history:
        try:
            await _repair_timeline_from_legacy(context_manager, timeline, chat_id)
            history = await timeline.history(chat_id)
        except Exception:
            pass
    elif event_log is None and not history:
        history = visible_legacy_history(
            await context_manager.get_chat_history_async(chat_id)
        )
    history.reverse()

    archived_files = (
        []
        if archive_index is not None
        else await context_manager.get_archived_files_async(chat_id)
    )
    protocol_count = 0
    if event_log is not None:
        protocol_count = sum(
            1
            for message in await event_log.history(chat_id, include_internal=True)
            if message.get("kind") in {"assistant_tool_call", "tool_result"}
        )
    if event_log is None and protocol_history is not None:
        try:
            protocol_count = int(
                (await protocol_history.session_summary(chat_id)).get(
                    "message_count", 0
                )
            )
            if protocol_count == 0:
                await _claim_legacy_protocol_history(
                    protocol_history, context_manager, chat_id
                )
                protocol_count = int(
                    (await protocol_history.session_summary(chat_id)).get(
                        "message_count", 0
                    )
                )
        except Exception:
            pass
    if protocol_count == 0 and event_log is None:
        try:
            protocol_count = len(
                await _legacy_protocol_history(context_manager, chat_id)
            )
        except Exception:
            pass

    return templates.TemplateResponse(
        request,
        "sessions/detail.html",
        {
            "request": request,
            "chat_id": chat_id,
            "messages": history,
            "archived_count": (
                sum(
                    1
                    for batch in await archive_index.list_for_webui(chat_id)
                    if batch.get("state") == "committed"
                )
                if archive_index is not None
                else len(archived_files)
            ),
            "protocol_count": protocol_count,
        },
    )


@router.get("/sessions/{chat_id}/protocol", response_class=HTMLResponse)
async def session_protocol_detail(
    request: Request,
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    protocol_history = managers.get("protocol_history")
    event_log = managers.get("conversation_event_log")
    messages = []
    pagination = None
    if event_log is not None:
        protocol_index = await event_log.protocol_turn_index(chat_id)
        selected_index, pagination = _paginate_cards(protocol_index, page, page_size)
        selected_turn_ids = [item["turn_id"] for item in selected_index]
        messages = [
            message
            for message in await event_log.history(
                chat_id, include_internal=True, turn_ids=selected_turn_ids
            )
            if message.get("kind") in {"assistant_tool_call", "tool_result"}
        ]
    if event_log is None and not messages and protocol_history is not None:
        messages = await protocol_history.history(chat_id)
        if not messages:
            await _claim_legacy_protocol_history(
                protocol_history, managers.get("context_manager"), chat_id
            )
            messages = await protocol_history.history(chat_id)
    if event_log is None and not messages:
        messages = await _legacy_protocol_history(
            managers.get("context_manager"), chat_id
        )
    if pagination is None:
        turns, pagination = _paginate_cards(
            _history_turn_cards(messages), page, page_size
        )
    else:
        turns = _history_turn_cards(messages)
    return templates.TemplateResponse(
        request,
        "sessions/protocol.html",
        {
            "request": request,
            "chat_id": chat_id,
            "messages": [],
            "turns": turns,
            "pagination": pagination,
            "pagination_query": _pagination_query(pagination),
        },
    )


def _ledger_events_view(events) -> list[dict]:
    return [_redact_message(event.to_history_dict()) for event in events]


def _history_turn_cards(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(messages):
        turn_id = str(message.get("turn_id") or f"message:{index}")
        card = cards.setdefault(
            turn_id,
            {
                "turn_id": turn_id,
                "turn_sequence": int(message.get("turn_sequence") or 0),
                "source_date": str(message.get("source_date") or ""),
                "status": str(message.get("terminal_status") or "unknown"),
                "events": [],
            },
        )
        card["turn_sequence"] = max(
            card["turn_sequence"], int(message.get("turn_sequence") or 0)
        )
        card["events"].append(_redact_message(message))
    return sorted(
        cards.values(),
        key=lambda item: (item["turn_sequence"], item["turn_id"]),
        reverse=True,
    )


def _ledger_turn_cards(events, statuses: Optional[dict[str, str]] = None) -> list[dict]:
    cards: dict[str, dict] = {}
    for event in events:
        card = cards.setdefault(
            event.turn_id,
            {
                "turn_id": event.turn_id,
                "turn_sequence": event.turn_sequence,
                "source_date": event.source_date,
                "events": [],
                "status": (statuses or {}).get(event.turn_id, "unknown"),
            },
        )
        card["events"].append(_redact_message(event.to_history_dict()))
    return sorted(cards.values(), key=lambda item: item["turn_sequence"], reverse=True)


async def _render_ledger_view(
    request: Request, chat_id: str, view: str, page: int = 1, page_size: int = 20
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    event_log = managers.get("conversation_event_log")
    projection = managers.get("prompt_history_projection")
    if event_log is None:
        raise HTTPException(status_code=503, detail="核心会话账本不可用")

    if view == "prompt":
        if projection is None:
            events = await event_log.snapshot_events(chat_id, include_internal=False)
            degraded_reason = "projection_unavailable"
        else:
            events = await projection.snapshot_for_prompt(chat_id)
            degraded_reason = events.degraded_reason
        title = "当前 Prompt 历史"
        description = "实际 bounded prompt history；归档正文不会因 WebUI 查询重新进入。"
        turn_snapshot = await event_log.snapshot_turns(chat_id, include_internal=False)
        statuses = {turn.turn_id: str(turn.status) for turn in turn_snapshot.turns}
        event_values = _ledger_events_view(events.events)
        turns, pagination = _paginate_cards(
            _ledger_turn_cards(events.events, statuses), page, page_size
        )
    elif view == "active":
        snapshot = (
            await projection.snapshot_for_active(chat_id)
            if projection is not None
            else await event_log.snapshot_events(chat_id, include_internal=False)
        )
        messages = [event.to_history_dict() for event in snapshot.events]
        turns, pagination = _paginate_cards(
            _ledger_turn_cards(snapshot.events), page, page_size
        )
        return templates.TemplateResponse(
            request,
            "sessions/projection.html",
            {
                "request": request,
                "chat_id": chat_id,
                "view_title": "Active History",
                "view_description": "当前热区兼容投影；它不等于完整账本，也不等于实际 Prompt。",
                "events": messages,
                "turns": turns,
                "stats": {
                    "event_count": len(messages),
                    "turn_count": len({event.turn_id for event in snapshot.events}),
                },
                "degraded_reason": "",
                "event_view": True,
                "pagination": pagination,
                "pagination_query": _pagination_query(pagination),
            },
        )
    else:
        title = "完整 Timeline"
        description = "核心账本中的完整可见 Timeline；工具协议请查看独立的协议视图，不代表每次模型调用都会加载。"
        degraded_reason = ""
        turn_snapshot = await event_log.snapshot_turns(chat_id, include_internal=False)
        all_turns = [
            {
                "turn_id": turn.turn_id,
                "turn_sequence": turn.turn_sequence,
                "source_date": turn.source_date,
                "status": str(turn.status),
                "events": [],
            }
            for turn in reversed(turn_snapshot.turns)
        ]
        selected_turns, pagination = _paginate_cards(all_turns, page, page_size)
        selected_turn_ids = [turn["turn_id"] for turn in selected_turns]
        events = await event_log.snapshot_events(
            chat_id, include_internal=False, turn_ids=selected_turn_ids
        )
        statuses = {turn["turn_id"]: turn["status"] for turn in selected_turns}
        event_values = _ledger_events_view(events.events)
        turns = _ledger_turn_cards(events.events, statuses)
    return templates.TemplateResponse(
        request,
        "sessions/projection.html",
        {
            "request": request,
            "chat_id": chat_id,
            "view_title": title,
            "view_description": description,
            "events": event_values,
            "turns": turns,
            "stats": {
                "event_count": len(event_values),
                "turn_count": len(_ledger_turn_cards(events.events, statuses)),
            },
            "degraded_reason": degraded_reason,
            "event_view": True,
            "pagination": pagination,
            "pagination_query": _pagination_query(pagination),
        },
    )


@router.get("/sessions/{chat_id}/active", response_class=HTMLResponse)
async def session_active_view(
    request: Request,
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    return await _render_ledger_view(request, chat_id, "active", page, page_size)


@router.get("/sessions/{chat_id}/timeline", response_class=HTMLResponse)
async def session_timeline_view(
    request: Request,
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    return await _render_ledger_view(request, chat_id, "timeline", page, page_size)


@router.get("/sessions/{chat_id}/prompt", response_class=HTMLResponse)
async def session_prompt_view(
    request: Request,
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    return await _render_ledger_view(request, chat_id, "prompt", page, page_size)


@router.get("/sessions/{chat_id}/summaries", response_class=HTMLResponse)
async def session_summaries_view(
    request: Request,
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    store = managers.get("turn_summary_store")
    summaries = (
        await store.list_for_webui(
            chat_id, limit=page_size, offset=(page - 1) * page_size
        )
        if store is not None
        else []
    )
    summary_total = (
        await store.count_for_webui(chat_id)
        if store is not None and hasattr(store, "count_for_webui")
        else len(summaries)
    )
    summary_pagination = {
        "page": max(1, page),
        "page_size": page_size,
        "total": summary_total,
        "total_pages": max(1, (summary_total + page_size - 1) // page_size),
    }
    return templates.TemplateResponse(
        request,
        "sessions/summaries.html",
        {
            "request": request,
            "chat_id": chat_id,
            "summaries": summaries,
            "pagination": summary_pagination,
            "pagination_query": _pagination_query(summary_pagination),
        },
    )


@router.get("/sessions/{chat_id}/model-context", response_class=HTMLResponse)
async def session_model_context_view(request: Request, chat_id: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    transcript = managers.get("model_context_transcript")
    report_store = managers.get("prompt_context_reports")
    reports = (
        await report_store.list_for_webui(chat_id) if report_store is not None else []
    )
    scopes = await transcript.scopes_for_chat(chat_id) if transcript is not None else ()
    scope_rows = []
    for scope in scopes:
        try:
            snapshot = await transcript.snapshot(scope, max_events=1000)
        except Exception as exc:
            _log.warning("读取模型上下文诊断失败 [%s..]: %s", chat_id[:12], exc)
            scope_rows.append(
                {
                    "scope": scope,
                    "event_count": "不可用",
                    "estimated_tokens": "不可用",
                    "roles": (),
                    "source_event_count": "不可用",
                    "error": str(exc),
                }
            )
            continue
        scope_rows.append(
            {
                "scope": scope,
                "event_count": len(snapshot.events),
                "estimated_tokens": sum(
                    max(1, len(event.content) // 4) for event in snapshot.events
                ),
                "roles": [event.role for event in snapshot.events],
                "source_event_count": len(snapshot.source_event_ids),
                "error": "",
            }
        )
    return templates.TemplateResponse(
        request,
        "sessions/model_context.html",
        {
            "request": request,
            "chat_id": chat_id,
            "scopes": scope_rows,
            "reports": reports,
        },
    )


@router.get("/sessions/{chat_id}/archive-history", response_class=HTMLResponse)
async def session_archive_history_view(
    request: Request,
    chat_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(_PAGE_SIZE_DEFAULT, ge=1, le=_PAGE_SIZE_MAX),
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    event_log = managers.get("conversation_event_log")
    projection = managers.get("prompt_history_projection")
    if event_log is None or projection is None:
        raise HTTPException(status_code=503, detail="归档投影不可用")
    hidden_ids = await projection.hidden_event_ids(chat_id)
    events = (
        await event_log.snapshot_events(
            chat_id,
            include_internal=False,
            event_ids=tuple(sorted(hidden_ids)),
        )
    ).events
    turn_snapshot = await event_log.snapshot_turns(chat_id, include_internal=True)
    statuses = {turn.turn_id: str(turn.status) for turn in turn_snapshot.turns}
    turns, pagination = _paginate_cards(
        _ledger_turn_cards(events, statuses), page, page_size
    )
    archive_index = managers.get("archive_index")
    archive_batches = (
        await archive_index.list_for_webui(chat_id) if archive_index is not None else []
    )
    return templates.TemplateResponse(
        request,
        "sessions/projection.html",
        {
            "request": request,
            "chat_id": chat_id,
            "view_title": "归档历史",
            "view_description": "核心账本中已从 Prompt 隐藏的完整事件；归档 JSONL 只是可选导出，不是读取来源。",
            "events": _ledger_events_view(events),
            "turns": turns,
            "stats": {"event_count": len(events)},
            "archive_batches": archive_batches,
            "degraded_reason": "",
            "event_view": True,
            "pagination": pagination,
            "pagination_query": _pagination_query(pagination),
        },
    )


@router.post("/sessions/{chat_id}/clear")
async def session_clear(request: Request, chat_id: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    context_manager = managers.get("context_manager")
    timeline = managers.get("conversation_timeline")
    protocol_history = managers.get("protocol_history")
    event_log = managers.get("conversation_event_log")
    prompt_projection = managers.get("prompt_history_projection")
    summary_store = managers.get("turn_summary_store")
    report_store = managers.get("prompt_context_reports")
    transcript = managers.get("model_context_transcript")
    archive_index = managers.get("archive_index")
    agent_engine = managers.get("agent_engine")

    clear_session = getattr(agent_engine, "clear_session_async", None)
    if callable(clear_session):
        await clear_session(chat_id)
        return _make_flash_redirect("/sessions", "success", "会话已清空")
    await context_manager.clear_chat_history_async(chat_id)
    if event_log is not None:
        await event_log.clear_chat(chat_id)
    if prompt_projection is not None:
        await prompt_projection.clear_chat(chat_id)
    if summary_store is not None:
        await summary_store.clear_chat(chat_id)
    if report_store is not None:
        await report_store.clear_chat(chat_id)
    if archive_index is not None:
        await archive_index.clear_chat(chat_id)
    if transcript is not None:
        await transcript.clear_chat(chat_id)
    if timeline is not None:
        await timeline.clear_chat(chat_id)
    if protocol_history is not None:
        await protocol_history.delete_chat(chat_id)
    return _make_flash_redirect("/sessions", "success", "会话已清空")


def _list_memory_summaries(memory_dir: Path, chat_id: str) -> list[dict]:
    mem_path = memory_dir / chat_id
    if not mem_path.is_dir():
        return []
    return [
        {"date": path.stem, "path": str(path), "size": path.stat().st_size}
        for path in sorted(mem_path.glob("*.md"), reverse=True)
    ]
