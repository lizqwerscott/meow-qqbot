import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER, HTTP_400_BAD_REQUEST

_log = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])

# chat_id 必须仅含字母数字、下划线、冒号、横线、点（避免路径遍历）
_CHAT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_:\-\.]+$")


def _validate_chat_id(chat_id: str) -> None:
    if not _CHAT_ID_PATTERN.match(chat_id):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="无效的 chat_id")


def _validate_date(date: str) -> None:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="无效的 date")


def _make_flash_redirect(url: str, category: str, message: str):
    separator = "&" if "?" in url else "?"
    return RedirectResponse(
        url=f"{url}{separator}flash_{category}={message}",
        status_code=HTTP_303_SEE_OTHER,
    )


@router.get("/sessions", response_class=HTMLResponse)
async def session_list(
    request: Request,
    q: Optional[str] = Query(None),
):
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")

    all_chat_ids = await context_manager.get_all_disk_chat_ids_async()
    if q:
        all_chat_ids = [cid for cid in all_chat_ids if q.lower() in cid.lower()]

    archived_counts = await context_manager.get_archived_sessions_summary_async()

    # 并行获取所有会话的摘要信息，提升性能
    summaries = await asyncio.gather(
        *[context_manager.get_session_summary_async(cid) for cid in all_chat_ids],
        return_exceptions=True
    )

    sessions = []
    for cid, summary in zip(all_chat_ids, summaries):
        # 处理可能的异常
        if isinstance(summary, Exception):
            sessions.append({
                "chat_id": cid,
                "message_count": 0,
                "last_activity": "-",
                "estimated_tokens": 0,
                "archived_count": archived_counts.get(cid, 0),
            })
        else:
            sessions.append({
                "chat_id": cid,
                "message_count": summary["message_count"],
                "last_activity": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(summary["last_activity"]),
                ),
                "estimated_tokens": summary["estimated_tokens"],
                "archived_count": archived_counts.get(cid, 0),
            })

    sessions.sort(key=lambda s: s["last_activity"], reverse=True)

    return templates.TemplateResponse(
        request,
        "sessions/list.html",
        {
            "request": request,
            "sessions": sessions,
            "query": q or "",
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

    archived_counts = await context_manager.get_archived_sessions_summary_async()
    all_archived_ids = sorted(archived_counts.keys())
    if q:
        all_archived_ids = [cid for cid in all_archived_ids if q.lower() in cid.lower()]

    sessions = []
    for cid in all_archived_ids:
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
):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")
    archive_manager = managers.get("archive_manager")

    files = await context_manager.get_archived_files_async(chat_id)
    memory_dir = (
        Path(getattr(archive_manager, "_memory_dir", "data/archives/memory"))
        if archive_manager
        else Path("data/archives/memory")
    )
    summaries = await asyncio.to_thread(_list_memory_summaries, memory_dir, chat_id)

    messages_by_file = {}
    if tab == "messages":
        for f in files[:3]:
            msgs = await context_manager.read_archived_messages_async(
                f["path"], max_messages=100
            )
            messages_by_file[f["timestamp_str"]] = msgs

    return templates.TemplateResponse(
        request,
        "sessions/archived_detail.html",
        {
            "request": request,
            "chat_id": chat_id,
            "archived_files": files,
            "summaries": summaries,
            "messages_by_file": messages_by_file,
            "tab": tab,
        },
    )


@router.get(
    "/sessions/archived/{chat_id}/messages/{timestamp}", response_class=HTMLResponse
)
async def archived_messages_full(request: Request, chat_id: str, timestamp: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")

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

    return templates.TemplateResponse(
        request,
        "sessions/archived_messages.html",
        {
            "request": request,
            "chat_id": chat_id,
            "timestamp": timestamp,
            "messages": messages,
        },
    )


@router.get("/sessions/archived/{chat_id}/summary/{date}", response_class=HTMLResponse)
async def archived_summary_view(request: Request, chat_id: str, date: str):
    _validate_chat_id(chat_id)
    _validate_date(date)
    managers = request.app.state.managers
    templates = request.app.state.templates
    archive_manager = managers.get("archive_manager")

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
        },
    )


@router.post("/sessions/archived/{chat_id}/delete/{timestamp}")
async def archived_delete(request: Request, chat_id: str, timestamp: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    context_manager = managers.get("context_manager")
    archive_manager = managers.get("archive_manager")

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

    history = await context_manager.get_chat_history_async(chat_id, max_messages=200)
    history.reverse()

    archived_files = await context_manager.get_archived_files_async(chat_id)

    return templates.TemplateResponse(
        request,
        "sessions/detail.html",
        {
            "request": request,
            "chat_id": chat_id,
            "messages": history,
            "archived_count": len(archived_files),
        },
    )


@router.post("/sessions/{chat_id}/clear")
async def session_clear(request: Request, chat_id: str):
    _validate_chat_id(chat_id)
    managers = request.app.state.managers
    context_manager = managers.get("context_manager")

    await context_manager.clear_chat_history_async(chat_id)
    return _make_flash_redirect("/sessions", "success", "会话已清空")


def _list_memory_summaries(memory_dir: Path, chat_id: str) -> list[dict]:
    mem_path = memory_dir / chat_id
    if not mem_path.is_dir():
        return []
    return [
        {"date": path.stem, "path": str(path), "size": path.stat().st_size}
        for path in sorted(mem_path.glob("*.md"), reverse=True)
    ]
