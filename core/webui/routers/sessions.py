import logging
import time
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

_log = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])


def _make_flash_redirect(url: str, category: str, message: str):
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}flash_{category}={message}", status_code=HTTP_303_SEE_OTHER)


@router.get("/sessions", response_class=HTMLResponse)
async def session_list(
    request: Request,
    q: Optional[str] = Query(None),
):
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")

    all_chat_ids = context_manager.get_all_disk_chat_ids()
    if q:
        all_chat_ids = [cid for cid in all_chat_ids if q.lower() in cid.lower()]

    sessions = []
    for cid in all_chat_ids:
        ctx = context_manager.get_context(cid)
        sessions.append({
            "chat_id": cid,
            "message_count": ctx.get_history_count(),
            "last_activity": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ctx.last_activity)),
            "estimated_tokens": ctx.estimate_tokens_for_history(),
        })

    sessions.sort(key=lambda s: s["last_activity"], reverse=True)

    return templates.TemplateResponse(request, "sessions/list.html", {
        "request": request,
        "sessions": sessions,
        "query": q or "",
    })


@router.get("/sessions/{chat_id}", response_class=HTMLResponse)
async def session_detail(request: Request, chat_id: str):
    managers = request.app.state.managers
    templates = request.app.state.templates
    context_manager = managers.get("context_manager")

    ctx = context_manager.get_context(chat_id)
    history = ctx.get_history_as_dicts(max_messages=200)
    history.reverse()

    return templates.TemplateResponse(request, "sessions/detail.html", {
        "request": request,
        "chat_id": chat_id,
        "messages": history,
    })


@router.post("/sessions/{chat_id}/clear")
async def session_clear(request: Request, chat_id: str):
    managers = request.app.state.managers
    context_manager = managers.get("context_manager")

    await context_manager.clear_chat_history_async(chat_id)
    return _make_flash_redirect("/sessions", "success", f"会话已清空")
