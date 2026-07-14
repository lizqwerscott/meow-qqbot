import logging
from typing import Optional, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

_log = logging.getLogger(__name__)

router = APIRouter(tags=["learners"])


def _make_flash_redirect(url: str, category: str, message: str):
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}flash_{category}={message}", status_code=HTTP_303_SEE_OTHER)


def _get_learner_data(store: Any) -> list:
    if store is None:
        return []
    try:
        items = store.get_all()
        return items
    except Exception:
        return []


@router.get("/learners/jargons", response_class=HTMLResponse)
async def jargon_list(
    request: Request,
    q: Optional[str] = Query(None),
):
    managers = request.app.state.managers
    templates = request.app.state.templates
    orchestrator = managers.get("learning_orchestrator")

    items = _get_learner_data(orchestrator.jargon.store if orchestrator else None)

    if q:
        q_lower = q.lower()
        items = [
            item for item in items
            if q_lower in item.get("term", "").lower()
            or q_lower in item.get("definition", "").lower()
        ]

    return templates.TemplateResponse(request, "learners/jargons.html", {
        "request": request,
        "items": items,
        "tab": "jargons",
        "query": q or "",
    })


@router.post("/learners/jargons/{term}/delete")
async def jargon_delete(request: Request, term: str):
    managers = request.app.state.managers
    orchestrator = managers.get("learning_orchestrator")

    if orchestrator and orchestrator.jargon:
        await orchestrator.jargon.store.delete(term)

    return _make_flash_redirect("/learners/jargons", "success", f"已删除词条: {term}")