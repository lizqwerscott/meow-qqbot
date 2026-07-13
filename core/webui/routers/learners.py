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


@router.get("/learners/expressions", response_class=HTMLResponse)
async def expression_list(request: Request):
    managers = request.app.state.managers
    templates = request.app.state.templates
    orchestrator = managers.get("learning_orchestrator")

    items = _get_learner_data(orchestrator.expression.store if orchestrator else None)

    return templates.TemplateResponse(request, "learners/expressions.html", {
        "request": request,
        "items": items,
        "tab": "expressions",
    })


@router.post("/learners/expressions/{emoji_hash}/delete")
async def expression_delete(request: Request, emoji_hash: str):
    managers = request.app.state.managers
    orchestrator = managers.get("learning_orchestrator")

    if orchestrator and orchestrator.expression:
        await orchestrator.expression.store.delete(emoji_hash)

    return _make_flash_redirect("/learners/expressions", "success", "已删除表情映射")


@router.get("/learners/behaviors", response_class=HTMLResponse)
async def behavior_list(request: Request):
    managers = request.app.state.managers
    templates = request.app.state.templates
    orchestrator = managers.get("learning_orchestrator")

    items = _get_learner_data(orchestrator.behavior.store if orchestrator else None)

    return templates.TemplateResponse(request, "learners/behaviors.html", {
        "request": request,
        "items": items,
        "tab": "behaviors",
    })


@router.post("/learners/behaviors/{idx}/delete")
async def behavior_delete(request: Request, idx: int):
    managers = request.app.state.managers
    orchestrator = managers.get("learning_orchestrator")

    if orchestrator and orchestrator.behavior:
        keys = orchestrator.behavior.store.keys()
        if 0 <= idx < len(keys):
            await orchestrator.behavior.store.delete(keys[idx])

    return _make_flash_redirect("/learners/behaviors", "success", "已删除行为模式")


@router.get("/learners/scenes", response_class=HTMLResponse)
async def scene_list(request: Request):
    managers = request.app.state.managers
    templates = request.app.state.templates
    orchestrator = managers.get("learning_orchestrator")

    items = _get_learner_data(orchestrator.scene.store if orchestrator else None)

    return templates.TemplateResponse(request, "learners/scenes.html", {
        "request": request,
        "items": items,
        "tab": "scenes",
    })


@router.post("/learners/scenes/{cluster_id}/delete")
async def scene_delete(request: Request, cluster_id: str):
    managers = request.app.state.managers
    orchestrator = managers.get("learning_orchestrator")

    if orchestrator and orchestrator.scene:
        await orchestrator.scene.store.delete(cluster_id)

    return _make_flash_redirect("/learners/scenes", "success", "已删除场景簇")
