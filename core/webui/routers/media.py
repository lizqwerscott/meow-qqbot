from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/media", tags=["media"])


@router.get("", response_class=HTMLResponse)
async def media_list(
    request: Request, order: str = Query("desc", pattern="^(asc|desc)$")
):
    service = request.app.state.managers.get("media_service")
    items = await service.list_media(descending=order == "desc") if service else []
    usage = await service.usage() if service else (0, 0)
    return request.app.state.templates.TemplateResponse(
        request,
        "media/list.html",
        {"items": items, "order": order, "usage": usage, "nav_active": "media"},
    )


@router.post("/{media_id}/delete")
async def media_delete(request: Request, media_id: str):
    service = request.app.state.managers.get("media_service")
    if service:
        await service.delete_media(media_id)
    return RedirectResponse("/media", status_code=303)
