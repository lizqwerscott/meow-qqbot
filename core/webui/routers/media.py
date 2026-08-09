import re

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

router = APIRouter(prefix="/media", tags=["media"])


@router.get("", response_class=HTMLResponse)
async def media_list(
    request: Request, order: str = Query("desc", pattern="^(asc|desc)$")
):
    service = request.app.state.managers.get("media_service")
    items = await service.list_media(descending=order == "desc") if service else []
    usage = await service.usage if service else (0, 0)
    return request.app.state.templates.TemplateResponse(
        request,
        "media/list.html",
        {"items": items, "order": order, "usage": usage, "nav_active": "media"},
    )


@router.get("/{media_id}", response_class=HTMLResponse)
async def media_detail(request: Request, media_id: str):
    service = request.app.state.managers.get("media_service")
    item = await service.get_media(media_id) if service else None
    if not item:
        raise HTTPException(status_code=404, detail="媒体不存在")
    text_preview = None
    if item["storage_status"] == "ready" and item["mime_type"] == "text/plain":
        text_preview = await service.get_text_preview(media_id)
    return request.app.state.templates.TemplateResponse(
        request,
        "media/detail.html",
        {"item": item, "nav_active": "media"} | {"text_preview": text_preview},
    )


@router.get("/{media_id}/content")
async def media_content(request: Request, media_id: str, download: bool = False):
    service = request.app.state.managers.get("media_service")
    result = (
        await (
            service.get_download_path(media_id)
            if download
            else service.get_preview_path(media_id)
        )
        if service
        else None
    )
    if not result:
        raise HTTPException(status_code=404, detail="媒体不可预览或超过大小限制")
    path, mime_type, filename = result
    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "media")[:120]
    response = FileResponse(
        path,
        media_type=mime_type,
        filename=safe_filename,
        content_disposition_type="attachment" if download else "inline",
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "private, no-store"
    return response


@router.post("/{media_id}/delete")
async def media_delete(request: Request, media_id: str):
    service = request.app.state.managers.get("media_service")
    if service:
        await service.delete_media(media_id)
    return RedirectResponse("/media", status_code=303)
