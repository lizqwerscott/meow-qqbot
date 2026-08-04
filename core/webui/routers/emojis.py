import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

_log = logging.getLogger(__name__)

router = APIRouter(tags=["emojis"])
PAGE_SIZE = 20


def _flash_url(url: str, category: str, message: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}flash_{category}={message}"


def _make_flash_redirect(url: str, category: str, message: str):
    return RedirectResponse(
        url=_flash_url(url, category, message), status_code=HTTP_303_SEE_OTHER
    )


@router.get("/emojis", response_class=HTMLResponse)
async def emoji_list(
    request: Request,
    page: int = Query(1, ge=1),
    q: Optional[str] = Query(None),
    sort_by: str = Query("has_tags"),
    sort_order: str = Query("desc"),
):
    managers = request.app.state.managers
    templates = request.app.state.templates
    emoji_manager = managers.get("emoji_manager")

    if q:
        all_items = (
            emoji_manager.find_emojis(q, max_results=100)
            if hasattr(emoji_manager, "find_emojis")
            else []
        )
        total = len(all_items)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    else:
        result = (
            emoji_manager.list_emojis(
                page=page, page_size=PAGE_SIZE, sort_by=sort_by, sort_order=sort_order
            )
            if emoji_manager
            else {"emojis": [], "total": 0}
        )
        all_items = result.get("emojis", [])
        total = result.get("total", 0)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    return templates.TemplateResponse(
        request,
        "emojis/list.html",
        {
            "request": request,
            "emojis": all_items,
            "page": page,
            "total_pages": total_pages,
            "query": q or "",
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
    )


@router.get("/emojis/{emoji_hash}", response_class=HTMLResponse)
async def emoji_detail(request: Request, emoji_hash: str):
    managers = request.app.state.managers
    templates = request.app.state.templates
    emoji_manager = managers.get("emoji_manager")

    record = (
        emoji_manager.get_info(emoji_hash)
        if hasattr(emoji_manager, "get_info")
        else None
    )
    if not record:
        return _make_flash_redirect(
            "/emojis", "error", f"未找到表情: {emoji_hash[:12]}.."
        )

    return templates.TemplateResponse(
        request,
        "emojis/detail.html",
        {
            "request": request,
            "emoji": record,
        },
    )


@router.post("/emojis/{emoji_hash}")
async def emoji_update(
    request: Request,
    emoji_hash: str,
    description: str = Form(""),
    tags: str = Form(""),
):
    managers = request.app.state.managers
    emoji_manager = managers.get("emoji_manager")

    tag_list = [
        t.strip()
        for t in tags.replace("，", ",").replace("、", ",").split(",")
        if t.strip()
    ]

    success = await emoji_manager.set_custom(
        emoji_hash, description=description or None, tags=tag_list or None
    )
    if not success:
        return _make_flash_redirect(f"/emojis/{emoji_hash}", "error", "更新失败")

    return _make_flash_redirect(f"/emojis/{emoji_hash}", "success", "已更新")


@router.post("/emojis/{emoji_hash}/reset")
async def emoji_reset(request: Request, emoji_hash: str):
    managers = request.app.state.managers
    emoji_manager = managers.get("emoji_manager")

    success = await emoji_manager.reset_to_auto(emoji_hash)
    if not success:
        return _make_flash_redirect(f"/emojis/{emoji_hash}", "error", "重置失败")

    return _make_flash_redirect(f"/emojis/{emoji_hash}", "success", "已重置为自动识别")


@router.post("/emojis/{emoji_hash}/delete")
async def emoji_delete(request: Request, emoji_hash: str):
    managers = request.app.state.managers
    emoji_manager = managers.get("emoji_manager")

    success = await emoji_manager.delete_emoji(emoji_hash)
    if not success:
        return _make_flash_redirect(
            "/emojis", "error", f"删除失败: {emoji_hash[:12]}.."
        )

    return _make_flash_redirect("/emojis", "success", f"已删除表情 {emoji_hash[:12]}..")


@router.post("/emojis/{emoji_hash}/reanalyze")
async def emoji_reanalyze(request: Request, emoji_hash: str):
    managers = request.app.state.managers
    emoji_manager = managers.get("emoji_manager")

    success = await emoji_manager.reanalyze_emoji(emoji_hash)
    if not success:
        return _make_flash_redirect(f"/emojis/{emoji_hash}", "error", "重新分析失败")

    return _make_flash_redirect(
        f"/emojis/{emoji_hash}", "success", "已重新使用 VLM 分析"
    )
