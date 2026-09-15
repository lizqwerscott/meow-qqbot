from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

router = APIRouter(tags=["nicknames"])


def _make_flash_redirect(url: str, category: str, message: str):
    separator = "&" if "?" in url else "?"
    return RedirectResponse(
        url=f"{url}{separator}flash_{category}={message}",
        status_code=HTTP_303_SEE_OTHER,
    )


@router.get("/nicknames", response_class=HTMLResponse)
async def nickname_list(request: Request):
    templates = request.app.state.templates
    identity_manager = request.app.state.managers.get("identity_manager")
    aliases = (
        identity_manager.list_channel_aliases()
        if identity_manager is not None
        else {"manual": {}, "auto": {}}
    )

    return templates.TemplateResponse(
        request,
        "nicknames/list.html",
        {
            "request": request,
            "manual_nicknames": aliases["manual"],
            "auto_nicknames": aliases["auto"],
        },
    )


@router.post("/nicknames/manual")
async def nickname_add_manual(
    request: Request,
    user_id: str = Form(...),
    nickname: str = Form(...),
):
    identity_manager = request.app.state.managers.get("identity_manager")
    if identity_manager is None:
        return _make_flash_redirect("/nicknames", "error", "身份管理器未就绪")
    identity_manager.set_channel_alias(user_id, nickname)
    return _make_flash_redirect("/nicknames", "success", f"已添加/更新昵称: {nickname}")


@router.post("/nicknames/manual/{user_id}/delete")
async def nickname_delete_manual(request: Request, user_id: str):
    identity_manager = request.app.state.managers.get("identity_manager")
    if identity_manager is not None:
        identity_manager.remove_channel_alias(user_id)
    return _make_flash_redirect("/nicknames", "success", "已删除")


@router.post("/nicknames/auto/{user_id}/delete")
async def nickname_delete_auto(request: Request, user_id: str):
    identity_manager = request.app.state.managers.get("identity_manager")
    if identity_manager is not None:
        identity_manager.remove_automatic_channel_aliases(user_id)
    return _make_flash_redirect("/nicknames", "success", "已删除自动昵称")


@router.post("/nicknames/auto/{user_id}/promote")
async def nickname_promote(request: Request, user_id: str):
    identity_manager = request.app.state.managers.get("identity_manager")
    if identity_manager is None:
        return _make_flash_redirect("/nicknames", "error", "身份管理器未就绪")
    if not identity_manager.promote_channel_alias(user_id):
        return _make_flash_redirect("/nicknames", "error", "未找到该昵称")
    return _make_flash_redirect("/nicknames", "success", "已提升为手动昵称")
