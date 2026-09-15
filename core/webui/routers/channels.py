from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from core.webui.group_directory import GroupDirectory

router = APIRouter(tags=["channels"])


@router.get("/channels", response_class=HTMLResponse)
async def channel_list(request: Request):
    managers = request.app.state.managers
    groups = GroupDirectory(
        identity_manager=managers.get("identity_manager"),
        channel_info_provider=managers.get("channel_info_provider"),
    ).list_groups()
    return request.app.state.templates.TemplateResponse(
        request,
        "channels/list.html",
        {"request": request, "groups": groups},
    )
