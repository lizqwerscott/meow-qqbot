from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

from core.session_identity import DeliveryTarget

router = APIRouter(tags=["identities"])


def _redirect(category: str, message: str) -> RedirectResponse:
    return RedirectResponse(
        url=f"/identities?flash_{category}={message}",
        status_code=HTTP_303_SEE_OTHER,
    )


@router.get("/identities", response_class=HTMLResponse)
async def identity_list(request: Request, ref: str = Query("")):
    manager = request.app.state.managers.get("identity_manager")
    highlight_ref = ref.strip()[:120]
    highlight_person_ref = ""
    chats = []
    suggestions = []
    if manager is None:
        rows = []
        direct_peers = []
        persons = []
    else:
        members = manager.list_members()
        direct_peers = manager.list_direct_peers()
        persons = manager.list_persons()
        chats = manager.list_chats()
        suggestions = manager.list_suggestions()
        if highlight_ref:
            matched = next(
                (
                    row
                    for row in [*members, *direct_peers]
                    if row.get("identity_ref") == highlight_ref
                ),
                None,
            )
            highlight_person_ref = (
                str(matched.get("person_ref") or "") if matched else ""
            )
            rows = [
                member
                for member in members
                if member.get("identity_ref") == highlight_ref
            ]
            direct_peers = [
                peer
                for peer in direct_peers
                if peer.get("identity_ref") == highlight_ref
            ]
        else:
            rows = members
    return request.app.state.templates.TemplateResponse(
        request,
        "identities/list.html",
        {
            "request": request,
            "members": rows,
            "direct_peers": direct_peers,
            "persons": persons,
            "chats": chats,
            "suggestions": suggestions,
            "highlight_ref": highlight_ref,
            "highlight_person_ref": highlight_person_ref,
        },
    )


@router.post("/identities/person")
async def create_person(request: Request, display_name: str = Form("")):
    manager = request.app.state.managers.get("identity_manager")
    if manager is None:
        return _redirect("error", "身份管理器未就绪")
    person_ref = manager.create_person(display_name.strip()[:80])
    return _redirect("success", f"已创建人物 {person_ref}")


@router.post("/identities/link")
async def link_person(
    request: Request,
    identity_ref: str = Form(...),
    person_ref: str = Form(...),
):
    manager = request.app.state.managers.get("identity_manager")
    if manager is None:
        return _redirect("error", "身份管理器未就绪")
    try:
        manager.link_person(identity_ref.strip(), person_ref.strip(), "webui-admin")
    except ValueError as exc:
        return _redirect("error", str(exc))
    except Exception:
        return _redirect("error", "身份关联失败")
    return _redirect("success", "已保存身份关联")


@router.post("/identities/alias")
async def set_chat_alias(
    request: Request,
    identity_ref: str = Form(...),
    chat_id: str = Form(...),
    alias: str = Form(...),
):
    manager = request.app.state.managers.get("identity_manager")
    if manager is None:
        return _redirect("error", "身份管理器未就绪")
    if not alias.strip():
        return _redirect("error", "群内称呼不能为空")
    try:
        manager.set_chat_alias_for_identity(
            DeliveryTarget("qq", "default", "group", chat_id.strip()),
            identity_ref.strip(),
            alias.strip()[:80],
        )
    except ValueError as exc:
        return _redirect("error", str(exc))
    return _redirect("success", "已保存群内称呼")


@router.post("/identities/suggest")
async def suggest_link(
    request: Request,
    identity_ref: str = Form(...),
    candidate_identity_ref: str = Form(...),
    reason: str = Form(""),
):
    manager = request.app.state.managers.get("identity_manager")
    if manager is None:
        return _redirect("error", "身份管理器未就绪")
    try:
        manager.suggest_link(
            identity_ref.strip(), candidate_identity_ref.strip(), reason.strip()
        )
    except ValueError as exc:
        return _redirect("error", str(exc))
    return _redirect("success", "已加入待确认关联")


@router.post("/identities/suggestions/generate")
async def generate_suggestions(request: Request):
    manager = request.app.state.managers.get("identity_manager")
    if manager is None:
        return _redirect("error", "身份管理器未就绪")
    created = manager.suggest_matching_identities()
    return _redirect("success", f"已生成 {created} 条待确认建议")


@router.post("/identities/suggestions/{suggestion_id}/resolve")
async def resolve_suggestion(
    request: Request,
    suggestion_id: int,
    decision: str = Form(...),
):
    manager = request.app.state.managers.get("identity_manager")
    if manager is None:
        return _redirect("error", "身份管理器未就绪")
    if decision not in {"accept", "reject"}:
        return _redirect("error", "无效的关联决定")
    try:
        manager.resolve_suggestion(
            suggestion_id,
            accepted=decision == "accept",
            decided_by="webui-admin",
        )
    except ValueError as exc:
        return _redirect("error", str(exc))
    return _redirect("success", "已处理身份关联建议")
