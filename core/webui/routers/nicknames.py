import json
import logging
import os

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

_log = logging.getLogger(__name__)

router = APIRouter(tags=["nicknames"])

MANUAL_PATH = "nicknames.json"
AUTO_PATH = "data/nicknames.json"


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            _log.warning(f"加载昵称文件失败 [{path}]: {e}")
            return {}
    return {}


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) if "/" in path else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _make_flash_redirect(url: str, category: str, message: str):
    separator = "&" if "?" in url else "?"
    return RedirectResponse(url=f"{url}{separator}flash_{category}={message}", status_code=HTTP_303_SEE_OTHER)


@router.get("/nicknames", response_class=HTMLResponse)
async def nickname_list(request: Request):
    templates = request.app.state.templates

    manual = _load_json(MANUAL_PATH)
    auto = _load_json(AUTO_PATH)

    return templates.TemplateResponse(request, "nicknames/list.html", {
        "request": request,
        "manual_nicknames": manual,
        "auto_nicknames": auto,
    })


@router.post("/nicknames/manual")
async def nickname_add_manual(
    request: Request,
    user_id: str = Form(...),
    nickname: str = Form(...),
):
    manual = _load_json(MANUAL_PATH)
    manual[user_id] = nickname
    _save_json(MANUAL_PATH, manual)

    # Also update the nickname manager if available
    nm = request.app.state.managers.get("nickname_manager")
    if nm:
        nm.nicknames = dict(manual)
        nm.save_auto()

    return _make_flash_redirect("/nicknames", "success", f"已添加/更新昵称: {nickname}")


@router.post("/nicknames/manual/{user_id}/delete")
async def nickname_delete_manual(request: Request, user_id: str):
    manual = _load_json(MANUAL_PATH)
    manual.pop(user_id, None)
    _save_json(MANUAL_PATH, manual)

    nm = request.app.state.managers.get("nickname_manager")
    if nm:
        nm.nicknames = dict(manual)

    return _make_flash_redirect("/nicknames", "success", "已删除")


@router.post("/nicknames/auto/{user_id}/delete")
async def nickname_delete_auto(request: Request, user_id: str):
    auto = _load_json(AUTO_PATH)
    auto.pop(user_id, None)
    _save_json(AUTO_PATH, auto)

    nm = request.app.state.managers.get("nickname_manager")
    if nm:
        nm.auto_nicknames = dict(auto)
        nm.save_auto()

    return _make_flash_redirect("/nicknames", "success", "已删除自动昵称")


@router.post("/nicknames/auto/{user_id}/promote")
async def nickname_promote(request: Request, user_id: str):
    auto = _load_json(AUTO_PATH)
    if user_id not in auto:
        return _make_flash_redirect("/nicknames", "error", "未找到该昵称")

    manual = _load_json(MANUAL_PATH)
    entry = auto[user_id]
    if isinstance(entry, dict):
        aliases = entry.get("aliases", [])
        latest = aliases[-1] if aliases else user_id
    else:
        latest = entry
    manual[user_id] = latest
    _save_json(MANUAL_PATH, manual)

    auto.pop(user_id)
    _save_json(AUTO_PATH, auto)

    nm = request.app.state.managers.get("nickname_manager")
    if nm:
        nm.nicknames = dict(manual)
        nm.auto_nicknames = dict(auto)
        nm.save_auto()

    return _make_flash_redirect("/nicknames", "success", "已提升为手动昵称")
