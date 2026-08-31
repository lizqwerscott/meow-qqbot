"""Authenticated WebUI controls for runtime engagement settings."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from core.runtime_settings import RuntimeSettingsConflict, RuntimeSettingsDegraded

router = APIRouter(prefix="/settings", tags=["settings"])
_log = logging.getLogger(__name__)
_NONCE_TTL = 600.0


def _coordinator(request: Request):
    coordinator = request.app.state.managers.get("runtime_settings")
    if coordinator is None:
        raise HTTPException(status_code=503, detail="runtime settings unavailable")
    return coordinator


def _target_digest(targets) -> str:
    value = sorted(target.chat_id for target in targets)
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _chat_id_digest(chat_ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(chat_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _issue_nonce(request: Request, revision: int, targets) -> str:
    nonce = secrets.token_urlsafe(24)
    pending = getattr(request.app.state, "settings_nonces", {})
    pending[nonce] = (revision, _target_digest(targets), time.time() + _NONCE_TTL)
    request.app.state.settings_nonces = pending
    return nonce


def _consume_nonce(request: Request, nonce: str, revision: int, targets) -> bool:
    pending = getattr(request.app.state, "settings_nonces", {})
    item = pending.pop(nonce, None)
    if item is None:
        return False
    expected_revision, expected_digest, expires_at = item
    return (
        time.time() <= expires_at
        and expected_revision == revision
        and expected_digest == _target_digest(targets)
    )


def _issue_confirmation(
    request: Request,
    revision: int,
    targets,
    active_chats: list[str],
    patch: dict,
) -> str:
    nonce = secrets.token_urlsafe(24)
    now = time.time()
    pending = getattr(request.app.state, "settings_confirmations", {})
    pending = {key: value for key, value in pending.items() if value[3] > now}
    pending[nonce] = (
        revision,
        _target_digest(targets),
        _chat_id_digest(active_chats),
        now + _NONCE_TTL,
        patch,
    )
    request.app.state.settings_confirmations = pending
    return nonce


def _consume_confirmation(
    request: Request,
    nonce: str,
    revision: int,
    targets,
) -> dict | None:
    pending = getattr(request.app.state, "settings_confirmations", {})
    item = pending.pop(nonce, None)
    if item is None:
        return None
    (
        expected_revision,
        expected_target_digest,
        expected_chat_digest,
        expires_at,
        patch,
    ) = item
    active_chats = _chat_ids(patch["group_ambient_active_chats"])
    if (
        time.time() > expires_at
        or expected_revision != revision
        or expected_target_digest != _target_digest(targets)
        or expected_chat_digest != _chat_id_digest(active_chats)
    ):
        return None
    return patch


async def _render(request: Request, error: str = ""):
    coordinator = _coordinator(request)
    snapshot = coordinator.snapshot()
    targets = await coordinator.targets()
    audits = await coordinator.audit(limit=30)
    group_catalog = await _group_catalog(request, targets, snapshot)
    nonce = _issue_nonce(request, snapshot.revision, targets)
    return request.app.state.templates.TemplateResponse(
        request,
        "settings/engagement.html",
        {
            "request": request,
            "snapshot": snapshot.to_dict(),
            "targets": targets,
            "group_catalog": group_catalog,
            "audits": audits,
            "active_nonce": nonce,
            "error": error,
            "degraded": coordinator.degraded,
            "degraded_reason": coordinator.degraded_reason,
        },
    )


async def _group_catalog(request: Request, targets, snapshot) -> list[dict]:
    """Build a conservative picker from groups the bot has already observed."""
    managers = request.app.state.managers
    context_manager = managers.get("context_manager")
    group_ids: set[str] = set(snapshot.config.group_ambient_active_chats)
    observed_group_ids: set[str] = set()

    if context_manager is not None:
        get_ids = getattr(context_manager, "get_all_disk_chat_ids_async", None)
        get_type = getattr(context_manager, "get_chat_type", None)
        if callable(get_ids) and callable(get_type):
            try:
                observed_group_ids.update(
                    chat_id for chat_id in await get_ids() if get_type(chat_id) is True
                )
            except Exception:
                _log.warning(
                    "unable to load group catalog for settings page", exc_info=True
                )

    target_by_id = {target.chat_id: target for target in targets}
    known_group_ids = set(observed_group_ids)
    group_ids.update(observed_group_ids)
    group_ids.update(
        target.chat_id for target in targets if target.verification_status != "removed"
    )
    catalog = []
    for chat_id in sorted(group_ids):
        target = target_by_id.get(chat_id)
        catalog.append(
            {
                "chat_id": chat_id,
                "display_name": (
                    "已配置目标"
                    if target is not None and chat_id in known_group_ids
                    else "已发现群聊" if chat_id in known_group_ids else "待确认目标"
                ),
                "verification_status": (
                    target.verification_status if target is not None else "unverified"
                ),
                "last_observed_at": (
                    target.last_observed_at if target is not None else None
                ),
                "selected": chat_id in snapshot.config.group_ambient_active_chats,
            }
        )
    return catalog


@router.get("/engagement", response_class=HTMLResponse)
async def engagement_settings(request: Request):
    return await _render(request)


def _chat_ids(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        values = raw.replace(",", "\n").splitlines()
    else:
        values = raw
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


@router.post("/engagement/update")
async def engagement_update(
    request: Request,
    revision: int = Form(...),
    mode: str = Form(...),
    active_chats: list[str] = Form([]),
    idle_ms: int = Form(...),
    cooldown_seconds: float = Form(...),
    quiet_cooldown_seconds: float = Form(...),
    window_seconds: float = Form(...),
    max_turns_per_window: int = Form(...),
    max_age_seconds: float = Form(...),
    min_messages: int = Form(...),
    allow_single_question: bool = Form(False),
    allow_single_media: bool = Form(False),
    quote: bool = Form(False),
    stale_quote_seconds: float = Form(...),
    active_nonce: str = Form(""),
):
    coordinator = _coordinator(request)
    targets = await coordinator.targets()
    active_chat_ids = _chat_ids(active_chats)
    patch = {
        "group_ambient_mode": mode,
        "group_ambient_active_chats": active_chat_ids,
        "group_ambient_idle_ms": idle_ms,
        "group_ambient_cooldown_seconds": cooldown_seconds,
        "group_ambient_quiet_cooldown_seconds": quiet_cooldown_seconds,
        "group_ambient_window_seconds": window_seconds,
        "group_ambient_max_turns_per_window": max_turns_per_window,
        "group_ambient_max_age_seconds": max_age_seconds,
        "group_ambient_min_messages": min_messages,
        "group_ambient_allow_single_question": allow_single_question,
        "group_ambient_allow_single_media": allow_single_media,
        "group_ambient_quote": quote,
        "group_ambient_stale_quote_seconds": stale_quote_seconds,
    }
    if mode == "active" and not _consume_nonce(
        request, active_nonce, revision, targets
    ):
        raise HTTPException(status_code=422, detail="active mode confirmation expired")
    if mode == "active":
        target_by_id = {target.chat_id: target for target in targets}
        selected_targets = [
            {
                "chat_id": chat_id,
                "verification_status": (
                    target_by_id[chat_id].verification_status
                    if chat_id in target_by_id
                    else "unverified"
                ),
            }
            for chat_id in active_chat_ids
        ]
        confirmation_nonce = _issue_confirmation(
            request, revision, targets, active_chat_ids, patch
        )
        return request.app.state.templates.TemplateResponse(
            request,
            "settings/engagement_confirm.html",
            {
                "request": request,
                "revision": revision,
                "confirmation_nonce": confirmation_nonce,
                "selected_targets": selected_targets,
                "verified_count": sum(
                    item["verification_status"] == "verified"
                    for item in selected_targets
                ),
                "target_limit": 100,
            },
        )
    try:
        await coordinator.update(
            expected_revision=revision,
            patch=patch,
            remote_ip=request.client.host if request.client else None,
        )
    except RuntimeSettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeSettingsDegraded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)


@router.post("/engagement/confirm")
async def engagement_confirm(
    request: Request,
    revision: int = Form(...),
    confirmation_nonce: str = Form(...),
):
    coordinator = _coordinator(request)
    targets = await coordinator.targets()
    patch = _consume_confirmation(request, confirmation_nonce, revision, targets)
    if patch is None:
        raise HTTPException(status_code=422, detail="active mode confirmation expired")
    try:
        await coordinator.update(
            expected_revision=revision,
            patch=patch,
            remote_ip=request.client.host if request.client else None,
        )
    except RuntimeSettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeSettingsDegraded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)


@router.post("/engagement/pause")
async def engagement_pause(request: Request, revision: int = Form(...)):
    coordinator = _coordinator(request)
    try:
        await coordinator.update(
            expected_revision=revision,
            patch={"group_ambient_mode": "off"},
            remote_ip=request.client.host if request.client else None,
        )
    except RuntimeSettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeSettingsDegraded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)


@router.post("/engagement/clear")
async def engagement_clear(
    request: Request,
    revision: int = Form(...),
    key: str | None = Form(None),
):
    coordinator = _coordinator(request)
    try:
        await coordinator.clear(
            expected_revision=revision,
            key=key or None,
            remote_ip=request.client.host if request.client else None,
        )
    except RuntimeSettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeSettingsDegraded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)


@router.post("/engagement/targets/{chat_id}/verify")
async def verify_target(request: Request, chat_id: str):
    try:
        await _coordinator(request).verify_target(chat_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)


@router.post("/engagement/targets/add")
async def add_target(
    request: Request,
    revision: int = Form(...),
    chat_id: str = Form(...),
):
    coordinator = _coordinator(request)
    current = coordinator.snapshot().config.group_ambient_active_chats
    try:
        await coordinator.update(
            expected_revision=revision,
            patch={
                "group_ambient_active_chats": list(
                    dict.fromkeys((*current, chat_id.strip()))
                )
            },
            remote_ip=request.client.host if request.client else None,
        )
    except RuntimeSettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)


@router.post("/engagement/targets/{chat_id}/remove")
async def remove_target(
    request: Request,
    chat_id: str,
    revision: int = Form(...),
):
    coordinator = _coordinator(request)
    current = coordinator.snapshot().config.group_ambient_active_chats
    try:
        await coordinator.update(
            expected_revision=revision,
            patch={
                "group_ambient_active_chats": [
                    item for item in current if item != chat_id
                ]
            },
            remote_ip=request.client.host if request.client else None,
        )
        await coordinator.remove_target(chat_id)
    except RuntimeSettingsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeSettingsDegraded as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RedirectResponse("/settings/engagement", status_code=303)
