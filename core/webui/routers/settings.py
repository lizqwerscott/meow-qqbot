"""Authenticated WebUI controls for runtime engagement settings."""

from __future__ import annotations

import hashlib
import json
import secrets
import time

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from core.runtime_settings import RuntimeSettingsConflict, RuntimeSettingsDegraded

router = APIRouter(prefix="/settings", tags=["settings"])
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


async def _render(request: Request, error: str = ""):
    coordinator = _coordinator(request)
    snapshot = coordinator.snapshot()
    targets = await coordinator.targets()
    audits = await coordinator.audit(limit=30)
    nonce = _issue_nonce(request, snapshot.revision, targets)
    return request.app.state.templates.TemplateResponse(
        request,
        "settings/engagement.html",
        {
            "request": request,
            "snapshot": snapshot.to_dict(),
            "targets": targets,
            "audits": audits,
            "active_nonce": nonce,
            "error": error,
            "degraded": coordinator.degraded,
            "degraded_reason": coordinator.degraded_reason,
        },
    )


@router.get("/engagement", response_class=HTMLResponse)
async def engagement_settings(request: Request):
    return await _render(request)


def _chat_ids(raw: str) -> list[str]:
    return [
        item.strip() for item in raw.replace(",", "\n").splitlines() if item.strip()
    ]


@router.post("/engagement/update")
async def engagement_update(
    request: Request,
    revision: int = Form(...),
    mode: str = Form(...),
    active_chats: str = Form(""),
    interval_seconds: int = Form(...),
    jitter_seconds: int = Form(...),
    active_hours_start: str = Form(...),
    active_hours_end: str = Form(...),
    timezone: str = Form(...),
    cooldown_seconds: float = Form(...),
    quiet_cooldown_seconds: float = Form(...),
    window_seconds: float = Form(...),
    max_turns_per_window: int = Form(...),
    reservation_seconds: float = Form(...),
    active_nonce: str = Form(""),
):
    coordinator = _coordinator(request)
    targets = await coordinator.targets()
    if mode == "active" and not _consume_nonce(
        request, active_nonce, revision, targets
    ):
        raise HTTPException(status_code=422, detail="active mode confirmation expired")
    patch = {
        "group_proactive_mode": mode,
        "group_proactive_active_chats": _chat_ids(active_chats),
        "group_proactive_interval_seconds": interval_seconds,
        "group_proactive_jitter_seconds": jitter_seconds,
        "group_proactive_active_hours_start": active_hours_start,
        "group_proactive_active_hours_end": active_hours_end,
        "group_proactive_timezone": timezone,
        "group_proactive_cooldown_seconds": cooldown_seconds,
        "group_proactive_quiet_cooldown_seconds": quiet_cooldown_seconds,
        "group_proactive_window_seconds": window_seconds,
        "group_proactive_max_turns_per_window": max_turns_per_window,
        "group_proactive_reservation_seconds": reservation_seconds,
    }
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
    current = coordinator.snapshot().config.group_proactive_active_chats
    try:
        await coordinator.update(
            expected_revision=revision,
            patch={
                "group_proactive_active_chats": list(
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
    current = coordinator.snapshot().config.group_proactive_active_chats
    try:
        await coordinator.update(
            expected_revision=revision,
            patch={
                "group_proactive_active_chats": [
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
