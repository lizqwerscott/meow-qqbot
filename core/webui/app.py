import ipaddress
import logging
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse as StarletteRedirectResponse
from starlette.status import HTTP_303_SEE_OTHER, HTTP_429_TOO_MANY_REQUESTS

from core.webui.auth import AuthMiddleware, verify_token
from core.webui.csrf import CSRFMiddleware, csrf_input

# 登录速率限制（内存中，每 IP 5 次/分钟）
_LOGIN_ATTEMPTS: Dict[str, list] = {}
_LOGIN_WINDOW = 60
_LOGIN_MAX_ATTEMPTS = 5
_LOGIN_MAX_IPS = 1000


def _check_login_rate(client_ip: str) -> None:
    now = time.time()
    attempts = _LOGIN_ATTEMPTS.setdefault(client_ip, [])
    # 清理过期的记录
    attempts[:] = [t for t in attempts if now - t < _LOGIN_WINDOW]
    if len(attempts) >= _LOGIN_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="登录尝试过于频繁，请稍后再试",
        )
    attempts.append(now)
    # 控制 IP 表上限，防止内存无限增长
    if len(_LOGIN_ATTEMPTS) > _LOGIN_MAX_IPS:
        for ip in list(_LOGIN_ATTEMPTS.keys()):
            if not _LOGIN_ATTEMPTS[ip]:
                del _LOGIN_ATTEMPTS[ip]
            if len(_LOGIN_ATTEMPTS) <= _LOGIN_MAX_IPS:
                break


def _client_ip(request: Request, webui_config: dict) -> str:
    peer = request.client.host if request.client else "unknown"
    configured = webui_config.get("trusted_proxies", ())
    if isinstance(configured, str):
        configured = (configured,)
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    trusted = False
    for value in configured or ():
        try:
            trusted = peer_address in ipaddress.ip_network(value, strict=False)
        except ValueError:
            continue
        if trusted:
            break
    if trusted:
        return request.headers.get("x-forwarded-for", peer).split(",")[0].strip()
    return peer


from core.webui.routers import (
    emojis,
    learners,
    media,
    nicknames,
    routing,
    sessions,
    settings,
    status,
    tasks,
)

_log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


def _format_timestamp(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _get_flashed_messages(request: Request) -> list:
    messages = []
    for cat in ("success", "error", "info"):
        key = f"flash_{cat}"
        val = request.query_params.get(key)
        if val:
            messages.append((cat, val))
    return messages


def create_app(managers: Dict[str, Any], webui_config: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="猫猫管理")

    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.globals["get_flashed_messages"] = _get_flashed_messages
    templates.env.globals["csrf_input"] = csrf_input
    templates.env.filters["format_timestamp"] = _format_timestamp

    app.state.managers = managers
    app.state.webui_config = webui_config
    app.state.templates = templates
    app.state.settings_nonces = {}
    app.add_middleware(CSRFMiddleware)

    # Mount emoji images directory first so it takes precedence over /static
    emoji_dir = Path("data/emojis")
    if emoji_dir.exists():
        app.mount(
            "/static/emojis", StaticFiles(directory=str(emoji_dir)), name="emoji-files"
        )

    # Static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Auth middleware
    token = webui_config.get("token", "")
    if token:
        app.add_middleware(AuthMiddleware, token=token)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, error: str = ""):
        return templates.TemplateResponse(
            request, "login.html", {"request": request, "error": error}
        )

    @app.post("/login")
    async def login_post(request: Request, token: str = Form(...)):
        client_ip = _client_ip(request, webui_config)
        _check_login_rate(client_ip)
        expected = webui_config.get("token", "")
        if token == expected:
            resp = RedirectResponse(url="/status", status_code=HTTP_303_SEE_OTHER)
            resp.set_cookie(
                key="webui_token",
                value=token,
                httponly=True,
                max_age=86400 * 7,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
            resp.set_cookie(
                "webui_csrf",
                secrets.token_urlsafe(32),
                httponly=True,
                max_age=86400,
                samesite="lax",
                secure=request.url.scheme == "https",
            )
            return resp
        return templates.TemplateResponse(
            request, "login.html", {"request": request, "error": "Token 无效"}
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/status")

    # Register routers
    app.include_router(status.router)
    app.include_router(emojis.router)
    app.include_router(nicknames.router)
    app.include_router(sessions.router)
    app.include_router(learners.router)
    app.include_router(tasks.router)
    app.include_router(media.router)
    app.include_router(routing.router)
    app.include_router(settings.router)

    return app


async def start_webui(app: FastAPI, webui_config: Dict[str, Any]) -> None:
    host = webui_config.get("host", "127.0.0.1")
    port = webui_config.get("port", 8080)
    log_level = webui_config.get("log_level", "warning")
    _log.info("WebUI 正在启动: http://%s:%s", host, port)
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    await server.serve()
