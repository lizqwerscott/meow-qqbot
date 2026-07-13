import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse as StarletteRedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

from core.webui.auth import AuthMiddleware, verify_token
from core.webui.routers import status, emojis, nicknames, sessions, learners, tasks

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
    templates.env.filters["format_timestamp"] = _format_timestamp

    app.state.managers = managers
    app.state.webui_config = webui_config
    app.state.templates = templates

    # Static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Mount emoji images directory if it exists
    emoji_dir = Path("data/emojis")
    if emoji_dir.exists():
        app.mount("/static/emojis", StaticFiles(directory=str(emoji_dir)), name="emoji-files")

    # Auth middleware
    token = webui_config.get("token", "")
    if token:
        app.add_middleware(AuthMiddleware, token=token)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, error: str = ""):
        return templates.TemplateResponse(request, "login.html", {"request": request, "error": error})

    @app.post("/login")
    async def login_post(request: Request, token: str = Form(...)):
        expected = webui_config.get("token", "")
        if token == expected:
            resp = RedirectResponse(url="/status", status_code=HTTP_303_SEE_OTHER)
            resp.set_cookie(key="webui_token", value=token, httponly=True, max_age=86400 * 7)
            return resp
        return templates.TemplateResponse(request, "login.html", {"request": request, "error": "Token 无效"})

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

    return app


async def start_webui(app: FastAPI, webui_config: Dict[str, Any]) -> None:
    host = webui_config.get("host", "127.0.0.1")
    port = webui_config.get("port", 8080)
    log_level = webui_config.get("log_level", "warning")
    _log.info("WebUI 正在启动: http://%s:%s", host, port)
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    await server.serve()
