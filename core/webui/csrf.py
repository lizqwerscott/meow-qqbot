"""Cookie-bound CSRF protection for WebUI mutations."""

from __future__ import annotations

import hmac
import secrets

from markupsafe import Markup
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse

COOKIE_NAME = "webui_csrf"
FORM_NAME = "_csrf"
MAX_MUTATION_BODY_BYTES = 64 * 1024


def csrf_token(request: Request) -> str:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return token
    token = getattr(request.state, "csrf_token", "")
    if token:
        return token
    token = secrets.token_urlsafe(32)
    request.state.csrf_token = token
    return token


def csrf_input(request: Request) -> Markup:
    return Markup(
        f'<input type="hidden" name="{FORM_NAME}" value="{csrf_token(request)}">'
    )


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.method == "POST" and request.url.path != "/login":
            if not request.app.state.webui_config.get(
                "token", ""
            ) and request.url.path.startswith("/settings/"):
                return PlainTextResponse("Forbidden", status_code=403)
            if not request.app.state.webui_config.get("token", ""):
                return await call_next(request)
            try:
                body = await request.body()
                if len(body) > MAX_MUTATION_BODY_BYTES:
                    return PlainTextResponse("Request too large", status_code=413)
                form = await request.form()
            except Exception:
                return PlainTextResponse("Invalid form", status_code=403)
            cookie = request.cookies.get(COOKIE_NAME, "")
            submitted = str(form.get(FORM_NAME, ""))
            if (
                not cookie
                or not submitted
                or not hmac.compare_digest(cookie, submitted)
            ):
                return PlainTextResponse("CSRF validation failed", status_code=403)

        response = await call_next(request)
        token = getattr(request.state, "csrf_token", "")
        if token:
            response.set_cookie(
                COOKIE_NAME,
                token,
                httponly=True,
                samesite="lax",
                secure=request.url.scheme == "https",
                max_age=86400,
            )
        return response
