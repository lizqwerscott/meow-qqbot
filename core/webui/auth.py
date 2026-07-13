import logging

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER, HTTP_403_FORBIDDEN

_log = logging.getLogger(__name__)


def verify_token(request: Request, token: str) -> bool:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == token:
        return True
    if request.query_params.get("token") == token:
        return True
    cookie_token = request.cookies.get("webui_token")
    if cookie_token == token:
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.url.path.startswith(("/static/", "/login")):
            return await call_next(request)

        if not verify_token(request, self._token):
            if request.url.path.startswith("/api/"):
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Forbidden")
            return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)

        return await call_next(request)
