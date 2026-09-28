import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

_log = logging.getLogger(__name__)

router = APIRouter(tags=["usage"])

_DIMENSIONS = ("day", "model", "source", "provider", "chat")
_PAGE_SIZE = 50


def _script_json(value: Any) -> str:
    """JSON safe to inline inside a <script> tag (escapes tag delimiters)."""
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _parse_day(value: str | None) -> float | None:
    """Parse ``YYYY-MM-DD`` (or an epoch) into a local-midnight timestamp."""
    if not value:
        return None
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%d"))
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


def _format_ts(value: float | None) -> str:
    if not value:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))


@router.get("/usage", response_class=HTMLResponse)
async def usage_page(request: Request):
    managers = request.app.state.managers
    templates = request.app.state.templates
    log = managers.get("token_usage_log")

    since_raw = request.query_params.get("since", "")
    until_raw = request.query_params.get("until", "")
    try:
        page = max(1, int(request.query_params.get("page", 1)))
    except ValueError:
        page = 1

    since = _parse_day(since_raw)
    until = _parse_day(until_raw)
    if until is not None:
        until += 86399.0  # include the whole end day

    context = {
        "request": request,
        "nav_active": "usage",
        "available": log is not None,
        "since": since_raw,
        "until": until_raw,
        "page": page,
        "page_size": _PAGE_SIZE,
        "summary": {},
        "breakdowns": {},
        "records": [],
        "total_records": 0,
        "total_pages": 1,
    }
    if log is None:
        return templates.TemplateResponse(request, "usage/index.html", context)

    try:
        summary = log.summary(since=since, until=until)
        breakdowns = {
            dimension: log.breakdown(dimension, since=since, until=until)
            for dimension in _DIMENSIONS
        }
        total_records = log.count(since=since, until=until)
        records = log.recent(
            limit=_PAGE_SIZE, offset=(page - 1) * _PAGE_SIZE, since=since, until=until
        )
        charts = {
            "day": log.timeseries("day", since=since, until=until),
            "hour": log.timeseries("hour", since=since, until=until, limit=72),
            "source": breakdowns["source"],
            "model": breakdowns["model"][:10],
            "provider": breakdowns["provider"],
            "chat": breakdowns["chat"][:10],
        }
    except Exception as exc:
        _log.warning("读取 token 用量账本失败: %s", exc)
        context["available"] = False
        return templates.TemplateResponse(request, "usage/index.html", context)

    for record in records:
        record["recorded_at_str"] = _format_ts(record.get("recorded_at"))
        record["cost_str"] = f"{float(record.get('cost') or 0.0):.6f}"

    context.update(
        {
            "summary": summary,
            "summary_earliest": _format_ts(summary.get("earliest")),
            "summary_latest": _format_ts(summary.get("latest")),
            "breakdowns": breakdowns,
            "records": records,
            "total_records": total_records,
            "total_pages": max(1, (total_records + _PAGE_SIZE - 1) // _PAGE_SIZE),
            "charts_json": _script_json(charts),
        }
    )
    return templates.TemplateResponse(request, "usage/index.html", context)
