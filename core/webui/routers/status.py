import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

_log = logging.getLogger(__name__)

router = APIRouter(tags=["status"])


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    managers = request.app.state.managers
    templates = request.app.state.templates

    agent_engine = managers.get("agent_engine")
    if agent_engine and agent_engine.hindsight:
        await agent_engine.hindsight.health()
    stats = agent_engine.get_stats() if agent_engine else {}

    # Add queue details
    queue_details = list(stats.get("queue_sizes", {}).items())
    stats["queue_details"] = queue_details
    stats["queue_total"] = sum(q for _, q in queue_details)

    return templates.TemplateResponse(request, "status/index.html", {
        "request": request,
        "stats": stats,
    })
