"""WebUI 路由 — 后台任务管理页面"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _get_managers(request: Request):
    return request.app.state.managers


@router.get("", response_class=HTMLResponse)
async def tasks_list(request: Request):
    templates = request.app.state.templates
    managers = _get_managers(request)
    agent_engine = managers.get("agent_engine")
    if not agent_engine:
        return templates.TemplateResponse(request, "tasks/list.html", {"tasks": [], "cron_jobs": []})

    task_manager = getattr(agent_engine, "_task_manager", None)
    cron_mgr = getattr(agent_engine, "_cron_job_manager", None)

    tasks = task_manager.list_tasks(limit=50) if task_manager else []
    cron_jobs = cron_mgr.list_jobs() if cron_mgr else []

    return templates.TemplateResponse(
        request,
        "tasks/list.html",
        {
            "request": request,
            "tasks": tasks,
            "cron_jobs": cron_jobs,
            "active_tasks": [t for t in tasks if t.status.value in ("pending", "running")],
        },
    )


@router.get("/detail/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: str):
    templates = request.app.state.templates
    managers = _get_managers(request)
    agent_engine = managers.get("agent_engine")
    task = None
    if agent_engine:
        task_manager = getattr(agent_engine, "_task_manager", None)
        if task_manager:
            task = task_manager.get_task(task_id)
    if not task:
        return templates.TemplateResponse(request, "tasks/detail.html", {"task": None})
    return templates.TemplateResponse(request, "tasks/detail.html", {"request": request, "task": task})
