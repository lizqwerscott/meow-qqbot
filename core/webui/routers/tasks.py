"""WebUI 路由 — 后台任务管理页面"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _get_managers(request: Request):
    return request.app.state.managers


def _get_mgrs(request: Request):
    """便捷获取 task_manager 和 cron_job_manager。"""
    mgrs = _get_managers(request)
    ae = mgrs.get("agent_engine")
    if not ae:
        return None, None
    return getattr(ae, "_task_manager", None), getattr(ae, "_cron_job_manager", None)


@router.get("", response_class=HTMLResponse)
async def tasks_list(request: Request):
    templates = request.app.state.templates
    task_manager, cron_mgr = _get_mgrs(request)

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
    task_manager, _ = _get_mgrs(request)
    task = task_manager.get_task(task_id) if task_manager else None
    if not task:
        return templates.TemplateResponse(request, "tasks/detail.html", {"task": None})
    return templates.TemplateResponse(request, "tasks/detail.html", {"request": request, "task": task})


# ── Cron Job 操作 ──


@router.post("/cron/{job_id}/delete")
async def cron_delete(request: Request, job_id: str):
    _, cron_mgr = _get_mgrs(request)
    if cron_mgr:
        cron_mgr.delete_job(job_id)
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/cron/{job_id}/pause")
async def cron_pause(request: Request, job_id: str):
    _, cron_mgr = _get_mgrs(request)
    if cron_mgr:
        cron_mgr.disable_job(job_id)
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/cron/{job_id}/resume")
async def cron_resume(request: Request, job_id: str):
    _, cron_mgr = _get_mgrs(request)
    if cron_mgr:
        cron_mgr.enable_job(job_id)
    return RedirectResponse(url="/tasks", status_code=303)


# ── Task 操作 ──


@router.post("/task/{task_id}/cancel")
async def task_cancel(request: Request, task_id: str):
    task_manager, _ = _get_mgrs(request)
    if task_manager:
        await task_manager.cancel_task(task_id)
    return RedirectResponse(url="/tasks", status_code=303)


@router.post("/task/{task_id}/delete")
async def task_delete(request: Request, task_id: str):
    task_manager, _ = _get_mgrs(request)
    if task_manager:
        task_manager._store.delete_task(task_id)
    return RedirectResponse(url="/tasks", status_code=303)
