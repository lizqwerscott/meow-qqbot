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
    stats = await agent_engine.get_stats() if agent_engine else {}
    get_engagement_status = getattr(agent_engine, "get_engagement_status", None)
    stats["engagement_status"] = (
        await get_engagement_status() if get_engagement_status is not None else {}
    )

    context_manager = managers.get("context_manager")
    archive_index = managers.get("archive_index")
    if archive_index is not None:
        try:
            summary_reader = getattr(archive_index, "chat_summaries_for_webui", None)
            if callable(summary_reader):
                summaries, _total = await summary_reader(limit=None)
                archived_counts = {
                    str(summary["chat_id"]): int(summary.get("archive_count", 0))
                    for summary in summaries
                    if int(summary.get("archive_count", 0)) > 0
                }
            else:
                archived_counts = {
                    str(chat_id): 1 for chat_id in await archive_index.chat_ids()
                }
        except Exception as exc:
            _log.warning("读取账本归档会话统计失败: %s", exc)
            archived_counts = {}
    else:
        archived_counts = (
            await context_manager.get_archived_sessions_summary_async()
            if context_manager
            else {}
        )

    queue_details = list(stats.get("queue_sizes", {}).items())
    stats["queue_details"] = queue_details
    stats["queue_total"] = sum(q for _, q in queue_details)
    stats["archived_count"] = len(archived_counts)
    media_service = managers.get("media_service")
    stats["media_provider_status"] = (
        media_service.provider_status() if media_service else {}
    )

    return templates.TemplateResponse(
        request,
        "status/index.html",
        {
            "request": request,
            "stats": stats,
        },
    )
