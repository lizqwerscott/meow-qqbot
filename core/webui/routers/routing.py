"""Read-only WebUI pages for content-free deterministic routing audit."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["routing"])


def _agent_engine(request: Request):
    engine = request.app.state.managers.get("agent_engine")
    if engine is None:
        raise HTTPException(status_code=503, detail="AgentEngine unavailable")
    return engine


@router.get("/routing", response_class=HTMLResponse)
async def routing_list(
    request: Request,
    mode: str = Query(""),
    reason_code: str = Query(""),
    source: str = Query(""),
    chat_prefix: str = Query(""),
):
    engine = _agent_engine(request)
    store = getattr(engine, "routing_audit_store", None)
    records = (
        await store.list_records(
            mode=mode or None,
            reason_code=reason_code or None,
            source=source or None,
            chat_prefix=chat_prefix.strip() or None,
        )
        if store is not None
        else []
    )
    return request.app.state.templates.TemplateResponse(
        request,
        "routing/list.html",
        {
            "request": request,
            "records": records,
            "filters": {
                "mode": mode,
                "reason_code": reason_code,
                "source": source,
                "chat_prefix": chat_prefix,
            },
        },
    )


@router.get("/routing/{record_id}", response_class=HTMLResponse)
async def routing_detail(request: Request, record_id: str):
    engine = _agent_engine(request)
    store = getattr(engine, "routing_audit_store", None)
    record = await store.get(record_id) if store is not None else None
    if record is None:
        raise HTTPException(status_code=404, detail="route audit record not found")
    return request.app.state.templates.TemplateResponse(
        request,
        "routing/detail.html",
        {"request": request, "record": record},
    )
