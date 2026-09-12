"""JSON/SSE endpoints for the WebUI conversation workbench."""

from __future__ import annotations

import json

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from core.webui.csrf import csrf_token
from core.webui.routers.sessions import (
    _attach_resource_views,
    _redact_message,
    _resource_view,
)

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/csrf")
async def get_csrf_token(request: Request):
    return {"token": csrf_token(request)}


def _gateway(request: Request):
    gateway = request.app.state.managers.get("webui_gateway")
    if gateway is None:
        raise HTTPException(status_code=503, detail="WebUI conversation unavailable")
    return gateway


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _not_found(exc: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"session not found: {exc.args[0]}")


async def _turn_dtos(page, *, session_id: str, request: Request) -> list[dict]:
    events_by_turn: dict[str, list] = {}
    for event in page.events:
        events_by_turn.setdefault(event.turn_id, []).append(event)
    cards = []
    for turn in reversed(page.turns):
        cards.append(
            {
                "events": [
                    _redact_message(event.to_history_dict())
                    for event in events_by_turn.get(turn.turn_id, [])
                ]
            }
        )
    await _attach_resource_views(request, page.chat_id, cards)
    result = []
    for turn, card in zip(reversed(page.turns), cards):
        events = events_by_turn.get(turn.turn_id, [])
        blocks = []
        timestamps = []
        tool_names: dict[str, str] = {}
        for event in events:
            for call in event.tool_calls:
                function = call.get("function") if isinstance(call, dict) else None
                call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                name = (
                    str(function.get("name") or "")
                    if isinstance(function, dict)
                    else ""
                )
                if call_id and name:
                    tool_names[call_id] = name
        for event, message in zip(events, card["events"]):
            timestamps.append(event.timestamp)
            sender_id = str(message.get("sender_id") or event.sender_id or "")
            content = str(message.get("content") or "").strip()
            reasoning = str(message.get("reasoning_content") or "").strip()
            if reasoning:
                blocks.append(
                    {
                        "type": "reasoning",
                        "role": "assistant",
                        "sender_id": sender_id,
                        "text": reasoning,
                    }
                )
            if content and not (event.role == "tool" and event.tool_call_id):
                blocks.append(
                    {
                        "type": "text",
                        "text": content,
                        "role": event.role,
                        "sender_id": sender_id,
                    }
                )
            for resource in message.get("resources") or ():
                resource_view = dict(resource)
                if "display_type" not in resource_view:
                    resource_view = _resource_view(resource_view)
                blocks.append(
                    {
                        "type": resource_view["resource_type"] or "file",
                        "role": event.role,
                        "sender_id": sender_id,
                        "resource": resource_view,
                    }
                )
            if message.get("tool_calls"):
                tool_calls = message["tool_calls"]
                first_call = tool_calls[0] if tool_calls else {}
                function = (
                    first_call.get("function") if isinstance(first_call, dict) else {}
                )
                blocks.append(
                    {
                        "type": "tool",
                        "role": event.role,
                        "sender_id": sender_id,
                        "tool_calls": tool_calls,
                        "tool_call_id": (
                            first_call.get("id", "")
                            if isinstance(first_call, dict)
                            else ""
                        ),
                        "tool_name": (
                            function.get("name", "")
                            if isinstance(function, dict)
                            else ""
                        ),
                        "arguments": (
                            function.get("arguments", "")
                            if isinstance(function, dict)
                            else ""
                        ),
                        "status": "called",
                    }
                )
            if event.role == "tool" and event.tool_call_id:
                blocks.append(
                    {
                        "type": "tool_result",
                        "role": event.role,
                        "sender_id": sender_id,
                        "tool_call_id": event.tool_call_id,
                        "tool_name": event.tool_name
                        or tool_names.get(event.tool_call_id, ""),
                        "status": "completed",
                        "result": content,
                        "text": content,
                    }
                )
        result.append(
            {
                "turn_id": turn.turn_id,
                "turn_sequence": turn.turn_sequence,
                "created_at": min(timestamps, default=turn.updated_at),
                "status": str(turn.status),
                "turn_kind": turn.turn_kind.value,
                "blocks": blocks,
                "metadata": {
                    "session_id": session_id,
                    "event_count": turn.event_count,
                    "terminal": turn.is_terminal,
                },
            }
        )
    return result


@router.post("/chat/sessions")
async def create_chat_session(request: Request):
    gateway = _gateway(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
        session = await gateway.create_session(
            title=body.get("title", ""), mode=body.get("mode", "agent")
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return {"session": session.to_dict()}


@router.get("/chat/sessions")
async def list_chat_sessions(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    cursor: str = Query(""),
):
    try:
        page = await _gateway(request).list_sessions(limit=limit, cursor=cursor)
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return page.to_dict()


@router.get("/chat/options")
async def chat_options(request: Request):
    return _gateway(request).chat_options()


@router.get("/chat/external-sessions")
async def list_external_chat_sessions(
    request: Request, limit: int = Query(100, ge=1, le=100)
):
    return (await _gateway(request).list_external_sessions(limit=limit)).to_dict()


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(request: Request, session_id: str):
    try:
        session = await _gateway(request).resolve_history_session(session_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    return {"session": session.to_dict()}


@router.patch("/chat/sessions/{session_id}")
async def rename_chat_session(request: Request, session_id: str):
    gateway = _gateway(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        session = await gateway.rename_session(session_id, body.get("title", ""))
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return {"session": session.to_dict()}


@router.post("/chat/sessions/{session_id}/turns")
async def submit_chat_turn(request: Request, session_id: str):
    gateway = _gateway(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        receipt = await gateway.submit(
            session_id,
            content=body.get("content", ""),
            resources=body.get("resources", ()),
            request_id=body.get("request_id", ""),
            mode=body.get("mode"),
            model_group=body.get("model_group"),
            reasoning_effort=body.get("reasoning_effort"),
        )
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return {"receipt": receipt.to_dict()}


@router.post("/chat/sessions/{session_id}/compact")
async def compact_chat_session(request: Request, session_id: str):
    gateway = _gateway(request)
    try:
        result = await gateway.compact_session(session_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"result": result}


@router.post("/chat/sessions/{session_id}/uploads")
async def upload_chat_resource(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
):
    gateway = _gateway(request)
    chunks = []
    total = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 25 * 1024 * 1024:
                raise ValueError("file exceeds the 25 MiB upload limit")
            chunks.append(chunk)
        resource = await gateway.upload_resource(
            session_id,
            filename=file.filename or "file",
            mime_type=file.content_type or "application/octet-stream",
            data=b"".join(chunks),
        )
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc
    finally:
        await file.close()
    return {"resource": resource}


@router.delete("/chat/sessions/{session_id}/uploads")
async def discard_chat_resource(request: Request, session_id: str):
    gateway = _gateway(request)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        await gateway.discard_upload(
            session_id,
            media_uri=str(body.get("media_uri") or ""),
            upload_id=str(body.get("upload_id") or ""),
        )
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return {"discarded": True}


@router.get("/chat/sessions/{session_id}/audit")
async def session_audit(request: Request, session_id: str, limit: int = 50):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    gateway = _gateway(request)
    try:
        items = gateway.list_audit(session_id, limit=limit)
    except KeyError as exc:
        raise _not_found(exc) from exc
    return {"items": items}


@router.get("/chat/sessions/{session_id}/approvals")
async def session_pending_approvals(request: Request, session_id: str):
    gateway = _gateway(request)
    try:
        items = gateway.list_pending_approvals(session_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    return {"items": items}


@router.post("/chat/approvals/resolve")
async def resolve_chat_approval(request: Request):
    gateway = _gateway(request)
    manager = request.app.state.managers.get("approval_manager")
    if manager is None or not callable(getattr(manager, "resolve_webui", None)):
        raise HTTPException(status_code=503, detail="approval unavailable")
    body = await request.json()
    if not isinstance(body, dict):
        raise _bad_request(ValueError("request body must be an object"))
    session_key = str(body.get("session_key") or "").strip()
    decision = str(body.get("decision") or "").strip().lower()
    if not session_key or len(session_key) > 512:
        raise _bad_request(ValueError("invalid approval session_key"))
    if decision not in {"allow-once", "allow-always", "deny"}:
        raise _bad_request(ValueError("invalid approval decision"))
    if not manager.resolve_webui(session_key, decision, gateway.operator_id):
        raise HTTPException(status_code=404, detail="approval not found or expired")
    return {"resolved": True, "decision": decision}


@router.get("/chat/sessions/{session_id}/events")
async def chat_events(
    request: Request,
    session_id: str,
    after_event_id: str = "",
    replay: bool = True,
):
    try:
        gateway = _gateway(request)
        gateway.get_session(session_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc

    after_event_id = after_event_id or request.headers.get("last-event-id", "")
    if not after_event_id and not replay:
        after_event_id = "__tail__"

    async def generate():
        async for event in gateway.hub.stream(
            session_id, after_event_id=after_event_id
        ):
            if await request.is_disconnected():
                break
            if event is None:
                yield ": keep-alive\n\n"
                continue
            payload = json.dumps(event.to_dict(), ensure_ascii=False)
            yield f"id: {event.event_id}\nevent: {event.event_type}\ndata: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{session_id}/turns")
async def session_turns(
    request: Request,
    session_id: str,
    limit: int = 30,
    before_turn_sequence: int | None = None,
    cutoff_sequence: int | None = None,
):
    if not session_id or "/" in session_id or "\x00" in session_id:
        raise HTTPException(status_code=400, detail="invalid session_id")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if before_turn_sequence is not None and before_turn_sequence < 1:
        raise HTTPException(status_code=400, detail="invalid before_turn_sequence")
    if cutoff_sequence is not None and cutoff_sequence < 0:
        raise HTTPException(status_code=400, detail="invalid cutoff_sequence")
    storage_session_id = session_id
    gateway = request.app.state.managers.get("webui_gateway")
    if gateway is not None:
        try:
            storage_session_id = (
                await gateway.resolve_history_session(session_id)
            ).session_key
        except KeyError as exc:
            raise _not_found(exc) from exc
        except ValueError as exc:
            raise _bad_request(exc) from exc
    event_log = request.app.state.managers.get("conversation_event_log")
    if event_log is None:
        raise HTTPException(status_code=503, detail="conversation history unavailable")
    page, has_more, next_before = await event_log.snapshot_turn_cursor(
        storage_session_id,
        limit=limit,
        before_turn_sequence=before_turn_sequence,
        cutoff_seq=cutoff_sequence,
        include_internal=True,
    )
    return {
        "session_id": session_id,
        "items": await _turn_dtos(page, session_id=session_id, request=request),
        "has_more": has_more,
        "next_before_turn_sequence": next_before,
        "cutoff_sequence": page.cutoff_seq,
        "unstable_cursor": False,
    }
