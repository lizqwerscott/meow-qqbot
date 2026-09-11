"""JSON/SSE endpoints for the WebUI conversation workbench."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.webui.routers.sessions import (
    _attach_resource_views,
    _redact_message,
    _resource_view,
)

router = APIRouter(prefix="/api", tags=["chat"])


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
        for event, message in zip(events, card["events"]):
            timestamps.append(event.timestamp)
            content = str(message.get("content") or "").strip()
            if content:
                blocks.append({"type": "text", "text": content, "role": event.role})
            for resource in message.get("resources") or ():
                resource_view = dict(resource)
                if "display_type" not in resource_view:
                    resource_view = _resource_view(resource_view)
                blocks.append(
                    {
                        "type": resource_view["resource_type"] or "file",
                        "role": event.role,
                        "resource": resource_view,
                    }
                )
            if message.get("tool_calls"):
                blocks.append(
                    {
                        "type": "tool",
                        "role": event.role,
                        "tool_calls": message["tool_calls"],
                    }
                )
            if event.role == "tool" and event.tool_call_id:
                blocks.append(
                    {
                        "type": "tool_result",
                        "role": event.role,
                        "tool_call_id": event.tool_call_id,
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
async def list_chat_sessions(request: Request):
    sessions = await _gateway(request).list_sessions()
    return {"items": [session.to_dict() for session in sessions]}


@router.patch("/chat/sessions/{session_id}")
async def rename_chat_session(request: Request, session_id: str):
    gateway = _gateway(request)
    try:
        body = await request.json()
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
        )
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc
    return {"receipt": receipt.to_dict()}


@router.get("/chat/sessions/{session_id}/events")
async def chat_events(request: Request, session_id: str, after_event_id: str = ""):
    try:
        gateway = _gateway(request)
        gateway.get_session(session_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc

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
            storage_session_id = gateway.get_session(session_id).session_key
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
