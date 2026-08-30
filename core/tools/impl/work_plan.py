"""WorkPlan tool adapter. Runtime context supplies identity; model cannot supply ACL context."""

from __future__ import annotations

import json

from core.tools._types import ToolEntry, ToolResult


def create_work_plan_entries(deps) -> list[ToolEntry]:
    service = getattr(deps, "work_plan_service", None)
    if service is None:
        return []

    async def work_plan(args, ctx):
        action = args.get("action", "list")
        principal = (
            service.principal_factory(ctx)
            if hasattr(service, "principal_factory")
            else None
        )
        if principal is None:
            return ToolResult(
                content=json.dumps(
                    {"error": "WorkPlan context unavailable"}, ensure_ascii=False
                )
            )
        try:
            if action == "list":
                plans = await service.list(principal)
                return ToolResult(
                    content=json.dumps([p.__dict__ for p in plans], ensure_ascii=False)
                )
            if action == "get":
                details = await service.details(principal, args.get("work_plan_id", ""))
                return ToolResult(content=json.dumps(details, ensure_ascii=False))
            if action == "list_steps":
                steps = await service.list_steps(
                    principal,
                    args["work_plan_id"],
                    ready_only=bool(args.get("ready_only", False)),
                )
                return ToolResult(
                    content=json.dumps(
                        [step.__dict__ for step in steps], ensure_ascii=False
                    )
                )
            if ctx.capabilities is not None and ctx.capabilities.mode.value != "agent":
                raise PermissionError("Chat can only read WorkPlans")
            if action == "acknowledge":
                if getattr(ctx, "consumer_evidence_callback", None) is None:
                    raise PermissionError(
                        "acknowledge is only valid for a WorkPlan consumer wake"
                    )
                return ToolResult(content=json.dumps({"acknowledged": True}))
            if action == "retry_background":
                runner = getattr(deps, "work_plan_runner", None)
                if runner is None:
                    raise RuntimeError("WorkPlan background runner is unavailable")
                task = await service.retry_background(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                    previous_task_id=args["previous_task_id"],
                    brief={
                        "task_summary": args.get("task_summary", ""),
                        "expected_result": args.get("expected_result", ""),
                    },
                    idempotency_key=str(args.get("idempotency_key", "")),
                )
                runner.start(task)
                return ToolResult(
                    content=json.dumps(
                        {"background_task_id": task.id, "status": task.status},
                        ensure_ascii=False,
                    )
                )
            if action == "update_step":
                step = await service.update_step(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                    step_id=args["step_id"],
                    status=args["step_status"],
                    result_summary=args.get("result_summary", ""),
                )
                return ToolResult(content=json.dumps(step.__dict__, ensure_ascii=False))
            if action == "add_step":
                step = await service.add_step(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                    title=args.get("title", ""),
                    description=args.get("description", ""),
                    depends_on=args.get("depends_on", []),
                    execution_mode=args.get("execution_mode", "foreground"),
                )
                return ToolResult(content=json.dumps(step.__dict__, ensure_ascii=False))
            if action == "create":
                plan = await service.create(principal, args.get("title", ""))
                return ToolResult(content=json.dumps(plan.__dict__, ensure_ascii=False))
            if action == "update":
                plan = await service.update_title(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                    title=args["title"],
                )
                return ToolResult(content=json.dumps(plan.__dict__, ensure_ascii=False))
            if action == "select":
                plan = await service.select(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                )
                return ToolResult(content=json.dumps(plan.__dict__, ensure_ascii=False))
            if action in {"pause", "resume", "complete"}:
                plan = await service.get(principal, args["work_plan_id"])
                statuses = {
                    "pause": "PAUSED",
                    "resume": "ACTIVE",
                    "complete": "COMPLETED",
                }
                plan = await service.transition(
                    principal,
                    plan.id,
                    expected_revision=int(args["expected_revision"]),
                    status=statuses[action],
                )
                return ToolResult(content=json.dumps(plan.__dict__, ensure_ascii=False))
            if action == "cancel":
                plan = await service.cancel(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                )
                runner = getattr(deps, "work_plan_runner", None)
                if runner is not None:
                    await runner.cancel_plan(plan.id)
                return ToolResult(content=json.dumps(plan.__dict__, ensure_ascii=False))
            if action == "reopen":
                plan = await service.reopen(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                )
                return ToolResult(content=json.dumps(plan.__dict__, ensure_ascii=False))
            if action == "delegate_background":
                runner = getattr(deps, "work_plan_runner", None)
                if runner is None:
                    raise RuntimeError("WorkPlan background runner is unavailable")
                task = await service.delegate_background(
                    principal,
                    args["work_plan_id"],
                    expected_revision=int(args["expected_revision"]),
                    brief={
                        "task_summary": args.get("task_summary", ""),
                        "expected_result": args.get("expected_result", ""),
                    },
                    required=bool(args.get("required", True)),
                    idempotency_key=str(args.get("idempotency_key", "")),
                    step_id=args.get("step_id") or None,
                    parallel=bool(args.get("parallel", False)),
                    depends_on_tasks=args.get("depends_on_tasks", []),
                )
                runner.start(task)
                return ToolResult(
                    content=json.dumps(
                        {"background_task_id": task.id, "status": task.status},
                        ensure_ascii=False,
                    )
                )
            if action == "share":
                await service.share(
                    principal,
                    args["work_plan_id"],
                    principal_id=args["principal_id"],
                    acl_role=args["acl_role"],
                    collaboration_enabled=bool(
                        args.get("collaboration_enabled", False)
                    ),
                )
                return ToolResult(content=json.dumps({"success": True}))
            return ToolResult(
                content=json.dumps(
                    {"error": "unsupported WorkPlan action"}, ensure_ascii=False
                )
            )
        except Exception as exc:
            return ToolResult(
                content=json.dumps({"error": str(exc)}, ensure_ascii=False)
            )

    params = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "get",
                    "create",
                    "update",
                    "select",
                    "pause",
                    "resume",
                    "complete",
                    "cancel",
                    "reopen",
                    "share",
                    "delegate_background",
                    "retry_background",
                    "list_steps",
                    "acknowledge",
                    "add_step",
                    "update_step",
                ],
            },
            "work_plan_id": {"type": "string"},
            "title": {"type": "string", "maxLength": 400},
            "depends_on_tasks": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 32,
            },
            "parallel": {"type": "boolean"},
            "previous_task_id": {"type": "string"},
            "step_id": {"type": "string"},
            "step_status": {
                "type": "string",
                "enum": ["PENDING", "ACTIVE", "BLOCKED", "DONE", "SKIPPED"],
            },
            "result_summary": {"type": "string", "maxLength": 2000},
            "description": {"type": "string", "maxLength": 2000},
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 32,
            },
            "execution_mode": {"type": "string", "enum": ["foreground", "background"]},
            "ready_only": {"type": "boolean"},
            "principal_id": {"type": "string"},
            "acl_role": {
                "type": "string",
                "enum": ["viewer", "contributor", "operator"],
            },
            "collaboration_enabled": {"type": "boolean"},
            "task_summary": {"type": "string", "maxLength": 600},
            "expected_result": {"type": "string", "maxLength": 600},
            "required": {"type": "boolean"},
            "idempotency_key": {"type": "string", "maxLength": 200},
        },
        "required": ["action"],
    }
    return [
        ToolEntry(
            name="work_plan",
            section="work_plan",
            description="查询或管理当前会话的持久工作计划。",
            parameters=params,
            handler=work_plan,
        )
    ]
