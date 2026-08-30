from core.engine.routing_metrics import RoutingMetrics


def test_routing_metrics_are_aggregate_and_bounded():
    metrics = RoutingMetrics()
    metrics.record_route(mode="chat", reason_code="default_chat")
    metrics.record_route(mode="agent", reason_code="work_request")
    metrics.record_handoff(status="accepted", latency_ms=10)
    metrics.record_handoff(status="duplicate")
    metrics.record_handoff(status="failed", latency_ms=30)
    metrics.record_background_terminal(status="completed", latency_ms=40)
    metrics.record_reconcile({"interrupted": 1, "queued": 2})
    metrics.record_work_plan_terminal(status="COMPLETED", latency_ms=50)

    assert metrics.snapshot() == {
        "routes_total": 2,
        "routes_mode:chat": 1,
        "routes_reason:default_chat": 1,
        "routes_mode:agent": 1,
        "routes_reason:work_request": 1,
        "handoffs_total": 3,
        "handoffs_status:accepted": 1,
        "handoffs_status:duplicate": 1,
        "handoffs_status:failed": 1,
        "handoff_latency_ms": {"count": 2, "avg": 20.0, "max": 30.0},
        "background_terminals_total": 1,
        "background_terminals_status:completed": 1,
        "background_terminal_latency_count": 1,
        "background_terminal_latency_total_ms": 40.0,
        "background_terminal_latency_max_ms": 40.0,
        "reconcile_runs": 1,
        "reconcile:interrupted": 1,
        "reconcile:queued": 2,
        "background_terminal_latency_ms": {"count": 1, "avg": 40.0, "max": 40.0},
        "work_plan_terminals_total": 1,
        "work_plan_terminals_status:COMPLETED": 1,
        "work_plan_terminal_latency_count": 1,
        "work_plan_terminal_latency_total_ms": 50.0,
        "work_plan_terminal_latency_max_ms": 50.0,
        "work_plan_terminal_latency_ms": {"count": 1, "avg": 50.0, "max": 50.0},
    }
