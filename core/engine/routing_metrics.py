"""Bounded in-process metrics for mode routing and Chat-to-Agent handoff."""

from collections import Counter
from typing import Optional


class RoutingMetrics:
    """Collect aggregate routing signals without retaining message content."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._handoff_latency_total_ms = 0.0
        self._handoff_latency_count = 0
        self._handoff_latency_max_ms = 0.0

    def record_route(self, *, mode: str, reason_code: str) -> None:
        self._counts["routes_total"] += 1
        self._counts[f"routes_mode:{mode}"] += 1
        self._counts[f"routes_reason:{reason_code}"] += 1

    def record_handoff(
        self, *, status: str, latency_ms: Optional[float] = None
    ) -> None:
        self._counts["handoffs_total"] += 1
        self._counts[f"handoffs_status:{status}"] += 1
        if latency_ms is not None:
            latency = max(0.0, float(latency_ms))
            self._handoff_latency_total_ms += latency
            self._handoff_latency_count += 1
            self._handoff_latency_max_ms = max(self._handoff_latency_max_ms, latency)

    def record_tool_rejection(self, *, tool_name: str, reason: str) -> None:
        self._counts["tool_rejections_total"] += 1
        self._counts[f"tool_rejections_tool:{tool_name}"] += 1
        self._counts[f"tool_rejections_reason:{reason}"] += 1

    def record_background_terminal(
        self, *, status: str, latency_ms: Optional[float] = None
    ) -> None:
        self._counts["background_terminals_total"] += 1
        self._counts[f"background_terminals_status:{status}"] += 1
        if latency_ms is not None:
            latency = max(0.0, float(latency_ms))
            self._counts["background_terminal_latency_count"] += 1
            self._counts["background_terminal_latency_total_ms"] += latency
            self._counts["background_terminal_latency_max_ms"] = max(
                self._counts.get("background_terminal_latency_max_ms", 0), latency
            )

    def record_work_plan_terminal(
        self, *, status: str, latency_ms: Optional[float] = None
    ) -> None:
        self._counts["work_plan_terminals_total"] += 1
        self._counts[f"work_plan_terminals_status:{status}"] += 1
        if latency_ms is not None:
            latency = max(0.0, float(latency_ms))
            self._counts["work_plan_terminal_latency_count"] += 1
            self._counts["work_plan_terminal_latency_total_ms"] += latency
            self._counts["work_plan_terminal_latency_max_ms"] = max(
                self._counts.get("work_plan_terminal_latency_max_ms", 0), latency
            )

    def record_reconcile(self, result: dict[str, int]) -> None:
        self._counts["reconcile_runs"] += 1
        for key, value in result.items():
            self._counts[f"reconcile:{key}"] += max(0, int(value))

    def snapshot(self) -> dict[str, object]:
        result: dict[str, object] = dict(self._counts)
        result["handoff_latency_ms"] = {
            "count": self._handoff_latency_count,
            "avg": (
                round(self._handoff_latency_total_ms / self._handoff_latency_count, 1)
                if self._handoff_latency_count
                else 0.0
            ),
            "max": round(self._handoff_latency_max_ms, 1),
        }
        result["background_terminal_latency_ms"] = {
            "count": self._counts.get("background_terminal_latency_count", 0),
            "avg": (
                round(
                    self._counts.get("background_terminal_latency_total_ms", 0.0)
                    / self._counts["background_terminal_latency_count"],
                    1,
                )
                if self._counts.get("background_terminal_latency_count", 0)
                else 0.0
            ),
            "max": round(
                self._counts.get("background_terminal_latency_max_ms", 0.0), 1
            ),
        }
        result["work_plan_terminal_latency_ms"] = {
            "count": self._counts.get("work_plan_terminal_latency_count", 0),
            "avg": (
                round(
                    self._counts.get("work_plan_terminal_latency_total_ms", 0.0)
                    / self._counts["work_plan_terminal_latency_count"],
                    1,
                )
                if self._counts.get("work_plan_terminal_latency_count", 0)
                else 0.0
            ),
            "max": round(self._counts.get("work_plan_terminal_latency_max_ms", 0.0), 1),
        }
        return result
