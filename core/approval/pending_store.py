"""Durable metadata for approvals that outlive one process instance."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from core.session_identity import DeliveryTarget


class PendingApprovalStore:
    """Atomically persist pending approval metadata without runtime futures."""

    VERSION = 1

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return
        raw_records = payload.get("items", []) if isinstance(payload, dict) else []
        if not isinstance(raw_records, list):
            return
        for raw_record in raw_records:
            record = self._normalise(raw_record)
            if record is not None:
                self._records[record["session_key"]] = record

    def _normalise(self, value: object) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        session_key = str(value.get("session_key") or "").strip()
        if not session_key or "\x00" in session_key:
            return None
        created_at = self._finite_number(value.get("created_at"))
        expires_at = self._finite_number(value.get("expires_at"))
        if created_at is None or expires_at is None or expires_at <= created_at:
            return None
        target = self._decode_target(value.get("delivery_target"))
        plan = value.get("plan")
        if plan is not None and not isinstance(plan, dict):
            plan = None
        try:
            timeout_sec = max(1, int(value.get("timeout_sec") or 120))
        except (TypeError, ValueError, OverflowError):
            timeout_sec = 120
        return {
            "session_key": session_key,
            "tool_name": str(value.get("tool_name") or ""),
            "details": str(value.get("details") or ""),
            "reason": str(value.get("reason") or ""),
            "created_at": created_at,
            "expires_at": expires_at,
            "delivery_target": target,
            "cwd": str(value.get("cwd") or ""),
            "timeout_sec": timeout_sec,
            "plan": plan,
        }

    @staticmethod
    def _finite_number(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _decode_target(value: object) -> DeliveryTarget | None:
        if not isinstance(value, dict):
            return None
        try:
            return DeliveryTarget(
                channel=str(value.get("channel") or ""),
                account_id=str(value.get("account_id") or ""),
                chat_type=str(value.get("chat_type") or "direct"),
                target_id=str(value.get("target_id") or ""),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _encode_target(target: DeliveryTarget | None) -> dict[str, str] | None:
        if target is None:
            return None
        return {
            "channel": target.channel,
            "account_id": target.account_id,
            "chat_type": target.chat_type,
            "target_id": target.target_id,
        }

    def records(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._records.items()}

    def put(
        self,
        session_key: str,
        info: dict[str, Any],
        plan: dict[str, Any] | None,
    ) -> None:
        encoded = {
            "session_key": session_key,
            "tool_name": info.get("tool_name", ""),
            "details": info.get("details", ""),
            "reason": info.get("reason", ""),
            "created_at": info.get("created_at", 0),
            "expires_at": info.get("expires_at", 0),
            "delivery_target": self._encode_target(info.get("delivery_target")),
            "cwd": info.get("cwd", ""),
            "timeout_sec": info.get("timeout_sec", 120),
            "plan": plan,
        }
        json.dumps(encoded, ensure_ascii=False)
        normalised = self._normalise(encoded)
        if normalised is None:
            raise ValueError("invalid pending approval record")
        self._records[session_key] = normalised
        self._save()

    def remove(self, session_key: str) -> None:
        if session_key not in self._records:
            return
        self._records.pop(session_key, None)
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        items = []
        for record in self._records.values():
            encoded = dict(record)
            encoded["delivery_target"] = self._encode_target(
                record.get("delivery_target")
            )
            items.append(encoded)
        temporary_path = self._path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {"version": self.VERSION, "items": items},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, self._path)
