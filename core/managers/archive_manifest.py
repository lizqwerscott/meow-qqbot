"""Atomic journal files for resumable archive operations."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

_STATES = (
    "prepared",
    "archive_written",
    "active_written",
    "summary_written",
    "committed",
)
_STATE_ORDER = {state: index for index, state in enumerate(_STATES)}


class ArchiveManifestStore:
    """Persist archive operation plans and their current commit state."""

    def __init__(self, root_dir: str):
        self._root = Path(root_dir) / "manifests"
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, operation_id: str) -> Path:
        safe_id = operation_id.replace(":", "_")
        return self._root / f"{safe_id}.json"

    def write(self, manifest: Dict[str, Any]) -> None:
        self._validate(manifest)
        path = self._path(str(manifest["operation_id"]))
        if path.is_file():
            existing = self.load(str(manifest["operation_id"]))
            previous_state = str(existing["state"])
            next_state = str(manifest["state"])
            if _STATE_ORDER[next_state] < _STATE_ORDER[previous_state]:
                raise RuntimeError(
                    "invalid archive manifest state transition: "
                    f"{previous_state} -> {next_state}"
                )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(manifest, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load_pending(self) -> List[Dict[str, Any]]:
        manifests: List[Dict[str, Any]] = []
        if not self._root.is_dir():
            return manifests
        for path in sorted(self._root.glob("*.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                _log.error("归档 manifest 无法读取 %s: %s", path, exc)
                continue
            try:
                self._validate(manifest)
            except ValueError as exc:
                _log.error("归档 manifest 格式无效 %s: %s", path, exc)
                continue
            if manifest.get("state") != "committed":
                manifests.append(manifest)
        return manifests

    def load(self, operation_id: str) -> Dict[str, Any] | None:
        path = self._path(operation_id)
        if not path.is_file():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"归档 manifest 无法读取 {path}: {exc}") from exc
        if not isinstance(manifest, dict) or not manifest.get("operation_id"):
            raise RuntimeError(f"归档 manifest 格式无效: {path}")
        self._validate(manifest)
        return manifest

    @staticmethod
    def _validate(manifest: Dict[str, Any]) -> None:
        if not isinstance(manifest, dict):
            raise ValueError("archive manifest must be an object")
        required = {"operation_id", "chat_id", "state", "batches"}
        missing = required - manifest.keys()
        if missing:
            raise ValueError(
                "archive manifest missing fields: " + ", ".join(sorted(missing))
            )
        state = manifest["state"]
        if state not in _STATE_ORDER:
            raise ValueError(f"unknown archive manifest state: {state}")
        if not isinstance(manifest["batches"], list):
            raise ValueError("archive manifest batches must be a list")
        for batch in manifest["batches"]:
            if not isinstance(batch, dict):
                raise ValueError("archive manifest batch must be an object")
            if not batch.get("batch_id") or not batch.get("partition_date"):
                raise ValueError("archive manifest batch identity is incomplete")
            batch_state = batch.get("state", "prepared")
            if batch_state not in _STATE_ORDER:
                raise ValueError(f"unknown archive batch state: {batch_state}")

    def pending_count(self, chat_id: str | None = None) -> int:
        manifests = self.load_pending()
        if chat_id is None:
            return len(manifests)
        return sum(1 for manifest in manifests if manifest.get("chat_id") == chat_id)

    def find_pending(self, chat_id: str, reason: str) -> Dict[str, Any] | None:
        candidates = [
            manifest
            for manifest in self.load_pending()
            if manifest.get("chat_id") == chat_id and manifest.get("reason") == reason
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda manifest: str(manifest["operation_id"]))
