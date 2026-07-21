import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

_log = logging.getLogger(__name__)

WHITELIST_PATH = "config/approval_whitelist.json"


class ApprovalManager:
    def __init__(self, api_client, admin_ids: list[str]):
        self._api = api_client
        self._admin_ids = set(admin_ids)
        self._pending: dict[str, asyncio.Future] = {}
        self._whitelist: dict = self._load_whitelist()

    # ── Whitelist persistence ──

    def _load_whitelist(self) -> dict:
        path = Path(WHITELIST_PATH)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                _log.warning("加载审批白名单失败: %s", e)
        return {"file_paths": [], "exec_commands": []}

    def _save_whitelist(self):
        path = Path(WHITELIST_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._whitelist, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def check_whitelist(self, tool_name: str, target: str) -> bool:
        if tool_name in ("read_file", "write_file", "edit_file", "apply_patch"):
            for entry in self._whitelist.get("file_paths", []):
                p = entry["path"]
                if target == p or target.startswith(p + "/"):
                    return True
            return False
        if tool_name == "exec":
            cmd_name = os.path.basename(target.split()[0])
            for entry in self._whitelist.get("exec_commands", []):
                if cmd_name == entry["command"]:
                    return True
            return False
        return False

    def add_to_whitelist(self, tool_name: str, target: str):
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if tool_name in ("read_file", "write_file", "edit_file", "apply_patch"):
            self._whitelist.setdefault("file_paths", [])
            if not any(e["path"] == target for e in self._whitelist["file_paths"]):
                self._whitelist["file_paths"].append({"path": target, "approved_at": now})
        elif tool_name == "exec":
            cmd_name = os.path.basename(target.split()[0])
            self._whitelist.setdefault("exec_commands", [])
            if not any(e["command"] == cmd_name for e in self._whitelist["exec_commands"]):
                self._whitelist["exec_commands"].append({"command": cmd_name, "approved_at": now})
        self._save_whitelist()

    # ── Approval flow ──

    async def request_approval(
        self,
        chat_id: str,
        tool_name: str,
        reason: str,
        details: str = "",
        timeout: int = 120,
    ) -> str:
        if not self._admin_ids:
            return "deny"

        admin_id = chat_id
        session_key = f"approval:{chat_id}:{tool_name}:{uuid.uuid4().hex[:8]}"
        future = asyncio.get_running_loop().create_future()
        self._pending[session_key] = future

        from qqbot_agent_sdk import ApprovalRequest, ApprovalSender

        req = ApprovalRequest(
            session_key=session_key,
            title=f"🔐 {tool_name} 审批请求",
            description=f"原因: {reason}",
            command_preview=details,
            severity="info",
            timeout_sec=timeout,
        )

        sender = ApprovalSender(self._api, log_tag="Approval")
        if not await sender.send(chat_type="c2c", chat_id=admin_id, req=req):
            self._pending.pop(session_key, None)
            _log.warning("审批消息发送失败: session_key=%s..", session_key[:20])
            return "deny"
        _log.info(
            "审批请求已发送到 admin %s..: tool=%s session_key=%s..",
            admin_id[:12], tool_name, session_key[:20],
        )

        loop = asyncio.get_running_loop()
        timeout_handle = loop.call_later(timeout, self._on_timeout, session_key)

        try:
            result = await future
            timeout_handle.cancel()
            if result == "allow-always":
                self.add_to_whitelist(tool_name, details)
            return result
        except asyncio.CancelledError:
            timeout_handle.cancel()
            self._pending.pop(session_key, None)
            raise

    def resolve(self, session_key: str, decision: str, approver_id: str) -> bool:
        if approver_id not in self._admin_ids:
            _log.warning("非管理员 %s.. 试图审批，忽略", approver_id[:16])
            return False
        future = self._pending.pop(session_key, None)
        if future and not future.done():
            future.set_result(decision)
            _log.info(
                "审批已处理: session_key=%s.. decision=%s", session_key[:20], decision
            )
            return True
        _log.warning("审批 future 不存在或已完成: %s..", session_key[:20])
        return False

    def _on_timeout(self, session_key: str):
        future = self._pending.pop(session_key, None)
        if future and not future.done():
            future.set_result("timeout")
            _log.info("审批超时: %s..", session_key[:20])
