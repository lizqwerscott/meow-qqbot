import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from core.approval.allowlist import AllowlistEntry
from core.approval.exec_policy import (
    ALLOW_DECISIONS,
    DECISION_ALLOW,
    DECISION_ALLOW_ALWAYS,
    DECISION_DENY,
    ExecPolicy,
)

_log = logging.getLogger(__name__)

WHITELIST_PATH = "config/approval_whitelist.json"

# 审批白名单 schema 版本（v2 = OpenClaw 风格 allowlist + defaults）
WHITELIST_VERSION = 2


def _parse_target(t: str) -> Optional[tuple[str, str]]:
    """解析转发目标：'c2c:<id>' / 'group:<id>'；裸 id 默认 c2c。

    非法目标（未知前缀 / 空 id / 空串）返回 None——拒绝而非静默当 chat id。
    """
    t = t.strip()
    if not t:
        return None
    if ":" in t:
        chat_type, _, chat_id = t.partition(":")
        if chat_type in ("c2c", "group") and chat_id:
            return chat_type, chat_id
        return None
    return "c2c", t


class ApprovalManager:
    def __init__(
        self,
        api_client,
        admin_ids: list[str],
        forward_to: list[str] = (),  # 2.3：审批卡转发目标（'c2c:<id>'/'group:<id>'）
    ):
        self._api = api_client
        self._admin_ids = set(admin_ids)
        self._forward_to = list(forward_to or ())
        self._pending: dict[str, asyncio.Future] = {}
        # 审批对应的 canonical plan（openclaw 风格：批准后须比对，防执行内容漂移）
        self._pending_plans: dict[str, dict] = {}
        # 2.3：待审批元信息（审批列表/超时通知用）
        self._pending_info: dict[str, dict] = {}
        # 2.4：allowlist 使用计数 dirty 标记 + 延迟落盘任务
        self._uses_dirty = False
        self._save_task_active = False
        self._whitelist: dict = self._load_whitelist()

    # ── Whitelist persistence ──

    def _load_whitelist(self) -> dict:
        path = Path(WHITELIST_PATH)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                _log.warning("加载审批白名单失败: %s", e)
                data = {}
        else:
            data = {}
        data = self._migrate_v1(data)
        data.setdefault("version", WHITELIST_VERSION)
        data.setdefault("defaults", {})
        data.setdefault("allowlist", [])
        data.setdefault("file_paths", [])
        data.setdefault("exec_commands", [])
        return data

    @staticmethod
    def _migrate_v1(data: dict) -> dict:
        """v1（exec_commands/file_paths）→ v2（+allowlist/defaults）。

        v1 的 exec_commands 条目迁移为 bare-name allowlist 条目（source=legacy），
        原字段保留镜像，兼容旧代码与旧测试。
        """
        if data.get("version", 1) >= 2 or not data.get("exec_commands"):
            return data
        migrated = [
            {
                "pattern": e["command"],
                "source": "legacy",
                "approved_at": e.get("approved_at", ""),
            }
            for e in data["exec_commands"]
            if e.get("command")
        ]
        data["allowlist"] = migrated + data.get("allowlist", [])
        return data

    def _save_whitelist(self):
        path = Path(WHITELIST_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self._whitelist)
        payload["version"] = WHITELIST_VERSION
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Host 策略面（对齐 openclaw exec-approvals.json defaults）──

    def get_host_policy(self) -> ExecPolicy:
        """host 审批文件的默认策略，仅能收紧 config 的 [exec] 段。

        非收紧性配置（safe_bins/safe_bin_profiles/approval_timeout）
        也在此读取——host 显式定义了才覆盖（effective_policy 判定），
        未定义时 approval_timeout 为 None（不覆盖 requested）。
        """
        d = self._whitelist.get("defaults", {}) or {}
        return ExecPolicy(
            security=d.get("security", "allowlist"),
            ask=d.get("ask", "on-miss"),
            ask_fallback=d.get("ask_fallback", "deny"),
            strict_inline_eval=d.get("strict_inline_eval", True),
            safe_bins=tuple(d.get("safe_bins") or ()),
            safe_bin_profiles=dict(d.get("safe_bin_profiles") or {}),
            approval_timeout=d.get("approval_timeout"),
        )

    def get_allowlist_entries(self) -> list[AllowlistEntry]:
        """返回运行时累积的 allowlist 条目（allow-always / legacy / manual）。"""
        entries: list[AllowlistEntry] = []
        for raw in self._whitelist.get("allowlist", []):
            if not raw or not raw.get("pattern"):
                continue
            entries.append(
                AllowlistEntry(
                    pattern=raw["pattern"],
                    arg_pattern=raw.get("arg_pattern"),
                    source=raw.get("source", "manual"),
                    id=raw.get("id", ""),
                    last_used_at=raw.get("last_used_at", 0),
                    last_used_command=raw.get("last_used_command", ""),
                    last_resolved_path=raw.get("last_resolved_path", ""),
                    uses=int(raw.get("uses", 0) or 0),  # 2.4 使用计数
                )
            )
        return entries

    # ── 2.4 审批管理：删除条目 / 使用计数 / 统计 ──

    def remove_allowlist_entry(
        self, pattern: str, source: str | None = None
    ) -> bool:
        """删除 allowlist 条目（精确 pattern 匹配，可选 source 限定），
        v1 镜像（exec_commands）同步清理。返回是否删除成功。"""
        entries = self._whitelist.get("allowlist", []) or []
        remaining = []
        removed = False
        for e in entries:
            if e.get("pattern") == pattern and (
                source is None or e.get("source") == source
            ):
                removed = True
                continue
            remaining.append(e)
        if not removed:
            return False
        self._whitelist["allowlist"] = remaining
        cmds = self._whitelist.get("exec_commands", []) or []
        self._whitelist["exec_commands"] = [
            e for e in cmds if e.get("command") != os.path.basename(pattern)
        ]
        try:
            self._save_whitelist()
        except Exception as e:
            _log.error("保存审批白名单失败: %s", e)
        return True

    def record_use(self, pattern: str):
        """记录 allowlist 条目使用次数（内存计数 + 延迟落盘）。

        由 exec 链路命中时调用；不实时写文件（避免每次执行都落盘），
        10s 内合并后统一保存。
        """
        for e in self._whitelist.get("allowlist", []) or []:
            if e.get("pattern") == pattern:
                e["uses"] = int(e.get("uses", 0) or 0) + 1
                e["last_used_at"] = int(time.time())
                self._uses_dirty = True
                self._schedule_use_save()
                return

    def _schedule_use_save(self):
        if self._save_task_active:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._save_task_active = True
        loop.call_later(10, self._flush_uses)

    def _flush_uses(self):
        self._save_task_active = False
        if not self._uses_dirty:
            return
        self._uses_dirty = False
        try:
            self._save_whitelist()
        except Exception as e:
            _log.error("保存审批白名单失败: %s", e)

    def flush(self):
        """落盘未保存的使用计数（进程正常关闭时调用）。

        10s 防抖窗口内的最后一次计数在关闭时强制落盘，避免丢失
        （对齐 nickname_manager.flush_save 的关闭模式）。幂等：无 dirty 不写。
        """
        self._flush_uses()

    def whitelist_stats(self) -> dict:
        """白名单规模 + 最近一次 allow-always 时间（猫猫状态 用）。"""
        entries = self._whitelist.get("allowlist", []) or []
        count = len([e for e in entries if e.get("pattern")])
        last = ""
        for e in entries:
            at = e.get("approved_at")
            if isinstance(at, (int, float)):
                # 手改/旧 v1 的 int 时间戳 → 归一为 ISO 字符串
                at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(at))
            at = at or ""
            if at > last:
                last = at
        return {"count": count, "last_allow_always_at": last}

    def check_whitelist(self, tool_name: str, target: str) -> bool:
        """旧式命令名/路径匹配（兼容层）。

        exec 工具新链路请用 get_allowlist_entries + match_allowlist；
        此方法保留给文件工具与旧调用方。
        """
        if not target:
            return False
        if tool_name in ("read_file", "write_file", "edit_file", "apply_patch"):
            for entry in self._whitelist.get("file_paths", []):
                p = entry.get("path", "")
                if not p:
                    continue
                if target == p or target.startswith(p + "/"):
                    return True
            return False
        if tool_name == "exec":
            parts = target.split()
            if not parts:
                return False
            cmd_name = os.path.basename(parts[0])
            for entry in self._whitelist.get("exec_commands", []):
                if cmd_name == entry.get("command"):
                    return True
            # 新 allowlist 条目中的 bare-name pattern 也命中
            for entry in self._whitelist.get("allowlist", []):
                pat = entry.get("pattern", "")
                if pat and "/" not in pat and pat == cmd_name:
                    return True
            return False
        return False

    def add_to_whitelist(self, tool_name: str, target: str, plan: dict | None = None):
        """allow-always 落白名单。

        exec：写入 bare-name allowlist 条目（对齐 openclaw：允许列表记
        resolved 路径 + 参数模式，而非裸命令名），同时双写 exec_commands
        兼容旧读取方。inline-eval 命令由调用方决定是否落白名单。

        包装器解包（2.1）：plan 的 ``persist_pattern``（调用方分析阶段算好的
        内层可执行路径，如 ``timeout 5 python3 x.py`` → ``python3``）优先；
        链式命令为 list（每个顶层段各一条）；无 plan 时回退现状（裸命令名）。
        """
        if not target:
            _log.warning("add_to_whitelist 收到空 target: tool=%s", tool_name)
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if tool_name in ("read_file", "write_file", "edit_file", "apply_patch"):
            self._whitelist.setdefault("file_paths", [])
            if not any(e.get("path") == target for e in self._whitelist["file_paths"]):
                self._whitelist["file_paths"].append(
                    {"path": target, "approved_at": now}
                )
        elif tool_name == "exec":
            parts = target.split()
            if not parts:
                _log.warning("add_to_whitelist exec 命令为空: target=%s", target)
                return
            # 2.1：包装器解包后持久化内层可执行路径；链式命令持久化所有顶层段；
            # 无法解包/无 plan 回退外层 basename
            persist = (plan or {}).get("persist_pattern") or [
                os.path.basename(parts[0])
            ]
            patterns = persist if isinstance(persist, list) else [persist]
            for pattern in patterns:
                cmd_name = os.path.basename(pattern)
                # v1 镜像（兼容旧读取方/旧测试）
                self._whitelist.setdefault("exec_commands", [])
                if not any(
                    e.get("command") == cmd_name
                    for e in self._whitelist["exec_commands"]
                ):
                    self._whitelist["exec_commands"].append(
                        {"command": cmd_name, "approved_at": now}
                    )
                # v2 allowlist 条目（bare-name 或绝对路径 pattern）
                self._whitelist.setdefault("allowlist", [])
                if not any(
                    e.get("pattern") == pattern
                    for e in self._whitelist["allowlist"]
                ):
                    self._whitelist["allowlist"].append(
                        {
                            "pattern": pattern,
                            "source": "allow-always",
                            "approved_at": now,
                            "last_used_at": int(time.time()),
                            "last_used_command": target,
                        }
                    )
        try:
            self._save_whitelist()
        except Exception as e:
            _log.error("保存审批白名单失败: %s", e)

    # ── Approval flow ──

    async def request_approval(
        self,
        chat_id: str,
        tool_name: str,
        reason: str,
        details: str = "",
        timeout: int = 120,
        *,
        plan: dict | None = None,
        ask_fallback: str | None = None,
        persist: bool = True,
        return_session_key: bool = False,
    ) -> str | tuple[str, str]:
        """发起审批。

        Args:
            plan: canonical 执行计划（command/argv/cwd/resolved_path），存入
                pending；调用方在审批通过后经 get_pending_plan 比对，
                防止执行内容与审批内容漂移（对齐 openclaw approval mismatch）。
            ask_fallback: 审批卡发送失败/超时的降级策略
                （"deny"|"allowlist"|"full"），默认取 host defaults。
            persist: allow-always 是否落白名单。strictInlineEval 的内联求值
                命令传 False（对齐 openclaw：allow-always 不持久化 inline-eval）。
            return_session_key: True 时返回 (decision, session_key)，供调用方
                执行前比对 plan。
        """
        if not self._admin_ids:
            return (DECISION_DENY, "") if return_session_key else DECISION_DENY
        fallback = ask_fallback or self.get_host_policy().ask_fallback

        admin_id = next(iter(self._admin_ids))
        session_key = f"approval:{chat_id}:{tool_name}:{uuid.uuid4().hex[:8]}"
        future = asyncio.get_running_loop().create_future()
        self._pending[session_key] = future
        if plan:
            # 深拷贝：审批期间调用方对 plan 的后续修改不污染 stored，
            # 使执行前的 _plan_mismatch 比对真实有效（防内容漂移）
            import copy

            self._pending_plans[session_key] = copy.deepcopy(plan)
        self._pending_info[session_key] = {
            "tool_name": tool_name,
            "details": details,
            "created_at": time.time(),
            "expires_at": time.time() + timeout,
        }

        from qqbot_agent_sdk import ApprovalRequest, ApprovalSender

        req = ApprovalRequest(
            session_key=session_key,
            title=f"🔐 {tool_name} 审批请求",
            description=f"原因: {reason}",
            command_preview=details,
            cwd=(plan or {}).get("cwd", "") if plan else "",
            severity="info",
            timeout_sec=timeout,
        )

        sender = ApprovalSender(self._api, log_tag="Approval")
        # 2.3 多目标转发：主目标（admin c2c）+ 配置的转发目标，同一 session_key；
        # 任一目标送达即视为可达（任一目标 resolve 都生效）。非法目标跳过。
        targets = [("c2c", admin_id)]
        for t in self._forward_to:
            parsed = _parse_target(t)
            if parsed is None:
                _log.warning("忽略非法审批转发目标: %r", t)
                continue
            targets.append(parsed)
        sent = False
        for chat_type, target_id in targets:
            try:
                if await sender.send(
                    chat_type=chat_type, chat_id=target_id, req=req
                ):
                    sent = True
            except Exception as e:
                _log.warning(
                    "审批消息发送异常 %s:%s..: %s", chat_type, target_id[:12], e
                )
        if not sent:
            self._pending.pop(session_key, None)
            self._pending_plans.pop(session_key, None)
            self._pending_info.pop(session_key, None)
            _log.warning("审批消息发送失败: session_key=%s..", session_key[:20])
            # 2.3 失败兜底：文本通知 admin（卡片未送达也可人工处理）
            self._spawn_admin_notice(
                f"⚠️ 审批请求未能送达（{session_key}）。\n"
                f"命令: {details[:120]}\n"
                f"已按策略自动处理；如需重新审批请重新发起该命令。"
            )
            return (
                (self._apply_fallback(fallback, details), session_key)
                if return_session_key
                else self._apply_fallback(fallback, details)
            )
        _log.info(
            "审批请求已发送到 admin %s..: tool=%s session_key=%s..",
            admin_id[:12],
            tool_name,
            session_key[:20],
        )

        loop = asyncio.get_running_loop()
        timeout_handle = loop.call_later(
            timeout, self._on_timeout, session_key, fallback, details
        )

        try:
            result = await future
            timeout_handle.cancel()
            if result == DECISION_ALLOW_ALWAYS:
                if persist:
                    self.add_to_whitelist(tool_name, details, plan=plan)
            if result not in ALLOW_DECISIONS:
                # 未放行：plan 由调用方 take_pending_plan 读取；非 allow 直接清理
                self._pending_plans.pop(session_key, None)
            self._pending_info.pop(session_key, None)
            return (result, session_key) if return_session_key else result
        except asyncio.CancelledError:
            timeout_handle.cancel()
            self._pending.pop(session_key, None)
            self._pending_plans.pop(session_key, None)
            self._pending_info.pop(session_key, None)
            raise

    def take_pending_plan(self, session_key: str) -> dict | None:
        """取出并删除审批对应的 canonical plan（调用方执行前比对后清理）。"""
        return self._pending_plans.pop(session_key, None)

    def list_pending(self) -> list[dict]:
        """2.3：待审批列表（文本命令/管理用）。

        Returns:
            每个 pending 的元信息（session_key/tool_name/details/created_at/
            remaining_secs），按创建时间倒序；不含已处理的条目。
        """
        now = time.time()
        out: list[dict] = []
        for key, future in self._pending.items():
            if future.done():
                continue
            info = self._pending_info.get(key, {})
            expires = info.get("expires_at", 0)
            out.append(
                {
                    "session_key": key,
                    "tool_name": info.get("tool_name", ""),
                    "details": info.get("details", ""),
                    "created_at": info.get("created_at", 0),
                    "remaining_secs": (
                        max(0, int(expires - now)) if expires else None
                    ),
                }
            )
        out.sort(key=lambda p: p["created_at"], reverse=True)
        return out

    def _spawn_admin_notice(self, text: str):
        """向 admin c2c 异步补发一条文本通知（审批卡失败/超时兜底，2.3）。"""
        if not self._admin_ids:
            return
        admin_id = next(iter(self._admin_ids))
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _notify():
            try:
                await self._api.send_text("c2c", admin_id, text)
            except Exception as e:
                _log.warning("审批通知文本发送失败: %s", e)

        try:
            loop.create_task(_notify())
        except Exception as e:
            _log.warning("审批通知任务创建失败: %s", e)

    def _apply_fallback(self, fallback: str, details: str) -> str:
        """审批不可达时的降级判定（对齐 openclaw askFallback）。

        返回 DECISION_ALLOW | DECISION_DENY：fallback=full → allow；
        fallback=allowlist 时按新 allowlist 逐段语义匹配（每个 segment 都命中
        才 allow）；fallback=deny（默认）→ deny。
        """
        if fallback == "full":
            return DECISION_ALLOW
        if fallback == "allowlist":
            from core.approval.allowlist import match_allowlist
            from core.tools.exec_analysis import analyze_command

            segments = analyze_command(details, env=os.environ)
            if not segments:
                return DECISION_DENY
            satisfied, _ = match_allowlist(segments, self.get_allowlist_entries())
            return DECISION_ALLOW if satisfied else DECISION_DENY
        return DECISION_DENY

    def resolve(self, session_key: str, decision: str, approver_id: str) -> bool:
        if approver_id not in self._admin_ids:
            _log.warning("非管理员 %s.. 试图审批，忽略", approver_id[:16])
            return False
        future = self._pending.pop(session_key, None)
        if future is None:
            # 2.3 文本命令兜底：支持唯一前缀匹配（卡片不展示完整 session key；
            # 支持尾部 uuid 片段，如 approval:chat:exec:abc12345 → abc）
            matches = [
                k
                for k in self._pending
                if k.startswith(session_key)
                or (":" in k and k.rsplit(":", 1)[1].startswith(session_key))
            ]
            if len(matches) == 1:
                session_key = matches[0]
                future = self._pending.pop(session_key, None)
        self._pending_info.pop(session_key, None)
        # plan 保留：由调用方 take_pending_plan 在比对后清理
        if future and not future.done():
            future.set_result(decision)
            _log.info(
                "审批已处理: session_key=%s.. decision=%s", session_key[:20], decision
            )
            return True
        _log.warning("审批 future 不存在或已完成: %s..", session_key[:20])
        return False

    def _on_timeout(self, session_key: str, fallback: str = "deny", details: str = ""):
        future = self._pending.pop(session_key, None)
        self._pending_info.pop(session_key, None)
        if future and not future.done():
            result = self._apply_fallback(fallback, details)
            future.set_result(result)
            _log.info("审批超时: %s.. (fallback=%s)", session_key[:20], fallback)
            # 2.3 前台兜底：文本通知 admin（超时后可凭 session key 了解/重试）
            self._spawn_admin_notice(
                f"⏰ 审批请求已超时（{session_key}），已按策略自动处理。\n"
                f"命令: {details[:120]}\n"
                f"如需重新审批，请重新发起该命令。"
            )
