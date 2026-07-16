"""ToolExecutor — 工具执行器

从 AgentEngine 剥离的纯执行逻辑，通过依赖注入获取所需的外部服务引用。
每个 executor 方法与 AgentEngine 中的原始实现一一对应。
"""

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE

from datetime import datetime, timedelta, timezone

from core.managers.nickname_manager import NicknameManager
from core.managers.workspace_manager import WorkspaceManager

_log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 上下文与返回值
# ════════════════════════════════════════════════════════════

@dataclass
class ToolContext:
    """每次工具调用的运行时上下文（与调用绑定的变量）。"""
    chat_id: str
    is_group: bool
    reply_to: str
    sender_id: str
    reply_callback: Callable
    delivery_channel: str = ""       # 真实聊天 ID（后台任务时与 chat_id 不同）
    reply_to_message_id: str = ""    # 原始消息 ID（后台任务时 reply_to 是合成 ID，此为真实的）


@dataclass
class ToolResult:
    """统一工具返回值。

    Attributes:
        content: 回吐给 AI 的文本（通常是 JSON 字符串）。
        sent_emoji: 仅 send_emoji 使用，标记是否成功发送了表情图片。
    """
    content: str
    sent_emoji: bool = False


# ════════════════════════════════════════════════════════════
# ToolExecutor
# ════════════════════════════════════════════════════════════

class ToolExecutor:
    """工具执行器。

    聚合所有工具的执行方法，通过构造函数接收所需的外部依赖。
    昵称数据通过 NicknameManager 实例统一访问，与 AgentEngine 共享同一实例。

    所有 handler 统一签名 ``(args: dict, ctx: ToolContext) -> ToolResult``。
    同步或异步由注册时的 ``is_async`` 标记决定。
    """

    def __init__(
        self,
        *,
        emoji_manager=None,
        media_uploader=None,
        api_client=None,
        hindsight=None,
        nickname_manager: Optional[NicknameManager] = None,
        bot_id: str = "",
        skill_managers=None,
        learning_orchestrator=None,
        admin_ids: Optional[list] = None,
        permission_manager=None,
    ):
        self._emoji_manager = emoji_manager
        self._media_uploader = media_uploader
        self._api_client = api_client
        self._hindsight = hindsight
        self._nm = nickname_manager
        self._bot_id = bot_id
        self._skill_managers = skill_managers
        self._learners = learning_orchestrator
        self._admin_ids = admin_ids or []
        self._perm = permission_manager
        self._bot_engine = None
        self._task_manager = None
        self._cron_job_manager = None
        self._background_task_runner = None
        self._workspace_manager = None
        self._heartbeat_response: dict = {}

        self._registry: Dict[str, tuple[Callable, bool]] = {}
        self._register_all()

    # ── 注册系统 ──

    def _register(self, name: str, handler: Callable, *, is_async: bool = False):
        self._registry[name] = (handler, is_async)

    def _register_all(self):
        self._register("search_emoji", self._exec_search_emoji)
        self._register("send_emoji", self._exec_send_emoji, is_async=True)
        self._register("search_user", self._exec_search_user)
        self._register("search_memory", self._exec_search_memory, is_async=True)
        self._register("mark_important", self._exec_mark_important, is_async=True)
        self._register("search_relation", self._exec_search_relation, is_async=True)
        self._register("rescan_skills", self._exec_rescan_skills)
        self._register("view_skill", self._exec_view_skill)
        self._register("execute_skill", self._exec_execute_skill, is_async=True)
        self._register("execute_command", self._exec_execute_command, is_async=True)
        self._register("define_jargon", self._exec_define_jargon, is_async=True)
        self._register("report_behavior_effect", self._exec_report_behavior_effect, is_async=True)
        self._register("create_task", self._exec_create_task, is_async=True)
        self._register("create_cron_job", self._exec_create_cron_job, is_async=True)
        self._register("cancel_task", self._exec_cancel_task, is_async=True)
        self._register("list_tasks", self._exec_list_tasks, is_async=True)
        self._register("list_cron_jobs", self._exec_list_cron_jobs)
        self._register("update_cron_job", self._exec_update_cron_job, is_async=True)
        self._register("delete_cron_job", self._exec_delete_cron_job, is_async=True)
        self._register("enable_cron_job", self._exec_enable_cron_job, is_async=True)
        self._register("disable_cron_job", self._exec_disable_cron_job, is_async=True)
        self._register("read_file", self._exec_read_file, is_async=True)
        self._register("write_file", self._exec_write_file, is_async=True)
        self._register("edit_file", self._exec_edit_file, is_async=True)
        self._register("list_files", self._exec_list_files, is_async=True)
        self._register("search_files", self._exec_search_files, is_async=True)
        self._register("heartbeat_respond", self._exec_heartbeat_respond)

    # ── 懒注入（AgentEngine 在运行时更新引用）──

    def set_media_uploader(self, uploader):
        self._media_uploader = uploader

    def set_api_client(self, client):
        self._api_client = client

    def set_bot_engine(self, engine):
        self._bot_engine = engine

    def set_nickname_manager(self, nm: NicknameManager):
        self._nm = nm

    def set_workspace_manager(self, wm: WorkspaceManager):
        self._workspace_manager = wm

    def set_task_managers(self, *, task_manager=None, cron_job_manager=None, background_task_runner=None):
        """注入任务管理器引用（后台任务工具需要）。"""
        self._task_manager = task_manager
        self._cron_job_manager = cron_job_manager
        self._background_task_runner = background_task_runner

    # ── 统一入口 ──

    async def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        """按工具名从 registry 查找 handler 并执行。"""
        entry = self._registry.get(name)
        if entry is None:
            _log.warning(f"未知工具调用: {name}")
            return ToolResult(content=json.dumps({"error": f"未知工具: {name}"}))

        # ── 工具权限检查 ──
        if self._perm:
            role = self._perm.get_user_role(ctx.sender_id)
            if not self._perm.can_use_tool(name, role):
                _log.warning(
                    f"工具权限拒绝: {name} role={role} sender={ctx.sender_id[:16]}.."
                )
                return ToolResult(content=json.dumps(
                    {"error": f"你没有权限使用该工具（需要 {self._perm._require_level(name)} 及以上）"},
                    ensure_ascii=False,
                ))

        _log.info(
            f"[工具调用] {name}: {json.dumps(args, ensure_ascii=False)[:200]}"
        )

        handler, is_async = entry
        if is_async:
            result = await handler(args, ctx)
        else:
            result = handler(args, ctx)

        _log.info(f"[工具调用] {name} 输出: {result.content[:200]}")
        return result

    # ════════════════════════════════════════════════════════
    # Emoji 工具
    # ════════════════════════════════════════════════════════

    def _exec_search_emoji(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 search_emoji — 按标签搜索表情。"""
        if self._emoji_manager is None:
            return ToolResult(content=json.dumps({"error": "表情管理器未就绪"}, ensure_ascii=False))

        query = args.get("query", "").strip()
        if not query:
            return ToolResult(content=json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False))

        results = self._emoji_manager.find_emojis(query, max_results=5)
        if not results:
            return ToolResult(content=json.dumps(
                {"error": "未找到匹配的表情", "query": query},
                ensure_ascii=False,
            ))

        result_data = []
        for r in results:
            desc = r.get("user_description") or r.get("auto_description", "") or "(无描述)"
            tags = r.get("user_tags") or r.get("auto_tags", []) or []
            result_data.append({
                "hash": r["hash"][:12],
                "description": desc,
                "tags": tags,
            })

        return ToolResult(content=json.dumps(result_data, ensure_ascii=False))

    async def _exec_send_emoji(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 send_emoji — 上传并发送表情图片。"""
        if self._emoji_manager is None or self._media_uploader is None:
            return ToolResult(content=json.dumps(
                {"success": False, "reason": "表情管理器或上传器未就绪"},
                ensure_ascii=False,
            ))

        emoji_hash = (args.get("emoji_hash") or "").strip()
        if not emoji_hash:
            return ToolResult(content=json.dumps(
                {"success": False, "reason": "未提供表情 hash"},
                ensure_ascii=False,
            ))

        # 后台任务时 delivery_channel 为真实聊天 ID，reply_to_message_id 为原始消息 ID
        # 后台任务 → 主动发送（不传 msg_id）；正常对话 → 回复
        effective_chat_id = ctx.delivery_channel or ctx.chat_id
        is_background = bool(ctx.delivery_channel)
        effective_reply_to = None if is_background else ctx.reply_to

        success, description, file_name, error = await self._send_emoji_by_hash(
            chat_id=effective_chat_id,
            emoji_hash=emoji_hash,
            is_group=ctx.is_group,
            reply_to=effective_reply_to,
        )

        if success:
            _log.info(f"表情已发送: {description}")
            return ToolResult(
                content=json.dumps({
                    "success": True,
                    "description": description,
                    "message": f"表情「{description}」已发送到聊天中",
                }, ensure_ascii=False),
                sent_emoji=True,
            )
        else:
            _log.warning(f"表情发送失败 [{emoji_hash[:12]}..]: {error}")
            return ToolResult(content=json.dumps({
                "success": False,
                "reason": error or "发送失败",
                "suggestion": "可以搜索其他表情试试，或直接用文字表达",
            }, ensure_ascii=False))

    async def _send_emoji_by_hash(
        self,
        chat_id: str,
        emoji_hash: str,
        is_group: bool,
        reply_to: Optional[str] = None,
    ) -> Tuple[bool, str, str, str]:
        """上传并发送已缓存的 emoji 图片到聊天。"""
        if self._emoji_manager is None or self._media_uploader is None:
            return False, "", "", "表情管理器或上传器未就绪"

        if len(emoji_hash) < 12:
            record = self._emoji_manager.get_info(emoji_hash)
            if not record:
                return False, "", "", f"未找到表情: {emoji_hash}"
        else:
            record = self._emoji_manager.find_by_hash(emoji_hash)
            if not record:
                return False, "", "", f"未找到表情: {emoji_hash[:12]}.."

        full_hash = record["hash"]
        file_name = record.get("file_name", "")
        local_path = self._emoji_manager._emoji_dir / file_name

        if not local_path.exists():
            return False, "", file_name, f"本地文件缺失: {local_path}"

        desc = record.get("user_description") or record.get("auto_description", "") or "表情"
        chat_type = "group" if is_group else "c2c"

        try:
            file_info = await self._media_uploader.upload(
                chat_type=chat_type,
                chat_id=chat_id,
                source=str(local_path),
                file_type=MEDIA_TYPE_IMAGE,
                file_name=file_name,
            )

            if reply_to:
                await self._bot_engine.send_reply(
                    chat_id=chat_id,
                    is_group=is_group,
                    message_id=reply_to,
                    media_file_info=file_info,
                )
            else:
                await self._bot_engine.send_proactive(
                    chat_id=chat_id,
                    is_group=is_group,
                    media_file_info=file_info,
                )

            record = self._emoji_manager.get_info(full_hash)
            if record:
                count = record.get("used_count", 0) + 1
                await self._emoji_manager.update_emoji(full_hash, used_count=count)

            _log.info(f"表情图片已发送 [{full_hash[:12]}..]: {desc}")
            return True, desc, file_name, ""

        except Exception as e:
            _log.error(f"发送表情图片失败 [{full_hash[:12]}..]: {e}")
            return False, desc, file_name, str(e)

    # ════════════════════════════════════════════════════════
    # 用户搜索工具
    # ════════════════════════════════════════════════════════

    def _exec_search_user(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 search_user — 按昵称模糊搜索群用户。"""
        query = args.get("query", "").strip().lower()
        if not query:
            return ToolResult(content=json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False))

        results = []
        seen = set()

        if self._nm:
            for uid, aliases in self._nm.iter_users():
                if uid in seen:
                    continue
                if query in uid.lower() or any(query in a.lower() for a in aliases):
                    seen.add(uid)
                    results.append({
                        "id": uid,
                        "nickname": aliases[-1] if aliases else uid,
                        "aliases": aliases,
                    })

        if not results:
            return ToolResult(content=json.dumps(
                {"error": "未找到匹配的用户", "query": query},
                ensure_ascii=False,
            ))

        return ToolResult(content=json.dumps(results[:10], ensure_ascii=False))

    # ════════════════════════════════════════════════════════
    # Hindsight 记忆工具
    # ════════════════════════════════════════════════════════

    async def _exec_search_memory(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 search_memory — 查询长期记忆。"""
        query = (args.get("query") or "").strip()
        if not query:
            return ToolResult(content=json.dumps({"error": "请输入搜索内容"}, ensure_ascii=False))

        person_name = (args.get("person_name") or "").strip()
        method = args.get("method", "hybrid")
        target_id = ctx.sender_id
        display_name = "当前用户"

        if person_name:
            if not ctx.is_group:
                return ToolResult(content=json.dumps(
                    {"error": "私聊中无法搜索其他人"}, ensure_ascii=False
                ))
            if not self._nm:
                return ToolResult(content=json.dumps(
                    {"error": "昵称管理器未就绪"}, ensure_ascii=False
                ))
            result_data = self._resolve_person(person_name, ctx)
            matched_id, display_name = result_data
            if not matched_id:
                if display_name.startswith("找到多个匹配"):
                    return ToolResult(content=json.dumps({"error": display_name}, ensure_ascii=False))
                return ToolResult(content=json.dumps(
                    {"error": f"在昵称列表中找不到叫「{person_name}」的人"},
                    ensure_ascii=False,
                ))
            target_id = matched_id

        if not self._hindsight:
            return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

        result = await self._hindsight.search(
            user_id=target_id,
            query=query,
            top_k=10,
            include_profile=True,
            method=method,
        )
        profiles = result.get("profiles", [])
        episodes = result.get("episodes", [])

        if not episodes and not profiles:
            return ToolResult(content=json.dumps(
                {"info": f"关于「{display_name}」暂无相关记忆记录"},
                ensure_ascii=False,
            ))

        lines = [f"关于「{display_name}」的记忆检索结果："]
        if profiles:
            lines.append("【人物画像】")
            for p in profiles[:3]:
                pd = p.get("profile_data", {})
                if isinstance(pd, dict):
                    for k, v in pd.items():
                        lines.append(f"- {k}: {v}")
        if episodes:
            lines.append("【相关记忆】")
            for e in episodes[:5]:
                content = e.get("summary", "") or e.get("subject", "") or e.get("episode", "")
                mem_type = e.get("memory_type", "episode")
                if content:
                    lines.append(f"- [{mem_type}] {content[:200]}")
        return ToolResult(content="\n".join(lines))

    async def _exec_mark_important(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 mark_important — 将重要信息注入记忆缓冲并触发提取。"""
        if not self._hindsight:
            return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

        profile_data_str = (args.get("profile_data") or "").strip()
        summary = (args.get("summary") or "").strip()
        stored = []

        if profile_data_str:
            try:
                profile_dict = json.loads(profile_data_str)
                if isinstance(profile_dict, dict) and profile_dict:
                    facts = "；".join(
                        f"{k}是{v}" for k, v in profile_dict.items()
                    )
                    profile_msg = (
                        f"[这是关于我（{ctx.sender_id}）的自我介绍，请记住这些信息] {facts}"
                    )
                    await self._hindsight.add_message(
                        session_id=ctx.chat_id,
                        sender_id=ctx.sender_id,
                        content=profile_msg,
                        role="user",
                        context="用户自我描述",
                        timestamp=None,
                    )
                    stored.append("画像信息")
            except json.JSONDecodeError:
                _log.warning(
                    f"mark_important profile_data JSON 解析失败: {profile_data_str[:100]}"
                )

        if summary:
            await self._hindsight.add_message(
                session_id=ctx.chat_id,
                sender_id=ctx.sender_id,
                content=f"[重要事件] {summary}",
                role="user",
                context="重要事件记录",
                timestamp=None,
            )
            stored.append("事件摘要")

        msg = "已标记为重要记忆。"
        if stored:
            msg += f" 已记录：{'、'.join(stored)}。"

        return ToolResult(content=json.dumps(
            {"success": True, "message": msg}, ensure_ascii=False,
        ))

    # ════════════════════════════════════════════════════════
    # 关系搜索工具
    # ════════════════════════════════════════════════════════

    async def _exec_search_relation(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 search_relation — 多维关系搜索。"""
        person_a_raw = (args.get("person_a") or "").strip()
        person_b_raw = (args.get("person_b") or "").strip()
        query = (args.get("query") or "").strip()
        method = args.get("method", "hybrid")

        if not person_a_raw or not person_b_raw:
            return ToolResult(content=json.dumps(
                {"error": "请指定两个人名或昵称"}, ensure_ascii=False
            ))

        if not self._hindsight:
            return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

        a_id, a_name = self._resolve_person(person_a_raw, ctx)
        b_id, b_name = self._resolve_person(person_b_raw, ctx)

        if a_id is None:
            return ToolResult(content=json.dumps(
                {"error": f"找不到「{person_a_raw}」对应的用户"}, ensure_ascii=False
            ))
        if b_id is None:
            return ToolResult(content=json.dumps(
                {"error": f"找不到「{person_b_raw}」对应的用户"}, ensure_ascii=False
            ))

        def make_query(base: str, target_name: str) -> str:
            return f"{base} {target_name}" if base else target_name

        query_a = make_query(query, b_name)
        query_b = make_query(query, a_name)
        query_speaker = make_query(query, f"{a_name} {b_name}")

        tasks = []
        task_labels: Dict[int, Tuple[str, str, str]] = {}

        tasks.append(self._hindsight.search(
            user_id=a_id, query=query_a, top_k=5, include_profile=True, method=method
        ))
        task_labels[len(tasks) - 1] = ("a", a_id, a_name)

        if b_id != a_id:
            tasks.append(self._hindsight.search(
                user_id=b_id, query=query_b, top_k=5, include_profile=True, method=method
            ))
            task_labels[len(tasks) - 1] = ("b", b_id, b_name)

        if ctx.sender_id not in (a_id, b_id):
            tasks.append(self._hindsight.search(
                user_id=ctx.sender_id, query=query_speaker, top_k=5,
                include_profile=False, method=method,
            ))
            task_labels[len(tasks) - 1] = ("speaker", ctx.sender_id, "当前用户")

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        lines = [f"关于「{a_name}」和「{b_name}」的关系检索结果："]
        seen_episodes: Set[str] = set()

        for idx, r in enumerate(raw_results):
            role, uid, label = task_labels[idx]

            if isinstance(r, Exception):
                _log.warning(f"search_relation {role}({uid}) 失败: {r}")
                continue

            profiles = r.get("profiles", [])
            episodes = r.get("episodes", [])

            if role == "a":
                if profiles:
                    lines.append(f"【{label} 的人物画像】")
                    for p in profiles[:3]:
                        pd = p.get("profile_data", {})
                        if isinstance(pd, dict):
                            for k, v in pd.items():
                                lines.append(f"- {k}: {v}")
                if episodes:
                    lines.append(f"【{label} 记忆中关于 {b_name} 的内容】")
                    self._append_deduped(lines, episodes, seen_episodes)

            elif role == "b":
                if profiles:
                    lines.append(f"【{label} 的人物画像】")
                    for p in profiles[:3]:
                        pd = p.get("profile_data", {})
                        if isinstance(pd, dict):
                            for k, v in pd.items():
                                lines.append(f"- {k}: {v}")
                if episodes:
                    lines.append(f"【{label} 记忆中关于 {a_name} 的内容】")
                    self._append_deduped(lines, episodes, seen_episodes)

            elif role == "speaker":
                if episodes:
                    lines.append("【你对两人的相关记载】")
                    self._append_deduped(lines, episodes, seen_episodes)

        if len(lines) == 1:
            lines.append("（未找到两人相关的记忆记录）")

        return ToolResult(content="\n".join(lines))

    # ════════════════════════════════════════════════════════
    # Skill 工具
    # ════════════════════════════════════════════════════════

    def _exec_rescan_skills(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 rescan_skills — 重新扫描技能。"""
        if not self._skill_managers:
            return ToolResult(content=json.dumps(
                {"error": "技能系统未就绪"}, ensure_ascii=False,
            ))
        result = self._skill_managers.rescan()
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    def _exec_view_skill(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 view_skill — 查看技能详细说明。"""
        if not self._skill_managers:
            return ToolResult(content=json.dumps(
                {"error": "技能系统未就绪"}, ensure_ascii=False,
            ))
        skill_name = (args.get("skill_name") or "").strip()
        if not skill_name:
            return ToolResult(content=json.dumps(
                {"error": "请提供技能名称"}, ensure_ascii=False,
            ))
        content = self._skill_managers.get_skill_detail(skill_name)
        return ToolResult(content=content)

    async def _exec_execute_skill(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 execute_skill — 运行技能自带的脚本。"""
        if not self._skill_managers:
            return ToolResult(content=json.dumps(
                {"error": "技能系统未就绪"}, ensure_ascii=False,
            ))
        skill_name = (args.get("skill_name") or "").strip()
        script_name = (args.get("script_name") or "").strip()
        if not skill_name or not script_name:
            return ToolResult(content=json.dumps(
                {"error": "请提供技能名称和脚本名称"}, ensure_ascii=False,
            ))
        arguments = args.get("arguments") or {}
        timeout = args.get("timeout", 30)

        result = self._skill_managers.execute_skill_script(
            skill_name=skill_name,
            script_name=script_name,
            arguments=arguments,
            timeout=timeout,
        )
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    async def _exec_execute_command(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 execute_command — 运行 bash 命令。"""
        if not self._skill_managers:
            return ToolResult(content=json.dumps(
                {"error": "技能系统未就绪"}, ensure_ascii=False,
            ))

        command = (args.get("command") or "").strip()
        if not command:
            return ToolResult(content=json.dumps(
                {"error": "请提供要执行的命令"}, ensure_ascii=False,
            ))

        timeout = args.get("timeout", 30)
        workdir = args.get("workdir")

        role = self._perm.get_user_role(ctx.sender_id) if self._perm else "admin"

        result = self._skill_managers.execute_command(
            command=command,
            timeout=timeout,
            workdir=workdir,
            user_role=role,
        )
        return ToolResult(content=json.dumps(result, ensure_ascii=False))

    # ════════════════════════════════════════════════════════
    # 学习工具
    # ════════════════════════════════════════════════════════

    async def _exec_define_jargon(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 define_jargon — AI 主动学习/解释俚语。"""
        if not self._learners:
            return ToolResult(content=json.dumps(
                {"error": "学习系统未就绪"}, ensure_ascii=False,
            ))

        term = (args.get("term") or "").strip()
        definition = (args.get("definition") or "").strip()
        example = (args.get("example") or "").strip()

        if not term or not definition:
            return ToolResult(content=json.dumps(
                {"error": "请提供俚语词汇和含义"}, ensure_ascii=False,
            ))

        examples = [example] if example else []
        await self._learners.add_jargon(
            term=term,
            definition=definition,
            examples=examples,
            added_by="AI",
            chat_id=ctx.chat_id,
        )

        return ToolResult(content=json.dumps({
            "success": True,
            "message": f"已学习俚语「{term}」: {definition}",
        }, ensure_ascii=False))

    async def _exec_report_behavior_effect(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 report_behavior_effect — AI 报告行为效果。"""
        if not self._learners:
            return ToolResult(content=json.dumps(
                {"error": "学习系统未就绪"}, ensure_ascii=False,
            ))

        scene = (args.get("scene_summary") or "").strip()
        action = (args.get("action_taken") or "").strip()
        effect = (args.get("effect") or "neutral").strip()

        if not scene or not action:
            return ToolResult(content=json.dumps(
                {"error": "请提供场景和行为描述"}, ensure_ascii=False,
            ))

        await self._learners.behavior.report_effect(
            scene_summary=scene,
            action_taken=action,
            effect=effect,
            chat_id=ctx.chat_id,
        )

        return ToolResult(content=json.dumps({
            "success": True,
            "message": f"已记录行为效果「{effect}」: {scene[:40]}..",
        }, ensure_ascii=False))

    # ── 辅助 ──

    def _resolve_person(
        self, raw: str, ctx: ToolContext
    ) -> Tuple[Optional[str], str]:
        raw_lower = raw.strip().lower()
        if not raw_lower:
            return None, ""

        if not self._nm:
            return None, ""

        sender_aliases = self._nm.get_aliases(ctx.sender_id)
        sender_display = sender_aliases[0] if sender_aliases else ctx.sender_id

        if raw_lower in ("我", "自己", "myself"):
            return ctx.sender_id, "当前用户"
        if raw_lower == ctx.sender_id.lower():
            return ctx.sender_id, sender_display
        if any(raw_lower == a.lower() for a in sender_aliases):
            return ctx.sender_id, sender_display

        if not ctx.is_group:
            return None, ""

        best_score = 0
        best_candidates = []

        for uid, aliases in self._nm.iter_users():
            score = 0
            if raw_lower == uid.lower():
                score = 10
            elif any(raw_lower == a.lower() for a in aliases):
                score = 8
            elif any(raw_lower in a.lower() for a in aliases):
                score = 3
            elif raw_lower in uid.lower():
                score = 1

            if score > best_score:
                best_score = score
                best_candidates = [(uid, aliases[-1] if aliases else uid)]
            elif score == best_score and score > 0:
                best_candidates.append((uid, aliases[-1] if aliases else uid))

        if not best_candidates:
            return None, ""
        if len(best_candidates) == 1:
            return best_candidates[0]

        names = "、".join(f"「{n}」({u[:12]}..)" for u, n in best_candidates)
        return None, f"找到多个匹配「{raw}」的用户: {names}"

    @staticmethod
    def _append_deduped(lines: List[str], episodes: list, seen: Set[str]):
        """将 episode 内容去重后追加到 lines。"""
        for e in episodes[:5]:
            content = e.get("summary", "") or e.get("subject", "") or e.get("episode", "")
            if content:
                dedup_key = content[:100]
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    lines.append(f"- {content[:200]}")

    # ════════════════════════════════════════════════════════
    # 后台任务工具
    # ════════════════════════════════════════════════════════

    async def _exec_create_task(self, args: dict, ctx: ToolContext) -> ToolResult:
        """创建一个一次性后台任务。"""
        if not self._task_manager or not self._background_task_runner:
            return ToolResult(content=json.dumps(
                {"error": "任务系统未就绪"}, ensure_ascii=False,
            ))

        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(content=json.dumps(
                {"error": "任务指令不能为空"}, ensure_ascii=False,
            ))

        # 在后台 fire-and-forget 执行
        task = await self._task_manager.create_task(
            prompt=prompt,
            task_type="manual",
            delivery_channel=ctx.chat_id,
            reply_to_message_id=ctx.reply_to,
        )
        asyncio.create_task(self._background_task_runner.run_task(task))

        return ToolResult(content=json.dumps({
            "success": True,
            "task_id": task.id[:16],
            "message": f"后台任务已创建！ID: {task.id[:16]}.. 执行完成后可使用 /tasks show {task.id[:12]} 查看结果",
        }, ensure_ascii=False))

    @staticmethod
    @staticmethod
    def _parse_iso_datetime(s: str) -> Optional[float]:
        """将 ISO 8601 时间字符串解析为 Unix 时间戳。
        
        没有时区信息的默认视为北京时间 (CST/UTC+8)。
        """
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                # 没有时区偏移 → 视为北京时间 CST (UTC+8)
                dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return None

    async def _exec_create_cron_job(self, args: dict, ctx: ToolContext) -> ToolResult:
        """创建定时或一次性任务。"""
        if not self._cron_job_manager:
            return ToolResult(content=json.dumps(
                {"error": "定时任务系统未就绪"}, ensure_ascii=False,
            ))

        name = (args.get("name") or "").strip()
        cron_expression = (args.get("cron_expression") or "").strip()
        at_str = (args.get("at") or "").strip()
        prompt = (args.get("prompt") or "").strip()
        session_mode = (args.get("session_mode") or "isolated").strip()
        session_id = (args.get("session_id") or "").strip()
        payload_type = (args.get("payload_type") or "message").strip()
        payload_command = (args.get("command") or "").strip()
        payload_model = (args.get("model") or "").strip() or None
        payload_thinking = (args.get("thinking") or "").strip() or None

        if not name:
            return ToolResult(content=json.dumps(
                {"error": "name 不能为空"}, ensure_ascii=False,
            ))
        if not prompt and payload_type not in ("command", "system_event"):
            return ToolResult(content=json.dumps(
                {"error": "prompt 不能为空"}, ensure_ascii=False,
            ))
        if not cron_expression and not at_str:
            return ToolResult(content=json.dumps(
                {"error": "cron_expression 和 at 必须至少提供一个"},
                ensure_ascii=False,
            ))

        # 解析一次性时间
        at_ts = None
        if at_str:
            at_ts = self._parse_iso_datetime(at_str)
            if at_ts is None:
                return ToolResult(content=json.dumps(
                    {"error": f"时间格式无法解析: {at_str}。请使用 ISO 8601 格式，如 '2027-01-01T08:00:00Z'"},
                    ensure_ascii=False,
                ))

        # 校验参数
        valid_modes = {"isolated", "custom", "main"}
        if session_mode not in valid_modes:
            session_mode = "isolated"
        custom_session_id = session_id if session_mode == "custom" else None

        valid_payloads = {"message", "command", "system_event"}
        if payload_type not in valid_payloads:
            payload_type = "message"
        if payload_type == "command" and not payload_command:
            return ToolResult(content=json.dumps(
                {"error": "payload_type=command 时 command 不能为空"},
                ensure_ascii=False,
            ))

        if payload_type == "message":
            payload_command = ""
        elif payload_type == "command":
            prompt = ""
        elif payload_type == "system_event":
            payload_command = ""

        job = await self._cron_job_manager.create_job(
            name=name,
            cron_expression=cron_expression,
            prompt=prompt,
            at=at_ts,
            delivery_channel=ctx.chat_id,
            is_group=ctx.is_group,
            session_mode=session_mode,
            custom_session_id=custom_session_id,
            payload_type=payload_type,
            command=payload_command,
            model=payload_model,
            thinking=payload_thinking,
        )

        # 构建描述
        payload_labels = {
            "message": "AI 消息",
            "command": f"Shell 命令",
            "system_event": "系统事件",
        }
        if job.is_one_shot:
            desc = f"🕐 一次性{payload_labels.get(payload_type, '任务')}「{name}」已创建！将在 {at_str} 执行。"
        else:
            desc = f"定时{payload_labels.get(payload_type, '任务')}「{name}」已创建！"

        if payload_type == "command":
            desc += f"\n命令: `{payload_command[:80]}`"

        # session 信息
        mode_desc = {
            "isolated": "每次执行使用全新隔离 session",
            "custom": f"在命名 session cron:{custom_session_id} 中执行（跨运行保留上下文）",
            "main": "在专用通道 cron:main 中执行",
        }.get(session_mode, "")
        if mode_desc:
            desc += f"\nSession 模式: {mode_desc}"

        return ToolResult(content=json.dumps({
            "success": True,
            "job_id": job.id[:16],
            "name": job.name,
            "cron_expression": job.cron_expression or "",
            "at": at_str,
            "session_mode": session_mode,
            "session_id": job.custom_session_id or "",
            "payload_type": payload_type,
            "command": payload_command or "",
            "model": payload_model or "",
            "thinking": payload_thinking or "",
            "message": desc,
        }, ensure_ascii=False))

    async def _exec_cancel_task(self, args: dict, ctx: ToolContext) -> ToolResult:
        """取消一个后台任务。"""
        if not self._task_manager:
            return ToolResult(content=json.dumps(
                {"error": "任务系统未就绪"}, ensure_ascii=False,
            ))

        task_id = (args.get("task_id") or "").strip()
        if not task_id:
            return ToolResult(content=json.dumps(
                {"error": "task_id 不能为空"}, ensure_ascii=False,
            ))

        # 先精确查找，再模糊匹配
        task = self._task_manager.get_task(task_id)
        if task is None:
            tasks = self._task_manager.list_tasks(limit=50)
            matched = [t for t in tasks if t.id.startswith(task_id)]
            if not matched:
                return ToolResult(content=json.dumps({
                    "error": f"未找到任务: {task_id}",
                }, ensure_ascii=False))
            task_id = matched[0].id

        success = await self._task_manager.cancel_task(task_id)
        if success:
            return ToolResult(content=json.dumps({
                "success": True,
                "task_id": task_id[:16],
                "message": f"任务 {task_id[:12]}.. 已取消。",
            }, ensure_ascii=False))
        return ToolResult(content=json.dumps({
            "error": f"无法取消任务 {task_id[:12]}..",
        }, ensure_ascii=False))

    async def _exec_list_tasks(self, args: dict, ctx: ToolContext) -> ToolResult:
        if not self._task_manager:
            return ToolResult(content=json.dumps(
                {"error": "任务系统未就绪"}, ensure_ascii=False,
            ))

        status_str = (args.get("status") or "").strip().lower()
        limit = args.get("limit") or 20
        if not isinstance(limit, int) or limit < 1:
            limit = 20
        limit = min(limit, 50)

        status_filter = None
        if status_str:
            from core.tasks.models import TaskStatus as TS
            try:
                status_filter = TS(status_str)
            except ValueError:
                pass

        tasks = self._task_manager.list_tasks(limit=limit, status=status_filter)
        if not tasks:
            return ToolResult(content=json.dumps({
                "tasks": [], "message": "暂无任务记录",
            }, ensure_ascii=False))

        result = []
        for t in tasks:
            result.append({
                "id": t.id,
                "type": t.type,
                "status": t.status.value,
                "created_at": t.created_at,
                "started_at": t.started_at,
                "finished_at": t.finished_at,
                "job_id": t.job_id,
                "prompt": t.prompt[:100] if t.prompt else "",
                "result": t.result[:200] if t.result else None,
                "error": t.error[:200] if t.error else None,
            })
        return ToolResult(content=json.dumps({
            "tasks": result,
            "total": len(result),
        }, ensure_ascii=False))

    # ── 辅助：按 job_id 或名称查找 cron job ──

    def _find_cron_job(self, job_id: str):
        if not self._cron_job_manager:
            return None
        job = self._cron_job_manager.get_job(job_id)
        if job is None:
            matched = self._cron_job_manager.find_jobs_by_name(job_id)
            if matched:
                job = matched[0]
        return job

    # ════════════════════════════════════════════════════════
    # 定时任务管理工具
    # ════════════════════════════════════════════════════════

    async def _exec_list_cron_jobs(self, args: dict, ctx: ToolContext) -> ToolResult:
        """列出所有定时任务。"""
        if not self._cron_job_manager:
            return ToolResult(content=json.dumps(
                {"error": "定时任务系统未就绪"}, ensure_ascii=False,
            ))
        jobs = self._cron_job_manager.list_jobs()
        if not jobs:
            return ToolResult(content=json.dumps({
                "jobs": [], "message": "暂无定时任务",
            }, ensure_ascii=False))
        result = []
        for j in jobs:
            result.append({
                "id": j.id,
                "name": j.name,
                "cron_expression": j.cron_expression or "",
                "at": j.at,
                "enabled": j.enabled,
                "next_run_at": j.next_run_at,
                "is_one_shot": j.is_one_shot,
                "session_mode": j.session_mode,
                "custom_session_id": j.custom_session_id or "",
                "payload_type": j.payload_type,
                "prompt": j.prompt[:100] if j.prompt else "",
                "command": j.command[:100] if j.command else "",
            })
        return ToolResult(content=json.dumps({
            "jobs": result,
            "total": len(result),
        }, ensure_ascii=False))

    async def _exec_update_cron_job(self, args: dict, ctx: ToolContext) -> ToolResult:
        """修改定时任务参数。"""
        if not self._cron_job_manager:
            return ToolResult(content=json.dumps(
                {"error": "定时任务系统未就绪"}, ensure_ascii=False,
            ))
        job_id = (args.get("job_id") or "").strip()
        if not job_id:
            return ToolResult(content=json.dumps(
                {"error": "job_id 不能为空"}, ensure_ascii=False,
            ))
        job = self._find_cron_job(job_id)
        if job is None:
            return ToolResult(content=json.dumps({
                "error": f"未找到定时任务: {job_id}",
            }, ensure_ascii=False))
        old_name = job.name
        changed = []

        # 逐个字段覆盖
        if "name" in args:
            job.name = (args["name"] or "").strip()
            changed.append("name")
        if "cron_expression" in args:
            job.cron_expression = (args["cron_expression"] or "").strip()
            job.at = None
            changed.append("cron_expression")
        if "at" in args:
            at_str = (args["at"] or "").strip()
            if at_str:
                at_ts = self._parse_iso_datetime(at_str)
                if at_ts is None:
                    return ToolResult(content=json.dumps({
                        "error": f"时间格式无法解析: {at_str}",
                    }, ensure_ascii=False))
                job.at = at_ts
                job.cron_expression = ""
                changed.append("at")
        if "prompt" in args:
            job.prompt = (args["prompt"] or "").strip()
            changed.append("prompt")
        if "enabled" in args:
            job.enabled = bool(args["enabled"])
            changed.append("enabled")
        if "session_mode" in args:
            mode = (args["session_mode"] or "").strip()
            valid_modes = {"isolated", "custom", "main"}
            if mode in valid_modes:
                job.session_mode = mode
                changed.append("session_mode")
        if "session_id" in args:
            sid = (args["session_id"] or "").strip()
            if job.session_mode == "custom":
                job.custom_session_id = sid
                changed.append("session_id")
        if "payload_type" in args:
            pt = (args["payload_type"] or "").strip()
            valid_payloads = {"message", "command", "system_event"}
            if pt in valid_payloads:
                job.payload_type = pt
                changed.append("payload_type")
        if "command" in args:
            job.command = (args["command"] or "").strip()
            changed.append("command")
        if "model" in args:
            job.model = (args["model"] or "").strip() or None
            changed.append("model")
        if "thinking" in args:
            job.thinking = (args["thinking"] or "").strip() or None
            changed.append("thinking")

        if not changed:
            return ToolResult(content=json.dumps({
                "error": "未提供要修改的字段",
            }, ensure_ascii=False))

        await self._cron_job_manager.update_job(job)
        return ToolResult(content=json.dumps({
            "success": True,
            "job_id": job.id[:16],
            "name": job.name,
            "changed": changed,
            "message": f"定时任务「{old_name}」已更新: {', '.join(changed)}",
        }, ensure_ascii=False))

    async def _exec_delete_cron_job(self, args: dict, ctx: ToolContext) -> ToolResult:
        """删除定时任务。"""
        if not self._cron_job_manager:
            return ToolResult(content=json.dumps(
                {"error": "定时任务系统未就绪"}, ensure_ascii=False,
            ))
        job_id = (args.get("job_id") or "").strip()
        if not job_id:
            return ToolResult(content=json.dumps(
                {"error": "job_id 不能为空"}, ensure_ascii=False,
            ))
        job = self._find_cron_job(job_id)
        if job is None:
            return ToolResult(content=json.dumps({
                "error": f"未找到定时任务: {job_id}",
            }, ensure_ascii=False))
        name = job.name
        await self._cron_job_manager.delete_job(job.id)
        return ToolResult(content=json.dumps({
            "success": True,
            "job_id": job.id[:16],
            "name": name,
            "message": f"定时任务「{name}」已删除",
        }, ensure_ascii=False))

    async def _exec_enable_cron_job(self, args: dict, ctx: ToolContext) -> ToolResult:
        """启用定时任务。"""
        if not self._cron_job_manager:
            return ToolResult(content=json.dumps(
                {"error": "定时任务系统未就绪"}, ensure_ascii=False,
            ))
        job_id = (args.get("job_id") or "").strip()
        if not job_id:
            return ToolResult(content=json.dumps(
                {"error": "job_id 不能为空"}, ensure_ascii=False,
            ))
        job = self._find_cron_job(job_id)
        if job is None:
            return ToolResult(content=json.dumps({
                "error": f"未找到定时任务: {job_id}",
            }, ensure_ascii=False))
        if job.enabled:
            return ToolResult(content=json.dumps({
                "success": True,
                "job_id": job.id[:16],
                "name": job.name,
                "message": f"定时任务「{job.name}」已是启用状态",
            }, ensure_ascii=False))
        success = await self._cron_job_manager.enable_job(job.id)
        return ToolResult(content=json.dumps({
            "success": success,
            "job_id": job.id[:16],
            "name": job.name,
            "message": f"定时任务「{job.name}」已启用",
        }, ensure_ascii=False))

    async def _exec_disable_cron_job(self, args: dict, ctx: ToolContext) -> ToolResult:
        """暂停定时任务。"""
        if not self._cron_job_manager:
            return ToolResult(content=json.dumps(
                {"error": "定时任务系统未就绪"}, ensure_ascii=False,
            ))
        job_id = (args.get("job_id") or "").strip()
        if not job_id:
            return ToolResult(content=json.dumps(
                {"error": "job_id 不能为空"}, ensure_ascii=False,
            ))
        job = self._find_cron_job(job_id)
        if job is None:
            return ToolResult(content=json.dumps({
                "error": f"未找到定时任务: {job_id}",
            }, ensure_ascii=False))
        if not job.enabled:
            return ToolResult(content=json.dumps({
                "success": True,
                "job_id": job.id[:16],
                "name": job.name,
                "message": f"定时任务「{job.name}」已是暂停状态",
            }, ensure_ascii=False))
        success = await self._cron_job_manager.disable_job(job.id)
        return ToolResult(content=json.dumps({
            "success": success,
            "job_id": job.id[:16],
            "name": job.name,
            "message": f"定时任务「{job.name}」已暂停",
        }, ensure_ascii=False))

    # ════════════════════════════════════════════════════════
    # 文件工具
    # ════════════════════════════════════════════════════════

    async def _exec_read_file(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 read_file — 读取工作区文件。"""
        if not self._workspace_manager:
            return ToolResult(content=json.dumps(
                {"error": "工作区未就绪"}, ensure_ascii=False,
            ))

        file_path = (args.get("file_path") or "").strip()
        if not file_path:
            return ToolResult(content=json.dumps(
                {"error": "请提供 file_path"}, ensure_ascii=False,
            ))

        admin_override = self._is_admin_private(ctx)
        try:
            target = self._workspace_manager.resolve_safe_path(
                ctx.is_group, ctx.chat_id, file_path, admin_override=admin_override,
            )
        except ValueError as e:
            return ToolResult(content=json.dumps(
                {"error": str(e)}, ensure_ascii=False,
            ))

        if not target.exists():
            return ToolResult(content=json.dumps(
                {"error": f"文件不存在: {file_path}"}, ensure_ascii=False,
            ))
        if not target.is_file():
            return ToolResult(content=json.dumps(
                {"error": f"路径不是文件: {file_path}"}, ensure_ascii=False,
            ))

        try:
            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except Exception as e:
            return ToolResult(content=json.dumps(
                {"error": f"读取失败: {e}"}, ensure_ascii=False,
            ))

        return ToolResult(content=json.dumps({
            "success": True,
            "content": content,
            "path": file_path,
        }, ensure_ascii=False))

    async def _exec_write_file(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 write_file — 写入文件到工作区。"""
        if not self._workspace_manager:
            return ToolResult(content=json.dumps(
                {"error": "工作区未就绪"}, ensure_ascii=False,
            ))

        file_path = (args.get("file_path") or "").strip()
        content = (args.get("content") or "")
        if not file_path:
            return ToolResult(content=json.dumps(
                {"error": "请提供 file_path"}, ensure_ascii=False,
            ))

        admin_override = self._is_admin_private(ctx)
        try:
            target = self._workspace_manager.resolve_safe_path(
                ctx.is_group, ctx.chat_id, file_path, admin_override=admin_override,
            )
        except ValueError as e:
            return ToolResult(content=json.dumps(
                {"error": str(e)}, ensure_ascii=False,
            ))

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        except Exception as e:
            return ToolResult(content=json.dumps(
                {"error": f"写入失败: {e}"}, ensure_ascii=False,
            ))

        return ToolResult(content=json.dumps({
            "success": True,
            "path": file_path,
            "size": len(content),
        }, ensure_ascii=False))

    async def _exec_edit_file(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 edit_file — 精确字符串替换编辑文件。"""
        if not self._workspace_manager:
            return ToolResult(content=json.dumps(
                {"error": "工作区未就绪"}, ensure_ascii=False,
            ))

        file_path = (args.get("file_path") or "").strip()
        old_string = (args.get("old_string") or "")
        new_string = (args.get("new_string") or "")
        replace_all = args.get("replace_all", False)

        if not file_path:
            return ToolResult(content=json.dumps(
                {"error": "请提供 file_path"}, ensure_ascii=False,
            ))
        if not old_string:
            return ToolResult(content=json.dumps(
                {"error": "请提供 old_string"}, ensure_ascii=False,
            ))

        admin_override = self._is_admin_private(ctx)
        try:
            target = self._workspace_manager.resolve_safe_path(
                ctx.is_group, ctx.chat_id, file_path, admin_override=admin_override,
            )
        except ValueError as e:
            return ToolResult(content=json.dumps(
                {"error": str(e)}, ensure_ascii=False,
            ))

        if not target.exists():
            return ToolResult(content=json.dumps(
                {"error": f"文件不存在: {file_path}"}, ensure_ascii=False,
            ))

        try:
            current = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except Exception as e:
            return ToolResult(content=json.dumps(
                {"error": f"读取失败: {e}"}, ensure_ascii=False,
            ))

        if replace_all:
            if old_string not in current:
                return ToolResult(content=json.dumps(
                    {"error": f"未找到匹配: {old_string[:60]}"}, ensure_ascii=False,
                ))
            new_content = current.replace(old_string, new_string)
        else:
            count = current.count(old_string)
            if count == 0:
                return ToolResult(content=json.dumps(
                    {"error": f"未找到匹配: {old_string[:60]}"}, ensure_ascii=False,
                ))
            if count > 1:
                return ToolResult(content=json.dumps(
                    {"error": f"找到 {count} 处匹配，请提供更多上下文或使用 replaceAll"},
                    ensure_ascii=False,
                ))
            new_content = current.replace(old_string, new_string, 1)

        try:
            await asyncio.to_thread(target.write_text, new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(content=json.dumps(
                {"error": f"写入失败: {e}"}, ensure_ascii=False,
            ))

        return ToolResult(content=json.dumps({
            "success": True,
            "path": file_path,
            "replaced": not replace_all,
        }, ensure_ascii=False))

    def _is_admin_private(self, ctx: ToolContext) -> bool:
        """私聊且发送者是管理员级别以上（含 system 角色）。

        通过 PermissionManager 的角色等级判断，system/admin 都能获得 workspace 根目录访问。
        """
        if ctx.is_group:
            return False
        if self._perm:
            role = self._perm.get_user_role(ctx.sender_id)
            return self._perm._role_level(role) >= 3  # admin=3, system=4
        return ctx.sender_id in self._admin_ids

    def _sandbox_target(self, is_group: bool, chat_id: str, rel_path: str, admin_override: bool = False) -> Path:
        """解析沙箱路径，支持 '.' 表示根目录。"""
        sandbox = self._workspace_manager.root_dir() if admin_override else self._workspace_manager.sandbox_dir(is_group, chat_id)
        if rel_path in ("", "."):
            return sandbox
        try:
            return self._workspace_manager.resolve_safe_path(is_group, chat_id, rel_path, admin_override=admin_override)
        except ValueError as e:
            raise

    async def _exec_list_files(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 list_files — 列出工作区文件。"""
        if not self._workspace_manager:
            return ToolResult(content=json.dumps(
                {"error": "工作区未就绪"}, ensure_ascii=False,
            ))

        admin_override = self._is_admin_private(ctx)
        rel_path = (args.get("path") or ".").strip()
        pattern = (args.get("pattern") or "").strip()

        try:
            target = self._sandbox_target(ctx.is_group, ctx.chat_id, rel_path, admin_override=admin_override)
        except ValueError as e:
            return ToolResult(content=json.dumps(
                {"error": str(e)}, ensure_ascii=False,
            ))

        if not target.exists():
            return ToolResult(content=json.dumps(
                {"error": f"路径不存在: {rel_path}"}, ensure_ascii=False,
            ))
        if not target.is_dir():
            return ToolResult(content=json.dumps(
                {"error": f"路径不是目录: {rel_path}"}, ensure_ascii=False,
            ))

        try:
            if pattern:
                items = list(target.rglob(pattern)) if "**" in pattern else list(target.glob(pattern))
                items.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
            else:
                items = sorted(
                    target.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
        except PermissionError:
            return ToolResult(content=json.dumps(
                {"error": "无权限访问该目录"}, ensure_ascii=False,
            ))

        sandbox = self._workspace_manager.root_dir() if admin_override else self._workspace_manager.sandbox_dir(ctx.is_group, ctx.chat_id)
        files_result = []
        dirs_result = []
        for item in items:
            try:
                rel = str(item.relative_to(sandbox))
            except ValueError:
                continue
            if item.is_dir():
                dirs_result.append(rel + "/")
            else:
                size = item.stat().st_size if item.is_file() else 0
                files_result.append({"path": rel, "size": size})

        return ToolResult(content=json.dumps({
            "success": True,
            "path": rel_path,
            "directories": dirs_result,
            "files": files_result,
            "total": len(dirs_result) + len(files_result),
        }, ensure_ascii=False))

    async def _exec_search_files(self, args: dict, ctx: ToolContext) -> ToolResult:
        """执行 search_files — 搜索文件内容（rg）。"""
        if not self._workspace_manager:
            return ToolResult(content=json.dumps(
                {"error": "工作区未就绪"}, ensure_ascii=False,
            ))

        pattern = (args.get("pattern") or "").strip()
        if not pattern:
            return ToolResult(content=json.dumps(
                {"error": "请提供搜索模式 pattern"}, ensure_ascii=False,
            ))

        admin_override = self._is_admin_private(ctx)
        rel_path = (args.get("path") or ".").strip()
        glob_filter = (args.get("glob") or "").strip()

        try:
            search_root = self._sandbox_target(ctx.is_group, ctx.chat_id, rel_path, admin_override=admin_override)
        except ValueError as e:
            return ToolResult(content=json.dumps(
                {"error": str(e)}, ensure_ascii=False,
            ))

        if not search_root.exists():
            return ToolResult(content=json.dumps(
                {"error": f"路径不存在: {rel_path}"}, ensure_ascii=False,
            ))

        sandbox = self._workspace_manager.root_dir() if admin_override else self._workspace_manager.sandbox_dir(ctx.is_group, ctx.chat_id)

        try:
            cmd = ["rg", "-n", "--no-heading", "--color", "never"]
            if glob_filter:
                cmd.extend(["-g", glob_filter])
            cmd.extend([pattern, str(search_root)])

            proc = await asyncio.to_thread(
                subprocess.run, cmd,
                capture_output=True, text=True, timeout=15,
            )

            if proc.returncode not in (0, 1):
                return ToolResult(content=json.dumps({
                    "error": f"搜索失败: {proc.stderr.strip()[:200]}",
                }, ensure_ascii=False))

            if not proc.stdout.strip():
                return ToolResult(content=json.dumps({
                    "success": True,
                    "pattern": pattern,
                    "matches": [],
                    "total": 0,
                }, ensure_ascii=False))

            matches = []
            MAX_MATCHES = 50
            for line in proc.stdout.splitlines():
                if len(matches) >= MAX_MATCHES:
                    break
                try:
                    raw_path, line_no, content = line.split(":", 2)
                    try:
                        rel = str(Path(raw_path).resolve().relative_to(sandbox))
                    except (ValueError, RuntimeError):
                        rel = raw_path
                    matches.append({
                        "file": rel,
                        "line": int(line_no) if line_no.isdigit() else 0,
                        "content": content.strip()[:200],
                    })
                except ValueError:
                    continue

            total = proc.stdout.count("\n")
            truncated = len(matches) < total

            return ToolResult(content=json.dumps({
                "success": True,
                "pattern": pattern,
                "matches": matches,
                "total": min(total, MAX_MATCHES),
                "truncated": truncated,
            }, ensure_ascii=False))

        except subprocess.TimeoutExpired:
            return ToolResult(content=json.dumps(
                {"error": "搜索超时（15秒）"}, ensure_ascii=False,
            ))
        except FileNotFoundError:
            return ToolResult(content=json.dumps(
                {"error": "rg (ripgrep) 未安装，请使用 execute_command 替代"}, ensure_ascii=False,
            ))
        except Exception as e:
            return ToolResult(content=json.dumps(
                {"error": f"搜索异常: {e}"}, ensure_ascii=False,
            ))

    # ── Heartbeat ──

    def _exec_heartbeat_respond(self, args: dict, ctx: ToolContext) -> ToolResult:
        """存储心跳响应状态，供 HeartbeatManager 读取。"""
        self._heartbeat_response = {
            "notify": bool(args.get("notify", False)),
            "notification_text": (args.get("notification_text") or "").strip(),
        }
        _log.info(
            f"心跳响应: notify={self._heartbeat_response['notify']} "
            f"text={self._heartbeat_response['notification_text'][:80]!r}"
        )
        return ToolResult(content=json.dumps({
            "success": True,
            "acknowledged": True,
        }, ensure_ascii=False))

    def consume_heartbeat_response(self) -> dict:
        """读取并清空心跳响应状态（供 HeartbeatManager 调用）。"""
        resp = self._heartbeat_response
        self._heartbeat_response = {}
        return resp
