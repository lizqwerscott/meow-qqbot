"""ToolExecutor — 工具执行器

从 AgentEngine 剥离的纯执行逻辑，通过依赖注入获取所需的外部服务引用。
每个 executor 方法与 AgentEngine 中的原始实现一一对应。
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE
from qqbot_agent_sdk.dto import MediaInfo, MessageToCreate, QQMessageType

from core.managers.nickname_manager import NicknameManager

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
        everos=None,
        nickname_manager: Optional[NicknameManager] = None,
        bot_id: str = "",
        skill_managers=None,
        learning_orchestrator=None,
        admin_ids: Optional[list] = None,
    ):
        self._emoji_manager = emoji_manager
        self._media_uploader = media_uploader
        self._api_client = api_client
        self._everos = everos
        self._nm = nickname_manager
        self._bot_id = bot_id
        self._skill_managers = skill_managers
        self._learners = learning_orchestrator
        self._admin_ids = admin_ids or []

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

    # ── 懒注入（AgentEngine 在运行时更新引用）──

    def set_media_uploader(self, uploader):
        self._media_uploader = uploader

    def set_api_client(self, client):
        self._api_client = client

    def set_nickname_manager(self, nm: NicknameManager):
        self._nm = nm

    # ── 统一入口 ──

    async def execute(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        """按工具名从 registry 查找 handler 并执行。"""
        entry = self._registry.get(name)
        if entry is None:
            _log.warning(f"未知工具调用: {name}")
            return ToolResult(content=json.dumps({"error": f"未知工具: {name}"}))

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

        success, description, file_name, error = await self._send_emoji_by_hash(
            chat_id=ctx.chat_id,
            emoji_hash=emoji_hash,
            is_group=ctx.is_group,
            reply_to=ctx.reply_to,
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
        reply_to: str,
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

            msg_seq = self._api_client.next_msg_seq() if self._api_client else 0
            msg = MessageToCreate(
                msg_type=QQMessageType.RICH_MEDIA,
                msg_seq=msg_seq,
                msg_id=reply_to,
                media=MediaInfo(file_info=file_info),
            )

            if is_group:
                await self._api_client.post_group_message(chat_id, msg)
            else:
                await self._api_client.post_c2c_message(chat_id, msg)

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

        for source_dict, source_name in (
            [(self._nm.nicknames, "手动"), (self._nm.auto_nicknames, "自动")]
            if self._nm else []
        ):
            for uid, nickname in source_dict.items():
                if uid in seen:
                    continue
                if query in nickname.lower() or query in uid.lower():
                    seen.add(uid)
                    results.append({
                        "id": uid,
                        "nickname": nickname,
                        "source": source_name,
                    })

        if not results:
            return ToolResult(content=json.dumps(
                {"error": "未找到匹配的用户", "query": query},
                ensure_ascii=False,
            ))

        return ToolResult(content=json.dumps(results[:10], ensure_ascii=False))

    # ════════════════════════════════════════════════════════
    # EverOS 记忆工具
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
            merged = self._nm.all_merged()

            matched_id = None
            for uid, nickname in merged.items():
                if person_name.lower() in nickname.lower() or person_name.lower() in uid.lower():
                    matched_id = uid
                    display_name = nickname
                    break

            if not matched_id:
                return ToolResult(content=json.dumps(
                    {"error": f"在昵称列表中找不到叫「{person_name}」的人"},
                    ensure_ascii=False,
                ))
            target_id = matched_id

        if not self._everos:
            return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))

        result = await self._everos.search(
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
        """执行 mark_important — 触发记忆提炼。"""
        if not self._everos:
            return ToolResult(content=json.dumps({"error": "记忆系统未就绪"}, ensure_ascii=False))
        await self._everos.flush(session_id=ctx.chat_id)
        return ToolResult(content=json.dumps(
            {"success": True, "message": "已标记当前对话为重要，正在整理记忆中。"},
            ensure_ascii=False,
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

        if not self._everos:
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

        tasks.append(self._everos.search(
            user_id=a_id, query=query_a, top_k=5, include_profile=True, method=method
        ))
        task_labels[len(tasks) - 1] = ("a", a_id, a_name)

        if b_id != a_id:
            tasks.append(self._everos.search(
                user_id=b_id, query=query_b, top_k=5, include_profile=True, method=method
            ))
            task_labels[len(tasks) - 1] = ("b", b_id, b_name)

        if ctx.sender_id not in (a_id, b_id):
            tasks.append(self._everos.search(
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

        result = self._skill_managers.execute_command(
            command=command,
            timeout=timeout,
            workdir=workdir,
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

        sender_nick = self._nm.get(ctx.sender_id)

        if raw_lower in ("我", "自己", "myself"):
            return ctx.sender_id, sender_nick if sender_nick != ctx.sender_id else "当前用户"
        if sender_nick and raw_lower == sender_nick.lower():
            return ctx.sender_id, sender_nick
        if raw_lower == ctx.sender_id.lower():
            return ctx.sender_id, sender_nick if sender_nick != ctx.sender_id else "当前用户"

        if not ctx.is_group:
            return None, ""

        merged = self._nm.all_merged()

        for uid, nickname in merged.items():
            if raw_lower in nickname.lower() or raw_lower in uid.lower():
                return uid, nickname

        return None, ""

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
