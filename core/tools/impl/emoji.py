import json
import logging

from qqbot_agent_sdk.constants import MEDIA_TYPE_IMAGE

from core.engine.delivery_ledger import DeliveryReceipt
from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)


def create_emoji_entries(deps: ToolDeps) -> list[ToolEntry]:

    async def _search_emoji(args: dict, ctx: ToolContext) -> ToolResult:
        emoji_manager = deps.emoji_manager
        if emoji_manager is None:
            return ToolResult(
                content=json.dumps({"error": "表情管理器未就绪"}, ensure_ascii=False)
            )

        query = args.get("query", "").strip()
        if not query:
            return ToolResult(
                content=json.dumps({"error": "搜索关键词为空"}, ensure_ascii=False)
            )

        results = emoji_manager.find_emojis(query, max_results=5)
        if not results:
            return ToolResult(
                content=json.dumps(
                    {"error": "未找到匹配的表情", "query": query},
                    ensure_ascii=False,
                )
            )

        result_data = []
        for r in results:
            desc = (
                r.get("user_description") or r.get("auto_description", "") or "(无描述)"
            )
            summary = r.get("auto_summary", "") or ""
            tags = r.get("user_tags") or r.get("auto_tags", []) or []
            result_data.append(
                {
                    "hash": r["hash"][:12],
                    "summary": summary,
                    "description": desc,
                    "tags": tags,
                }
            )

        return ToolResult(content=json.dumps(result_data, ensure_ascii=False))

    async def _send_emoji(args: dict, ctx: ToolContext) -> ToolResult:
        emoji_manager = deps.emoji_manager
        media_uploader = deps.media_uploader.value
        bot_engine = deps.bot_engine.value

        if emoji_manager is None or media_uploader is None:
            return ToolResult(
                content=json.dumps(
                    {"success": False, "reason": "表情管理器或上传器未就绪"},
                    ensure_ascii=False,
                )
            )

        emoji_hash = (args.get("emoji_hash") or "").strip()
        if not emoji_hash:
            return ToolResult(
                content=json.dumps(
                    {"success": False, "reason": "未提供表情 hash"},
                    ensure_ascii=False,
                )
            )

        effective_chat_id = ctx.delivery_channel or ctx.chat_id
        is_background = bool(ctx.delivery_channel)
        effective_reply_to = None if is_background else ctx.reply_to

        if (
            ctx.turn_active_callback is not None
            and not await ctx.turn_active_callback()
        ):
            return ToolResult(
                content=json.dumps(
                    {"success": False, "reason": "当前 turn 已终结"},
                    ensure_ascii=False,
                ),
                delivery_receipt=DeliveryReceipt(
                    status="failed",
                    error_code="turn_not_active",
                    retryable=False,
                ),
            )

        success, description, file_name, error, receipt = await _send_emoji_by_hash(
            emoji_manager=emoji_manager,
            media_uploader=media_uploader,
            bot_engine=bot_engine,
            chat_id=effective_chat_id,
            emoji_hash=emoji_hash,
            is_group=ctx.is_group,
            reply_to=effective_reply_to,
        )

        if success:
            _log.info(f"表情已发送: {description}")
            return ToolResult(
                content=json.dumps(
                    {
                        "success": True,
                        "description": description,
                        "message": f"表情「{description}」已发送到聊天中",
                    },
                    ensure_ascii=False,
                ),
                sent_emoji=True,
                delivery_receipt=receipt,
            )
        else:
            _log.warning(f"表情发送失败 [{emoji_hash[:12]}..]: {error}")
            return ToolResult(
                content=json.dumps(
                    {
                        "success": False,
                        "reason": error or "发送失败",
                        "suggestion": "可以搜索其他表情试试，或直接用文字表达",
                    },
                    ensure_ascii=False,
                ),
                delivery_receipt=receipt,
            )

    async def _send_emoji_by_hash(
        emoji_manager,
        media_uploader,
        bot_engine,
        chat_id: str,
        emoji_hash: str,
        is_group: bool,
        reply_to: str | None = None,
    ) -> tuple[bool, str, str, str, DeliveryReceipt | None]:
        record = emoji_manager.find_by_hash(emoji_hash)
        if not record:
            return False, "", "", f"未找到表情: {emoji_hash[:12]}", None

        full_hash = record["hash"]
        file_name = record.get("file_name", "")
        local_path = emoji_manager._emoji_dir / file_name

        if not local_path.exists():
            return False, "", file_name, f"本地文件缺失: {local_path}", None

        desc = (
            record.get("user_description")
            or record.get("auto_description", "")
            or "表情"
        )
        chat_type = "group" if is_group else "c2c"

        try:
            file_info = await media_uploader.upload(
                chat_type=chat_type,
                chat_id=chat_id,
                source=str(local_path),
                file_type=MEDIA_TYPE_IMAGE,
                file_name=file_name,
            )

            if reply_to:
                transport_result = await bot_engine.send_reply(
                    chat_id=chat_id,
                    is_group=is_group,
                    message_id=reply_to,
                    media_file_info=file_info,
                )
            else:
                transport_result = await bot_engine.send_proactive(
                    chat_id=chat_id,
                    is_group=is_group,
                    media_file_info=file_info,
                )
            receipt = (
                transport_result
                if isinstance(transport_result, DeliveryReceipt)
                else DeliveryReceipt(
                    status="accepted",
                    logical_delivery_id=f"emoji:{chat_id}:{full_hash}",
                )
            )

            if receipt.status not in {"accepted", "partial"}:
                return False, desc, file_name, "表情发送未确认", receipt

            record = emoji_manager.get_info(full_hash)
            if record:
                count = record.get("used_count", 0) + 1
                await emoji_manager.update_emoji(full_hash, used_count=count)

            _log.info(f"表情图片已发送 [{full_hash[:12]}..]: {desc}")
            return True, desc, file_name, "", receipt

        except Exception as e:
            _log.error(f"发送表情图片失败 [{full_hash[:12]}..]: {e}")
            return (
                False,
                desc,
                file_name,
                str(e),
                DeliveryReceipt(
                    status="failed",
                    logical_delivery_id=f"emoji:{chat_id}:{full_hash}",
                    error_code="transport_exception",
                    retryable=True,
                ),
            )

    EMOJI_SEARCH_PARAMS = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用于搜索的关键词，多个词用空格分隔，例如：开心 撒娇 白猫 挥手。关键词越具体搜索越精准。",
            }
        },
        "required": ["query"],
    }

    EMOJI_SEND_PARAMS = {
        "type": "object",
        "properties": {
            "emoji_hash": {
                "type": "string",
                "description": "表情的唯一标识 hash（完整 hash 或前 12 位短 hash），通过 search_emoji 获取",
            },
            "reason": {
                "type": "string",
                "description": "发送这个表情的原因或想表达的情绪，仅用于记录",
            },
        },
        "required": ["emoji_hash", "reason"],
    }

    return [
        ToolEntry(
            name="search_emoji",
            section="emoji",
            description="搜索表情图片。输入一个或多个关键词（标签、描述词均可），用空格分开。系统会匹配任意关键词，按匹配数量排序返回。关键词越具体搜索越精准。",
            parameters=EMOJI_SEARCH_PARAMS,
            handler=_search_emoji,
        ),
        ToolEntry(
            name="send_emoji",
            section="emoji",
            description="发送一个指定的表情图片到聊天中。需要提供通过 search_emoji 获取到的表情 hash。一条回复最多发送 1 个表情。",
            parameters=EMOJI_SEND_PARAMS,
            handler=_send_emoji,
            delivery_kind="emoji",
        ),
    ]
