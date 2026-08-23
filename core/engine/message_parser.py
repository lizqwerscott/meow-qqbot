"""MessageParser — 从原始 QQ 事件解析为 ParsedMessage。

职责：纯解析，无副作用。
副作用（昵称收集）由调用方 BotEngine 在 parse() 返回后执行。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from qqbot_agent_sdk import EventParser

from core.card_parser import parse_card
from core.managers.emoji_manager import is_custom_emoji
from core.message import MessageType, ResourceMeta

_log = logging.getLogger(__name__)


@dataclass
class ParsedMessage:
    id: str
    sender_id: str
    chat_id: str
    content: str
    chat_scope: str
    msg_type: MessageType
    resources: List[ResourceMeta]
    mentioned_ids: List[str]
    is_at_mention: bool
    replied_content: str
    replied_author: str
    replied_author_id: str
    author_id: str
    author_username: str
    replied_message_id: str = ""
    task_correlation_id: str = ""
    mention_entries: List[Tuple[str, str]] = field(default_factory=list)
    reply_author_entries: List[Tuple[str, str]] = field(default_factory=list)
    replied_resources: List[ResourceMeta] = field(default_factory=list)


@dataclass
class MessageParserDeps:
    emoji_manager: Any | None = None


class MessageParser:

    def __init__(self, deps: MessageParserDeps):
        self._deps = deps

    async def parse(self, event_type: str, raw: dict) -> ParsedMessage | None:
        event = EventParser().parse(event_type, raw)
        if event is None:
            return None

        msg_type: MessageType = MessageType.TEXT
        resources: List[ResourceMeta] = []

        emoji_manager = self._deps.emoji_manager

        if is_custom_emoji(event.content, event.attachments):
            _log.info(f"检测到自定义表情，用户: {event.user_id}")
            msg_type = MessageType.EMOJI
            try:
                if emoji_manager:
                    summary, desc, tags, emoji_hash = await emoji_manager.get_or_build(
                        event.attachments[0]
                    )
                    tag_str = " ".join(tags) if tags else ""
                    event.content = f"[表情: {summary}]"
                    if tag_str:
                        event.content += f" [情绪: {tag_str}]"
                    resources = [
                        ResourceMeta(
                            resource_type="emoji",
                            resource_id=emoji_hash,
                            hash=emoji_hash,
                            mime_type=event.attachments[0].content_type or "",
                            filename=event.attachments[0].filename or "",
                            size=event.attachments[0].size or 0,
                        )
                    ]
                else:
                    event.content = "[自定义表情]"
            except Exception as e:
                _log.error(f"自定义表情处理失败: {e}")
                event.content = "[自定义表情]"

        elif event.raw and (
            "ark" in event.raw
            or "ark_data" in event.raw
            or "embed" in event.raw
            or event.message_type in (3, 4)
        ):
            _log.info(f"Card raw: {event.raw}")
            card_text = parse_card(event.raw or {}, event.message_type)
            if card_text:
                _log.info(f"检测到卡片消息，解析为: {card_text}")
                event.content = card_text
                msg_type = MessageType.CARD
            else:
                _log.info("卡片消息解析失败，跳过")
                return None

        elif event.attachments:
            ct = (event.attachments[0].content_type or "").lower()
            if ct.startswith("image/"):
                msg_type = MessageType.IMAGE
            elif (
                "voice" in ct
                or "audio" in ct
                or ct.endswith(".silk")
                or ct.endswith(".amr")
            ):
                msg_type = MessageType.VOICE
            elif ct.startswith("video/"):
                msg_type = MessageType.VIDEO
            else:
                msg_type = MessageType.FILE
            _log.info(f"检测到{msg_type}消息: {event.attachments[0].filename}")
            resources = [
                ResourceMeta(
                    resource_type=str(msg_type),
                    resource_id=att.url.strip(),
                    source_url=att.url.strip(),
                    mime_type=att.content_type or "",
                    height=att.height or 0,
                    width=att.width or 0,
                    size=att.size or 0,
                    filename=att.filename or "",
                )
                for att in event.attachments
            ]

        else:
            stripped = event.content.strip()
            if not stripped:
                _log.debug(
                    "跳过硬解消息: event_type=%s, sender=%s",
                    event_type,
                    event.author_id[:12] if event.author_id else "?",
                )
                return None
            cleaned = re.sub(r"<faceType=\d+,[^>]+>", "", stripped).strip()
            if not cleaned:
                _log.debug(
                    "消息仅含表情面: event_type=%s, content=%s",
                    event_type,
                    stripped[:80],
                )
                return None

        mentioned_ids = []
        mentions_data = raw.get("mentions", [])
        mention_entries: List[Tuple[str, str]] = []
        for m in mentions_data:
            uid = m.get("id")
            if uid:
                mentioned_ids.append(uid)
                event.content = event.content.replace(f"<@{uid}>", f"@{uid}")
        event.content = event.content.strip()

        is_at_mention = any(m.get("is_you") for m in mentions_data)
        for m in mentions_data:
            mention_entries.append((m.get("id", ""), m.get("username", "")))

        replied_content = ""
        replied_resources: List[ResourceMeta] = []
        replied_author = ""
        replied_author_id = ""
        replied_message_id = ""
        reply_author_entries: List[Tuple[str, str]] = []
        raw_elems = raw.get("msg_elements", [])
        if event.msg_elements:
            elem = event.msg_elements[0]
            if raw_elems:
                replied_author_data = raw_elems[0].get("author", {})
                replied_author = replied_author_data.get("username", "")
                replied_author_id = replied_author_data.get("id", "")
                replied_message_id = (
                    raw_elems[0].get("id")
                    or raw_elems[0].get("message_id")
                    or raw_elems[0].get("messageId")
                    or ""
                )
            if not replied_message_id:
                replied_message_id = (
                    raw.get("replied_message_id")
                    or raw.get("reply_message_id")
                    or raw.get("message_reference", {}).get("message_id", "")
                    or ""
                )
            for raw_elem in raw_elems:
                reply_author_entries.append(
                    (
                        raw_elem.get("author", {}).get("id", ""),
                        raw_elem.get("author", {}).get("username", ""),
                    )
                )
            if elem.attachments and is_custom_emoji(
                elem.content or "", elem.attachments
            ):
                try:
                    if emoji_manager:
                        summary, desc, tags, _ = await emoji_manager.get_or_build(
                            elem.attachments[0]
                        )
                        tag_str = " ".join(tags) if tags else ""
                        replied_content = f"[表情: {summary}]"
                        if tag_str:
                            replied_content += f" [情绪: {tag_str}]"
                except Exception as e:
                    _log.error(f"解析引用消息中的自定义表情失败: {e}")
                    replied_content = "[引用消息: 自定义表情]"
            elif elem.attachments:
                replied_content = (elem.content or "") + " [含附件]"
                replied_resources = [
                    ResourceMeta(
                        resource_type=(
                            "image"
                            if (att.content_type or "").lower().startswith("image/")
                            else "file"
                        ),
                        resource_id=(att.url or "").strip(),
                        source_url=(att.url or "").strip(),
                        mime_type=att.content_type or "",
                        height=att.height or 0,
                        width=att.width or 0,
                        size=att.size or 0,
                        filename=att.filename or "",
                    )
                    for att in elem.attachments
                ]
            else:
                replied_content = elem.content or ""

        author_id = raw.get("author", {}).get("id", "")
        author_username = raw.get("author", {}).get("username", "")
        task_metadata = raw.get("task")
        if not isinstance(task_metadata, dict):
            task_metadata = {}

        return ParsedMessage(
            id=event.message_id,
            sender_id=event.user_id,
            chat_id=event.chat_id,
            content=event.content,
            chat_scope=event.chat_scope,
            msg_type=msg_type,
            resources=resources,
            mentioned_ids=mentioned_ids,
            is_at_mention=is_at_mention,
            replied_content=replied_content,
            replied_author=replied_author,
            replied_author_id=replied_author_id,
            replied_message_id=replied_message_id,
            task_correlation_id=str(
                raw.get("task_correlation_id")
                or task_metadata.get("correlation_id", "")
                or ""
            ),
            author_id=author_id,
            author_username=author_username,
            mention_entries=mention_entries,
            reply_author_entries=reply_author_entries,
            replied_resources=replied_resources,
        )
