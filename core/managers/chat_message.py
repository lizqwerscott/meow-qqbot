import logging
import re

_log = logging.getLogger(__name__)

# 匹配 to_dict() 中 user 消息添加的 [发言人 在 YYYY-MM-DD HH:MM:SS]: 前缀
_RE_PREFIX = re.compile(r"^\[.*? 在 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]:\s*")


def strip_content_prefix(content: str) -> str:
    """移除消息内容中由旧 to_dict() 添加的 [NAME 在 TIME]: 前缀。

    仅在从旧 JSONL 数据恢复时需要（新数据通过 raw_content 字段避免污染）。
    """
    while _RE_PREFIX.match(content):
        content = _RE_PREFIX.sub("", content)
    return content


def normalize_legacy_content(data: dict) -> str:
    """Read user content from old display-formatted or storage-formatted records."""
    role = data.get("role", "user")
    content = str(data.get("raw_content", data.get("content", "")) or "")
    if role != "user":
        return content
    if "raw_content" not in data:
        return strip_content_prefix(content)
    displayed = str(data.get("content", "") or "")
    displayed_without_outer = _RE_PREFIX.sub("", displayed, count=1)
    if _RE_PREFIX.match(content) and displayed_without_outer == content:
        return strip_content_prefix(content)
    return content
