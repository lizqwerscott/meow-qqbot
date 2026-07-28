"""Card message parser — 解析 QQ 卡片消息(ARK/EMBED)为统一分享文本格式。"""

import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


def _find_kv(kv_list: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for item in kv_list:
        if isinstance(item, dict) and item.get("key") == key:
            return item
    return None


def _extract_ark_title(ark: Dict[str, Any]) -> str:
    """从 ar.kv 中提取标题。"""
    if lst := _find_kv(ark.get("kv", []), "#LIST#"):
        objs = lst.get("obj", [])
        for obj in objs:
            obj_kv = obj.get("obj_kv", [])
            for kv in obj_kv:
                if kv.get("key") in ("title", "desc"):
                    val = (kv.get("value") or "").strip()
                    if val:
                        return val
    if desc := _find_kv(ark.get("kv", []), "#DESC#"):
        val = (desc.get("value") or "").strip()
        if val:
            return val
    if meta := _find_kv(ark.get("kv", []), "#META#"):
        val = (meta.get("value") or "").strip()
        if val:
            return val
    return ""


def _extract_ark_url(ark: Dict[str, Any]) -> str:
    """从 ar.kv 中提取链接。"""
    if lst := _find_kv(ark.get("kv", []), "#LIST#"):
        objs = lst.get("obj", [])
        for obj in objs:
            obj_kv = obj.get("obj_kv", [])
            for kv in obj_kv:
                if kv.get("key") == "url":
                    val = (kv.get("value") or "").strip()
                    if val:
                        return val
    return ""


def _extract_ark_source(ark: Dict[str, Any]) -> str:
    """从 ar.kv 中提取来源名称。"""
    if lst := _find_kv(ark.get("kv", []), "#LIST#"):
        objs = lst.get("obj", [])
        for obj in objs:
            obj_kv = obj.get("obj_kv", [])
            for kv in obj_kv:
                if kv.get("key") == "source":
                    val = (kv.get("value") or "").strip()
                    if val:
                        return val
    if prompt := _find_kv(ark.get("kv", []), "#PROMPT#"):
        val = (prompt.get("value") or "").strip()
        if val:
            return val
    return ""


def parse_ark(ark: dict) -> Optional[str]:
    """解析 ARK 消息为统一分享格式。

    支持两种格式：
    - QQ 小程序格式 (``ark_type=miniapp``)，字段在 ``fields`` 字典中
    - 标准 ARK 格式，数据在 ``kv[]`` 数组中
    """
    if not isinstance(ark, dict):
        return None

    title = source = url = ""

    # 格式A: QQ 小程序 / 图文H5 (fields + prompt)
    if "fields" in ark:
        fields = ark.get("fields", {}) or {}
        title = fields.get("title", "") or ""
        source = fields.get("source") or fields.get("tag") or ""
        url = fields.get("jump_url") or ""

    # 格式B: 标准 ARK (kv 数组)
    if not title:
        title = _extract_ark_title(ark)
        url = _extract_ark_url(ark)
    if not source:
        source = _extract_ark_source(ark)

    parts = []
    if source:
        parts.append(f"[分享 | {source}]")
    else:
        parts.append("[分享]")
    if title:
        parts.append(title)
    if url:
        parts.append(url)

    text = " ".join(parts)
    return text if text != "[分享]" else None


def parse_embed(embed: dict) -> Optional[str]:
    """解析 EMBED 消息为统一分享格式。"""
    if not isinstance(embed, dict):
        return None

    title = (embed.get("title") or "").strip()
    prompt = (embed.get("prompt") or "").strip()

    parts = ["[分享]"]
    if title:
        parts.append(title)
    if prompt:
        parts.append(prompt)

    text = " ".join(parts)
    return text if text != "[分享]" else None


def parse_card(raw: dict, message_type: int) -> Optional[str]:
    """解析卡片消息，返回统一分享格式文本。

    :param raw: 消息的原始 dict（含 ``ark`` / ``ark_data`` / ``embed`` 字段）
    :param message_type: 3=ARK, 4=EMBED
    :returns: 统一格式文本，解析失败返回 None
    """
    # 自动检测：message_type 非 3/4 但 raw 里有 ark_data/ark/embed
    if message_type not in (3, 4):
        if "ark" in raw or "ark_data" in raw:
            message_type = 3
        elif "embed" in raw:
            message_type = 4

    if message_type == 3:
        ark_data = raw.get("ark") or raw.get("ark_data")
        if ark_data:
            result = parse_ark(ark_data)
            if result:
                return result
        _log.info("CardParser: ARK 消息解析失败或无数据")

    elif message_type == 4:
        if "embed" in raw:
            result = parse_embed(raw["embed"])
            if result:
                return result
        _log.info("CardParser: EMBED 消息解析失败或无数据")

    return None
