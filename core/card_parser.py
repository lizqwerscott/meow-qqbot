"""Card message parser — 解析 QQ 卡片消息(ARK/EMBED)为统一分享文本格式。"""

import logging
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)


def _find_kv(kv_list: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for item in kv_list:
        if item.get("key") == key:
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

    QQ 小程序/分享卡片通常在 ``ark.kv[].obj[].obj_kv`` 中携带
    title / desc / url / source 等字段。
    """
    if not isinstance(ark, dict):
        return None

    title = _extract_ark_title(ark)
    url = _extract_ark_url(ark)
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

    :param raw: 消息的原始 dict（含 ``ark`` 或 ``embed`` 字段）
    :param message_type: 3=ARK, 4=EMBED
    :returns: 统一格式文本，解析失败返回 None
    """
    if message_type == 3:
        if "ark" in raw:
            result = parse_ark(raw["ark"])
            if result:
                return result
        _log.info("CardParser: ARK 消息解析失败或无数据，使用原始 content")

    elif message_type == 4:
        if "embed" in raw:
            result = parse_embed(raw["embed"])
            if result:
                return result
        _log.info("CardParser: EMBED 消息解析失败或无数据，使用原始 content")

    return None
