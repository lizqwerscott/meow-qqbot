"""Delivery normalization — 投递标准化管道。

对应 OpenClaw heartbeat-delivery-normalization.ts。
"""

import re
from typing import Tuple

ACK_MAX_CHARS_DEFAULT = 300


def strip_heartbeat_token(text: str, ack_max_chars: int = ACK_MAX_CHARS_DEFAULT) -> Tuple[str, bool]:
    """剥离 HEARTBEAT_OK / NO_REPLY token。

    返回 (清理后文本, 是否应跳过投递)。
    对应 OpenClaw stripHeartbeatToken(mode="heartbeat")。
    """
    if not text or not text.strip():
        return "", True

    # 剥离 HTML tags 并修整空白
    cleaned = re.sub(r'<[^>]*>', ' ', text).strip()

    token, alt_token = "HEARTBEAT_OK", "NO_REPLY"
    cleaned_upper = cleaned.upper()
    has_token = token in cleaned_upper or alt_token in cleaned_upper

    # 无 token → 不跳过，不剥离（openclaw 规则）
    if not has_token:
        return cleaned, False

    # 有 token：先剥离可能的 markdown 包裹，防止 **HEARTBEAT_OK** 无法识别
    cleaned = cleaned.strip("*`~_")
    cleaned_upper = cleaned.upper()

    # 循环剥离首尾 token（大小写不敏感）
    changed = True
    while changed:
        changed = False
        for tok in (token, alt_token):
            stripped = cleaned
            if stripped.upper().startswith(tok):
                stripped = stripped[len(tok):].lstrip()
            m = re.search(re.escape(tok) + r'[^\w]{0,4}$', stripped, re.IGNORECASE)
            if m:
                stripped = stripped[:m.start()]
            if stripped != cleaned:
                cleaned = stripped
                changed = True

    if not cleaned.strip():
        return "", True

    rest = cleaned.strip()
    if len(rest) <= ack_max_chars:
        return "", True
    return rest, False


def strip_trailing_notify_false(text: str) -> Tuple[str, bool]:
    """剥离行尾 notify=false 指令。

    对应 OpenClaw stripTrailingHeartbeatNotifyFalse。
    """
    m = re.search(
        r'(?:^|[\r\n])[ \t]*notify\s*=\s*false[ \t]*(?:\r?\n[ \t]*)*$',
        text,
        re.IGNORECASE,
    )
    if m:
        return text[:m.start()].rstrip(), True
    return text, False


def normalize_heartbeat_reply(
    text: str,
    ack_max_chars: int = ACK_MAX_CHARS_DEFAULT,
) -> Tuple[str, bool]:
    """完整标准化管道。

    1. strip_heartbeat_token（剥离 token + markdown 包裹）
    2. strip_trailing_notify_false（剥离 notify=false）
    返回 (清洗后文本, 是否应跳过投递)。
    """
    cleaned, should_skip = strip_heartbeat_token(text, ack_max_chars)
    if should_skip:
        return "", True
    cleaned, had_notify_false = strip_trailing_notify_false(cleaned)
    if had_notify_false and not cleaned.strip():
        return "", True
    return cleaned, False
