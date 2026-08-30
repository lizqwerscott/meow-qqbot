"""Normalized configuration for conversation collection and engagement."""

import logging
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EngagementConfig:
    conversation_collect_idle_ms: int = 700
    conversation_collect_max_wait_ms: int = 1500
    conversation_collect_max_messages: int = 8
    conversation_collect_max_chars: int = 6000
    private_conversation_delivery_mode: str = "automatic"
    direct_task_delivery_mode: str = "automatic"
    group_ambient_delivery_mode: str = "message_tool_only"
    direct_task_collaboration_enabled: bool = False
    mode_routing_enabled: bool = False
    chat_search_enabled: bool = True
    planner_wait_max_seconds: int = 300
    planner_max_consecutive_waits: int = 2
    group_reply_necessity_threshold: int = 80
    group_reply_frequency: float = 1.0
    group_ambient_mode: str = "off"
    group_ambient_active_chats: tuple[str, ...] = ()
    group_ambient_idle_ms: int = 1000
    group_ambient_cooldown_seconds: float = 30.0
    group_ambient_quiet_cooldown_seconds: float = 10.0
    group_ambient_window_seconds: float = 300.0
    group_ambient_max_turns_per_window: int = 4
    group_ambient_max_age_seconds: float = 600.0
    group_ambient_min_messages: int = 2
    group_ambient_allow_single_question: bool = True
    group_ambient_allow_single_media: bool = False
    group_ambient_quote: bool = False
    group_ambient_stale_quote_seconds: float = 120.0
    group_proactive_mode: str = "off"
    group_proactive_active_chats: tuple[str, ...] = ()
    group_proactive_interval_seconds: int = 900
    group_proactive_cooldown_seconds: float = 900.0
    group_proactive_quiet_cooldown_seconds: float = 300.0
    group_proactive_window_seconds: float = 3600.0
    group_proactive_max_turns_per_window: int = 2
    group_proactive_reservation_seconds: float = 120.0
    group_proactive_active_hours_start: str = "09:00"
    group_proactive_active_hours_end: str = "23:00"
    group_proactive_timezone: str = "Asia/Shanghai"
    group_proactive_jitter_seconds: int = 0
    delivery_recovery_after_seconds: float = 60.0
    delivery_retry_base_seconds: float = 30.0
    delivery_retry_max_attempts: int = 5
    media_batch_max_resources: int = 8
    media_batch_max_chars: int = 12000
    media_batch_max_download_bytes: int = 100 * 1024 * 1024
    media_batch_capability_timeout_seconds: float = 120.0


_DEFAULTS = EngagementConfig()


def _number(
    raw: Mapping[str, Any],
    key: str,
    default: int | float,
    *,
    minimum: int | float,
    maximum: int | float,
    logger: logging.Logger,
    integer: bool = False,
) -> int | float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        logger.warning("[ai].%s 无效，使用默认值 %s", key, default)
        return default
    if value < minimum or value > maximum:
        logger.warning("[ai].%s 超出范围，使用默认值 %s", key, default)
        return default
    if integer and int(value) != value:
        logger.warning("[ai].%s 必须为整数，使用默认值 %s", key, default)
        return default
    return int(value) if integer else float(value)


def _enum(
    raw: Mapping[str, Any],
    key: str,
    default: str,
    choices: set[str],
    logger: logging.Logger,
) -> str:
    value = raw.get(key, default)
    if value not in choices:
        logger.warning("[ai].%s 无效，使用默认值 %s", key, default)
        return default
    return value


def _boolean(
    raw: Mapping[str, Any], key: str, default: bool, logger: logging.Logger
) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        logger.warning("[ai].%s 必须为布尔值，使用默认值 %s", key, default)
        return default
    return value


def _chat_allowlist(raw: Mapping[str, Any], logger: logging.Logger) -> tuple[str, ...]:
    value = raw.get("group_ambient_active_chats", ())
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple, set)) or any(
        not isinstance(chat_id, str) or not chat_id.strip() for chat_id in value
    ):
        logger.warning("[ai].group_ambient_active_chats 无效，使用空 allowlist")
        return ()
    return tuple(dict.fromkeys(chat_id.strip() for chat_id in value))


def _proactive_allowlist(raw: Mapping[str, Any], logger: logging.Logger) -> tuple[str, ...]:
    value = raw.get("group_proactive_active_chats", ())
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple, set)) or any(
        not isinstance(chat_id, str) or not chat_id.strip() for chat_id in value
    ):
        logger.warning("[ai].group_proactive_active_chats 无效，使用空 allowlist")
        return ()
    return tuple(dict.fromkeys(chat_id.strip() for chat_id in value))


def _clock_value(raw: Mapping[str, Any], key: str, default: str, logger: logging.Logger) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        logger.warning("[ai].%s 无效，使用默认值 %s", key, default)
        return default
    try:
        hour, minute = (int(part) for part in value.split(":"))
    except ValueError:
        logger.warning("[ai].%s 无效，使用默认值 %s", key, default)
        return default
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        logger.warning("[ai].%s 无效，使用默认值 %s", key, default)
        return default
    return value
def normalize_engagement_config(
    raw: Mapping[str, Any] | None,
    *,
    logger: logging.Logger = logging.getLogger(__name__),
) -> EngagementConfig:
    """Normalize all conversation/engagement settings at one boundary."""
    raw = raw or {}
    return EngagementConfig(
        conversation_collect_idle_ms=int(
            _number(
                raw,
                "conversation_collect_idle_ms",
                _DEFAULTS.conversation_collect_idle_ms,
                minimum=0,
                maximum=60000,
                logger=logger,
                integer=True,
            )
        ),
        conversation_collect_max_wait_ms=int(
            _number(
                raw,
                "conversation_collect_max_wait_ms",
                _DEFAULTS.conversation_collect_max_wait_ms,
                minimum=0,
                maximum=120000,
                logger=logger,
                integer=True,
            )
        ),
        conversation_collect_max_messages=int(
            _number(
                raw,
                "conversation_collect_max_messages",
                _DEFAULTS.conversation_collect_max_messages,
                minimum=1,
                maximum=100,
                logger=logger,
                integer=True,
            )
        ),
        conversation_collect_max_chars=int(
            _number(
                raw,
                "conversation_collect_max_chars",
                _DEFAULTS.conversation_collect_max_chars,
                minimum=1,
                maximum=1000000,
                logger=logger,
                integer=True,
            )
        ),
        private_conversation_delivery_mode=_enum(
            raw,
            "private_conversation_delivery_mode",
            "automatic",
            {"automatic", "message_tool_only"},
            logger,
        ),
        direct_task_delivery_mode=_enum(
            raw,
            "direct_task_delivery_mode",
            "automatic",
            {"automatic", "message_tool_only"},
            logger,
        ),
        group_ambient_delivery_mode=_enum(
            raw,
            "group_ambient_delivery_mode",
            "message_tool_only",
            {"automatic", "message_tool_only"},
            logger,
        ),
        direct_task_collaboration_enabled=_boolean(
            raw,
            "direct_task_collaboration_enabled",
            _DEFAULTS.direct_task_collaboration_enabled,
            logger,
        ),
        mode_routing_enabled=_boolean(
            raw, "mode_routing_enabled", _DEFAULTS.mode_routing_enabled, logger
        ),
        chat_search_enabled=_boolean(
            raw, "chat_search_enabled", _DEFAULTS.chat_search_enabled, logger
        ),
        planner_wait_max_seconds=int(
            _number(
                raw,
                "planner_wait_max_seconds",
                _DEFAULTS.planner_wait_max_seconds,
                minimum=1,
                maximum=300,
                logger=logger,
                integer=True,
            )
        ),
        planner_max_consecutive_waits=int(
            _number(
                raw,
                "planner_max_consecutive_waits",
                _DEFAULTS.planner_max_consecutive_waits,
                minimum=1,
                maximum=10,
                logger=logger,
                integer=True,
            )
        ),
        group_reply_necessity_threshold=int(
            _number(
                raw,
                "group_reply_necessity_threshold",
                _DEFAULTS.group_reply_necessity_threshold,
                minimum=0,
                maximum=100,
                logger=logger,
                integer=True,
            )
        ),
        group_reply_frequency=_number(
            raw,
            "group_reply_frequency",
            _DEFAULTS.group_reply_frequency,
            minimum=0,
            maximum=1,
            logger=logger,
        ),
        group_ambient_mode=_enum(
            raw, "group_ambient_mode", "off", {"off", "shadow", "active"}, logger
        ),
        group_ambient_active_chats=_chat_allowlist(raw, logger),
        group_ambient_idle_ms=int(
            _number(
                raw,
                "group_ambient_idle_ms",
                1000,
                minimum=0,
                maximum=120000,
                logger=logger,
                integer=True,
            )
        ),
        group_ambient_cooldown_seconds=_number(
            raw,
            "group_ambient_cooldown_seconds",
            30.0,
            minimum=0,
            maximum=86400,
            logger=logger,
        ),
        group_ambient_quiet_cooldown_seconds=_number(
            raw,
            "group_ambient_quiet_cooldown_seconds",
            10.0,
            minimum=0,
            maximum=86400,
            logger=logger,
        ),
        group_ambient_window_seconds=_number(
            raw,
            "group_ambient_window_seconds",
            300.0,
            minimum=1,
            maximum=604800,
            logger=logger,
        ),
        group_ambient_max_turns_per_window=int(
            _number(
                raw,
                "group_ambient_max_turns_per_window",
                4,
                minimum=1,
                maximum=1000,
                logger=logger,
                integer=True,
            )
        ),
        group_ambient_max_age_seconds=_number(
            raw,
            "group_ambient_max_age_seconds",
            600.0,
            minimum=1,
            maximum=604800,
            logger=logger,
        ),
        group_ambient_min_messages=int(
            _number(
                raw,
                "group_ambient_min_messages",
                2,
                minimum=1,
                maximum=100,
                logger=logger,
                integer=True,
            )
        ),
        group_ambient_allow_single_question=_boolean(
            raw, "group_ambient_allow_single_question", True, logger
        ),
        group_ambient_allow_single_media=_boolean(
            raw, "group_ambient_allow_single_media", False, logger
        ),
        group_ambient_quote=_boolean(raw, "group_ambient_quote", False, logger),
        group_ambient_stale_quote_seconds=_number(
            raw,
            "group_ambient_stale_quote_seconds",
            120.0,
            minimum=0,
            maximum=604800,
            logger=logger,
        ),
        group_proactive_mode=_enum(
            raw, "group_proactive_mode", "off", {"off", "shadow", "active"}, logger
        ),
        group_proactive_active_chats=_proactive_allowlist(raw, logger),
        group_proactive_interval_seconds=int(
            _number(
                raw,
                "group_proactive_interval_seconds",
                900,
                minimum=60,
                maximum=86400,
                logger=logger,
                integer=True,
            )
        ),
        group_proactive_cooldown_seconds=_number(
            raw,
            "group_proactive_cooldown_seconds",
            900.0,
            minimum=0,
            maximum=604800,
            logger=logger,
        ),
        group_proactive_quiet_cooldown_seconds=_number(
            raw,
            "group_proactive_quiet_cooldown_seconds",
            300.0,
            minimum=0,
            maximum=604800,
            logger=logger,
        ),
        group_proactive_window_seconds=_number(
            raw,
            "group_proactive_window_seconds",
            3600.0,
            minimum=1,
            maximum=604800,
            logger=logger,
        ),
        group_proactive_max_turns_per_window=int(
            _number(
                raw,
                "group_proactive_max_turns_per_window",
                2,
                minimum=1,
                maximum=1000,
                logger=logger,
                integer=True,
            )
        ),
        group_proactive_reservation_seconds=_number(
            raw,
            "group_proactive_reservation_seconds",
            120.0,
            minimum=1,
            maximum=3600,
            logger=logger,
        ),
        group_proactive_active_hours_start=_clock_value(
            raw, "group_proactive_active_hours_start", "09:00", logger
        ),
        group_proactive_active_hours_end=_clock_value(
            raw, "group_proactive_active_hours_end", "23:00", logger
        ),
        group_proactive_timezone=str(
            raw.get("group_proactive_timezone", "Asia/Shanghai")
            if isinstance(raw.get("group_proactive_timezone", "Asia/Shanghai"), str)
            else "Asia/Shanghai"
        ),
        group_proactive_jitter_seconds=int(
            _number(
                raw,
                "group_proactive_jitter_seconds",
                0,
                minimum=0,
                maximum=3600,
                logger=logger,
                integer=True,
            )
        ),
        delivery_recovery_after_seconds=_number(
            raw,
            "delivery_recovery_after_seconds",
            60.0,
            minimum=1,
            maximum=604800,
            logger=logger,
        ),
        delivery_retry_base_seconds=_number(
            raw,
            "delivery_retry_base_seconds",
            30.0,
            minimum=1,
            maximum=86400,
            logger=logger,
        ),
        delivery_retry_max_attempts=int(
            _number(
                raw,
                "delivery_retry_max_attempts",
                5,
                minimum=1,
                maximum=100,
                logger=logger,
                integer=True,
            )
        ),
        media_batch_max_resources=int(
            _number(
                raw,
                "media_batch_max_resources",
                8,
                minimum=1,
                maximum=100,
                logger=logger,
                integer=True,
            )
        ),
        media_batch_max_chars=int(
            _number(
                raw,
                "media_batch_max_chars",
                12000,
                minimum=1,
                maximum=1_000_000,
                logger=logger,
                integer=True,
            )
        ),
        media_batch_max_download_bytes=int(
            _number(
                raw,
                "media_batch_max_download_bytes",
                100 * 1024 * 1024,
                minimum=1,
                maximum=2 * 1024 * 1024 * 1024,
                logger=logger,
                integer=True,
            )
        ),
        media_batch_capability_timeout_seconds=_number(
            raw,
            "media_batch_capability_timeout_seconds",
            120.0,
            minimum=0.1,
            maximum=1800,
            logger=logger,
        ),
    )
