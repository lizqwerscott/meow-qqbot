"""Normalized configuration for conversation collection and engagement."""

import logging
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

GROUP_REPLY_FREQUENCY_MIN = 0.125


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
    group_reply_trigger_mode: str = "frequency"
    group_reply_necessity_threshold: int = 80
    group_reply_frequency: float = 0.25
    group_reply_chat_overrides: tuple[tuple[str, "GroupReplySettings"], ...] = ()
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
    delivery_recovery_after_seconds: float = 60.0
    delivery_retry_base_seconds: float = 30.0
    delivery_retry_max_attempts: int = 5
    media_batch_max_resources: int = 8
    media_batch_max_chars: int = 12000
    media_batch_max_download_bytes: int = 100 * 1024 * 1024
    media_batch_capability_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class GroupReplySettings:
    trigger_mode: str = "frequency"
    frequency: float = 0.25


_DEFAULTS = EngagementConfig()


def get_group_reply_settings(
    config: EngagementConfig, chat_id: str
) -> GroupReplySettings:
    """Return the per-group reply policy, falling back to global defaults."""
    for configured_chat_id, settings in config.group_reply_chat_overrides:
        if configured_chat_id == chat_id:
            return settings
    return GroupReplySettings(
        trigger_mode=config.group_reply_trigger_mode,
        frequency=config.group_reply_frequency,
    )


def _valid_reply_frequency(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and 0 <= value <= 1
        and (value == 0 or value >= GROUP_REPLY_FREQUENCY_MIN)
    )


def _group_reply_overrides(
    raw: Mapping[str, Any],
    logger: logging.Logger,
) -> tuple[tuple[str, GroupReplySettings], ...]:
    value = raw.get("group_reply_chat_overrides", {})
    if value in (None, {}):
        return ()
    if not isinstance(value, Mapping):
        logger.warning("[ai].group_reply_chat_overrides 无效，使用空的群聊覆盖配置")
        return ()

    result: list[tuple[str, GroupReplySettings]] = []
    for chat_id, settings in value.items():
        if (
            not isinstance(chat_id, str)
            or not chat_id.strip()
            or not isinstance(settings, Mapping)
        ):
            logger.warning("[ai].group_reply_chat_overrides 包含无效项，跳过该群聊配置")
            continue
        trigger_mode = settings.get("trigger_mode", _DEFAULTS.group_reply_trigger_mode)
        if trigger_mode not in {"frequency", "reply_necessity"}:
            logger.warning(
                "[ai].group_reply_chat_overrides[%s].trigger_mode 无效，使用 frequency",
                chat_id,
            )
            trigger_mode = "frequency"
        frequency = settings.get("frequency", _DEFAULTS.group_reply_frequency)
        if not _valid_reply_frequency(frequency):
            logger.warning(
                "[ai].group_reply_chat_overrides[%s].frequency 无效，使用 %.2f",
                chat_id,
                _DEFAULTS.group_reply_frequency,
            )
            frequency = _DEFAULTS.group_reply_frequency
        result.append(
            (
                chat_id.strip(),
                GroupReplySettings(trigger_mode, float(frequency)),
            )
        )
    return tuple(result)


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


def _reply_frequency(
    raw: Mapping[str, Any],
    key: str,
    default: float,
    logger: logging.Logger,
) -> float:
    value = raw.get(key, default)
    if not _valid_reply_frequency(value):
        logger.warning("[ai].%s 无效，使用默认值 %s", key, default)
        return default
    return float(value)


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
        group_reply_trigger_mode=_enum(
            raw,
            "group_reply_trigger_mode",
            _DEFAULTS.group_reply_trigger_mode,
            {"frequency", "reply_necessity"},
            logger,
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
        group_reply_frequency=_reply_frequency(
            raw,
            "group_reply_frequency",
            _DEFAULTS.group_reply_frequency,
            logger,
        ),
        group_reply_chat_overrides=_group_reply_overrides(raw, logger),
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
