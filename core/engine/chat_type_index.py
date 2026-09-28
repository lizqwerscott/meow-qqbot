"""Chat type lookup backed by the delivery target catalog.

The delivery target catalog is written on every inbound message with the real
``DeliveryTarget`` (channel/account/chat_type/target_id), so a raw target id is
the authoritative group/private source.  Canonical session keys never match a raw
target id, so those are resolved through the identity resolver when available.

This module replaces the retired ``ChatContext`` active store, which used to
persist ``context_chat_types`` in ``conversation_context.sqlite3``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

_log = logging.getLogger(__name__)


class ChatTypeIndex:
    """Resolve whether a session is a group or a private chat."""

    def __init__(self, catalog: Any = None, resolver: Any = None) -> None:
        self._catalog = catalog
        self._resolver = resolver

    def observe(self, target: Any) -> None:
        """Record a delivery target's chat type (idempotent)."""
        if target is None or self._catalog is None:
            return
        try:
            self._catalog.observe(target, source="inbound")
        except Exception as exc:
            target_id = str(getattr(target, "target_id", ""))
            _log.warning("登记会话类型失败 [%s..]: %s", target_id[:12], exc)

    def get(self, key: str) -> Optional[bool]:
        """Return ``True`` for groups, ``False`` for private, ``None`` if unknown."""
        if not key:
            return None
        catalog_type = self._from_catalog(key)
        if catalog_type is not None:
            return catalog_type == "group"
        resolved_type = self._from_resolver(key)
        if resolved_type is not None:
            return resolved_type == "group"
        return None

    def _from_catalog(self, target_id: str) -> Optional[str]:
        if self._catalog is None:
            return None
        try:
            return self._catalog.get_chat_type(target_id)
        except Exception:
            return None

    def _from_resolver(self, key: str) -> Optional[str]:
        if self._resolver is None:
            return None
        try:
            ref = self._resolver.resolve_legacy(key)
        except Exception:
            return None
        target = getattr(ref, "target", None)
        chat_type = getattr(target, "chat_type", None) if target is not None else None
        return str(chat_type) if chat_type else None
