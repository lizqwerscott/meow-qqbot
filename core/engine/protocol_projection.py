"""Provider protocol view derived from the conversation event ledger."""

from typing import Any

from core.engine.conversation_event_log import ConversationEvent, ConversationEventLog


class ProtocolProjection:
    """Expose strict assistant/tool wire history without another fact store."""

    def __init__(self, event_log: ConversationEventLog):
        self._event_log = event_log

    async def snapshot(
        self, turn_id: str, *, chat_id: str | None = None
    ) -> tuple[ConversationEvent, ...]:
        return await self._event_log.protocol_snapshot(turn_id, chat_id=chat_id)

    async def snapshot_wire(
        self, turn_id: str, *, chat_id: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._event_log.protocol_wire(turn_id, chat_id=chat_id)

    @staticmethod
    def to_wire_messages(
        events: tuple[ConversationEvent, ...] | list[ConversationEvent],
    ) -> list[dict[str, Any]]:
        return [event.to_wire() for event in events]
