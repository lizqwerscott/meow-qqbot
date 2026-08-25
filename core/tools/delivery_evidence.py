"""Classify completed tool deliveries for one reply segment."""

from dataclasses import dataclass, field

from core.engine.delivery_ledger import DeliveryReceipt

_TEXT_DELIVERY_KIND = "message"
_COMPLETED_STATUSES = frozenset({"accepted", "partial"})


@dataclass(frozen=True)
class _DeliveryRecord:
    kind: str
    target_chat_id: str
    receipt: DeliveryReceipt


@dataclass
class DeliveryEvidence:
    """Remember tool deliveries and answer whether text already reached a target."""

    _records: list[_DeliveryRecord] = field(default_factory=list)

    def record(
        self,
        *,
        kind: str,
        target_chat_id: str,
        receipt: DeliveryReceipt | None,
    ) -> None:
        if not kind or not target_chat_id or receipt is None:
            return
        self._records.append(
            _DeliveryRecord(
                kind=kind,
                target_chat_id=target_chat_id,
                receipt=receipt,
            )
        )

    def has_completed_text_reply(self, target_chat_id: str) -> bool:
        return any(
            record.kind == _TEXT_DELIVERY_KIND
            and record.target_chat_id == target_chat_id
            and record.receipt.status in _COMPLETED_STATUSES
            for record in self._records
        )

    def reset(self) -> None:
        self._records.clear()
