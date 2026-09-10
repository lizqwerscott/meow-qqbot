from types import SimpleNamespace

import pytest

from core.command_handlers.cost import CostCommand
from core.message import InputMessage
from core.session_identity import (
    DeliveryTarget,
    SessionIdentityRegistry,
    SessionIdentityResolver,
)


@pytest.mark.asyncio
async def test_cost_lookup_resolves_legacy_session_alias(tmp_path):
    registry = SessionIdentityRegistry(tmp_path / "identity.sqlite3")
    registry.register_chat(
        DeliveryTarget("qq", "default", "group", "123"),
        legacy_key="123",
        state="verified",
        hindsight_document_id="session-123",
    )
    resolver = SessionIdentityResolver(registry, canonical_enabled=True)
    canonical_key = "agent:main:qq:default:group:123"
    stats = SimpleNamespace(
        turn_count=1,
        prompt_tokens=10,
        cache_hit_tokens=0,
        cache_miss_tokens=10,
        cache_hit_rate=0.0,
        completion_tokens=5,
        cost=0.01,
        total_tokens=15,
    )
    tracker = SimpleNamespace(
        get_global_stats=lambda: stats,
        get_session_stats=lambda key: stats if key == canonical_key else None,
        get_all_sessions=lambda: {canonical_key: stats},
    )
    command = CostCommand(
        SimpleNamespace(
            cost_tracker=tracker,
            session_identity_resolver=resolver,
        )
    )

    replies = await command.execute(InputMessage("m1", "admin", "123", "", True), "123")

    assert "→ `agent:main:qq:default:group:123`" in replies[0]["content"]
    registry.close()
