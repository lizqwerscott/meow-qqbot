from types import SimpleNamespace

import pytest

from core.engine.dynamic_context.memory import MemoryBlockBuilder
from core.managers.identity_manager import IdentityManager
from core.message import InputMessage
from core.session_identity import DeliveryTarget


@pytest.mark.asyncio
async def test_memory_prompt_projects_legacy_platform_mentions(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    ref = manager.observe(target, "actor-1", "小明")
    hindsight = SimpleNamespace(
        search=lambda **kwargs: None,
    )

    async def search(**kwargs):
        return {"episodes": [{"summary": "@actor-1 之前提到过这个"}], "profiles": []}

    hindsight.search = search
    builder = MemoryBlockBuilder(hindsight, 3, None, None, manager)
    message = InputMessage(
        id="m1",
        sender_id="actor-1",
        chat_id="g1",
        content="这个是什么",
        is_group=True,
        delivery_target=target,
    )

    text = await builder.build_memory_context("actor-1", message)

    assert f"@{ref.identity_ref}" in text
    assert "@actor-1" not in text
