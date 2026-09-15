import json
from unittest.mock import AsyncMock

import pytest

from core.managers.identity_manager import IdentityManager
from core.session_identity import DeliveryTarget
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.user import create_user_entries


@pytest.mark.asyncio
async def test_search_user_returns_identity_refs_without_platform_ids(tmp_path):
    target = DeliveryTarget("qq", "default", "group", "g1")
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    ref = manager.observe(target, "actor-1", "小明")
    deps = ToolDeps(identity_manager=manager)
    context = ToolContext(
        chat_id="g1",
        is_group=True,
        reply_to="",
        sender_id="actor-1",
        reply_callback=AsyncMock(),
        delivery_target=target,
    )

    result = await create_user_entries(deps)[0].handler({"query": "小明"}, context)

    assert json.loads(result.content)[0]["identity_ref"] == ref.identity_ref
    assert "actor-1" not in result.content


@pytest.mark.asyncio
async def test_search_user_does_not_fallback_to_raw_ids(tmp_path):
    target = DeliveryTarget("qq", "default", "direct", "u1")
    deps = ToolDeps()
    context = ToolContext(
        chat_id="u1",
        is_group=False,
        reply_to="",
        sender_id="actor-1",
        reply_callback=AsyncMock(),
        delivery_target=target,
    )

    result = await create_user_entries(deps)[0].handler({"query": "小明"}, context)

    assert "actor-1" not in result.content
    assert "身份管理器未就绪" in result.content
