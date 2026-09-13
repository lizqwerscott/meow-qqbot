from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.managers.identity_manager import IdentityManager
from core.session_identity import DeliveryTarget
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.memory import create_memory_entries


def _context(target: DeliveryTarget) -> ToolContext:
    return ToolContext(
        chat_id=target.target_id,
        is_group=target.chat_type == "group",
        reply_to="",
        sender_id="actor-1",
        reply_callback=AsyncMock(),
        delivery_target=target,
    )


def _entry(deps: ToolDeps, name: str):
    return next(entry for entry in create_memory_entries(deps) if entry.name == name)


@pytest.fixture
def memory_setup(tmp_path):
    target = DeliveryTarget("qq", "default", "group", "g1")
    identities = IdentityManager(tmp_path / "identity.sqlite3")
    sender = identities.observe(target, "actor-1", "甲")
    mentioned = identities.observe(target, "actor-2", "乙")
    result = {
        "profiles": [{"profile_data": {"fact": "@actor-2 喜欢咖啡"}}],
        "episodes": [{"summary": "actor-1 和（actor-2）一起讨论过项目"}],
    }
    hindsight = SimpleNamespace(
        search=AsyncMock(return_value=result),
        search_shared=AsyncMock(return_value=result),
        add_message=AsyncMock(return_value=True),
    )
    deps = ToolDeps(identity_manager=identities, hindsight=hindsight)
    return target, sender, mentioned, deps


@pytest.mark.asyncio
async def test_memory_search_projects_profiles_and_episodes(memory_setup):
    target, sender, mentioned, deps = memory_setup

    result = await _entry(deps, "memory").handler(
        {"action": "search", "query": "项目"}, _context(target)
    )

    assert "actor-1" not in result.content
    assert "actor-2" not in result.content
    assert sender.identity_ref in result.content
    assert mentioned.identity_ref in result.content


@pytest.mark.asyncio
async def test_shared_and_relation_memory_project_results(memory_setup):
    target, sender, mentioned, deps = memory_setup
    context = _context(target)
    memory = _entry(deps, "memory")

    shared = await memory.handler({"action": "search_shared", "query": "项目"}, context)
    relation = await memory.handler(
        {
            "action": "relation",
            "person_a": sender.identity_ref,
            "person_b": mentioned.identity_ref,
        },
        context,
    )

    assert "actor-1" not in shared.content
    assert "actor-2" not in shared.content
    assert "actor-1" not in relation.content
    assert "actor-2" not in relation.content


@pytest.mark.asyncio
async def test_mark_important_persists_projected_identity_refs(memory_setup):
    target, sender, mentioned, deps = memory_setup

    result = await _entry(deps, "mark_important").handler(
        {
            "profile_data": '{"note":"@actor-2 是同事"}',
            "summary": "actor-1 和（actor-2）完成了任务",
        },
        _context(target),
    )

    assert "success" in result.content
    stored = [
        call.kwargs["content"] for call in deps.hindsight.add_message.call_args_list
    ]
    assert all(
        "actor-1" not in content and "actor-2" not in content for content in stored
    )
    assert sender.identity_ref in stored[0]
    assert mentioned.identity_ref in stored[0]
    assert mentioned.identity_ref in stored[1]
