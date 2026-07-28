import pytest

from core.managers.nickname_manager import NicknameManager


@pytest.fixture
def nm():
    return NicknameManager(bot_id="bot_001")


@pytest.mark.asyncio
async def test_collect_new_user(nm):
    await nm.collect("user_001", "Alice")
    assert "user_001" in nm.auto_nicknames
    assert nm.auto_nicknames["user_001"]["aliases"] == ["Alice"]


@pytest.mark.asyncio
async def test_collect_duplicate(nm):
    await nm.collect("user_001", "Alice")
    await nm.collect("user_001", "Alice")
    assert nm.auto_nicknames["user_001"]["aliases"] == ["Alice"]


@pytest.mark.asyncio
async def test_collect_new_alias_moves_to_end(nm):
    await nm.collect("user_001", "Alice")
    await nm.collect("user_001", "Bob")
    assert nm.auto_nicknames["user_001"]["aliases"] == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_collect_existing_alias_moves_to_end(nm):
    await nm.collect("user_001", "Alice")
    await nm.collect("user_001", "Bob")
    await nm.collect("user_001", "Alice")
    assert nm.auto_nicknames["user_001"]["aliases"] == ["Bob", "Alice"]


@pytest.mark.asyncio
async def test_collect_ignores_bot_id(nm):
    await nm.collect("bot_001", "Bot")
    assert "bot_001" not in nm.auto_nicknames


@pytest.mark.asyncio
async def test_collect_ignores_empty(nm):
    await nm.collect("", "")
    assert len(nm.auto_nicknames) == 0


def test_get_manual_first(nm):
    nm.nicknames["user_001"] = "手动名"
    nm.auto_nicknames["user_001"] = {"aliases": ["自动名"], "updated_at": 0}
    assert nm.get("user_001") == "手动名"


def test_get_auto_fallback(nm):
    nm.auto_nicknames["user_001"] = {"aliases": ["自动名"], "updated_at": 0}
    assert nm.get("user_001") == "自动名"


def test_get_fallback_to_id(nm):
    assert nm.get("unknown") == "unknown"


def test_get_aliases_manual_only(nm):
    nm.nicknames["user_001"] = "手动名"
    assert nm.get_aliases("user_001") == ["手动名"]


def test_get_aliases_merged(nm):
    nm.nicknames["user_001"] = "手动名"
    nm.auto_nicknames["user_001"] = {"aliases": ["自动名1", "自动名2"], "updated_at": 0}
    assert nm.get_aliases("user_001") == ["手动名", "自动名1", "自动名2"]


def test_flush_save_no_error(nm):
    import asyncio
    asyncio.run(nm.flush_save())
