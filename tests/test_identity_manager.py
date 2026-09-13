from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.engine.dynamic_context.social import SocialBlockBuilder
from core.managers.identity_manager import IdentityManager
from core.message import InputMessage
from core.session_identity import DeliveryTarget


def test_identity_ref_is_stable_and_chat_aliases_are_isolated(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    first_chat = DeliveryTarget("qq", "default", "group", "g1")
    second_chat = DeliveryTarget("qq", "default", "group", "g2")

    first = manager.observe(first_chat, "actor-1", "小明")
    manager.observe(second_chat, "actor-1", "明明")

    assert manager.get_ref(first_chat, "actor-1").identity_ref == first.identity_ref
    assert manager.get_ref(first_chat, "actor-1").chat_name == "小明"
    assert manager.get_ref(second_chat, "actor-1").chat_name == "明明"
    assert manager.list_members()[0]["identity_ref"] == first.identity_ref


def test_project_recent_sorts_by_activity_and_keeps_forced_users(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    manager.observe(target, "actor-1", "甲")
    manager.observe(target, "actor-2", "乙")
    message = InputMessage(
        id="m",
        sender_id="actor-1",
        chat_id="g1",
        content="hello",
        is_group=True,
        delivery_target=target,
        mentioned_ids=["actor-1"],
    )
    events = [
        SimpleNamespace(role="user", sender_id="actor-2", timestamp=1e20),
        SimpleNamespace(role="user", sender_id="actor-2", timestamp=1e20),
        SimpleNamespace(role="user", sender_id="actor-2", timestamp=1e20),
    ]

    block = __import__("asyncio").run(
        SocialBlockBuilder(None, "bot", manager).build(
            chat_id="g1",
            is_group=True,
            has_users=True,
            input_message=message,
            recent_events=events,
            max_users=2,
        )
    )

    assert "identity_ref:" in block
    assert "actor-1" not in block
    assert "actor-2" not in block


def test_project_text_replaces_only_current_chat_mentions(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    ref = manager.observe(target, "actor-1", "甲")

    assert manager.project_text(target, "@actor-1 你好") == f"@{ref.identity_ref} 你好"
    assert manager.resolve_identity(target, ref.identity_ref) == "actor-1"


def test_project_text_does_not_replace_actor_id_prefixes(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    ref = manager.observe(target, "actor-1", "甲")

    assert manager.project_text(target, "@actor-10 @actor-1") == (
        f"@actor-10 @{ref.identity_ref}"
    )


def test_chat_type_is_part_of_membership_and_alias_scope(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    group = DeliveryTarget("qq", "default", "group", "same-id")
    direct = DeliveryTarget("qq", "default", "direct", "same-id")

    manager.observe(group, "actor-1", "群名")
    manager.observe(direct, "actor-1", "私聊名")

    assert manager.get_ref(group, "actor-1").chat_name == "群名"
    assert manager.get_ref(direct, "actor-1").chat_name == "私聊名"
    assert manager.resolve_identity(
        group, manager.get_ref(group, "actor-1").identity_ref
    )
    assert len(manager._conn.execute("SELECT * FROM chat_memberships").fetchall()) == 2


def test_observe_legacy_history_backfills_members_and_names(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")

    observed = manager.observe_legacy_history(
        target,
        [
            {
                "role": "user",
                "sender_id": "actor-1",
                "name": "历史名",
                "timestamp": 100,
            },
            {"role": "assistant", "content": "reply"},
        ],
    )

    ref = manager.get_ref(target, "actor-1")
    assert observed == 1
    assert ref is not None
    assert ref.chat_name == "历史名"
    assert manager.list_chats()[0]["chat_id"] == "g1"
    assert (
        manager.observe_legacy_history(
            target, [{"role": "user", "sender_id": "actor-1", "name": "历史名"}]
        )
        == 0
    )
    membership = manager._conn.execute(
        "SELECT message_count FROM chat_memberships WHERE actor_id = ?", ("actor-1",)
    ).fetchone()
    assert membership["message_count"] == 1


def test_old_membership_table_is_rebuilt_with_chat_type(tmp_path):
    import sqlite3

    database = tmp_path / "identity.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("""
        CREATE TABLE chat_memberships (
            channel TEXT NOT NULL,
            account_id TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            mention_count INTEGER NOT NULL DEFAULT 0,
            reply_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(channel, account_id, chat_id, actor_id)
        )
        """)
    connection.execute(
        "INSERT INTO chat_memberships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("qq", "default", "g1", "actor-1", 1, 1, 1, 0, 0),
    )
    connection.commit()
    connection.close()

    manager = IdentityManager(database)
    columns = manager._conn.execute("PRAGMA table_info(chat_memberships)").fetchall()
    primary_key = [
        row["name"] for row in sorted(columns, key=lambda row: row["pk"]) if row["pk"]
    ]

    assert primary_key == ["channel", "account_id", "chat_id", "chat_type", "actor_id"]
    manager.observe(DeliveryTarget("qq", "default", "direct", "g1"), "actor-1")
    assert len(manager._conn.execute("SELECT * FROM chat_memberships").fetchall()) == 2


def test_outbound_identity_mention_resolves_in_current_chat(tmp_path):
    from core.engine.client import BotEngine

    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    ref = manager.observe(target, "actor-1", "甲")
    engine = BotEngine.__new__(BotEngine)
    engine.identity_manager = manager

    content = engine._project_outbound_mentions(
        target, f'<qqbot-at-user id="{ref.identity_ref}" /> 你好'
    )

    assert content == "<@actor-1> 你好"


def test_hindsight_content_uses_anonymous_identity_refs():
    from core.engine.agent_engine import AgentEngine

    content = AgentEngine._format_hindsight_content(
        "@actor-2 你好",
        "actor-1",
        ["actor-2"],
        identity_ref="member_sender",
        mentioned_identity_refs={"actor-2": "member_mentioned"},
    )

    assert content == "[member_sender]: @member_mentioned 你好"
    assert "actor-1" not in content
    assert "actor-2" not in content


@pytest.mark.asyncio
async def test_live_admission_projects_sender_reply_and_mentions(tmp_path):
    from core.engine.agent_engine import AgentEngine

    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    sender = manager.observe(target, "actor-1", "甲")
    mentioned = manager.observe(target, "actor-2", "乙")
    replied = manager.observe(target, "actor-3", "丙")
    engine = AgentEngine.__new__(AgentEngine)
    engine._identity_manager = manager
    message = InputMessage(
        id="m1",
        sender_id="actor-1",
        chat_id="g1",
        content="@actor-2 你好",
        is_group=True,
        delivery_target=target,
        mentioned_ids=["actor-2"],
        replied_content="旧消息",
        replied_author="actor-3",
        replied_author_id="actor-3",
        identity_ref=sender.identity_ref,
        mentioned_identity_refs={"actor-2": mentioned.identity_ref},
        replied_identity_ref=replied.identity_ref,
    )

    pending = await engine._prepare_pending_inbound(
        message,
        intent=None,
    )

    assert pending.prepared_content == (
        f"[正在回复 {replied.identity_ref}: 旧消息]\n" f"@{mentioned.identity_ref} 你好"
    )
    assert (
        engine._anonymous_actor_ref(
            message, "actor-1", fallback="actor-1", known_ref=message.identity_ref
        )
        == sender.identity_ref
    )


@pytest.mark.asyncio
async def test_hindsight_side_effect_persists_anonymous_content(tmp_path):
    from core.engine.agent_engine import AgentEngine

    manager = IdentityManager(tmp_path / "identity.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "g1")
    sender = manager.observe(target, "actor-1", "甲")
    mentioned = manager.observe(target, "actor-2", "乙")
    add_message = AsyncMock(return_value=True)
    engine = AgentEngine.__new__(AgentEngine)
    engine._identity_manager = manager
    engine._nm = None
    engine.session_identity_resolver = None
    engine.hindsight = SimpleNamespace(
        add_message=add_message,
        msg_type_to_context=lambda _: None,
    )

    assert await engine._run_hindsight_side_effect(
        {
            "chat_id": "g1",
            "content": "@actor-2 你好",
            "sender_id": "actor-1",
            "identity_ref": sender.identity_ref,
            "mentioned_ids": ["actor-2"],
            "mentioned_identity_refs": {"actor-2": mentioned.identity_ref},
            "delivery_target": list(target.catalog_key),
            "msg_type": "text",
        }
    )

    persisted_content = add_message.call_args.kwargs["content"]
    assert (
        persisted_content == f"[{sender.identity_ref}]: @{mentioned.identity_ref} 你好"
    )
    assert "actor-1" not in persisted_content
    assert "actor-2" not in persisted_content
    assert add_message.call_args.kwargs["sender_id"] == "actor-1"


def test_legacy_nickname_migration_is_idempotent_and_does_not_create_chat(tmp_path):
    import json

    manual_path = tmp_path / "config" / "nicknames.json"
    auto_path = tmp_path / "data" / "nicknames.json"
    manual_path.parent.mkdir()
    auto_path.parent.mkdir()
    manual_path.write_text(json.dumps({"actor-1": "手动名"}), encoding="utf-8")
    auto_path.write_text(
        json.dumps({"actor-1": {"aliases": ["旧名", "新名"]}}), encoding="utf-8"
    )

    manager = IdentityManager(tmp_path / "identity.sqlite3")
    first = manager.migrate_legacy_nicknames(
        manual_path=manual_path, auto_path=auto_path
    )
    second = manager.migrate_legacy_nicknames(
        manual_path=manual_path, auto_path=auto_path
    )
    assert manager.list_chats() == []
    target = DeliveryTarget("qq", "default", "group", "g1")
    manager.observe(target, "actor-1")
    ref = manager.get_ref(target, "actor-1")

    assert first == {"manual": 1, "auto": 1}
    assert second == {"manual": 1, "auto": 1}
    assert ref is not None
    assert ref.historical_names[0] == "手动名"
    assert {"手动名", "旧名", "新名"}.issubset(ref.historical_names)
    assert not manager.list_chats() == []
    assert manual_path.with_name("nicknames.json.legacy.bak").exists()


def test_identity_suggestion_requires_confirmation(tmp_path):
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    first_chat = DeliveryTarget("qq", "default", "group", "g1")
    second_chat = DeliveryTarget("telegram", "default", "group", "g2")
    first = manager.observe(first_chat, "qq-user", "小明")
    second = manager.observe(second_chat, "telegram-user", "Ming")
    manager.observe(second_chat, "telegram-user-2", "小明")
    generated = manager.suggest_matching_identities()
    suggestion_id = manager.suggest_link(
        first.identity_ref, second.identity_ref, "same avatar"
    )

    assert generated == 1
    assert manager.get_ref(first_chat, "qq-user").person_ref == ""
    manager.resolve_suggestion(suggestion_id, accepted=True, decided_by="admin")
    assert manager.get_ref(first_chat, "qq-user").person_ref
    assert (
        manager.get_ref(second_chat, "telegram-user").person_ref
        == manager.get_ref(first_chat, "qq-user").person_ref
    )


def test_runtime_account_health_callback_triggers_once_after_three_failures():
    from types import SimpleNamespace

    from core.engine.client import BotEngine

    calls = []
    engine = BotEngine.__new__(BotEngine)
    engine._consecutive_account_info_failures = 0
    engine._runtime_shutdown_started = False
    engine._runtime_shutdown_callback = lambda: calls.append("stop")
    failed = SimpleNamespace(self_info=None)

    for _ in range(3):
        engine._record_account_info_health(failed)
    engine._record_account_info_health(failed)

    assert calls == ["stop"]


@pytest.mark.asyncio
async def test_bot_engine_inbound_path_mounts_identity_and_channel_info(tmp_path):
    from types import SimpleNamespace

    from core.engine.client import BotEngine
    from core.session_identity.channel_info import ChannelInfoSnapshot

    target = DeliveryTarget("qq", "default", "group", "g1")
    manager = IdentityManager(tmp_path / "identity.sqlite3")
    routed = []

    async def route(**kwargs):
        routed.append(kwargs["input_message"])

    class Adapter:
        async def get_info(self, target):
            return ChannelInfoSnapshot(availability="complete")

    class Registry:
        def for_target(self, value):
            assert value == target
            return Adapter()

    parsed = SimpleNamespace(
        id="m1",
        sender_id="actor-1",
        author_id="actor-1",
        author_username="小明",
        chat_id="g1",
        chat_scope="group",
        content="hello",
        msg_type="text",
        resources=[],
        mentioned_ids=["actor-2"],
        is_at_mention=False,
        mention_entries=[("actor-2", "小红")],
        reply_author_entries=[],
        replied_author_id="",
        replied_content="",
        replied_author="",
        replied_message_id="",
        task_correlation_id="",
        replied_resources=[],
    )

    async def parse(*_args):
        return parsed

    engine = BotEngine.__new__(BotEngine)
    engine.parser = SimpleNamespace(parse=parse)
    engine.nickname_manager = SimpleNamespace(get=lambda user_id: user_id)
    engine.identity_manager = manager
    engine.channel_registry = Registry()
    engine.agent_engine = SimpleNamespace(resolve_message_identity=lambda message: None)
    engine.media_service = None
    engine.router = SimpleNamespace(route=route)
    engine._bot_id = "bot"
    engine._consecutive_account_info_failures = 0
    engine._runtime_shutdown_started = False
    engine._runtime_shutdown_callback = None

    await engine._on_message_event("GROUP_AT_MESSAGE_CREATE", {})

    assert len(routed) == 1
    assert routed[0].delivery_target == target
    assert routed[0].identity_ref.startswith("member_")
    assert routed[0].mentioned_identity_refs["actor-2"].startswith("member_")
    assert routed[0].channel_info.availability == "complete"
