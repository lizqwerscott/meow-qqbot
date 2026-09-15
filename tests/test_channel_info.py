import pytest

from core.engine.dynamic_context.social import SocialBlockBuilder
from core.message import InputMessage
from core.session_identity import DeliveryTarget
from core.session_identity.channel_info import (
    ChannelActorInfo,
    ChannelChatInfo,
    ChannelInfoSnapshot,
    QQChannelInfoProvider,
)


class FakeApi:
    def __init__(self):
        self.calls = []

    async def request(self, method, path, timeout=0):
        self.calls.append(path)
        if path == "/users/@me":
            return {"id": "bot", "display_name": "猫猫", "username": "cat"}
        if path.endswith("/info"):
            return {
                "group_openid": "group-1",
                "group_name": "测试群",
                "member_count": 4,
            }
        return {"member_openid": "bot", "role": "admin", "can_send": True}


@pytest.mark.asyncio
async def test_qq_channel_info_provider_persists_and_hits_cache(tmp_path):
    api = FakeApi()
    provider = QQChannelInfoProvider(api, db_path=tmp_path / "channel.sqlite3")
    target = DeliveryTarget("qq", "default", "group", "group-1")

    first = await provider.get_info(target)
    second = await provider.get_info(target)

    assert first.availability == "complete"
    assert first.channel == "qq"
    assert first.channel_name == "QQ"
    assert first.prompt_summary()["bot"] == {
        "display_name": "猫猫",
        "username": "cat",
        "is_bot": True,
    }
    assert first.chat_info.title == "测试群"
    assert second.availability == "complete"
    assert second.chat_info.title == "测试群"
    assert api.calls == [
        "/users/@me",
        "/v2/groups/group-1/info",
        "/v2/groups/group-1/bot_state",
    ]
    statuses = provider.cache_status()
    assert {item["scope"] for item in statuses} == {
        "account",
        "chat",
        "membership",
    }
    assert all(
        len(item["target_fingerprint"]) == 10 or item["scope"] == "account"
        for item in statuses
    )


@pytest.mark.asyncio
async def test_qq_channel_info_provider_subject_is_capability_unavailable():
    api = FakeApi()
    provider = QQChannelInfoProvider(api, db_path=":memory:")
    snapshot = await provider.get_info(
        DeliveryTarget("qq", "default", "group", "group-1"),
        subject_id="other",
    )

    assert "capability_unavailable" in snapshot.unavailable_reasons


@pytest.mark.asyncio
async def test_qq_channel_info_provider_cools_down_failed_component(tmp_path):
    class FailingApi:
        calls = 0

        async def request(self, method, path, timeout=0):
            self.calls += 1
            raise RuntimeError("temporary failure")

    api = FailingApi()
    provider = QQChannelInfoProvider(api, db_path=tmp_path / "channel.sqlite3")
    first = await provider.get_info()
    second = await provider.get_info()

    assert first.availability == "unavailable"
    assert second.availability == "unavailable"
    assert api.calls == 1


def test_channel_prompt_summary_contains_channel_and_bot_names():
    snapshot = ChannelInfoSnapshot(
        channel="qq",
        channel_name="QQ",
        self_info=ChannelActorInfo(
            actor_id="hidden", display_name="猫猫", username="cat", is_bot=True
        ),
        chat_info=ChannelChatInfo(chat_id="hidden", chat_type="group", title="测试群"),
        availability="complete",
    )

    summary = snapshot.prompt_summary(include_runtime_status=False)
    text = SocialBlockBuilder._format_summary(summary)

    assert "渠道名称：QQ" in text
    assert "机器人名称：猫猫" in text
    assert "机器人用户名：cat" in text
    assert "聊天名称：测试群" in text
    assert "hidden" not in text


@pytest.mark.asyncio
async def test_direct_qq_prompt_includes_channel_info():
    snapshot = ChannelInfoSnapshot(
        channel="qq",
        channel_name="QQ",
        self_info=ChannelActorInfo(actor_id="hidden", display_name="猫猫"),
        availability="complete",
    )
    message = InputMessage(
        id="m1",
        sender_id="user",
        chat_id="user",
        content="你好",
        is_group=False,
        delivery_target=DeliveryTarget("qq", "default", "direct", "user"),
        channel_info=snapshot,
    )

    block = await SocialBlockBuilder("bot").build_stable(
        chat_id="user",
        is_group=False,
        has_users=False,
        input_message=message,
    )

    assert "渠道名称：QQ" in block
    assert "机器人名称：猫猫" in block
