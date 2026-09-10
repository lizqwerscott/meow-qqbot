import pytest

from core.media.service import MediaService
from core.message import InputMessage
from core.tools._types import ToolContext
from core.tools.impl._delivery import resolve_transport_target


def test_input_message_derives_canonical_target_and_session_key():
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="123",
        content="hello",
        is_group=True,
    )

    assert message.delivery_target.channel == "qq"
    assert message.delivery_target.account_id == "default"
    assert message.delivery_target.chat_type == "group"
    assert message.delivery_target.target_id == "123"
    assert message.session_key == "agent:main:qq:default:group:123"


def test_input_message_preserves_explicit_identity():
    from core.session_identity import DeliveryTarget

    target = DeliveryTarget("telegram", "work", "direct", "abc:123")
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="legacy",
        content="hello",
        is_group=False,
        delivery_target=target,
        session_key="agent:support:telegram:work:direct:abc%3A123",
    )

    assert message.chat_id == "legacy"
    assert message.delivery_target == target
    assert message.session_key.endswith("abc%3A123")


def test_internal_input_message_does_not_invent_qq_target():
    message = InputMessage(
        id="wake",
        sender_id="system",
        chat_id="heartbeat:events",
        content="wake",
        is_group=False,
    )

    assert message.delivery_target is None
    assert message.session_key == "heartbeat:events"


@pytest.mark.parametrize(
    "chat_id",
    ["work-plan:plan-1", "agent:main:work-plan:plan-1", "subagent:run-1"],
)
def test_all_internal_input_messages_do_not_invent_qq_target(chat_id):
    message = InputMessage(
        id="internal",
        sender_id="system",
        chat_id=chat_id,
        content="work",
        is_group=False,
    )

    assert message.delivery_target is None
    assert message.session_key == chat_id


def test_tool_transport_resolves_canonical_key_to_raw_target():
    target = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="123",
        content="hello",
        is_group=True,
    ).delivery_target

    class Bot:
        @staticmethod
        def resolve_delivery_target(value, *, is_group=None):
            assert value == "agent:main:qq:default:group:123"
            assert is_group is True
            return target

    context = ToolContext(
        chat_id="agent:main:qq:default:group:123",
        is_group=True,
        reply_to="m1",
        sender_id="u1",
        reply_callback=None,
    )

    assert resolve_transport_target(context, Bot()) == ("123", False)


def test_tool_transport_keeps_reply_anchor_for_regular_turn_target():
    context = ToolContext(
        chat_id="agent:main:qq:default:group:123",
        is_group=True,
        reply_to="m1",
        sender_id="u1",
        reply_callback=None,
        delivery_channel="123",
        internal_control=False,
    )

    assert resolve_transport_target(context, None) == ("123", False)


@pytest.mark.parametrize("chat_id", ["heartbeat:events", "task:t1", "subagent:s1"])
def test_tool_transport_rejects_internal_session_without_target(chat_id):
    context = ToolContext(
        chat_id=chat_id,
        is_group=False,
        reply_to="",
        sender_id="system",
        reply_callback=None,
    )

    assert resolve_transport_target(context, None) == ("", False)


def test_media_service_uses_canonical_session_partition_when_enabled(tmp_path):
    class Resolver:
        canonical_enabled = True

    service = MediaService(
        http_client=None,
        storage_dir=tmp_path,
        session_identity_resolver=Resolver(),
    )
    message = InputMessage(
        id="m1",
        sender_id="u1",
        chat_id="123",
        content="",
        is_group=True,
        session_key="agent:main:qq:default:group:123",
    )

    assert service._session_key_for_message(message) == message.session_key
