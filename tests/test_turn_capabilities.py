from core.engine.turn_capabilities import TurnCapabilities
from core.managers.session_manager import InboundIntent


def tool(name):
    return {"type": "function", "function": {"name": name}}


def test_ambient_capabilities_allow_only_low_risk_tools():
    capabilities = TurnCapabilities.for_intent(
        InboundIntent.GROUP_AMBIENT,
        allowed_media_uris=frozenset({"media://inbound/allowed"}),
    )

    assert capabilities.allow_automatic_reply is False
    assert capabilities.allows_delivery_kind("message")
    assert capabilities.allows_delivery_kind("emoji")
    assert not capabilities.allows_delivery_kind("automatic")
    assert capabilities.allows_tool("send_message") is True
    assert capabilities.allows_tool("image") is True
    assert capabilities.allows_tool("exec") is False
    assert [
        item["function"]["name"]
        for item in capabilities.filter_tools(
            [tool("exec"), tool("image"), tool("send_message")]
        )
    ] == ["image", "send_message"]
    assert capabilities.allows_tool_args(
        "image", {"media_uri": "media://inbound/allowed"}
    )
    assert not capabilities.allows_tool_args(
        "image", {"media_uri": "media://inbound/other"}
    )
    assert capabilities.allows_tool_args(
        "read_file", {"media_uri": "media://inbound/allowed"}
    )
    assert not capabilities.allows_tool_args(
        "read_file",
        {"media_uri": "media://inbound/allowed", "file_path": "secrets.txt"},
    )


def test_direct_capabilities_preserve_existing_tool_surface():
    capabilities = TurnCapabilities.for_intent(InboundIntent.DIRECT_TASK)

    assert capabilities.allow_automatic_reply is True
    assert capabilities.allows_delivery_kind("automatic")


def test_capabilities_reject_context_from_another_turn():
    capabilities = TurnCapabilities.for_intent(
        InboundIntent.DIRECT_TASK,
        chat_id="chat-a",
        sender_id="user-a",
        reply_to="message-a",
    )

    assert capabilities.allows_context(
        chat_id="chat-a", sender_id="user-a", reply_to="message-a"
    )
    assert not capabilities.allows_context(
        chat_id="chat-b", sender_id="user-a", reply_to="message-a"
    )
    assert not capabilities.allows_context(
        chat_id="chat-a", sender_id="user-b", reply_to="message-a"
    )
    assert not capabilities.allows_context(
        chat_id="chat-a", sender_id="user-a", reply_to="message-b"
    )
