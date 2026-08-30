from core.engine.prompt_snapshot import PromptMode
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


def test_ambient_chat_profile_exposes_limited_control_actions():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="group_ambient",
        intent=InboundIntent.GROUP_AMBIENT,
    )

    schemas = capabilities.model_tool_schemas([])
    actions = schemas[-1]["function"]["parameters"]["oneOf"]
    assert {item["properties"]["action"]["const"] for item in actions} == {
        "wait",
        "no_reply",
    }


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


def test_chat_capability_profiles_filter_high_risk_tools_and_actions():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="private_chat",
        intent=InboundIntent.PRIVATE_CONVERSATION,
    )

    assert capabilities.planner_actions == frozenset(
        {"wait", "no_reply", "request_agent"}
    )
    assert capabilities.allows_tool("send_emoji")
    assert capabilities.allows_tool("web_search")
    assert capabilities.allows_tool("memory")
    assert not capabilities.allows_tool("exec")
    assert not capabilities.allows_tool("write_file")
    assert not capabilities.allows_tool("cron")
    assert not capabilities.allows_tool("spawn_subagent")
    assert [
        item["function"]["name"]
        for item in capabilities.filter_tools(
            [tool("exec"), tool("web_search"), tool("write_file"), tool("send_emoji")]
        )
    ] == ["web_search", "send_emoji"]


def test_chat_memory_execution_rejects_cross_user_scope_and_relations():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="group_explicit",
        intent=InboundIntent.DIRECT_TASK,
    )

    assert capabilities.allows_tool_args("memory", {"action": "search"})
    assert not capabilities.allows_tool_args(
        "memory", {"action": "search", "person_name": "其他群友"}
    )
    assert not capabilities.allows_tool_args(
        "memory", {"action": "relation", "person_a": "甲", "person_b": "乙"}
    )


def test_group_reply_shared_memory_schema_matches_execution_policy():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="group_reply",
        intent=InboundIntent.DIRECT_TASK,
    )

    assert capabilities.allows_tool_args("memory", {"action": "search_shared"})
    assert not capabilities.allows_tool_args(
        "memory", {"action": "search_shared", "user_id": "another-user"}
    )


def test_ambient_chat_exposes_wait_and_attachment_only_media_reads():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="group_ambient",
        intent=InboundIntent.GROUP_AMBIENT,
        allowed_media_uris=frozenset({"media://inbound/allowed"}),
    )

    assert capabilities.planner_actions == frozenset({"wait", "no_reply"})
    assert not capabilities.allow_automatic_reply
    assert not capabilities.allows_tool("memory")
    assert capabilities.allows_tool_args(
        "read_file", {"media_uri": "media://inbound/allowed"}
    )
    assert not capabilities.allows_tool_args(
        "read_file",
        {"media_uri": "media://inbound/allowed", "file_path": "secret.txt"},
    )


def test_agent_mode_preserves_the_existing_unrestricted_tool_surface():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.AGENT,
        capability_profile="ignored-by-agent",
        intent=InboundIntent.DIRECT_TASK,
    )

    assert capabilities.capability_profile == "agent_full"
    assert capabilities.allows_tool("exec")
    assert capabilities.allows_tool("write_file")


def test_chat_model_schemas_add_one_profile_filtered_planner_control_tool():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.CHAT,
        capability_profile="group_reply",
        intent=InboundIntent.DIRECT_TASK,
    )

    schemas = capabilities.model_tool_schemas(
        [tool("web_search"), tool("exec"), tool("planner_control")]
    )

    assert [schema["function"]["name"] for schema in schemas] == [
        "web_search",
        "planner_control",
    ]
    actions = schemas[-1]["function"]["parameters"]["oneOf"]
    assert {item["properties"]["action"]["const"] for item in actions} == {
        "wait",
        "no_reply",
    }


def test_agent_model_schemas_do_not_add_planner_control():
    capabilities = TurnCapabilities.for_mode(
        mode=PromptMode.AGENT,
        capability_profile="agent_full",
        intent=InboundIntent.DIRECT_TASK,
    )

    assert [
        schema["function"]["name"]
        for schema in capabilities.model_tool_schemas([tool("exec")])
    ] == ["exec"]
