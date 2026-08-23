from core.engine.delivery_prompt_contract import DeliveryPromptContract
from core.managers.session_manager import InboundIntent


def test_automatic_delivery_contract_declares_final_text_and_fixed_target():
    contract = DeliveryPromptContract(
        intent=InboundIntent.DIRECT_TASK,
        delivery_mode="automatic",
        reply_target="message-1",
    )

    rendered = contract.render(
        [
            {"function": {"name": "send_message"}},
            {"function": {"name": "read_file"}},
        ]
    )

    assert rendered.startswith("<delivery_contract>")
    assert "不含 tool call 的最终 assistant 文本会自动发送" in rendered
    assert "固定投递目标：message-1" not in rendered
    assert contract.render_target() == (
        "固定投递目标：message-1。不得选择、推断或改写其他 chat 的投递目标。"
    )
    assert "send_message" in rendered
    assert "read_file" not in rendered


def test_tool_only_ambient_contract_requires_explicit_delivery_or_no_reply():
    contract = DeliveryPromptContract(
        intent=InboundIntent.GROUP_AMBIENT,
        delivery_mode="message_tool_only",
        reply_target="ambient-anchor",
    )

    rendered = contract.render([{"function": {"name": "send_message"}}])

    assert "任何 assistant 文本，包括最终文本，都不会自动发送" in rendered
    assert "必须调用可见投递工具" in rendered
    assert "群聊主动参与" in rendered
    assert "NO_REPLY" in rendered
    assert "固定投递目标：ambient-anchor" not in rendered


def test_contract_only_lists_delivery_tools_present_in_turn_toolset():
    contract = DeliveryPromptContract(
        intent=InboundIntent.PRIVATE_CONVERSATION,
        delivery_mode="automatic",
        reply_target="message-1",
    )

    rendered = contract.render([{"function": {"name": "read_file"}}])

    assert "可见显式投递工具：无。" in rendered


def test_contract_fingerprint_ignores_per_turn_reply_target():
    tools = [{"function": {"name": "send_message"}}]
    first = DeliveryPromptContract(
        intent=InboundIntent.PRIVATE_CONVERSATION,
        delivery_mode="automatic",
        reply_target="message-1",
    )
    second = DeliveryPromptContract(
        intent=InboundIntent.PRIVATE_CONVERSATION,
        delivery_mode="automatic",
        reply_target="message-2",
    )

    assert first.fingerprint(tools) == second.fingerprint(tools)
