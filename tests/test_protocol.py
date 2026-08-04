"""AI 协议抽象层测试 — AssistantMessage / AssistantToolCall wire 格式。"""

from core.ai.protocol import (
    AssistantMessage,
    AssistantToolCall,
    ensure_messages_consistent,
)


def _tc(tc_id: str = "call_1", name: str = "web_search", args: str = "{}"):
    return AssistantToolCall(id=tc_id, name=name, arguments=args)


class TestAssistantToolCallToWire:
    def test_to_wire_matches_openai_format(self):
        """逐字段对照 openai wire 格式（id/type/function.name/arguments）。"""
        tc = _tc()
        assert tc.to_wire() == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": "{}"},
        }


class TestAssistantMessageToWire:
    def test_with_tool_calls_and_reasoning(self):
        """完整形态：content + tool_calls + reasoning_content。"""
        msg = AssistantMessage(
            content="你好",
            tool_calls=[_tc(), _tc("call_2", "send_emoji")],
            reasoning_content="思考中",
        )
        assert msg.to_wire() == {
            "role": "assistant",
            "content": "你好",
            "reasoning_content": "思考中",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "send_emoji", "arguments": "{}"},
                },
            ],
        }

    def test_without_reasoning_omits_key(self):
        """无 reasoning 时不携带该键（与旧 ToolLoop 条件设置一致）。"""
        msg = AssistantMessage(content="你好", tool_calls=[_tc()])
        wire = msg.to_wire()
        assert "reasoning_content" not in wire
        assert wire["tool_calls"] == msg.tool_calls_data

    def test_without_tool_calls_omits_key(self):
        """无 tool_calls 时不携带该键（空数组会触发部分 API 400）。"""
        msg = AssistantMessage(content="你好")
        wire = msg.to_wire()
        assert wire == {"role": "assistant", "content": "你好"}
        assert "tool_calls" not in wire

    def test_content_none_serializes_as_none(self):
        """content 为 None 时保持 None（不降级为空串）。"""
        msg = AssistantMessage(tool_calls=[_tc()])
        assert msg.to_wire()["content"] is None


class TestEnsureMessagesConsistent:
    def test_removes_orphan_tool_call(self):
        """孤立 tool_calls（无对应 tool 响应）的 assistant 消息被移除。"""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {}}],
            },
        ]
        ensure_messages_consistent(messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_keeps_complete_tool_roundtrip(self):
        """完整 assistant(tool_calls) + tool 响应配对保留原序。"""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        ensure_messages_consistent(messages)
        assert [m["role"] for m in messages] == ["user", "assistant", "tool"]

    def test_reorders_interleaved_user_message(self):
        """tool 响应被用户消息插队时，重新排序使 tool 紧跟 assistant。"""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {}}],
            },
            {"role": "user", "content": "interleaved"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        ensure_messages_consistent(messages)
        assert [m["role"] for m in messages] == ["assistant", "tool", "user"]

    def test_removes_orphan_tool_response(self):
        """无 assistant(tool_calls) 配对的孤立 tool 响应被移除。"""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "call_9", "content": "orphan"},
        ]
        ensure_messages_consistent(messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
