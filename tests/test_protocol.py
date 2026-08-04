"""AI 协议抽象层测试 — AssistantMessage / AssistantToolCall wire 格式。"""

import pytest

from core.ai.protocol import (
    AssistantMessage,
    AssistantToolCall,
    StreamBuffer,
    ensure_messages_consistent,
    log_llm_error,
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


class TestStreamBuffer:
    """流式聚合缓冲：消灭服务层平行 list 参数组后的唯一拼装点。"""

    def test_assemble_full_message(self):
        """文本 + 思维链 + 工具调用 → 完整 AssistantMessage。"""
        buf = StreamBuffer()
        buf.text_parts.extend(["你好", "世界"])
        buf.reasoning_parts.append("思考中")
        buf.tool_calls.append(
            AssistantToolCall(id="c1", name="web_search", arguments="{}")
        )
        message, usage = buf.assemble({"prompt_tokens": 1})
        assert message is not None
        assert message.content == "你好世界"
        assert message.reasoning_content == "思考中"
        assert message.tool_calls[0].name == "web_search"
        assert usage == {"prompt_tokens": 1}

    def test_assemble_empty_returns_none(self):
        """无任何内容 → (None, usage)，调用方走失败/回退路径。"""
        message, usage = StreamBuffer().assemble(None)
        assert message is None and usage is None

    def test_assemble_drops_incomplete_tool_calls(self):
        """无 name 的未完成调用被过滤（对齐旧 _assemble_stream_result 语义）。"""
        buf = StreamBuffer()
        buf.tool_calls.append(AssistantToolCall(id="c1", name="", arguments=""))
        message, usage = buf.assemble(None)
        assert message is None

    def test_reset_clears_all(self):
        """reset：降级重试前清空（重试是全新生成）。"""
        buf = StreamBuffer()
        buf.text_parts.append("旧内容")
        buf.tool_calls.append(AssistantToolCall(id="c1", name="t", arguments=""))
        buf.reset()
        message, usage = buf.assemble(None)
        assert message is None


class TestLogLlmError:
    """统一错误分类日志（限流/服务不可用/超时/其他）。"""

    @pytest.mark.parametrize(
        "err, level",
        [
            (RuntimeError("429 Too Many Requests"), "WARNING"),
            (RuntimeError("rate_limit_exceeded"), "WARNING"),
            (RuntimeError("502 Bad Gateway"), "ERROR"),
            (RuntimeError("503 Service Unavailable"), "ERROR"),
            (RuntimeError("service_unavailable"), "ERROR"),
            (RuntimeError("request timeout after 30s"), "WARNING"),
            (RuntimeError("connection refused"), "ERROR"),
        ],
    )
    def test_classification(self, caplog, err, level):
        """429/rate_limit/timeout → warning；502/503/其他 → error。"""
        with caplog.at_level("WARNING", logger="core.ai.protocol"):
            log_llm_error(err, "model-x")
        assert any(r.levelname == level for r in caplog.records)
        assert all(
            r.levelname == level for r in caplog.records if "model-x" in r.getMessage()
        )

    def test_service_and_tag_format(self, caplog):
        """service/tag 参数控制日志措辞（DeepSeek 流式等）。"""
        with caplog.at_level("WARNING", logger="core.ai.protocol"):
            log_llm_error(RuntimeError("429"), "m1", service="DeepSeek", tag="（流式）")
        assert any(
            "DeepSeek 请求被限流（流式） [m1]" in r.getMessage() for r in caplog.records
        )

    def test_status_code_takes_priority(self, caplog):
        """openai 风格异常自带 status_code 时优先于字符串子串判定。"""

        class _StatusErr(Exception):
            def __init__(self, status, msg):
                super().__init__(msg)
                self.status_code = status

        # 429 但消息里没有任何限流关键词：status 判定必须生效
        with caplog.at_level("WARNING", logger="core.ai.protocol"):
            log_llm_error(_StatusErr(429, "weird upstream response"), "m1")
        assert any("请求被限流" in r.getMessage() for r in caplog.records)
        # 503 同理
        with caplog.at_level("WARNING", logger="core.ai.protocol"):
            log_llm_error(_StatusErr(503, "weird upstream response"), "m1")
        assert any("服务不可用" in r.getMessage() for r in caplog.records)
        # 无 status_code 的异常回退子串判定（既有行为不变）
        with caplog.at_level("WARNING", logger="core.ai.protocol"):
            log_llm_error(RuntimeError("502 Bad Gateway"), "m1")
        assert any("服务不可用" in r.getMessage() for r in caplog.records)
