"""AI 协议抽象层测试 — AssistantMessage / AssistantToolCall wire 格式。"""

import pytest

from core.ai.protocol import (
    AssistantMessage,
    AssistantToolCall,
    StreamCallbacks,
    StreamState,
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


class TestStreamState:
    """版本化流状态的 interface 测试。"""

    @pytest.mark.asyncio
    async def test_snapshots_are_cumulative_and_previous_values_are_stable(self):
        snapshots = []

        async def on_snapshot(snapshot):
            snapshots.append(snapshot)

        state = StreamState(StreamCallbacks(on_snapshot=on_snapshot))
        await state.append_text("旧")
        await state.append_text("内容")
        await state.append_reasoning("思考")

        assert [(item.generation, item.text, item.reasoning) for item in snapshots] == [
            (0, "旧", None),
            (0, "旧内容", None),
            (0, "旧内容", "思考"),
        ]
        assert snapshots[0].text == "旧"
        assert state.content == "旧内容"
        assert state.reasoning_content == "思考"

    @pytest.mark.asyncio
    async def test_replace_generation_clears_all_state_and_emits_event(self):
        resets = []

        async def on_reset(event):
            resets.append(event)

        state = StreamState(StreamCallbacks(on_reset=on_reset))
        await state.append_text("草稿")
        await state.upsert_tool_call("0", call_id="draft", name="search")
        state.finalize_tool_call("0")
        state.set_usage({"completion_tokens": 3})

        await state.replace_generation("provider_revision")

        assert state.generation == 1
        assert not state.has_output
        assert state.assemble() == (None, None)
        assert resets[0].previous_generation == 0
        assert resets[0].generation == 1
        assert resets[0].reason == "provider_revision"

    @pytest.mark.asyncio
    async def test_tool_calls_require_completion_and_preserve_order(self):
        state = StreamState()
        await state.upsert_tool_call(
            "first", call_id="call-1", name="search", arguments_delta='{"q":'
        )
        await state.upsert_tool_call("second", name="emoji", arguments_complete="")
        await state.upsert_tool_call("first", arguments_delta='"cats"}')

        message, usage = state.assemble()
        assert message is None
        assert usage is None
        assert state.has_output

        state.finalize_tool_call("second")
        state.finalize_tool_call("first")
        message, _ = state.assemble()
        assert message is not None
        assert [
            (call.id, call.name, call.arguments) for call in message.tool_calls
        ] == [
            ("call-1", "search", '{"q":"cats"}'),
            ("second", "emoji", ""),
        ]

    @pytest.mark.asyncio
    async def test_finalize_all_and_usage_are_isolated(self):
        usage = {"prompt_tokens": {"cached": 2}}
        state = StreamState()
        await state.upsert_tool_call("0", name="search", arguments_delta="{}")
        state.set_usage(usage)
        usage["prompt_tokens"]["cached"] = 99
        state.finalize_all_tool_calls()

        message, result_usage = state.assemble()
        assert message is not None
        assert result_usage == {"prompt_tokens": {"cached": 2}}
        result_usage["prompt_tokens"]["cached"] = 100
        message.tool_calls[0].arguments = "mutated"
        message_again, usage_again = state.assemble()
        assert message_again.tool_calls[0].arguments == "{}"
        assert usage_again == {"prompt_tokens": {"cached": 2}}

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_roll_back_state(self):
        async def on_snapshot(_snapshot):
            raise RuntimeError("callback failed")

        state = StreamState(StreamCallbacks(on_snapshot=on_snapshot))
        with pytest.raises(RuntimeError, match="callback failed"):
            await state.append_text("已写入")
        assert state.content == "已写入"

    @pytest.mark.asyncio
    async def test_callback_reentrancy_fails_fast(self):
        state = None

        async def on_snapshot(_snapshot):
            await state.append_text("重入")

        state = StreamState(StreamCallbacks(on_snapshot=on_snapshot))
        with pytest.raises(RuntimeError, match="callback reentrancy"):
            await state.append_text("外层")
        assert state.content == "外层"

    @pytest.mark.asyncio
    async def test_reset_callback_reentrancy_fails_fast(self):
        state = None

        async def on_reset(_event):
            await state.replace_generation("retry")

        state = StreamState(StreamCallbacks(on_reset=on_reset))
        with pytest.raises(RuntimeError, match="callback reentrancy"):
            await state.replace_generation("retry")
        assert state.generation == 1


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
