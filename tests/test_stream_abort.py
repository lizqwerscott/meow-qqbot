"""流式断流契约测试 — 服务层不得把半截聚合当完整结果返回。

回归（code-review Spec 轴 c1/c2）：修复前 AIService / DeepSeekResponsesService
在流迭代中途失败时吞掉异常、返回部分聚合，ToolLoop 误判成功 → mark_success
+ 投递截断回复 + 跳过 fallback 链（即使零转发）。修复后抛 StreamAbortedError，
由 ToolLoop 依转发状态决定回退（零转发）或终止（已转发部分文本，避免双回复）。
"""

from types import SimpleNamespace as NS

import pytest

from core.ai.deepseek_service import DeepSeekResponsesService
from core.ai.protocol import StreamAbortedError, StreamCallbacks
from core.ai.service import AIService


def _chunk(text=None, usage=None, reasoning=None, tool_calls=None):
    """最小 chat-completions SSE chunk 形状（_consume_chunk_stream 消费面）。"""
    if text is None and reasoning is None and tool_calls is None:
        return NS(choices=[], usage=usage)
    return NS(
        choices=[
            NS(
                delta=NS(
                    content=text,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                )
            )
        ],
        usage=usage,
    )


class _OkStream:
    """按顺序产出 chunk 后正常结束的流。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    async def __aiter__(self):
        for c in self._chunks:
            yield c

    async def close(self):
        self.closed = True


class _FailingStream:
    """产出若干 chunk 后抛异常的流（模拟中途断流）。"""

    def __init__(self, chunks, error):
        self._chunks = list(chunks)
        self._error = error
        self.closed = False

    async def __aiter__(self):
        for c in self._chunks:
            yield c
        raise self._error

    async def close(self):
        self.closed = True


def _make_ai_service(create_fn):
    svc = AIService(api_key="test")
    svc.client = NS(chat=NS(completions=NS(create=create_fn)))
    return svc


def _make_deepseek_service(create_fn):
    svc = DeepSeekResponsesService(api_key="test")
    svc.client = NS(responses=NS(create=create_fn))
    return svc


# ── AIService（chat-completions SSE） ──


class TestAIServiceStreamAbort:
    def test_mid_stream_failure_raises(self):
        """流迭代中途抛异常 → StreamAbortedError（绝不返回半截聚合）。"""

        async def create(**kwargs):
            return _FailingStream(
                [_chunk(text="已生成的部分内容")], RuntimeError("connection reset")
            )

        svc = _make_ai_service(create)
        with pytest.raises(StreamAbortedError, match="流式响应中断"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )

    def test_create_failure_raises(self):
        """create() 早期失败（未消费任何 chunk）→ 同样抛 StreamAbortedError。"""

        async def create(**kwargs):
            raise RuntimeError("429 rate limit")

        svc = _make_ai_service(create)
        with pytest.raises(StreamAbortedError, match="流式响应中断"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )

    def test_stream_closed_on_abort(self):
        """断流时流对象仍要 close（finally 兜底），不泄漏连接。"""

        stream = _FailingStream([_chunk(text="x")], RuntimeError("boom"))

        async def create(**kwargs):
            return stream

        svc = _make_ai_service(create)
        with pytest.raises(StreamAbortedError):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )
        assert stream.closed

    def test_complete_stream_returns_message(self):
        """正常结束不受影响：完整聚合返回 AssistantMessage。"""

        async def create(**kwargs):
            return _OkStream([_chunk(text="完"), _chunk(text="整文本")])

        svc = _make_ai_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert message is not None and message.content == "完整文本"

    def test_chat_completion_adapter_emits_state_snapshots_and_aggregates_tools(self):
        """chat-completions adapter 只翻译事件，状态快照与工具调用由协议模块拥有。"""
        snapshots = []

        async def on_snapshot(snapshot):
            snapshots.append((snapshot.generation, snapshot.text, snapshot.reasoning))

        async def create(**kwargs):
            return _OkStream(
                [
                    _chunk(text="答案"),
                    _chunk(reasoning="思考"),
                    _chunk(
                        tool_calls=[
                            NS(
                                index=0,
                                id="call_1",
                                function=NS(name="search", arguments='{"q":'),
                            )
                        ]
                    ),
                    _chunk(
                        tool_calls=[
                            NS(
                                index=0,
                                id=None,
                                function=NS(name=None, arguments='"cat"}'),
                            )
                        ]
                    ),
                    _chunk(usage=NS(model_dump=lambda: {"total_tokens": 7})),
                ]
            )

        svc = _make_ai_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(
                messages=[{"role": "user", "content": "hi"}],
                callbacks=StreamCallbacks(on_snapshot=on_snapshot),
            )
        )
        assert snapshots == [(0, "答案", None), (0, "答案", "思考")]
        assert message is not None
        assert message.tool_calls[0].arguments == '{"q":"cat"}'
        assert usage == {"total_tokens": 7}

    def test_stream_options_retry_discards_partial(self):
        """stream_options 降级重试：首流半截内容必须丢弃（重试是全新生成）。

        回归（c2）：修复前重试直接追加到未清空的 text_parts → 内容重复。
        """
        calls = {"n": 0}

        async def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # 网关在流中途拒绝 stream_options：首流已有半截内容
                return _FailingStream(
                    [_chunk(text="首流半截内容")],
                    RuntimeError("server does not support stream_options"),
                )
            return _OkStream([_chunk(text="重试后的完整内容")])

        svc = _make_ai_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert calls["n"] == 2
        assert (
            message is not None and message.content == "重试后的完整内容"
        ), "重试结果不得混入首流半截内容"

    def test_retry_emits_on_reset(self):
        """降级重试前触发 on_reset：调用方归零转发偏移（新生成从 0 开始）。"""
        calls = {"n": 0, "resets": 0}

        async def on_reset(_event):
            calls["resets"] += 1

        async def create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FailingStream(
                    [_chunk(text="首流半截")],
                    RuntimeError("stream_options unsupported"),
                )
            return _OkStream([_chunk(text="重试后的内容")])

        svc = _make_ai_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(
                messages=[{"role": "user", "content": "hi"}],
                callbacks=StreamCallbacks(on_reset=on_reset),
            )
        )
        assert calls["n"] == 2 and calls["resets"] == 1
        assert message is not None and message.content == "重试后的内容"

    def test_retry_failure_raises(self):
        """降级重试也失败 → 抛 StreamAbortedError（不再静默返回半截聚合）。"""

        async def create(**kwargs):
            raise RuntimeError("stream_options unsupported")

        svc = _make_ai_service(create)
        with pytest.raises(StreamAbortedError, match="流式降级重试失败"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )


# ── DeepSeekResponsesService（Responses API SSE） ──


def _ev(t, **kw):
    return NS(type=t, **kw)


class TestDeepSeekStreamAbort:
    def test_failed_event_raises(self):
        """response.failed 事件 → StreamAbortedError（API 侧失败不算正常完成）。"""

        async def create(**kwargs):
            return _OkStream(
                [
                    _ev("response.output_text.delta", delta="已生成部分"),
                    _ev("response.failed", response=NS(error="server error")),
                ]
            )

        svc = _make_deepseek_service(create)
        with pytest.raises(StreamAbortedError, match="Responses API 流式失败"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )

    def test_mid_stream_exception_raises(self):
        """流迭代抛异常 → StreamAbortedError（不返回半截聚合）。"""

        async def create(**kwargs):
            return _FailingStream(
                [_ev("response.output_text.delta", delta="部分")],
                RuntimeError("socket closed"),
            )

        svc = _make_deepseek_service(create)
        with pytest.raises(StreamAbortedError, match="流式响应中断"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )

    def test_draft_then_final_keeps_only_final(self):
        """修订输出（草稿+终稿两个 message item）：缓冲只保留终稿。

        回归（线上事故）：DeepSeek v4-flash 先流草稿再流终稿，旧实现把两个
        item 的 output_text.delta 全部 append → 内容 = 草稿+终稿翻倍，用户
        看到两条消息（首块切在草稿末尾 + 补发终稿）。
        """
        p1 = "第一段"
        p2 = "第二段"
        p3 = "第三段补充"

        class _DraftFinalStream:
            async def __aiter__(self):
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="item_draft"),
                )
                for ch in p1 + "\n\n" + p2:
                    yield _ev(
                        "response.output_text.delta", delta=ch, item_id="item_draft"
                    )
                yield _ev(
                    "response.output_item.done",
                    item=NS(type="message", id="item_draft"),
                )
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="item_final"),
                )
                for ch in p1 + "\n\n" + p2 + "\n\n" + p3:
                    yield _ev(
                        "response.output_text.delta", delta=ch, item_id="item_final"
                    )
                yield _ev(
                    "response.output_item.done",
                    item=NS(type="message", id="item_final"),
                )
                yield _ev("response.completed", response=NS(usage=None))

            async def close(self):
                pass

        async def create(**kwargs):
            return _DraftFinalStream()

        svc = _make_deepseek_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert message is not None
        assert (
            message.content == p1 + "\n\n" + p2 + "\n\n" + p3
        ), "内容必须只含终稿，草稿文本不得混入"

    def test_revision_emits_on_reset_and_drops_draft_tool_calls(self):
        """修订时触发 on_reset（调用方归零转发偏移）+ 草稿工具调用丢弃。

        回归：草稿被转发过（超长/空闲 flush）后若不触发 on_reset，终稿会从
        旧偏移切片；草稿的 function_call 不丢弃会被幽灵执行。
        """
        resets = {"n": 0}

        async def on_reset(_event):
            resets["n"] += 1

        class _ReviseStream:
            async def __aiter__(self):
                # 草稿轮：文本 + 工具调用
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="m_draft"),
                )
                for ch in "草稿文本":
                    yield _ev("response.output_text.delta", delta=ch, item_id="m_draft")
                yield _ev(
                    "response.output_item.done",
                    item=NS(type="message", id="m_draft"),
                )
                yield _ev(
                    "response.output_item.added",
                    item=NS(
                        type="function_call",
                        id="fc_draft",
                        name="web_search",
                        call_id="call_draft",
                    ),
                )
                yield _ev(
                    "response.function_call_arguments.delta",
                    item_id="fc_draft",
                    delta='{"q": "旧查询"}',
                )
                yield _ev(
                    "response.function_call_arguments.done",
                    item_id="fc_draft",
                    arguments='{"q": "旧查询"}',
                )
                yield _ev(
                    "response.output_item.done",
                    item=NS(type="function_call", id="fc_draft"),
                )
                # 终稿轮：新 message item 取代草稿
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="m_final"),
                )
                for ch in "终稿文本内容":
                    yield _ev("response.output_text.delta", delta=ch, item_id="m_final")
                yield _ev(
                    "response.output_item.done",
                    item=NS(type="message", id="m_final"),
                )
                yield _ev(
                    "response.output_item.added",
                    item=NS(
                        type="function_call",
                        id="fc_final",
                        name="web_search",
                        call_id="call_final",
                    ),
                )
                yield _ev(
                    "response.function_call_arguments.delta",
                    item_id="fc_final",
                    delta='{"q": "新查询"}',
                )
                yield _ev(
                    "response.function_call_arguments.done",
                    item_id="fc_final",
                    arguments='{"q": "新查询"}',
                )
                yield _ev(
                    "response.output_item.done",
                    item=NS(type="function_call", id="fc_final"),
                )
                yield _ev("response.completed", response=NS(usage=None))

            async def close(self):
                pass

        async def create(**kwargs):
            return _ReviseStream()

        svc = _make_deepseek_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(
                messages=[{"role": "user", "content": "hi"}],
                callbacks=StreamCallbacks(on_reset=on_reset),
            )
        )
        assert resets["n"] == 1, "终稿 message item 开始时必须触发一次 on_reset"
        assert message is not None
        assert message.content == "终稿文本内容"
        assert message.tool_calls is not None and len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "call_final", "草稿工具调用必须被丢弃"
        assert message.tool_calls[0].arguments == '{"q": "新查询"}'

    def test_revision_preserves_reasoning_before_final_message(self):
        """终稿 reasoning 先到时，切代后仍归入终稿。"""

        class _ReasoningBeforeFinalMessageStream:
            async def __aiter__(self):
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="m_draft"),
                )
                yield _ev("response.output_text.delta", delta="草稿")
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="function_call", id="fc_draft", name="search"),
                )
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="reasoning", id="r_final"),
                )
                yield _ev(
                    "response.reasoning_text.delta",
                    item_id="r_final",
                    delta="终稿前思考第一段",
                )
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="reasoning", id="r_final_2"),
                )
                yield _ev(
                    "response.reasoning_text.delta",
                    item_id="r_final_2",
                    delta="终稿前思考",
                )
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="m_final"),
                )
                yield _ev("response.output_text.delta", delta="终稿")
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="reasoning", id="r_after"),
                )
                yield _ev(
                    "response.reasoning_text.delta",
                    item_id="r_after",
                    delta="终稿后思考",
                )
                yield _ev("response.completed", response=NS(usage=None))

            async def close(self):
                pass

        async def create(**kwargs):
            return _ReasoningBeforeFinalMessageStream()

        svc = _make_deepseek_service(create)
        message, _ = asyncio_run(
            svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert message is not None
        assert message.content == "终稿"
        assert message.reasoning_content == "终稿前思考第一段终稿前思考终稿后思考"

    def test_ambiguous_reasoning_delta_aborts_stream(self):
        """缺少 item 关联且无法依赖事件顺序时，适配器必须 fail closed。"""

        class _AmbiguousReasoningStream:
            async def __aiter__(self):
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="m1"),
                )
                yield _ev("response.output_text.delta", delta="已有文本")
                yield _ev("response.reasoning_text.delta", delta="无法归属")

            async def close(self):
                pass

        async def create(**kwargs):
            return _AmbiguousReasoningStream()

        svc = _make_deepseek_service(create)
        with pytest.raises(StreamAbortedError, match="无法关联到 output item"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )

    def test_unknown_reasoning_item_id_aborts_stream(self):
        """reasoning delta 引用未知 item 时，适配器必须 fail closed。"""

        class _UnknownReasoningItemStream:
            async def __aiter__(self):
                yield _ev(
                    "response.output_item.added",
                    item=NS(type="message", id="m1"),
                )
                yield _ev("response.output_text.delta", delta="已有文本")
                yield _ev(
                    "response.reasoning_text.delta",
                    item_id="unknown-reasoning-item",
                    delta="无法归属",
                )

            async def close(self):
                pass

        async def create(**kwargs):
            return _UnknownReasoningItemStream()

        svc = _make_deepseek_service(create)
        with pytest.raises(StreamAbortedError, match="未知 output item"):
            asyncio_run(
                svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
            )

    def test_parse_output_revision_keeps_only_final(self):
        """非流式 _parse_output 对齐修订语义：只保留最新 message/工具调用。"""

        def _msg_item(i, text):
            return NS(
                type="message", id=f"m{i}", content=[NS(type="output_text", text=text)]
            )

        def _fc_item(i):
            return NS(
                type="function_call",
                id=f"fc{i}",
                call_id=f"call_{i}",
                name="web_search",
                arguments='{"q": "q%d"}' % i,
            )

        response = NS(
            output=[
                _msg_item(1, "草稿文本"),
                _fc_item(1),
                _msg_item(2, "终稿文本"),
                _fc_item(2),
            ]
        )
        from core.ai.deepseek_service import _parse_output

        msg = _parse_output(response)
        assert msg is not None
        assert msg.content == "终稿文本"
        assert len(msg.tool_calls) == 1 and msg.tool_calls[0].id == "call_2"

    def test_parse_output_keeps_reasoning_adjacent_to_latest_message(self):
        """最新 message 前后的 reasoning 都属于最新 generation。"""

        def _msg(text):
            return NS(type="message", content=[NS(type="output_text", text=text)])

        def _reasoning(text):
            return NS(type="reasoning", content=[NS(type="reasoning_text", text=text)])

        response = NS(
            output=[
                _msg("草稿"),
                _reasoning("终稿前思考第一段"),
                _reasoning("终稿前思考"),
                _msg("终稿"),
                _reasoning("终稿后思考"),
            ]
        )
        from core.ai.deepseek_service import _parse_output

        msg = _parse_output(response)
        assert msg is not None
        assert msg.content == "终稿"
        assert msg.reasoning_content == "终稿前思考第一段终稿前思考终稿后思考"

    def test_revision_without_item_ids_still_starts_new_generation(self):
        """缺失 item ID 时，已有输出后的新 message 仍必须切代。"""
        resets = {"n": 0}

        async def on_reset(_event):
            resets["n"] += 1

        class _NoIdRevisionStream:
            async def __aiter__(self):
                yield _ev(
                    "response.output_item.added", item=NS(type="message", id=None)
                )
                yield _ev("response.output_text.delta", delta="草稿")
                yield _ev(
                    "response.output_item.added", item=NS(type="message", id=None)
                )
                yield _ev("response.output_text.delta", delta="终稿")
                yield _ev("response.completed", response=NS(usage=None))

            async def close(self):
                pass

        async def create(**kwargs):
            return _NoIdRevisionStream()

        svc = _make_deepseek_service(create)
        message, _ = asyncio_run(
            svc.chat_completion_stream(
                messages=[{"role": "user", "content": "hi"}],
                callbacks=StreamCallbacks(on_reset=on_reset),
            )
        )
        assert resets["n"] == 1
        assert message is not None and message.content == "终稿"

    def test_function_call_alias_keeps_canonical_arguments(self):
        """item ID 与 call ID 不同时，arguments 事件仍归并到同一调用。"""

        class _AliasedToolStream:
            async def __aiter__(self):
                yield _ev(
                    "response.output_item.added",
                    item=NS(
                        type="function_call",
                        id="item-1",
                        call_id="call-1",
                        name="search",
                    ),
                )
                yield _ev(
                    "response.function_call_arguments.delta",
                    item_id="call-1",
                    delta='{"q":"stale"}',
                )
                yield _ev(
                    "response.function_call_arguments.done",
                    item_id="call-1",
                    arguments="",
                )
                yield _ev("response.completed", response=NS(usage=None))

            async def close(self):
                pass

        async def create(**kwargs):
            return _AliasedToolStream()

        svc = _make_deepseek_service(create)
        message, _ = asyncio_run(
            svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert message is not None and message.tool_calls is not None
        assert message.tool_calls[0].id == "call-1"
        assert message.tool_calls[0].arguments == ""

    def test_complete_stream_returns_message(self):
        """response.completed 正常收尾不受影响。"""

        async def create(**kwargs):
            return _OkStream(
                [
                    _ev("response.output_text.delta", delta="完整"),
                    _ev("response.output_text.delta", delta="回复"),
                    _ev("response.completed", response=NS(usage=None)),
                ]
            )

        svc = _make_deepseek_service(create)
        message, usage = asyncio_run(
            svc.chat_completion_stream(messages=[{"role": "user", "content": "hi"}])
        )
        assert message is not None and message.content == "完整回复"


def asyncio_run(coro):
    """统一入口：后续如需切 loop 策略只改这里。"""
    import asyncio

    return asyncio.run(coro)
