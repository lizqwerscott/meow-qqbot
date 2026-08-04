"""流式断流契约测试 — 服务层不得把半截聚合当完整结果返回。

回归（code-review Spec 轴 c1/c2）：修复前 AIService / DeepSeekResponsesService
在流迭代中途失败时吞掉异常、返回部分聚合，ToolLoop 误判成功 → mark_success
+ 投递截断回复 + 跳过 fallback 链（即使零转发）。修复后抛 StreamAbortedError，
由 ToolLoop 依转发状态决定回退（零转发）或终止（已转发部分文本，避免双回复）。
"""

from types import SimpleNamespace as NS

import pytest

from core.ai.deepseek_service import DeepSeekResponsesService
from core.ai.protocol import StreamAbortedError
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
