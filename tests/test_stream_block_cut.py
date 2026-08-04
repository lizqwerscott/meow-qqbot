"""流式 block 切块安全测试 — markdown 表格/链接/围栏不切断。

背景（线上事故）：r/emacs 推送消息在 QQ 上"被打乱"——内容重复、链接半截
（`announcing_now` 截断）。根因：流式 block 切块按字符数切，块边界落在
链接行中间；且 `_markdown_safe_cut` 在 pending 恰好达到块大小、末尾是
半截行（流式生成中，无换行结尾）时，safe 被更新为 len+1 超出文本长度，
调用方"不切"→ 半截链接整段发出。

本测试用线上真实消息（REAL_R_EMACS_MESSAGE）走完整流式链路回归。
"""

import asyncio
from types import SimpleNamespace as NS

import pytest

from core.ai.protocol import AssistantMessage
from core.markdown_split import (
    markdown_safe_cut,
    pending_starts_incomplete,
    trailing_structure,
)
from core.tools.tool_loop import ToolLoop

# ── 线上真实消息（2026-08-04 16:19 FreshRSS r/emacs 推送，jsonl 原文） ──
REAL_R_EMACS_MESSAGE = (
    "来啦主人！这是 FreshRSS「论坛」分类里最新的 **r/emacs 帖子**（8月2日~4日），"
    "干净的列表版喵～(ฅ´ω`ฅ)\n\n"
    "**📌 r/emacs 最新热帖：**\n\n"
    "1. 🧠 **The Shape of Things to Come, Part 1: The Continuous Thunderdome — "
    "Steve Yegge**\n"
    "   [reddit.com/r/emacs/comments/1vewowe]"
    "(https://www.reddit.com/r/emacs/comments/1vewowe/not_mine_the_shape_of_things_to_come_part_1_the/)\n"
    '   *Steve Yegge 的长文，讨论"未来形态"话题，回帖挺热闹的*\n\n'
    "2. 📖 **bible-gateway package update**（触摸打字背经文 + 阅读计划 + transient 菜单）\n"
    "   [reddit.com/r/emacs/comments/1veofgj]"
    "(https://www.reddit.com/r/emacs/comments/1veofgj/biblegateway_package_update_touchtyping_verse/)\n\n"
    "3. 📱 **Best android client for org agenda/todos?（不要 orgzly）**\n"
    "   [reddit.com/r/emacs/comments/1vel9ze]"
    "(https://www.reddit.com/r/emacs/comments/1vel9ze/best_android_client_for_org_agendatodos_not_orgzly/)\n"
    "   *求安卓端 Org agenda 客户端推荐*\n\n"
    "4. 🎵 **Announcing now-playing.el**（macOS Music 应用的远程控制接口）\n"
    "   [reddit.com/r/emacs/comments/1vel7ne]"
    "(https://www.reddit.com/r/emacs/comments/1vel7ne/announcing_nowplayingel_a_remote_control/)\n\n"
    "5. 🖥️ **A Real-Time Display Editor**\n"
    "   [reddit.com/r/emacs/comments/1veh2nf]"
    "(https://www.reddit.com/r/emacs/comments/1veh2nf/a_realtime_display_editor/)\n\n"
    "6. 🤖 **Claude Code Layer Spacemacs not working**"
    "（Claude Code 层在 Spacemacs 里不工作，求助帖）\n"
    "   [reddit.com/r/emacs/comments/1vefzcy]"
    "(https://www.reddit.com/r/emacs/comments/1vefzcy/claude_code_layer_spacemacs_not_working/)\n\n"
    "7. 🐛 **[patch] GNU emacs bug#68339 (w32 ime font size)**\n"
    "   [reddit.com/r/emacs/comments/1ve3g80]"
    "(https://www.reddit.com/r/emacs/comments/1ve3g80/patch_gnu_emacs_bug68339_w32_ime_font_size/)\n\n"
    "8. 🎨 **I decided to customize imenu-list a bit more**\n"
    "   [reddit.com/r/emacs/comments/1vdziaa]"
    "(https://www.reddit.com/r/emacs/comments/1vdziaa/i_decided_to_customize_imenulist_a_bit_more/)\n\n"
    "9. 🌱 **15 years old looking for advice**（15 岁新人求入坑建议）\n"
    "   [reddit.com/r/emacs/comments/1vdy2pg]"
    "(https://www.reddit.com/r/emacs/comments/1vdy2pg/15_years_old_looking_for_advice/)\n\n"
    "10. 🐢 **eglot extremely slow when doing xref-find-definitions on large file**"
    "（eglot 大文件 xref 卡顿）\n"
    "    [reddit.com/r/emacs/comments/1vdx80z]"
    "(https://www.reddit.com/r/emacs/comments/1vdx80z/eglot_extremely_slow_when_doing/)\n\n"
    "---\n\n"
    "这版清爽多了吧喵～主人对哪条感兴趣？比如 Steve Yegge 那篇长文或者 "
    "Claude Code 相关的，猫猫可以帮你抓全文来看看！(｡･ω･｡)ﾉ♡"
)


# ── _markdown_safe_cut 单元测试 ──


class TestMarkdownSafeCut:
    def test_plain_text_cuts_at_line_end(self):
        """多行普通文本：切点 ≤ limit+一行余量，且落在行尾。"""
        text = "第一行普通文本。\n第二行普通文本。\n第三行普通文本。\n" * 20
        cut = markdown_safe_cut(text, 100)
        assert 0 < cut <= 111
        assert text[cut - 1] == "\n"

    def test_single_long_line_no_cut(self):
        """单行超长（无换行）：无有效切点，交给 _split_markdown 兜底。"""
        one = "这是一段普通的文本。" * 50
        cut = markdown_safe_cut(one, 100)
        assert not (0 < cut < len(one))

    def test_table_not_split_at_limit(self):
        """limit 落在表格中间：切点退回表格前（表格整体留到下一块）。"""
        prefix = "表格测试开始。\n"
        table = "| 列1 | 列2 |\n| --- | --- |\n| a | b |\n| c | d |\n| e | f |\n| g | h |\n| i | j |\n"
        cut = markdown_safe_cut(prefix + table, len(prefix) + 20)
        assert cut == len(prefix)
        assert "|" not in (prefix + table)[:cut]

    def test_complete_table_stays_in_block(self):
        """表格完整在 limit 内：切点在表格后，表格整体在块内。"""
        prefix = "表格测试开始。\n"
        table = "| 列1 | 列2 |\n| --- | --- |\n| a | b |\n| c | d |\n| e | f |\n| g | h |\n| i | j |\n"
        text = prefix + table + "表格结束后的普通文本。" * 5
        cut = markdown_safe_cut(text, len(prefix) + len(table) + 10)
        assert cut >= len(prefix) + len(table)
        block = text[:cut]
        assert block.count("| --- |") == 1
        assert "| i | j |" in block

    def test_half_row_never_cut(self):
        """流式生成中的半截行（无换行结尾）：切点退回最后一个完整行尾。

        回归：修复前 safe 被半截行更新为 len+1，调用方不切，
        半截链接/单词整段发出（QQ 半截链接的根因）。
        """
        # 半截链接行
        t = "第一行。\n第二行。\n[链接](https://www.reddit.com/r/emacs/comments/1vel7ne/announc"
        assert markdown_safe_cut(t, 100) == len("第一行。\n第二行。\n")
        # 文本恰好 = limit 且末尾半截链接
        t2 = "第一行。\n[链接](https://example.com/abc"
        assert markdown_safe_cut(t2, len(t2)) == len("第一行。\n")
        # 半截表格行：表格整体（表头+分隔+半截行）留到下一块，切点退回表格前
        t3 = "表格测试开始。\n| 列1 | 列2 |\n| --- | --- |\n| 未完成 | 行"
        assert markdown_safe_cut(t3, len(t3) + 1) == len("表格测试开始。\n")

    def test_complete_end_no_cut(self):
        """末尾是完整行：切点 ≥ len（调用方不切，整块发出）。"""
        t = "第一行。\n第二行。\n"
        cut = markdown_safe_cut(t, len(t) + 5)
        assert not (0 < cut < len(t))

    def test_fence_body_never_cut(self):
        """limit 落在代码围栏体内：切点退回围栏前。"""
        fence_text = (
            "代码开始：\n```python\nprint('hello')\nprint('world')\n```\n继续普通文本。"
            * 3
        )
        assert markdown_safe_cut(fence_text, 30) == len("代码开始：\n")
        # 末尾半截围栏
        half_fence = "代码开始：\n```python\nprint('x')\n"
        assert markdown_safe_cut(half_fence, len(half_fence) + 1) == len("代码开始：\n")

    def test_table_continuation_cuts_at_row_end(self):
        """延续表格（initial_in_table）：数据行尾可切，半截行被切掉。"""
        tail = "| 数据01 | 7 | 完整行 |\n| 数据02 | 14 | 半截"
        cut = markdown_safe_cut(tail, 100, initial_in_table=True)
        assert cut == len("| 数据01 | 7 | 完整行 |\n")

    def test_fence_continuation(self):
        """围栏延续（initial_in_fence）：围栏体内不可切，结束行后可切。"""
        fence_cont = "print('y')\nprint('z')\n```\n普通文本继续。\n"
        cut = markdown_safe_cut(
            fence_cont, 100, initial_in_fence=True, initial_fence_marker="```"
        )
        assert cut >= len("print('y')\nprint('z')\n```\n")
        assert (
            markdown_safe_cut(
                "print('y')\nprint('z')\n",
                100,
                initial_in_fence=True,
                initial_fence_marker="```",
            )
            == 0
        )


class TestTrailingStructure:
    def test_table_state(self):
        """已发文本末尾状态：表内 / 围栏内（含 marker）。"""
        assert trailing_structure("表头\n| --- | --- |\n| 数据 | 值 |\n") == (
            True,
            False,
            None,
        )
        # 末尾换行不重置表内状态（修复：rstrip 后再 split）
        assert trailing_structure("| 数据 | 值 |\n") == (True, False, None)
        assert trailing_structure("普通文本\n```python\n") == (False, True, "```")
        assert trailing_structure("```python\nprint(1)\n```\n") == (False, False, None)
        assert trailing_structure(
            "表头\n| --- | --- |\n| 数据 | 值 |\n\n普通文本\n"
        ) == (
            False,
            False,
            None,
        )


class TestPendingStartsIncomplete:
    def test_incomplete_structures(self):
        """空闲 flush 跳过条件：表格行开头 / 围栏体开头。"""
        assert pending_starts_incomplete("| 行数据 | 值 |", "表头已发") is True
        assert pending_starts_incomplete("| --- | --- |", "表头已发") is True
        assert pending_starts_incomplete("print('x')", "```python\n") is True
        assert pending_starts_incomplete("普通文本", "") is False
        assert pending_starts_incomplete("```python\n", "") is False


# ── ToolLoop 流式集成测试 ──


class _MockCost:
    def record_turn(self, chat_id, model, usage):
        pass


class _MockCtx:
    async def add_assistant_message_async(self, *a, **k):
        pass

    async def add_tool_result_async(self, *a, **k):
        pass


class _MockSvc:
    """逐字符喂 on_text，模拟真实 SSE 流式到达；支持断流/停顿。"""

    def __init__(self, text="", throw_after=None, tail_sleep=0.0):
        self.text, self.throw_after, self.tail_sleep = text, throw_after, tail_sleep

    @property
    def model(self):
        return "mock"

    async def chat_completion_stream(
        self,
        messages,
        tools=None,
        model=None,
        temperature=None,
        max_tokens=None,
        callbacks=None,
    ):
        buf = ""
        for i, ch in enumerate(self.text):
            await asyncio.sleep(0)
            buf += ch
            if callbacks and callbacks.on_text:
                await callbacks.on_text(buf)
            if self.throw_after is not None and i >= self.throw_after:
                raise RuntimeError("boom mid-stream")
        if self.tail_sleep:
            await asyncio.sleep(self.tail_sleep)
        return AssistantMessage(content=self.text or None), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


def _make_loop(svc, block_chars=800, idle_ms=1000):
    ctx = NS(
        ai=NS(
            ai_service=svc,
            model_registry=None,
            max_tool_rounds=5,
            stream_reply=True,
            stream_block_chars=block_chars,
            stream_block_idle_ms=idle_ms,
        ),
        mgmt=NS(
            permission_manager=None,
            cost_tracker=_MockCost(),
            context_manager=_MockCtx(),
        ),
        memory=NS(hindsight_memory=None),
    )
    return ToolLoop(ctx)


async def _run_case(svc, block_chars=800, idle_ms=1000):
    loop = _make_loop(svc, block_chars, idle_ms)
    sent, streamed = [], []

    async def reply_cb(chat_id, content, message_id, is_group):
        sent.append(content)

    async def stream_cb(chunk):
        streamed.append(chunk)

    ret = await loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        chat_id="c1",
        is_group=True,
        reply_to="m1",
        reply_callback=reply_cb,
        stream_callback=stream_cb,
    )
    await asyncio.sleep(0.2)  # 给残留 timer 机会（幽灵文本检测）
    return ret, sent, streamed


def _assert_no_cut_markdown(blocks):
    """任何块不得含半截链接/半截表格行（行中间切断）。"""
    problems = []
    for i, blk in enumerate(blocks):
        for ln in blk.split("\n"):
            s = ln.strip()
            if s.startswith("[") and "](" in s and not s.rstrip().endswith(")"):
                problems.append(f"块{i} 半截链接: {s[-40:]!r}")
            if s.startswith("|") and not s.endswith("|"):
                problems.append(f"块{i} 半截表格行: {s[-40:]!r}")
    assert not problems, problems


class TestStreamBlockIntegration:
    def test_real_r_emacs_message_no_cut_links(self):
        """线上真实消息走流式 block=800：无半截链接、拼接完整。

        回归：修复前块边界切在 `1vel7ne` 链接中间，
        QQ 上显示半截链接 + 链接"重复"（块尾+块首）。
        """
        content = REAL_R_EMACS_MESSAGE
        ret, sent, streamed = asyncio.run(_run_case(_MockSvc(text=content)))
        blocks = streamed + sent

        assert "".join(blocks) == content, "拼接必须完整"
        assert len(blocks) >= 2, "长消息应切多块"
        _assert_no_cut_markdown(blocks)
        # 每条消息都以完整行结尾
        for blk in blocks:
            assert blk.endswith("\n") or blk == blocks[-1], "非末块应以换行结尾"

    def test_real_r_emacs_message_smaller_blocks(self):
        """block=500/400 同样无半截链接（多块场景）。"""
        content = REAL_R_EMACS_MESSAGE
        for bs in (500, 400):
            ret, sent, streamed = asyncio.run(
                _run_case(_MockSvc(text=content), block_chars=bs)
            )
            blocks = streamed + sent
            assert "".join(blocks) == content
            _assert_no_cut_markdown(blocks)

    def test_no_reply_silent(self):
        """静默回复（含前导空白）：不转发不发送。"""
        ret, sent, streamed = asyncio.run(_run_case(_MockSvc(text="NO_REPLY")))
        assert streamed == [] and sent == []
        ret, sent, streamed = asyncio.run(_run_case(_MockSvc(text="   NO_REPLY")))
        assert streamed == [] and sent == []

    def test_abort_after_probe_stops_loop(self):
        """探测期后断流：已转发 → 不再回退，未发出的尾巴补发（不丢结尾）。

        回归：修复前断流时尾巴（st.text[st.sent:]）静默丢弃，用户丢失回复结尾；
        修复后已发前缀 + 补发尾巴 = 已生成的全部文本，且无重复。
        """
        text = "这是一段会在探测期之后被打断的文本内容，流式转发此时已经开始工作。" * 5
        throw_after = 130
        ret, sent, streamed = asyncio.run(
            _run_case(_MockSvc(text=text, throw_after=throw_after))
        )
        assert streamed != [] and ret == (False, True)
        delivered = "".join(streamed) + "".join(sent)
        # 流在 i=throw_after 后中断：已生成 = 前 throw_after+1 字符
        assert (
            delivered == text[: throw_after + 1]
        ), "已发前缀 + 补发尾巴必须等于已生成部分，无重复无丢失"

    def test_abort_before_probe_falls_back(self):
        """探测期内断流：零转发 → 兜底消息。"""
        text = "这是一段会在探测期之内被打断的文本内容。" * 5
        ret, sent, streamed = asyncio.run(
            _run_case(_MockSvc(text=text, throw_after=50))
        )
        assert streamed == [] and sent == ["AI 服务异常"]

    def test_abort_no_ghost_text_from_orphan_timer(self):
        """timer 已安排未运行即断流：不得泄漏幽灵文本。

        回归：修复前异常路径不取消 timer，fallback 后旧 timer
        触发 flush 向用户发送第一模型的半截内容（双回复）。
        """
        text = "幽灵文本测试内容，用于验证异常后旧定时器不得泄漏。" * 5
        ret, sent, streamed = asyncio.run(
            _run_case(_MockSvc(text=text, throw_after=113))
        )
        assert streamed == [] and sent == ["AI 服务异常"]

    def test_idle_flush_after_pause(self):
        """空闲超时强制发块（未达块大小也不冷场）。"""
        text = "空闲超时测试。" * 50
        ret, sent, streamed = asyncio.run(
            _run_case(_MockSvc(text=text, tail_sleep=0.15), idle_ms=50)
        )
        assert "".join(streamed) + "".join(sent) == text
        assert len(streamed) == 2 and sent == []
