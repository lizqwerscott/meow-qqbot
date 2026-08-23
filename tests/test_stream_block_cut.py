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
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.ai.protocol import (
    AssistantMessage,
    AssistantToolCall,
    StreamAbortedError,
    StreamReset,
)
from core.markdown_split import (
    TrailingState,
    is_heading_line,
    is_lead_in_line,
    is_list_marker,
    markdown_safe_cut,
    pending_starts_incomplete,
    trailing_structure,
)
from core.tools import tool_loop as tool_loop_module
from core.tools._types import ToolResult
from core.tools.tool_loop import ToolLoop
from tests.stream_test_helpers import emit_snapshot

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
        """末尾是完整行：最后一行不设切点（宁慢勿断——它可能是条目标题，
        等下一行确认；收尾补发兜底），cut 退回倒数第二行行尾。"""
        t = "第一行。\n第二行。\n"
        cut = markdown_safe_cut(t, len(t) + 5)
        assert cut == len("第一行。\n")

    def test_lead_in_line_binds_to_following_list(self):
        """引导行（「：」结尾）绑定其后列表：行尾不设切点，首个列表项
        随引导行同块（tmp/new_chat_v2.txt 的「…只用它做：」+「- 事后分析」）。"""
        lead = "近期的研究发现：模型内部状态会影响行为。以前大家只用它做："
        # 引导行后暂停：持有
        assert markdown_safe_cut(lead, len(lead)) == 0
        # 列表第一项到达：仍持有（引导行 + 首项同块）
        one = lead + "\n- 事后分析（post-hoc analysis）"
        assert markdown_safe_cut(one, len(one)) == 0
        # 第二项到达：切在第二项起点（引导行 + 首项已同块）
        two = one + "\n- 直接输出引导（output steering）"
        cut = markdown_safe_cut(two, len(two))
        assert two[:cut] == one + "\n"

    def test_lead_in_binds_across_heading_chain(self):
        """长标题行 + 短标题（引导形态）+ 列表：前导行随同绑定，标题链不拆。"""
        title = '🧠 Emotion2Skill：让 LLM 用"内心情绪"选技能'
        sub = "📋 基本信息："
        # 长标题 + 短标题：无切点（等列表确认）
        assert markdown_safe_cut(title + "\n" + sub, len(title) + len(sub) + 1) == 0
        # 列表第一项到达：仍持有（长标题 + 短标题 + 首项同块）
        one = title + "\n" + sub + "\n- 论文：arXiv:2608.09248"
        assert markdown_safe_cut(one, len(one)) == 0

    def test_preceding_line_binds_when_next_is_lead_in(self):
        """普通行后跟引导行：前导行行尾不切（过渡句随同绑定，不孤悬块尾）。"""
        lead = "基于技能的 Agent 在选择用哪个技能时，只靠文本级信号做判断："
        trans = "但 LLM 内部自己的表征状态完全没被利用！"
        # 前导行 + 引导行：切点退回更早（不在过渡句行尾）
        text = trans + "\n" + lead
        cut = markdown_safe_cut(text, len(text))
        assert cut == 0 or not text[:cut].endswith(trans + "\n")

    def test_lead_in_inside_list_item_binds_next_marker(self):
        """列表项内（无空行分隔）的引导行：绑定其后首个列表项——
        「- 代码…」+「🎯 要解决什么问题？」+「…判断：」+「- 任务描述」
        切点落在绑定项之后的项边界，引导行不孤悬块尾。"""
        text = (
            "- 代码已开源：github.com/BoHan-LIN04/Emotion2Skill\n"
            "🎯 要解决什么问题？\n"
            "基于技能的 LLM Agent（把任务拆成技能库来调用）只靠文本级信号做判断：\n"
            "- 任务描述、口头反思、经验规则……\n"
            "但 LLM 内部自己的表征状态完全没被利用！\n"
            '💡 核心思路：给 Agent 加"直觉"\n'
            "近期的可解释性研究发现：情绪表征会因果地影响行为。以前大家只用它做：\n"
            "- 事后分析（post-hoc analysis）\n"
            "- 直接输出引导（output steering）"
        )
        cut = markdown_safe_cut(text, len(text))
        assert 0 < cut < len(text)
        assert text[:cut].endswith("- 事后分析（post-hoc analysis）\n")

    def test_heading_inside_list_item_binds_next_marker(self):
        """列表项内无空行分隔的标题形态短行（「🔧 具体做法」）：绑定其后
        首个列表项，标题不孤悬块尾。"""
        text = (
            "Emotion2Skill 的突破：把情绪表征用到 Agent 级决策上！\n"
            "🔧 具体做法\n"
            "1. 提取情绪向量：从残差流里提取 27 维情绪状态向量\n"
            "2. 置信度门控摘要：映射成带置信度门控的摘要注入路由提示"
        )
        cut = markdown_safe_cut(text, len(text))
        assert 0 < cut < len(text)
        assert text[:cut].endswith(
            "1. 提取情绪向量：从残差流里提取 27 维情绪状态向量\n"
        )

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
        cut = markdown_safe_cut(tail, 100, initial=TrailingState(in_table=True))
        assert cut == len("| 数据01 | 7 | 完整行 |\n")

    def test_fence_continuation(self):
        """围栏延续（initial_in_fence）：围栏体内不可切，结束行后可切。"""
        fence_cont = "print('y')\nprint('z')\n```\n普通文本继续。\n"
        cut = markdown_safe_cut(
            fence_cont, 100, initial=TrailingState(in_fence=True, fence_marker="```")
        )
        assert cut >= len("print('y')\nprint('z')\n```\n")
        assert (
            markdown_safe_cut(
                "print('y')\nprint('z')\n",
                100,
                initial=TrailingState(in_fence=True, fence_marker="```"),
            )
            == 0
        )


class TestTrailingStructure:
    def test_table_state(self):
        """已发文本末尾状态：表内 / 围栏内（含 marker）。"""
        assert trailing_structure(
            "表头\n| --- | --- |\n| 数据 | 值 |\n"
        ) == TrailingState(in_table=True)
        # 末尾换行不重置表内状态（修复：rstrip 后再 split）
        assert trailing_structure("| 数据 | 值 |\n") == TrailingState(in_table=True)
        assert trailing_structure("普通文本\n```python\n") == TrailingState(
            in_fence=True, fence_marker="```"
        )
        assert trailing_structure("```python\nprint(1)\n```\n") == TrailingState()
        assert (
            trailing_structure("表头\n| --- | --- |\n| 数据 | 值 |\n\n普通文本\n")
            == TrailingState()
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
    def record_turn(self, chat_id, model, usage, metadata=None):
        pass


class _MockCtx:
    async def add_assistant_message_async(self, *a, **k):
        pass

    async def add_tool_result_async(self, *a, **k):
        pass


async def _emit_reset(callbacks, reason):
    if callbacks and callbacks.on_reset:
        await callbacks.on_reset(
            StreamReset(previous_generation=0, generation=1, reason=reason)
        )


class _MockSvc:
    """逐字符喂 snapshot，模拟真实 SSE 流式到达；支持断流/停顿。"""

    def __init__(
        self, text="", throw_after=None, tail_sleep=0.0, step=0.0, pause_lines=()
    ):
        self.text, self.throw_after, self.tail_sleep, self.step = (
            text,
            throw_after,
            tail_sleep,
            step,
        )
        self.pause_lines = set(pause_lines)

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
        line_no = 0
        for i, ch in enumerate(self.text):
            if ch == "\n":
                line_no += 1
            await asyncio.sleep(self.step)
            buf += ch
            await emit_snapshot(callbacks, buf)
            if self.throw_after is not None and i >= self.throw_after:
                raise RuntimeError("boom mid-stream")
            if line_no in self.pause_lines:
                await asyncio.sleep(0.15)  # 超过 idle(1000ms) 的停顿
        if self.tail_sleep:
            await asyncio.sleep(self.tail_sleep)
        return AssistantMessage(content=self.text or None), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class _SuppressReplySvc(_MockSvc):
    def __init__(self, text):
        super().__init__(text=text)
        self.calls = 0

    async def chat_completion_stream(
        self,
        messages,
        tools=None,
        model=None,
        temperature=None,
        max_tokens=None,
        callbacks=None,
    ):
        self.calls += 1
        if self.calls > 1:
            return AssistantMessage(), {"prompt_tokens": 1, "completion_tokens": 1}

        await emit_snapshot(callbacks, self.text)
        return AssistantMessage(
            content=self.text,
            tool_calls=[
                AssistantToolCall(
                    id="call-1",
                    name="heartbeat_respond",
                    arguments='{"notify": false}',
                )
            ],
        ), {"prompt_tokens": 1, "completion_tokens": 1}


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


async def _run_case(svc, block_chars=800, idle_ms=1000, cb_sleep=0.0):
    """跑一轮 ToolLoop 并收集投递内容。

    cb_sleep>0 时 stream_callback 模拟 QQ API 发送延迟（发送在途窗口），
    用于复现 flush 在途期间的并发竞态。
    """
    loop = _make_loop(svc, block_chars, idle_ms)
    sent, streamed = [], []

    async def reply_cb(chat_id, content, message_id, is_group):
        sent.append(content)

    async def stream_cb(chunk):
        if cb_sleep:
            await asyncio.sleep(cb_sleep)  # 发送在途窗口
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

    def test_suppressed_reply_cancels_idle_flush(self, monkeypatch):
        """抑制回复的工具轮次不得遗留流式 idle timer。"""
        svc = _SuppressReplySvc("抑制回复的流式文本" * 20)
        loop = _make_loop(svc, idle_ms=50)
        streamed, sent = [], []

        async def reply_cb(chat_id, content, message_id, is_group):
            sent.append(content)

        async def stream_cb(chunk):
            streamed.append(chunk)

        async def execute_suppressed(*args, **kwargs):
            await asyncio.sleep(0.1)
            return ToolResult(content="ok", no_reply=True)

        monkeypatch.setattr(tool_loop_module, "execute_tool", execute_suppressed)
        ret = asyncio.run(
            loop.run(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                chat_id="c1",
                is_group=True,
                reply_to="m1",
                reply_callback=reply_cb,
                stream_callback=stream_cb,
            )
        )
        assert ret == (False, False)
        assert streamed == [] and sent == []

    def test_abort_after_probe_stops_loop(self):
        """探测期后断流：已转发 → 不再回退，未发出的尾巴补发（不丢结尾）。

        回归：修复前断流时尾巴（st.text[st.sent:]）静默丢弃，用户丢失回复结尾；
        修复后已发前缀 + 补发尾巴 = 已生成的全部文本，且无重复。
        """
        text = (
            "这是一段会在探测期之后被打断的文本内容，流式转发此时已经开始工作。\n" * 5
        )
        throw_after = 130
        ret, sent, streamed = asyncio.run(
            _run_case(_MockSvc(text=text, throw_after=throw_after, step=0.01))
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
        text = "空闲超时测试。\n" * 50
        ret, sent, streamed = asyncio.run(
            _run_case(_MockSvc(text=text, tail_sleep=0.15), idle_ms=50)
        )
        assert "".join(streamed) + "".join(sent) == text

    def test_heading_lead_in_never_orphaned_at_block_tail(self):
        """模型在标题/引导行后停顿：标题/引导句不得孤悬块尾（其列表同块）。

        回归：tmp/new_chat_v2.txt 事故——长引导行（「…只用它做：」>24 字符
        不满足短行标题启发式）在列表第一项到达时被切在行尾，列表掉进下一块；
        无空行分隔的「列表→标题→引导行→列表」结构同样把引导句孤悬块尾。
        修复后：引导行/标题行绑定首个列表项（含列表项内形态判定），
        任意停顿点切出的块尾行都不是「其后还有内容」的标题/引导行。
        """
        content = (
            "抓到啦主人！这篇论文挺有想法的，猫猫给你讲讲～(ฅ´ω`ฅ)📖\n"
            '🧠 Emotion2Skill：让 LLM 用"内心情绪"选技能\n'
            "📋 基本信息：\n"
            "- 论文：arXiv:2608.09248（8月10日提交，11日修订）\n"
            "- 作者：林博涵等 8 人（中国团队）\n"
            "- 代码已开源：github.com/BoHan-LIN04/Emotion2Skill\n"
            "🎯 要解决什么问题？\n"
            '基于技能的 LLM Agent（把任务拆成"技能库"里的可复用流程来调用）在选择用哪个技能时，只靠文本级信号做判断：\n'
            "- 任务描述、口头反思、经验规则……\n"
            '但 LLM 内部自己的表征状态（模型"心里"在想什么）完全没被利用！\n'
            '💡 核心思路：给 Agent 加"直觉"\n'
            '近期的可解释性研究发现：LLM 内部存在线性的"情绪表征"，并且会因果地影响行为。以前大家只用它做：\n'
            "- 事后分析（post-hoc analysis）\n"
            "- 直接输出引导（output steering）\n"
            "Emotion2Skill 的突破：把情绪表征用到 Agent 级决策上！\n"
            "🔧 具体做法\n"
            "1. 提取情绪向量：从残差流里提取 27 维情绪状态向量\n"
            "2. 置信度门控摘要：映射成带置信度门控的摘要注入路由提示\n"
            '3. 情绪轨迹分析：检测"内部状态突变"定位有问题的技能调用\n'
            "4. 定向 SOP 重写：精准重写对应的标准操作流程\n"
            "📊 效果\n"
            "在 WebShop 和 ALFWorld 基准上：\n"
            "- Emotion2Skill + Qwen3-8B vs Zero-Shot 基线：成功率提升 +26.9%！\n"
            "\n"
            '猫猫的理解：这相当于给 AI Agent 装了个"直觉系统"。\n'
            "主人觉得这个方向有意思吗？要不要猫猫再看看它的 GitHub？(ฅ´ω`ฅ)💕"
        )
        # 停顿点 = 原捕获文件的分隔位置（标题/引导行之后），覆盖三种形态：
        # 短标题（📋 基本信息：）、长引导行（…只用它做：）、列表项（4. 定向 SOP…）
        for pauses in ({3, 12, 20}, {3, 8, 16, 21}, {2, 7, 13}):
            ret, sent, streamed = asyncio.run(
                _run_case(
                    _MockSvc(text=content, pause_lines=pauses),
                    idle_ms=50,
                )
            )
            blocks = streamed + sent
            assert "".join(blocks) == content, "拼接必须完整"
            _assert_no_cut_markdown(blocks)
            for i, blk in enumerate(blocks[:-1]):
                tail = blk.rstrip("\n").split("\n")[-1]
                if is_list_marker(tail):
                    continue  # 列表项行是合法块尾（项边界可切）
                assert not is_heading_line(
                    tail, True
                ), f"块{i} 以标题行结尾，其后还有内容: {tail!r}"
                assert not is_lead_in_line(
                    tail
                ), f"块{i} 以引导行结尾，其后还有内容: {tail!r}"


# ── c1 主契约：模型链下断流回退决策（零转发→回退 / 已转发→终止） ──


class _AbortSvc:
    """流式调用立即抛 StreamAbortedError（零转发）。"""

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
        raise StreamAbortedError("connection reset")


class _ResetSvc:
    """模拟服务内部降级重试：先流半截 → on_reset → 从零开始新生成。"""

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
        # 第一次生成：半截（流中途失败前已触发回调）
        buf = ""
        generation = 0
        for ch in "半截内容会被丢弃":
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        # 服务内部重试
        await _emit_reset(callbacks, "retry")
        generation += 1
        # 第二次生成（全新）
        buf = ""
        full = "这是重试后的完整回复内容，长度足以越过静默探测期。" * 3
        for ch in full:
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        return AssistantMessage(content=buf), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class TestToolLoopAbortFallback:
    """模型链下断流的回退决策（c1 主契约，此前只有服务层测试）。"""

    def _make_registry(self, resolves):
        reg = MagicMock()
        reg.resolve_model_chain = AsyncMock(side_effect=resolves)
        reg.get = MagicMock(return_value=resolves[-1][1])
        reg.get_session_config = MagicMock(return_value=(30, 600))
        reg.cooldown_manager = MagicMock(
            is_cooled_down=AsyncMock(return_value=False),
            record_success=AsyncMock(),
            record_failure=AsyncMock(),
        )
        return reg

    def _run_chain(self, svc_a, svc_b, sent, streamed):
        reg = self._make_registry([("a", svc_a), ("b", svc_b)])
        ctx = NS(
            ai=NS(
                ai_service=svc_a,
                model_registry=reg,
                max_tool_rounds=5,
                stream_reply=True,
                stream_block_chars=800,
                stream_block_idle_ms=1000,
            ),
            mgmt=NS(
                permission_manager=None,
                cost_tracker=_MockCost(),
                context_manager=_MockCtx(),
            ),
            memory=NS(hindsight_memory=None),
        )
        loop = ToolLoop(ctx)

        async def reply_cb(chat_id, content, message_id, is_group):
            sent.append(content)

        async def stream_cb(chunk):
            streamed.append(chunk)

        async def run():
            return await loop.run(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                chat_id="c1",
                is_group=True,
                reply_to="m1",
                reply_callback=reply_cb,
                stream_callback=stream_cb,
                model_chain=["a", "b"],
            )

        return asyncio.run(run()), reg

    def test_abort_before_forward_falls_back_in_chain(self):
        """零转发断流 → 回退下一模型：兜底回复投递，失败模型记冷却。"""
        svc_a, svc_b = _AbortSvc(), _MockSvc(text="兜底模型的完整回复内容")
        sent, streamed = [], []
        ret, _ = self._run_chain(svc_a, svc_b, sent, streamed)
        assert streamed == []
        assert sent == ["兜底模型的完整回复内容"], "回退模型必须完成投递"
        assert ret == (False, True)

    def test_abort_before_forward_records_cooldown(self):
        """断流（异常）→ record_cooldown=True：失败模型写入冷却。"""
        svc_a, svc_b = _AbortSvc(), _MockSvc(text="x" * 20)
        sent, streamed = [], []
        _, reg = self._run_chain(svc_a, svc_b, sent, streamed)
        reg.cooldown_manager.record_failure.assert_awaited_once_with("a")
        reg.cooldown_manager.record_success.assert_awaited_once_with("b")

    def test_abort_after_forward_terminates_no_fallback(self):
        """已转发部分文本后断流 → 终止：不回退、尾巴补发、记冷却。"""
        text = (
            "这是一段会在探测期之后被打断的文本内容，流式转发此时已经开始工作。\n" * 5
        )
        svc_a = _MockSvc(text=text, throw_after=130, step=0.01)
        svc_b = _MockSvc(text="不应被调用的模型B")
        sent, streamed = [], []
        _, reg = self._run_chain(svc_a, svc_b, sent, streamed)
        # 只 acquire 过一次（首模型），回退模型从未被触碰
        reg.resolve_model_chain.assert_awaited_once()
        reg.cooldown_manager.record_failure.assert_awaited_once_with("a")
        reg.cooldown_manager.record_success.assert_not_awaited()
        # 已发前缀 + 补发尾巴 = 已生成部分
        delivered = "".join(streamed) + "".join(sent)
        assert delivered == text[:131], "尾巴必须补发，且无重复"


class _ReviseSvc:
    """模拟 DeepSeek 修订输出：长草稿（超探测期，已被转发）→ on_reset → 终稿。"""

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
        # 长草稿：> 800 块大小，达块强制转发（无法撤回）
        draft = "这是很长很长的草稿内容，主人听我说。\n" * 50
        buf = ""
        generation = 0
        for ch in draft:
            await asyncio.sleep(0)  # 让出事件循环，空闲 flush 定时器才能跑
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        # 模型修订：触发 on_reset（真实 DeepSeek 服务在终稿 message item 时触发）
        await _emit_reset(callbacks, "provider_revision")
        generation += 1
        # 终稿：全新生成
        buf = ""
        final = "这是终稿的完整回复内容，主人下班辛苦了好好休息。" * 5
        for ch in final:
            await asyncio.sleep(0)
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        return AssistantMessage(content=buf), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class TestToolLoopStreamReset:
    """服务内部重试（on_reset）：新文本从 0 偏移转发，半截旧文本不混入。"""

    def test_retry_restarts_forward_offset(self):
        ret, sent, streamed = asyncio.run(_run_case(_ResetSvc()))
        delivered = "".join(streamed) + "".join(sent)
        full = "这是重试后的完整回复内容，长度足以越过静默探测期。" * 3
        assert delivered == full, "重试文本必须从 0 偏移完整投递"
        assert "半截内容" not in delivered

    def test_revision_after_forwarded_draft_delivers_full_final(self):
        """草稿已被转发后的修订：终稿必须从头完整投递（不得从旧偏移切片）。

        回归（第二轮审查）：旧实现清缓冲但不触发 on_reset，st.sent 停在草稿
        长度 → 终稿开头被跳过/整条被吞，双气泡症状在长草稿下依旧存在。
        """
        ret, sent, streamed = asyncio.run(_run_case(_ReviseSvc()))
        delivered = "".join(streamed) + "".join(sent)
        final = "这是终稿的完整回复内容，主人下班辛苦了好好休息。" * 5
        assert streamed != [], "长草稿应先被转发"
        assert delivered.endswith(final), "终稿必须完整收尾，开头不得被跳过"
        sentence = "这是终稿的完整回复内容，主人下班辛苦了好好休息。"
        assert delivered.count(sentence) == 5, "终稿不得重复"


# ── 重复气泡回归（18:59 线上事故）：flush 发送在途时的并发捕获 ──


class _SteppedStreamSvc:
    """逐字符流式服务（可指定字符步进间隔），配合 _run_case 的 cb_sleep
    复现「发送在途时 snapshot / reset 并发」的竞态窗口。
    """

    def __init__(self, text: str, step: float = 0.002):
        self.text = text
        self.step = step

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
        generation = 0
        for ch in self.text:
            await asyncio.sleep(self.step)
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        return AssistantMessage(content=buf), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class _SteppedReviseSvc(_SteppedStreamSvc):
    """长草稿流式到一半（flush 发送在途）触发修订 → 终稿从零重流。"""

    def __init__(self, draft: str, final: str, step: float = 0.002):
        super().__init__(draft, step)
        self.final = final

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
        generation = 0
        for ch in self.text:  # 草稿（基类 text）
            await asyncio.sleep(self.step)
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        await _emit_reset(callbacks, "provider_revision")
        generation += 1
        buf = ""
        for ch in self.final:
            await asyncio.sleep(self.step)
            buf += ch
            await emit_snapshot(callbacks, buf, generation)
        return AssistantMessage(content=buf), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class _PauseFlushSvc:
    """流式到触发首块 flush 后显式停顿：发送在途期间第二个空闲定时器
    必然触发（复现 18:59 重复气泡的确定性条件：flush 在途 + 再次调度）。"""

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
        line = "这是第一段完整内容，主人下班辛苦啦。\n"
        first = line * 18  # 216 字符 > 112 探测期，触发首块 flush
        buf = ""
        for ch in first:
            await asyncio.sleep(0.001)
            buf += ch
            await emit_snapshot(callbacks, buf)
        # 停顿：首块 flush 的发送（0.2s）在途期间，空闲定时器（50ms）
        # 必然触发第二次 flush —— 旧实现以旧偏移二次捕获同一段文本
        await asyncio.sleep(0.3)
        rest = "第二段收尾内容，猫猫陪你休息。\n"
        buf = ""
        for ch in rest:
            await asyncio.sleep(0.001)
            buf += ch
            await emit_snapshot(callbacks, buf)
        return AssistantMessage(content=first + rest), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class _ConcurrentResetSvc:
    """空闲 flush 发送在途时触发 reset，再发送新 generation。"""

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
        draft = "草稿内容，不能推进终稿游标。\n" * 10
        await emit_snapshot(callbacks, draft, 0)
        # idle=50ms 的 flush 获得运行机会，并在 stream_callback 中等待。
        await asyncio.sleep(0.08)
        await _emit_reset(callbacks, "provider_revision")
        final = "终稿从零开始，不能被旧 generation 吞掉。" * 4
        await emit_snapshot(callbacks, final, 1)
        return AssistantMessage(content=final), {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


class TestToolLoopFlushRace:
    """flush 发送在途时的并发竞态：同一段文本不得被二次捕获（重复气泡）。"""

    def test_second_idle_flush_never_recaptures_inflight_segment(self):
        """18:59 重复气泡事故回归。

        旧实现发送完成才推进 st.sent；发送在途时 snapshot 会再次调度空闲
        定时器，新 flush 以旧偏移二次捕获同一段文本（逐字节相同的重复
        气泡，用户看到 [P1][P2][P2][P3]，而上下文历史只有一份）。
        发送前推进后，任何并发 snapshot 看到的偏移都已包含在途块，
        同一段不可能再被捕获。
        """
        line = "这是第一段完整内容，主人下班辛苦啦。\n"
        svc = _PauseFlushSvc()
        ret, sent, streamed = asyncio.run(_run_case(svc, idle_ms=50, cb_sleep=0.2))
        delivered = "".join(streamed) + "".join(sent)
        assert streamed, "长文本应先被转发"
        assert delivered.count(line) == 18, (
            f"第一段不得被二次捕获（重复气泡事故）：出现 {delivered.count(line)} 次，"
            f"delivered={delivered!r}"
        )

    def test_reset_during_inflight_flush_keeps_final_complete(self):
        """修订（on_reset）在 flush 发送在途时到达：终稿必须从头完整投递。

        旧实现发送后才推进 st.sent，on_reset 把 sent 清零后再 += 在途块
        长度（0 + len(pending)）→ 偏移损坏 → 终稿开头被跳过/整条被吞。
        """
        draft = "这是草稿内容。\n" * 120  # 840 字符：达块强制转发，首块 flush
        # 发送（0.25s）在途窗口充足（旧 12ms 太窄；短草稿从不转发）
        final = "这是终稿内容，主人辛苦啦。" * 8  # 104 字符，全程 < 探测期
        svc = _SteppedReviseSvc(draft=draft, final=final, step=0.002)
        ret, sent, streamed = asyncio.run(_run_case(svc, idle_ms=1000, cb_sleep=0.25))
        delivered = "".join(streamed) + "".join(sent)
        assert streamed, "长草稿应先被转发"
        assert delivered.endswith(final), (
            f"修订后的终稿必须从头完整投递（不得被旧偏移吞掉）："
            f"delivered={delivered!r}"
        )

    def test_reset_replaces_generation_while_idle_flush_is_inflight(self):
        """旧 generation 的在途 flush 不得推进终稿 generation 的游标。"""
        ret, sent, streamed = asyncio.run(
            _run_case(_ConcurrentResetSvc(), idle_ms=50, cb_sleep=0.25)
        )
        final = "终稿从零开始，不能被旧 generation 吞掉。" * 4
        delivered = "".join(streamed) + "".join(sent)
        assert streamed, "旧 generation 应已开始发送"
        assert delivered.endswith(final)
        assert delivered.count(final) == 1
