"""验证流式块切分不再拆散列表项（终稿标记格式 + 草稿无标记格式）。

背景：tmp/chat.txt 中列表项被拆到两个气泡（如 Tramp 项标题+URL 在块3、
"——"注解在块4；乐子帖第3条被收尾句隔开）。修复：
1. flush_lock 串行化 → 顺序不乱（dev/test_stream_order.py 验证）
2. markdown_safe_cut 列表状态机 → 切点只落在列表项边界/空行，
   「——」注解行与裸 URL 行永远不与其条目拆开
3. pending_starts_incomplete → 空闲 flush 不以列表续行开头

本脚本在 chat.txt 实际断点位置强制停顿，检查每个气泡边界：
- 拼接还原 == 原文
- 任何气泡不以「——」注解或裸 URL 开头
- 标记格式下，切点不落在列表项内部（标记行+续行+注解 必须同气泡）
- 草稿格式下，条目标题后停顿不会让标题单独成块（标题+URL+注解整体）
- 任何气泡不以标题行结尾（标题与其列表同气泡，不孤悬块尾）

另含：断流补发（force）路径、split_markdown 超长列表切分、标题跨空行绑定、
后置处理约束（短注解不迁移 / 3600 字节上限不破）测试。

用法: PYTHONPATH=. uv run python dev/test_list_split.py
"""

import asyncio
from types import SimpleNamespace as NS

from core.ai.protocol import AssistantMessage
from core.markdown_split import (
    is_annotation_continuation,
    is_continuation_line,
    is_fence_line,
    is_heading_line,
    is_list_marker,
    is_table_row,
    is_table_separator,
    is_url_line,
    split_markdown,
)
from core.tools.tool_loop import ToolLoop
from tests.stream_test_helpers import emit_snapshot

# ── 测试数据内嵌（不依赖 tmp/，测试自包含）──
# FINAL_TEXT = 事故当次回复的终稿（jsonl 第 365 条，1881 字符，标记格式）
# DRAFT_TEXT = 用户捕获到的流式气泡拼接（tmp/chat.txt，1306 字符，无标记格式）
FINAL_TEXT = r"""来啦主人！这是 **8月10日~12日** 的最新 r/emacs 帖子，猫猫总结好了～(ฅ´ω`ฅ)📋

**⭐ 值得一看的：**

1. **Fortnightly Tips, Tricks, and Questions — 2026-08-11 / week 32** 📌
   [reddit.com/r/emacs/comments/1vl5lkv](https://www.reddit.com/r/emacs/comments/1vl5lkv/fortnightly_tips_tricks_and_questions_20260811/)
   —— *r/emacs 双周精华帖，各种小技巧大杂烩，每次都值得翻！*

2. **Tramp for Homelab?** 🏠
   [reddit.com/r/emacs/comments/1vlq9tu](https://www.reddit.com/r/emacs/comments/1vlq9tu/tramp_for_homelab/)
   —— *用 TRAMP 管理自建服务器（主人也有 homelab，可以看看大家怎么玩的！）*

3. **gptel-agent users: how are you using the web search tool?** 🤖
   [reddit.com/r/emacs/comments/1vll30j](https://www.reddit.com/r/emacs/comments/1vll30j/gptelagent_users_how_are_you_using_the_web_search/)
   —— *gptel-agent 的 web search 用法交流（AI 相关的 Emacs 玩法）*

4. **What I learned about widget.el while building TextUI 0.2–0.5** 📖
   [reddit.com/r/emacs/comments/1vlcqtu](https://www.reddit.com/r/emacs/comments/1vlcqtu/what_i_learned_about_widgetel_while_building/)
   —— *作者用 widget.el 做 TextUI 的心得分享，技术含量高*

5. **Continuous clocking across days** ⏱️
   [reddit.com/r/emacs/comments/1vlrw2x](https://www.reddit.com/r/emacs/comments/1vlrw2x/continuous_clocking_across_days/)
   —— *Org-mode 跨天计时问题（时间管理用户会关心）*

**😂 乐子帖：**

6. **How do I quit vim?** —— *在 r/emacs 问怎么退出 vim，这波是挑衅还是求救哈哈哈*
7. **You understand my pain** —— *"你们懂我的痛"共鸣帖*
8. **It have been such a long time since i have a good experience updating my code editor** —— *好久没体验过"更新编辑器是件好事"了（懂的都懂）*

**🔧 实用求助/讨论：**
- LSP-Mode and eldoc?
- Workflows hard to reproduce outside Emacs?
- Indentation on emacs?（nvim 转来的对齐问题）
- Rust-analyzer with eglot and large codebases（大项目性能）
- [eaf] ebook-viewer（EAF 电子书阅读器，支持 Calibre）
- Displaying an input language indicator in mode-line（doom-modeline 输入法指示器）
- Question about ivy/counsel-fzf

---

主人对哪个感兴趣？猫猫可以展开讲讲！比如那个 **Tramp for Homelab**，感觉跟主人的 192.168.100.203 会很搭喵～(ฅ´ω`ฅ)💕"""

DRAFT_TEXT = r"""好嘞主人！猫猫拉一下最新的 r/emacs 帖子～(ฅ´ω`ฅ)📡来啦主人！这是 8月10日~12日 的最新 r/emacs 帖子，猫猫总结好了～(ฅ´ω`ฅ)📋
⭐ 值得一看的：Fortnightly Tips, Tricks, and Questions — 2026-08-11 / week 32 📌
reddit.com/r/emacs/comments/1vl5lkv
—— r/emacs 双周精华帖，各种小技巧大杂烩，每次都值得翻！
Tramp for Homelab? 🏠
reddit.com/r/emacs/comments/1vlq9tu—— 用 TRAMP 管理自建服务器（主人也有 homelab，可以看看大家怎么玩的！）
gptel-agent users: how are you using the web search tool? 🤖
reddit.com/r/emacs/comments/1vll30j
—— gptel-agent 的 web search 用法交流（AI 相关的 Emacs 玩法）
What I learned about widget.el while building TextUI 0.2–0.5 📖
reddit.com/r/emacs/comments/1vlcqtu
—— 作者用 widget.el 做 TextUI 的心得分享，技术含量高Continuous clocking across days ⏱️
reddit.com/r/emacs/comments/1vlrw2x
—— Org-mode 跨天计时问题（时间管理用户会关心）
😂 乐子帖：
How do I quit vim? —— 在 r/emacs 问怎么退出 vim，这波是挑衅还是求救哈哈哈
You understand my pain —— "你们懂我的痛"共鸣帖主人对哪个感兴趣？猫猫可以展开讲讲！比如那个 Tramp for Homelab，感觉跟主人的 192.168.100.203 会很搭喵～(ฅ´ω`ฅ)💕It have been such a long time since i have a good experience updating my code editor —— 好久没体验过"更新编辑器是件好事"了（懂的都懂）
🔧 实用求助/讨论：
LSP-Mode and eldoc?
Workflows hard to reproduce outside Emacs?
Indentation on emacs?（nvim 转来的对齐问题）
Rust-analyzer with eglot and large codebases（大项目性能）
[eaf] ebook-viewer（EAF 电子书阅读器，支持 Calibre）
Displaying an input language indicator in mode-line（doom-modeline 输入法指示器）
Question about ivy/counsel-fzf"""

FINAL = FINAL_TEXT
DRAFT = DRAFT_TEXT

# 无标记格式 + 段落级标题（tmp/new_chat.txt 机器之心+arXiv 段，不含开场气泡）：
# 📄 标题后停顿（流式增量只到第一行条目）是本次修复的事故断点——
# 标题行不得孤悬块尾（标题与其列表同气泡）。
HEADING_TEXT = (
    "🧠 机器之心（今日 3 篇）\n"
    "🎓 2026云程奖启动！我们要给「在校AI本硕博」发奖学金\n"
    "jiqizhixin.com/articles/2026-08-12-7\n"
    "—— 机器之心自己的奖学金项目，面向在校 AI 本硕博\n"
    "🐧 终于！Linux用户等来了官方ChatGPT桌面版\n"
    "jiqizhixin.com/articles/2026-08-12-6\n"
    "—— ChatGPT 官方桌面版终于上 Linux 了！（用 Emacs 的 Linux 用户狂喜？）\n"
    "⚖️ Agentic RL 后训练资源怎么分？港中文、恒生大学提出 Libra，吞吐最高提升 3 倍\n"
    "jiqizhixin.com/articles/2026-08-12-5\n"
    "—— 港中文等提出 Libra，优化 Agentic RL 后训练资源分配，吞吐最高 3 倍\n"
    "📄 arXiv cs.AI（今日更新）\n"
    "TongGuOCR：面向中国历史文献的布局感知 OCR 多模态大模型\n"
    "Thought-Level Beam Search for Reasoning：思维级束搜索提升推理\n"
    "StructReward：结构化过程奖励，用于多模态推理自纠错\n"
    "SDDBMs：软去噪扩散桥模型（生成模型方向）\n"
    "ForestBench：多智能体协作评估的统一图框架\n"
    "Emotion2Skill：模型内部情绪信号驱动技能选择（挺有意思的方向！）\n"
    "Entropy-based Code Adversarial Translation：基于熵的代码对抗翻译，用于真实仓库迁移\n"
    "主人对哪篇感兴趣？猫猫可以深挖摘要！比如 ChatGPT Linux 桌面版或者 Libra 都挺热的喵～(ฅ´ω`ฅ)💕"
)


class MockCost:
    def record_turn(self, chat_id, model, usage):
        pass


class MockCtx:
    async def add_assistant_message_async(self, *a, **k):
        pass


class MockSvc:
    """逐字符喂 snapshot；在 pause_at（字符下标）处停顿 > idle_ms，模拟模型节奏。"""

    def __init__(self, text, pause_at):
        self.text = text
        self.pause_at = set(pause_at)

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
            if i in self.pause_at:
                await asyncio.sleep(1.5)  # > stream_block_idle_ms(1000)
            buf += ch
            await emit_snapshot(callbacks, buf)
        return AssistantMessage(content=self.text or None), None


class Capture:
    """按投递顺序捕获流式块与收尾补发（锁保证顺序）。"""

    def __init__(self):
        self.received: list[str] = []

    async def stream(self, chunk):
        self.received.append(chunk)

    async def tail(self, chat_id, content, message_id, is_group):
        if content:
            self.received.append(content)


def line_ends(text):
    """每行的行尾下标（含换行）。"""
    ends, pos = [], 0
    for line in text.split("\n"):
        pos += len(line) + 1
        ends.append(pos)
    return ends


def protected_regions(text):
    """标记格式：每个列表项 = 标记行 + 后续续行（直到空行/新标记/文本尾）。

    返回 (start, end) 区间列表（字符下标，含行尾换行）。
    """
    lines = text.split("\n")
    regions = []
    starts = []  # (行起始下标, 行结束下标)
    pos = 0
    for line in lines:
        starts.append((pos, pos + len(line) + 1))
        pos += len(line) + 1
    i = 0
    while i < len(lines):
        if is_list_marker(lines[i]):
            s = starts[i][0]
            e = starts[i][1]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not is_list_marker(lines[j]):
                e = starts[j][1]
                j += 1
            regions.append((s, e))
            i = j
        else:
            i += 1
    return regions


async def run_case(label, text, pause_at):
    ctx = NS(
        ai=NS(
            ai_service=MockSvc(text, pause_at),
            model_registry=None,
            max_tool_rounds=5,
            stream_reply=True,
            stream_block_chars=800,
            stream_block_idle_ms=1000,
        ),
        mgmt=NS(
            permission_manager=None,
            cost_tracker=MockCost(),
            context_manager=MockCtx(),
        ),
        memory=NS(hindsight_memory=None),
    )
    cap = Capture()
    loop = ToolLoop(ctx)
    await loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        chat_id="c1",
        is_group=True,
        reply_to="m1",
        reply_callback=cap.tail,
        stream_callback=cap.stream,
    )
    return cap.received


def verify(label, text, blocks):
    print(f"=== {label} ===")
    print(f"  共 {len(blocks)} 块")
    for i, b in enumerate(blocks):
        first = b.split("\n", 1)[0].strip()
        last = b.rstrip("\n").split("\n")[-1].strip()
        print(
            f"    [{i+1}] {len(b):4d}字符 | 首行 {first[:22]!r} | 末行 {last[-22:]!r}"
        )

    joined = "".join(blocks)
    ok = True
    if joined != text:
        print(f"  ❌ 拼接不完整: {len(joined)} != {len(text)}")
        return False

    # 边界位置（前一块结尾）
    boundaries = []
    pos = 0
    for b in blocks[:-1]:
        pos += len(b)
        boundaries.append(pos)

    # 1) 任何块不以「——」注解开头（注解不脱离其条目）
    for i, b in enumerate(blocks):
        first = b.split("\n", 1)[0].strip()
        if is_annotation_continuation(first):
            print(f"  ❌ 块{i+1} 以注解开头: {first[:30]!r}")
            ok = False

    # 1b) 任何块不以裸 URL 开头（URL 是上一行条目的续行，不脱离标题）
    for i, b in enumerate(blocks):
        first = b.split("\n", 1)[0].strip()
        if is_url_line(first):
            print(f"  ❌ 块{i+1} 以裸 URL 开头: {first[:36]!r}")
            ok = False

    # 2) 标记格式：切点不落在列表项内部
    regions = protected_regions(text)
    for p in boundaries:
        for s, e in regions:
            if s < p < e:
                prev = text[:p].rstrip("\n").split("\n")[-1]
                print(f"  ❌ 切点 {p} 落在列表项内部（{s},{e}）: 前一行 {prev[:30]!r}")
                ok = False

    # 3) 任何块不以标题行结尾（标题与其列表同气泡，不孤悬块尾）
    def _is_trailing_heading(block):
        lines = block.rstrip("\n").split("\n")
        if not lines or not lines[-1].strip():
            return False
        last = lines[-1].strip()
        if (
            is_annotation_continuation(last)
            or is_url_line(last)
            or is_list_marker(last)
        ):
            return False
        prev = lines[-2].strip() if len(lines) >= 2 else ""
        prev_boundary = not prev or bool(prev and not is_continuation_line(prev))
        return is_heading_line(last, prev_boundary)

    for i, b in enumerate(blocks):
        if _is_trailing_heading(b):
            print(
                f"  ❌ 块{i+1} 以标题行结尾（标题孤悬块尾）: "
                f"{b.rstrip().splitlines()[-1]!r}"
            )
            ok = False

    if ok:
        print("  ✅ 顺序完整、无注解孤儿、列表项整体未拆、标题不孤悬块尾")
    return ok


async def test_stream_abort_tail_force():
    """断流补发（flush_incomplete=True）：生成到列表项中间断流。

    已转发的块（含安全切点）+ force 补发块（无视结构完整性）拼接后必须
    == 断流点前的全文；且不得死循环（有超时兜底）。
    """
    ABORT_AT = 250  # 项1 的 URL 中间

    class AbortSvc:
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
            for i, ch in enumerate(FINAL):
                if i == 200:
                    await asyncio.sleep(1.5)  # 让首块先转发
                if i == ABORT_AT:
                    raise RuntimeError("模拟断流")
                buf += ch
                await emit_snapshot(callbacks, buf)

    ctx = NS(
        ai=NS(
            ai_service=AbortSvc(),
            model_registry=None,
            max_tool_rounds=5,
            stream_reply=True,
            stream_block_chars=800,
            stream_block_idle_ms=1000,
        ),
        mgmt=NS(
            permission_manager=None,
            cost_tracker=MockCost(),
            context_manager=MockCtx(),
        ),
        memory=NS(hindsight_memory=None),
    )
    cap = Capture()
    await asyncio.wait_for(
        ToolLoop(ctx).run(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            chat_id="c1",
            is_group=True,
            reply_to="m1",
            reply_callback=cap.tail,
            stream_callback=cap.stream,
        ),
        timeout=15,
    )
    joined = "".join(cap.received)
    ok = joined == FINAL[:ABORT_AT]
    print(f"=== 断流补发（断流点@{ABORT_AT}） ===")
    print(f"  共 {len(cap.received)} 块")
    for i, b in enumerate(cap.received):
        print(f"    [{i+1}] {len(b):4d}字符 | {b[:28]!r} ... {b[-24:]!r}")
    if ok:
        print("  ✅ 已转发 + force 补发 == 断流点前全文，无死循环")
    else:
        print(f"  ❌ 补发不完整: {len(joined)} != {ABORT_AT}")
    assert ok


def test_split_markdown_big_list() -> bool:
    """split_markdown 超长标记列表（>3600 字节）：切块不拆列表项。

    块边界可能丢失行尾换行（split_markdown 既有行为，markdown 渲染等价），
    因此断言改为：每个列表项（标记行+续行）完整落在某个块内、不跨块。
    """
    big = "\n".join(
        f"{i+1}. **项目 {i+1}** —— 这是一段比较长的说明文字，用来撑大段落体积。\n"
        f"   [链接](https://example.com/item{i+1}/path/to/some/long/url/with/more/segments)"
        for i in range(40)
    )
    chunks = split_markdown(big)
    ok = True
    # 逐项提取：标记行 + 后续非标记行（直到下一个标记）
    lines = big.split("\n")
    items, cur = [], []
    for ln in lines:
        if is_list_marker(ln):
            if cur:
                items.append("\n".join(cur))
            cur = [ln]
        elif cur:
            cur.append(ln)
    if cur:
        items.append("\n".join(cur))
    joined = "".join(chunks)
    missing = [it for it in items if not any(it in c for c in chunks)]
    if missing:
        ok = False
        for it in missing[:3]:
            print(f"  ❌ 列表项跨块/丢失: {it[:40]!r}")
    # 块首必须是完整列表项（无续行孤儿）
    for c in chunks[1:]:
        first = c.split("\n", 1)[0]
        if not is_list_marker(first):
            print(f"  ❌ 块以续行开头: {first[:40]!r}")
            ok = False
    print(f"=== split_markdown 超长列表（{len(big.encode())} 字节） ===")
    print(f"  共 {len(chunks)} 块, {len(items)} 个列表项")
    for i, c in enumerate(chunks):
        print(
            f"    [{i+1}] {len(c.encode('utf-8')):4d}字节 | 首行 {c.split(chr(10))[0][:24]!r}"
        )
    print("  ✅ 所有列表项完整、块首均为完整列表项" if ok else "  ❌ 列表项被拆散")
    assert ok


async def test_stream_hardcut_oversized():
    """字节上限硬切（>3600 字节、无安全切点）分支测试。

    场景 1：单行无换行超长文本 → 无换行可用 → utf8_prefix 按 UTF-8 安全
    字节边界切（不裂中文/emoji，拼接可还原）。
    场景 2：超限列表项（标记行 + 4000 字符 URL 续行）→ 先按行尾硬切，
    剩余无换行部分再走 utf8_prefix。
    断言：拼接 == 原文、每个流式块 ≤ 3600 字节、无死循环。
    """
    huge_line = "超长单行内容测试，没有换行的超长文本段落。" * 400  # ~22KB 无换行
    big_item = "1. **超长列表项** —— 说明\n" + "https://example.com/" + "x" * 4000
    ok = True
    for label, text in (("单行无换行", huge_line), ("超限列表项", big_item)):
        ctx = NS(
            ai=NS(
                ai_service=MockSvc(text, pause_at=set()),
                model_registry=None,
                max_tool_rounds=5,
                stream_reply=True,
                stream_block_chars=800,
                stream_block_idle_ms=1000,
            ),
            mgmt=NS(
                permission_manager=None,
                cost_tracker=MockCost(),
                context_manager=MockCtx(),
            ),
            memory=NS(hindsight_memory=None),
        )
        cap = Capture()
        await asyncio.wait_for(
            ToolLoop(ctx).run(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                chat_id="c1",
                is_group=True,
                reply_to="m1",
                reply_callback=cap.tail,
                stream_callback=cap.stream,
            ),
            timeout=15,
        )
        blocks = cap.received
        joined = "".join(blocks)
        sizes = [len(b.encode("utf-8")) for b in blocks]
        ok_j = joined == text
        ok_b = all(s <= 3600 for s in sizes)
        ok = ok and ok_j and ok_b
        print(f"=== 硬切·{label}（{len(text.encode('utf-8'))} 字节） ===")
        print(f"  共 {len(blocks)} 块, 块字节数={sizes}")
        if ok_j and ok_b:
            print("  ✅ 拼接完整（UTF-8 安全）、每块 ≤ 3600 字节、无死循环")
        else:
            print(
                f"  ❌ 拼接={'OK' if ok_j else 'FAIL'} 字节上限={'OK' if ok_b else 'FAIL'}"
            )
    assert ok


def test_predicates() -> bool:
    """谓词边界行为（锁定三项修复的取舍）。"""
    ok = True
    # 1) 单/双 em-dash 都视为注解续行（单「—」注解不再成孤儿）
    assert is_annotation_continuation("— 单破折号注解")
    assert is_annotation_continuation("—— 双破折号注解")
    assert not is_annotation_continuation("普通行")
    # 2) 裸 IP 行不是 URL（普通文本）；带端口/路径的才是
    assert not is_url_line("192.168.100.203")
    assert is_url_line("192.168.100.203:8050")
    assert is_url_line("192.168.100.203/status")
    assert is_url_line("reddit.com/r/emacs/comments/1vl5lkv")
    assert is_url_line("https://www.reddit.com/r/emacs/")
    assert not is_url_line("192.168.100.203 会很搭喵～")
    # 3) split_markdown 超限嵌套列表：缩进保留（strip("\n") 而非 strip()）
    from core.markdown_split import split_markdown

    nested = "\n".join(
        ["- 顶层项 A", "  - 嵌套子项 a", "  - 嵌套子项 b", "- 顶层项 B"] * 30
    )
    chunks = split_markdown(nested)
    joined = "".join(chunks)
    if "  - 嵌套子项 a" not in joined:
        print("  ❌ 嵌套列表缩进丢失")
        ok = False
    # 4) 标题行判定：ATX 无条件；短行需「前一行不是续行」；句末标点/围栏/表行排除
    assert is_heading_line("## 每日推荐")
    assert is_heading_line("📄 arXiv cs.AI（今日更新）", after_boundary=True)
    assert is_heading_line("🧠 机器之心（今日 3 篇）", after_boundary=True)
    assert is_heading_line("🔧 实用求助/讨论：", after_boundary=True)
    assert not is_heading_line(
        "TongGuOCR：面向中国历史文献的布局感知 OCR 多模态大模型", after_boundary=True
    )  # 太长
    assert not is_heading_line(
        "ForestBench：多智能体协作评估的统一图框架"
    )  # 短但前一行是续行
    assert not is_heading_line("明白了！", after_boundary=True)  # 句末标点收尾
    assert not is_heading_line("```")
    assert not is_heading_line("| a | b |", after_boundary=True)
    # 5) 注解/URL 续行行内排除（短注解不被当标题移走；URL 不构成标题）
    assert not is_heading_line("—— 不错", after_boundary=True)
    assert not is_heading_line("a.co/x", after_boundary=True)
    assert is_continuation_line("reddit.com/r/emacs/comments/1vl5lkv")
    assert is_continuation_line("| a | b |")
    assert not is_continuation_line("普通内容行")
    print("=== 谓词边界 ===")
    print(
        "  ✅ 单/双破折号注解、裸 IP vs 带端口/路径 IP、嵌套缩进、标题行/续行判定 全部符合预期"
        if ok
        else "  ❌ 有断言失败"
    )
    assert ok


def test_split_markdown_table_no_rows() -> bool:
    """表头+分隔行而无数据行的表格（2 行）：回归文本行，不丢弃（宁断勿丢）。

    模糊测试发现：`| a | b |` + `| --- | --- |` 后接非表行时，_flush_table
    对不足 3 行的表格直接 return，表头两行凭空消失（丢内容）。
    """
    ok = True
    # 中段：2 行表后接普通内容
    t1 = "\n".join(
        ["列表内容填充" * 30] * 10
        + ["| a | b |", "| --- | --- |", "后面还有内容", "- 条目 X"]
        + ["列表内容填充" * 30] * 10
    )
    c1 = split_markdown(t1)
    flat1 = []
    for c in c1:
        parts = c.split("\n")
        if parts and parts[-1] == "":
            parts.pop()
        flat1.extend(parts)
    if flat1 != t1.split("\n"):
        print(f"  ❌ 中段 2 行表丢失: {flat1 != t1.split(chr(10))}")
        ok = False
    # 尾部：文本以 2 行表结尾（EOF 路径）——表行恢复为文本行，
    # 以段落边界（\n\n）衔接（非空行内容保全即可，空行插入无害）
    t2 = "\n".join(["列表内容填充" * 30] * 10 + ["| a | b |", "| --- | --- |"])
    c2 = split_markdown(t2)
    flat2 = []
    for c in c2:
        parts = c.split("\n")
        if parts and parts[-1] == "":
            parts.pop()
        flat2.extend(parts)
    nb2 = [ln for ln in flat2 if ln.strip()]
    if nb2 != [ln for ln in t2.split("\n") if ln.strip()]:
        print("  ❌ EOF 2 行表丢失")
        ok = False
    # 正常 3 行表不受影响（仍按表格整体保留）
    t3 = "\n".join(
        ["列表内容填充" * 30] * 10 + ["| a | b |", "| --- | --- |", "| 1 | 2 |"]
    )
    c3 = split_markdown(t3)
    if "| --- | --- |" not in "".join(c3):
        print("  ❌ 正常 3 行表被破坏")
        ok = False
    print("=== 2 行表不丢弃 ===")
    print("  ✅ 中段/EOF 2 行表回归文本行、3 行表正常" if ok else "  ❌ 有断言失败")
    assert ok


def test_split_markdown_heading_cap() -> bool:
    """后置处理两项约束（评审缺陷修复）：
    1. 短「——」注解不被当标题移走（注解孤儿）——注解行内排除；
    2. 标题移块不突破 3600 字节上限——下一块近满时尾部行级联顺延。
    """
    ok = True
    filler = "填充内容" * 40  # 480B/行
    annotation = "—— 不错"
    heading = "📄 arXiv cs.AI（今日更新）"

    def flatten(chunks):
        """行级压平：块尾换行是块边界分隔不计入内容行（块内空行保留）。"""
        out = []
        for c in chunks:
            parts = c.split("\n")
            if parts and parts[-1] == "":
                parts.pop()
            out.extend(parts)
        return out

    # case1: 块 0 以短注解结尾（7×481B + 注解 14B = 3381 ≤ 3600，
    # 条目 400B 放不下）→ 注解必须留在原块，不得移到下一块开头
    item400 = "Y" * 399  # 400B/行
    t1 = "\n".join([filler] * 7 + [annotation] + [item400] * 4)
    c1 = split_markdown(t1)
    if flatten(c1) != t1.split("\n"):
        print("  ❌ case1 行级拼接不完整")
        ok = False
    if c1[0].rstrip("\n").splitlines()[-1] != annotation:
        print(f"  ❌ case1 短注解被当标题移走: {c1[0].rstrip().splitlines()[-1]!r}")
        ok = False
    if c1[1].splitlines()[0] != item400:
        print("  ❌ case1 下一块首行不是条目")
        ok = False

    # case2: 块 0 以标题结尾（7×481B + 标题 25B = 3392），块 1 已近满
    # （16×224B = 3584，标题 25B 放不下 3609 > 3600）→ 尾部行顺延新块
    item224 = "X" * 223  # 224B/行
    t2 = "\n".join([filler] * 7 + [heading] + [item224] * 20)
    c2 = split_markdown(t2)
    sizes = [len(ch.encode("utf-8")) for ch in c2]
    if flatten(c2) != t2.split("\n"):
        print("  ❌ case2 行级拼接不完整")
        ok = False
    if max(sizes) > 3600:
        print(f"  ❌ case2 块超 3600B: {sizes}")
        ok = False
    heads = [ch.splitlines()[0] for ch in c2]
    if heading not in heads:
        print("  ❌ case2 标题不在任何块开头")
        ok = False
    if any(ch.rstrip("\n").splitlines()[-1] == heading for ch in c2[:-1]):
        print("  ❌ case2 标题孤悬块尾")
        ok = False

    print("=== 后置处理约束（注解不迁移 / 字节上限） ===")
    print(f"  case1 注解留在原块末行: {c1[0].rstrip().splitlines()[-1]!r}")
    print(f"  case2 块字节数: {sizes} (max {max(sizes)})")
    print("  ✅ 全部符合" if ok else "  ❌ 有断言失败")
    assert ok


def test_heading_blank_line() -> bool:
    """CommonMark 风格「标题 + 空行 + 列表」：标题后第一个空行被标题绑定。

    模型按 prompt 用规范 markdown 后常写「标题\n\n- 条目」：空行行尾与
    标题后首个标记行起点都不设切点，否则标题在空行/标记前孤悬块尾。
    """
    from core.markdown_split import markdown_safe_cut, trailing_structure

    ok = True
    # 1) 标题+空行后停顿（列表尚未生成）→ 持有等待（cut=0，宁慢勿断）
    c1 = markdown_safe_cut(
        "📄 arXiv cs.AI（今日更新）\n\n",
        len("📄 arXiv cs.AI（今日更新）\n\n"),
        initial=trailing_structure(""),
    )
    # 2) 标题+空行+两列表项 → 切点在第二个标记行起点（标题+空行+首项同气泡）
    t2 = (
        "📄 arXiv cs.AI（今日更新）\n\n"
        "- TongGuOCR：面向中国历史文献的布局感知 OCR 多模态大模型\n"
        "- Thought-Level Beam Search for Reasoning：思维级束搜索提升推理"
    )
    c2 = markdown_safe_cut(t2, len(t2), initial=trailing_structure(""))
    want2 = t2.index("- Thought-Level")
    # 3) 普通（长）行 + 空行 → 空行仍是切点（非标题场景不受影响）
    t3 = "这是一段比较长的普通段落文字内容超过二十四个字符的限制\n\n第二段内容\n"
    c3 = markdown_safe_cut(t3, len(t3), initial=trailing_structure(""))
    want3 = t3.index("\n\n") + 2
    # 4) 无标题的普通列表 → 标记行起点照常可切（回归）
    t4 = "- 条目一\n- 条目二\n"
    c4 = markdown_safe_cut(t4, len(t4), initial=trailing_structure(""))
    want4 = t4.index("- 条目二")
    print("=== 标题跨空行绑定 ===")
    print(
        f"  标题+空行停顿 cut={c1}(期望0) | 标题+空行+列表 cut={c2}(期望{want2}) | "
        f"普通空行 cut={c3}(期望{want3}) | 普通列表 cut={c4}(期望{want4})"
    )
    if c1 != 0 or c2 != want2 or c3 != want3 or c4 != want4:
        print("  ❌ 有断言失败")
        ok = False
    else:
        print("  ✅ 标题跨空行绑定、普通空行/列表切点不受影响")
    assert ok


def test_split_markdown_heading() -> bool:
    """split_markdown（>3600 字节）标题不孤悬块尾。

    字节精确构造：17 个填充行 + 「——」注解 + 标题行 = 3439 字节（≤3600），
    下一条目 217 字节放不下 → 标题本会落在块 0 末尾（孤悬）；修复后标题
    移到块 1 开头，与其列表同块。
    """
    ok = True
    filler = "列表内容填充" * 11  # 198 字节/行
    annotation = "—— 以上是填充内容"
    heading = "📄 arXiv cs.AI（今日更新）"
    item = "TongGuOCR：面向中国历史文献的布局感知 OCR 多模态大模型" * 3  # 216 字节
    lines = [filler] * 17 + [annotation, heading] + [item] * 8
    text = "\n".join(lines)
    chunks = split_markdown(text)
    joined = "".join(chunks)
    if joined != text:
        print(f"  ❌ 拼接不完整: {len(joined)} != {len(text)}")
        ok = False
    # 标题不得是任何块（末块除外）的最后一行；且被移到下一块开头
    for i, c in enumerate(chunks[:-1]):
        last = c.rstrip("\n").split("\n")[-1]
        if last.strip() == heading:
            print(f"  ❌ 块{i+1} 以标题行结尾（孤悬）: {last!r}")
            ok = False
    if not chunks[1].startswith(heading + "\n"):
        print(f"  ❌ 标题未移到块 2 开头: 块2 首行 {chunks[1].splitlines()[0]!r}")
        ok = False
    # 填充 + 注解仍留在块 0（注解不跟着标题走）
    if not chunks[0].endswith(annotation + "\n"):
        print(
            "  ❌ 块 1 不应包含标题（标题已移走）: 末行",
            chunks[0].splitlines()[-1][:20],
        )
        ok = False
    print("=== split_markdown 标题不孤悬（%d 字节） ===" % len(text.encode("utf-8")))
    print("  ✅ 标题与列表同块、注解留在原块、拼接完整" if ok else "  ❌ 有断言失败")
    assert ok


async def test_stream_full_suite():
    """全量回归：断流补发 / 字节上限硬切 / 超长列表 / 谓词 / 标题 / 注解 / 2行表。"""


async def test_stream_final_marked():
    """终稿（标记格式）：在 chat.txt 实际断点位置停顿。"""
    lines = FINAL.split("\n")
    ends = line_ends(FINAL)
    # 断点：Tramp 注解行尾 / widget.el 注解行尾 / "乐子帖"标题后 / 求助第3项后
    targets = [
        "—— *用 TRAMP 管理自建服务器（主人也有 homelab，可以看看大家怎么玩的！）*",
        "—— *作者用 widget.el 做 TextUI 的心得分享，技术含量高*",
        "**😂 乐子帖：**",
        "Indentation on emacs?（nvim 转来的对齐问题）",
    ]
    pause_at = [ends[i] - 1 for i, ln in enumerate(lines) if ln in targets]
    assert verify(
        "终稿（标记格式，1881字符）", FINAL, await run_case("final", FINAL, pause_at)
    )


async def test_stream_draft_unmarked():
    """草稿（无标记格式）：Tramp URL 行尾停顿。"""
    dlines = DRAFT.split("\n")
    dends = line_ends(DRAFT)
    dtargets = [
        "reddit.com/r/emacs/comments/1vlq9tu",
        'You understand my pain —— "你们懂我的痛"共鸣帖',
        "Tramp for Homelab? 🏠",  # 标题后停顿（URL 尚未生成）→ 标题不得单独成块
    ]
    dpause = [dends[i] - 1 for i, ln in enumerate(dlines) if ln in dtargets]
    assert verify(
        "草稿（无标记格式，1306字符）", DRAFT, await run_case("draft", DRAFT, dpause)
    )


async def test_stream_heading_not_orphaned():
    """标题不孤悬块尾（new_chat.txt 事故）：📄 标题后停顿。"""
    hlines = HEADING_TEXT.split("\n")
    hends = line_ends(HEADING_TEXT)
    htargets = [
        "—— 港中文等提出 Libra，优化 Agentic RL 后训练资源分配，吞吐最高 3 倍",
        "📄 arXiv cs.AI（今日更新）",
    ]
    hpause = set()
    for i, ln in enumerate(hlines):
        if ln in htargets:
            # 标题行尾 + 8 字符（第一行 arXiv 条目写到一半）——事故断点：
            # 空闲 flush 时 pending 以标题行结尾，不得把标题单独发出
            hpause.add(hends[i] + 8)
    assert verify(
        "标题不孤悬（无标记格式，arXiv 断点）",
        HEADING_TEXT,
        await run_case("heading", HEADING_TEXT, sorted(hpause)),
    )

    # ── 断流补发（force 路径）：生成中途断流，已转发+补发 == 断流点前全文 ──
    await test_stream_abort_tail_force()

    # ── 字节上限硬切（>3600 无安全切点）：单行 + 超限列表项 ──
    await test_stream_hardcut_oversized()

    # ── split_markdown 超长列表（>3600 字节）：块首必须是完整列表项 ──
    test_split_markdown_big_list()

    # ── 标题不孤悬块尾（new_chat.txt 事故）：📄 标题后停顿 ──
    hlines = HEADING_TEXT.split("\n")
    hends = line_ends(HEADING_TEXT)
    htargets = [
        "—— 港中文等提出 Libra，优化 Agentic RL 后训练资源分配，吞吐最高 3 倍",
        "📄 arXiv cs.AI（今日更新）",
    ]
    hpause = set()
    for i, ln in enumerate(hlines):
        if ln in htargets:
            # 标题行尾 + 8 字符（第一行 arXiv 条目写到一半）——事故断点：
            # 空闲 flush 时 pending 以标题行结尾，不得把标题单独发出
            hpause.add(hends[i] + 8)
    ok7 = verify(
        "标题不孤悬（无标记格式，arXiv 断点）",
        HEADING_TEXT,
        await run_case("heading", HEADING_TEXT, sorted(hpause)),
    )

    # ── 谓词边界（单/双破折号、裸 IP、嵌套缩进、标题行判定）──
    test_predicates()

    # ── split_markdown 标题不孤悬（>3600 字节字节精确构造）──
    test_split_markdown_heading()

    # ── 标题跨空行绑定（CommonMark 风格「标题 + 空行 + 列表」）──
    test_heading_blank_line()

    # ── 后置处理约束（注解不迁移 / 3600 字节上限不破）──
    test_split_markdown_heading_cap()

    # ── 2 行表（表头+分隔行）不丢弃（宁断勿丢）──
    test_split_markdown_table_no_rows()
