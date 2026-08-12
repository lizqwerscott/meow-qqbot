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

另含：断流补发（force）路径与 split_markdown 超长列表切分测试。

用法: PYTHONPATH=. uv run python dev/test_list_split.py
"""

import asyncio
import json
import sys
from types import SimpleNamespace as NS

from core.ai.protocol import AssistantMessage
from core.markdown_split import (
    is_annotation_continuation,
    is_list_marker,
    is_url_line,
    split_markdown,
)
from core.tools.tool_loop import ToolLoop

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


class MockCost:
    def record_turn(self, chat_id, model, usage):
        pass


class MockCtx:
    async def add_assistant_message_async(self, *a, **k):
        pass


class MockSvc:
    """逐字符喂 on_text；在 pause_at（字符下标）处停顿 > idle_ms，模拟模型节奏。"""

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
            if callbacks and callbacks.on_text:
                await callbacks.on_text(buf)
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

    if ok:
        print("  ✅ 顺序完整、无注解孤儿、列表项整体未拆")
    return ok


async def run_abort_case() -> bool:
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
                if callbacks and callbacks.on_text:
                    await callbacks.on_text(buf)

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
    return ok


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
    return ok


async def run_hardcut_case() -> bool:
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
    return ok


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
    print("=== 谓词边界 ===")
    print(
        "  ✅ 单/双破折号注解、裸 IP vs 带端口/路径 IP、嵌套缩进 全部符合预期"
        if ok
        else "  ❌ 有断言失败"
    )
    return ok


async def main():
    # ── 终稿（标记格式）：在 chat.txt 实际断点位置停顿 ──
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
    ok1 = verify(
        "终稿（标记格式，1881字符）", FINAL, await run_case("final", FINAL, pause_at)
    )

    # ── 草稿（无标记格式）：Tramp URL 行尾（chat.txt 块3/块4 断点）停顿 ──
    dlines = DRAFT.split("\n")
    dends = line_ends(DRAFT)
    dtargets = [
        "reddit.com/r/emacs/comments/1vlq9tu",
        'You understand my pain —— "你们懂我的痛"共鸣帖',
        "Tramp for Homelab? 🏠",  # 标题后停顿（URL 尚未生成）→ 标题不得单独成块
    ]
    dpause = [dends[i] - 1 for i, ln in enumerate(dlines) if ln in dtargets]
    ok2 = verify(
        "草稿（无标记格式，1306字符）", DRAFT, await run_case("draft", DRAFT, dpause)
    )

    # ── 断流补发（force 路径）：生成中途断流，已转发+补发 == 断流点前全文 ──
    ok3 = await run_abort_case()

    # ── 字节上限硬切（>3600 无安全切点）：单行 + 超限列表项 ──
    ok5 = await run_hardcut_case()

    # ── split_markdown 超长列表（>3600 字节）：块首必须是完整列表项 ──
    ok4 = test_split_markdown_big_list()

    # ── 谓词边界（单/双破折号、裸 IP、嵌套缩进）──
    ok6 = test_predicates()

    return 0 if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
