"""验证流式块发送的并发竞态与 flush_lock 修复（确定性复现）。

线上事故形态（tmp/chat.txt 块6/块7）：模型先输出列表尾巴（421 字符，
QQ API 发送慢 / 重试退避），紧接着输出收尾句（77 字符，发送快）。
旧实现中两个 idle flush 并发在途 → 后发先至，收尾句插到列表中间。

本测试用纯段落文本（无列表/URL/注解，切点恒存在、不被列表保护跳过），
让第一个块的发送在途 2.5s，随后块的定时器在发送期间触发——无锁时后块
并发发出（回调进入时间早于前块退出），有锁（flush_lock）时严格串行
（进入时间 >= 前块退出）。断言逐块回调的时间戳序列 + 拼接完整性。

用法: PYTHONPATH=. uv run python dev/test_stream_order.py
"""

import asyncio
import time
from types import SimpleNamespace as NS

from core.ai.protocol import AssistantMessage
from core.tools.tool_loop import ToolLoop
from tests.stream_test_helpers import emit_snapshot

# 纯段落文本：每行一句、行尾切点恒存在 → 空闲 flush 不会被列表保护跳过
FULL = (
    "第一段内容，猫猫在认真组织这段话，保证行尾都是安全切点。\n" * 20
    + "\n\n"
    + "第二段内容，继续输出更多文字，模拟模型生成节奏。\n" * 20
    + "\n\n"
    + "第三段内容，收尾补充，测试拼接完整性。\n" * 15
)


class MockCost:
    def record_turn(self, chat_id, model, usage):
        pass


class MockCtx:
    async def add_assistant_message_async(self, *a, **k):
        pass


class BurstSvc:
    """分 4 段突发生成：burst 后停 1.5s（> idle 1s），触发空闲 flush。"""

    def __init__(self, text, burst, pause):
        self.text = text
        self.burst = burst
        self.pause = pause

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
            if i > 0 and i % self.burst == 0:
                await asyncio.sleep(self.pause)
            buf += ch
            await emit_snapshot(callbacks, buf)
        return AssistantMessage(content=self.text or None), None


class SlowCb:
    """首个块发送慢（2.5s，模拟 QQ API 慢/重试退避），其余快（0.01s）。

    记录每个回调的进入/退出时刻 → 有锁时进入时刻严格不早于前块退出时刻。
    """

    def __init__(self):
        self.times: list[tuple[float, float]] = []  # (entry, exit)
        self.blocks: list[str] = []
        self._n = 0

    async def __call__(self, chunk):
        entry = time.monotonic()
        delay = 2.5 if self._n == 0 else 0.01
        self._n += 1
        await asyncio.sleep(delay)
        self.times.append((entry, time.monotonic()))
        self.blocks.append(chunk)


async def test_stream_blocks_serialized():
    ctx = NS(
        ai=NS(
            ai_service=BurstSvc(FULL, burst=300, pause=1.5),
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
    cb = SlowCb()
    tailed: list[str] = []

    async def tail(chat_id, content, message_id, is_group):
        if content:
            tailed.append(content)

    await ToolLoop(ctx).run(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        chat_id="c1",
        is_group=True,
        reply_to="m1",
        reply_callback=tail,
        stream_callback=cb,
    )

    blocks = cb.blocks + tailed
    print(f"流式 {len(cb.blocks)} 块 + 收尾 {len(tailed)} 块")
    for i, (entry, exit_) in enumerate(cb.times):
        print(
            f"  块{i+1}: 进入 {entry- cb.times[0][0]:6.3f}s 退出 {exit_-cb.times[0][0]:6.3f}s"
        )

    ok = True
    # 1) 串行化：后一块进入时刻 >= 前一块退出时刻
    for i in range(1, len(cb.times)):
        if cb.times[i][0] < cb.times[i - 1][1] - 1e-6:
            print(f"  ❌ 块{i+1} 进入早于块{i} 退出 → 并发在途（乱序风险）")
            ok = False
    # 2) 拼接完整 == 原文（顺序与完整性）
    joined = "".join(blocks)
    if joined != FULL:
        print(f"  ❌ 拼接不完整/乱序: {len(joined)} != {len(FULL)}")
        ok = False
    # 3) 切点都在行尾
    pos = 0
    for b in blocks[:-1]:
        pos += len(b)
        if FULL[pos - 1] != "\n":
            print(f"  ❌ 中线切点 @{pos}")
            ok = False
    # 4) 确实演练了并发窗口（至少 2 块流式）
    if len(cb.times) < 2:
        print("  ⚠️ 只有 1 块流式，未演练并发窗口（测试编排失效）")
        ok = False

    assert ok, "流式块串行化失败：见上方 ❌ 诊断"
    print("✅ 串行化成立：后块进入时刻 >= 前块退出时刻，拼接完整，无中线切点")
