"""core/markdown_split 测试 — 非流式拆块（原 BotEngine._split_markdown）。

从 BotEngine 迁出到独立模块后补的直接测试：字节上限、围栏整体性、表格分页。
流式切点（markdown_safe_cut 等）的测试在 test_stream_block_cut.py。
"""

from core.markdown_split import (
    MARKDOWN_SAFE_CHUNK_BYTE_LIMIT,
    split_markdown,
    utf8len,
)


def test_short_text_single_chunk():
    """不超过上限 → 原样单块。"""
    assert split_markdown("你好世界") == ["你好世界"]
    assert split_markdown("") == []


def test_byte_limit_respected():
    """所有块 utf-8 字节数 ≤ 上限；块间/句间只插入 \n 分隔，内容不丢不增。"""
    text = "这是一段足够长的普通文本。" * 300
    chunks = split_markdown(text, max_bytes=800)
    assert "".join(chunks).replace("\n", "") == text
    assert all(utf8len(c) <= 800 for c in chunks)
    assert len(chunks) > 1


def test_fence_never_split_across_chunks():
    """代码围栏自闭合：任何块的围栏段都成对出现（超长围栏拆为多个闭合段）。"""
    body = "\n".join(f"print({i})" for i in range(300))
    text = f"开头说明。\n```python\n{body}\n```\n结尾说明。" * 2
    chunks = split_markdown(text, max_bytes=600)
    for c in chunks:
        assert c.count("```") % 2 == 0, "块内围栏必须成对闭合，绝不跨块"
    joined = "\n".join(chunks)
    assert "print(0)" in joined and "print(299)" in joined, "围栏内容完整"


def test_table_split_keeps_header():
    """超长表格分页：续页块都以「表头+分隔行」开头，内容不丢。"""
    rows = "\n".join(f"| 行{i} | 数据{i} |" for i in range(200))
    text = f"表格如下：\n| 列A | 列B |\n| --- | --- |\n{rows}"
    chunks = split_markdown(text, max_bytes=500)
    assert "表格如下：" in chunks[0]
    header = "| 列A | 列B |\n| --- | --- |\n"
    assert all(c.startswith(header) for c in chunks[1:]), "续页必须以表头开头"
    assert "| 行199 | 数据199 |" in chunks[-1], "最后一行不丢"
