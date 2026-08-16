"""read_file 分页语义测试：core/text_paging.py 核心逻辑 + media 服务层 + 工作区工具层。"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.managers.workspace_manager import WorkspaceManager
from core.media.service import MediaService
from core.text_paging import (
    BinaryFileError,
    OffsetOutOfRangeError,
    TruncateReason,
    build_pagination_hint,
    paginate_text,
    read_text_page,
)
from core.tools._types import ToolContext
from core.tools.deps import ToolDeps
from core.tools.impl.file import create_file_entries

TEXT = "\n".join(f"line{i}" for i in range(1, 11))  # 10 行: line1..line10


# ── paginate_text 核心逻辑 ──


def test_full_read_no_truncation():
    page = paginate_text(TEXT, max_chars=20000)
    assert page.content == TEXT
    assert not page.truncated
    assert page.next_offset == 0
    assert page.total_lines == 10
    assert page.output_lines == 10


def test_line_offset_and_limit():
    page = paginate_text(TEXT, offset=2, limit=3, max_chars=20000)
    assert page.content == "line2\nline3\nline4"
    assert page.start_line == 2
    assert page.truncated
    assert page.truncated_by.value == "lines"
    assert page.next_offset == 5  # 第 5 行开始续读
    assert "offset=5" in build_pagination_hint(page)


def test_paging_is_lossless():
    """两页拼接 == 全文（不重不漏）。"""
    first = paginate_text(TEXT, offset=1, limit=4, max_chars=20000)
    second = paginate_text(TEXT, offset=first.next_offset, limit=4, max_chars=20000)
    third = paginate_text(TEXT, offset=second.next_offset, limit=4, max_chars=20000)
    assert first.content + "\n" + second.content + "\n" + third.content == TEXT
    assert third.next_offset == 0


def test_exact_limit_reaches_end_no_truncation():
    page = paginate_text(TEXT, offset=7, limit=4, max_chars=20000)
    assert page.content == "line7\nline8\nline9\nline10"
    assert not page.truncated
    assert page.next_offset == 0


def test_char_cap_keeps_complete_lines():
    # 每行 5 字符 + 1 换行；max_chars=13 → line1(5)+换行(1)+line2(5)=11，line3 加上换行会超 13
    page = paginate_text(TEXT, max_chars=13)
    assert page.content == "line1\nline2"
    assert page.truncated
    assert page.truncated_by.value == "chars"
    assert page.next_offset == 3


def test_single_line_exceeding_cap_returns_prefix_and_marks_partial():
    page = paginate_text(TEXT, max_chars=4)
    assert page.content == "line"  # "line1" 前 4 字符
    assert page.last_line_partial
    assert page.truncated_by.value == "chars"
    assert page.next_offset == 2  # 跳过被截断行
    assert "内容不完整" in build_pagination_hint(page)


def test_offset_out_of_range_raises():
    with pytest.raises(OffsetOutOfRangeError):
        paginate_text(TEXT, offset=11, max_chars=20000)


def test_offset_below_one_raises():
    with pytest.raises(ValueError):
        paginate_text(TEXT, offset=0, max_chars=20000)


def test_empty_file():
    page = paginate_text("", max_chars=20000)
    assert page.content == ""
    assert not page.truncated


# ── read_text_page 磁盘窗口读 ──


def _write(path, text: str):
    Path(path).write_text(text, encoding="utf-8")
    return Path(path)


def test_window_read_matches_paginate_semantics(tmp_path):
    p = _write(tmp_path / "f.txt", TEXT)
    expected = paginate_text(TEXT, max_chars=20000)
    page = read_text_page(p, max_chars=20000)
    assert page.content == expected.content == TEXT
    assert page.total_lines == expected.total_lines == 10
    assert page.next_offset == 0


def test_window_read_offset_matches_paginate(tmp_path):
    p = _write(tmp_path / "f.txt", TEXT)
    expected = paginate_text(TEXT, offset=2, limit=3, max_chars=20000)
    page = read_text_page(p, offset=2, limit=3, max_chars=20000)
    assert page.content == expected.content == "line2\nline3\nline4"
    assert page.next_offset == expected.next_offset == 5


def test_window_read_large_file_no_size_limit(tmp_path):
    """窗口读不整读文件：>1MB 文件也能翻页。"""
    p = tmp_path / "big.log"
    with p.open("w", encoding="utf-8") as f:
        for i in range(60_000):
            f.write(f"row{i:06d}-{'x' * 40}\n")  # ~2.8MB
    assert p.stat().st_size > 1024 * 1024
    page = read_text_page(p, offset=50_000, limit=3, max_chars=20000)
    assert page.content.startswith("row049999-")
    assert page.start_line == 50_000
    assert page.total_lines == 60_001  # 末尾空行（split 语义）
    assert page.next_offset == 50_003


def test_window_read_trailing_newline_keeps_empty_last_line(tmp_path):
    """以 \n 结尾的文件，split 语义末尾有空行。"""
    p = _write(tmp_path / "f.txt", "a\nb\n")
    page = read_text_page(p, offset=3, limit=1, max_chars=20000)
    assert page.content == ""
    assert page.total_lines == 3
    assert page.output_lines == 1
    assert not page.truncated
    assert page.next_offset == 0


def test_window_read_crlf_matches_paginate(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"a\r\nb\r\nc")
    page = read_text_page(p, max_chars=20000)
    assert page.content == "a\nb\nc"  # universal newlines 转换
    assert page.total_lines == 3


def test_window_read_empty_file(tmp_path):
    p = _write(tmp_path / "f.txt", "")
    page = read_text_page(p, max_chars=20000)
    assert page.content == ""
    assert page.total_lines == 1
    assert not page.truncated


def test_window_read_offset_out_of_range(tmp_path):
    p = _write(tmp_path / "f.txt", TEXT)
    with pytest.raises(OffsetOutOfRangeError):
        read_text_page(p, offset=11, max_chars=20000)


def test_window_read_binary_rejected(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"\x00\x01\x02\xff" * 100)
    with pytest.raises(BinaryFileError):
        read_text_page(p, max_chars=20000)


def test_window_read_non_utf8_rejected(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes("中文内容".encode("gbk"))
    with pytest.raises(BinaryFileError):
        read_text_page(p, max_chars=20000)


def test_window_read_char_cap_keeps_complete_lines(tmp_path):
    p = _write(tmp_path / "f.txt", TEXT)
    expected = paginate_text(TEXT, max_chars=13)
    page = read_text_page(p, max_chars=13)
    assert page.content == expected.content == "line1\nline2"
    assert page.truncated_by is expected.truncated_by is TruncateReason.CHARS
    assert page.next_offset == expected.next_offset == 3


def test_window_read_single_long_line_prefix(tmp_path):
    p = _write(tmp_path / "f.txt", "a" * 100 + "\n" + TEXT)
    page = read_text_page(p, max_chars=10)
    assert page.content == "a" * 10
    assert page.last_line_partial
    assert page.next_offset == 2  # 跳过第 1 行


def test_window_read_paging_is_lossless(tmp_path):
    p = _write(tmp_path / "f.txt", TEXT)
    first = read_text_page(p, offset=1, limit=4, max_chars=20000)
    second = read_text_page(p, offset=first.next_offset, limit=4, max_chars=20000)
    third = read_text_page(p, offset=second.next_offset, limit=4, max_chars=20000)
    assert first.content + "\n" + second.content + "\n" + third.content == TEXT
    assert third.next_offset == 0


# ── 修复回归测试（code review 发现） ──


def test_cjk_file_crossing_8kb_boundary_not_misclassified(tmp_path):
    """S9 Blocker：CJK 多字节字符跨 8KB 边界不得误判为二进制。"""
    body = "\n".join(f"第{i}行内容" for i in range(3000))
    p = _write(tmp_path / "cn.txt", body)
    assert len(body.encode()) > 8192
    page = read_text_page(p, offset=2800, limit=5, max_chars=20000)
    assert page.content.startswith(
        "第2799行内容"
    )  # offset 1-based → 0-based 第 2799 行
    assert page.start_line == 2800


def test_invalid_utf8_in_read_region_rejected(tmp_path):
    """S9：目标行段内非法 UTF-8 必须拒绝（不再被 replace 静默吞掉）。"""
    p = tmp_path / "bad.txt"
    p.write_bytes(b"line1\n" + b"good" * 3000 + b"\nline3\n" + b"\xff\xfe bad\n")
    with pytest.raises(BinaryFileError):
        read_text_page(p, offset=4, max_chars=20000)
    # 未读取区域（跳过行）的非法字节不校验——窗口读固有权衡
    page = read_text_page(p, offset=1, limit=1, max_chars=20000)
    assert page.content == "line1"


def test_partial_last_line_hint_has_no_invalid_offset(tmp_path):
    """S3/S4：单行文件超限时 hint 不得输出非法 offset=0。"""
    p = _write(tmp_path / "one.txt", "a" * 100)
    page = read_text_page(p, max_chars=10)
    assert page.last_line_partial
    assert page.next_offset == 0
    hint = build_pagination_hint(page)
    assert "offset=0" not in hint
    assert "最后一行" in hint


def test_limit_zero_rejected_on_both_paths(tmp_path):
    """limit<1 双路径统一拒绝（不再静默转 1）。"""
    with pytest.raises(ValueError):
        paginate_text(TEXT, limit=0, max_chars=20000)
    p = _write(tmp_path / "f.txt", TEXT)
    with pytest.raises(ValueError):
        read_text_page(p, limit=0, max_chars=20000)


def test_leading_empty_line_budget_no_overflow(tmp_path):
    """空行开头时输出不得超过 max_chars。"""
    text = "\nabcdefghij"  # 空行 + 10 字符行
    page = paginate_text(text, max_chars=5)
    assert len(page.content) <= 5
    assert page.last_line_partial
    assert page.content == "\nabcd"  # 4 内容 + 1 前导换行 = 5


def test_crlf_parity_with_paginate(tmp_path):
    """S11：CRLF 文件经 universal newlines 归一后与 read_text+paginate 一致。"""
    p = tmp_path / "crlf.txt"
    p.write_bytes(b"line1\r\nline2\r\nline3\r\n")
    page = read_text_page(p, max_chars=20000)
    expected = paginate_text(p.read_text(encoding="utf-8"), max_chars=20000)
    assert page.content == expected.content
    assert page.total_lines == expected.total_lines == 4  # 末尾空行


def test_cr_only_file_line_count_consistent(tmp_path):
    """\r 单独换行文件：行数与 universal newlines 读取一致。"""
    p = tmp_path / "cr.txt"
    p.write_bytes(b"line1\rline2\rline3\r")
    page = read_text_page(p, offset=2, limit=1, max_chars=20000)
    assert page.content == "line2"
    assert page.total_lines == 4  # 3 行 + 末尾空行


def test_oversized_line_only_buffers_prefix(tmp_path):
    """S7：超长单行只缓冲前缀，后续行读取不受影响。"""
    huge = "x" * 100_000
    p = _write(tmp_path / "huge.txt", f"{huge}\nline2\nline3")
    page = read_text_page(p, offset=1, limit=3, max_chars=1000)
    assert page.last_line_partial
    assert len(page.content) == 1000
    assert page.next_offset == 2
    # 跳过超长行后正常读取后续行
    page2 = read_text_page(p, offset=2, limit=2, max_chars=20000)
    assert page2.content == "line2\nline3"
    assert page2.start_line == 2


def test_skip_phase_does_not_buffer_oversized_lines(tmp_path):
    """跳行阶段遇到超长行（如 5MB minified JSON 单行）不缓冲、能跳过。"""
    huge = "y" * 5_000_000
    p = _write(tmp_path / "minified.json", f"line1\n{huge}\nline3")
    page = read_text_page(p, offset=3, limit=1, max_chars=20000)
    assert page.content == "line3"
    assert page.start_line == 3


def test_max_chars_clamped_in_function(tmp_path):
    """max_chars 钳制收敛在函数内（调用方不再重复）。"""
    from core.text_paging import MAX_PAGE_CHARS

    page = paginate_text(TEXT, max_chars=10**9)
    assert page.max_chars == MAX_PAGE_CHARS
    page2 = read_text_page(_write(tmp_path / "f.txt", TEXT), max_chars=0)
    assert page2.max_chars == 1


# ── 工作区 read_file 工具层 ──


def _read_entry():
    return next(e for e in create_file_entries(ToolDeps()) if e.name == "read_file")


def _ctx():
    return ToolContext(
        chat_id="c1",
        is_group=False,
        reply_to="",
        sender_id="u1",
        reply_callback=AsyncMock(),
    )


def _make_entry(tmp_path):
    wm = WorkspaceManager(root=str(tmp_path))
    entries = create_file_entries(ToolDeps(workspace_manager=wm))
    return next(e for e in entries if e.name == "read_file"), wm


def _write_workspace_file(wm: WorkspaceManager, name: str, content: str):
    path = wm.sandbox_dir(is_group=False, chat_id="c1") / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_workspace_read_file_with_offset_and_limit(tmp_path):
    entry, wm = _make_entry(tmp_path)
    _write_workspace_file(wm, "notes.txt", TEXT)
    result = await entry.handler(
        {"file_path": "notes.txt", "offset": 2, "limit": 3}, _ctx()
    )
    data = json_loads(result.content)
    assert data["success"]
    assert data["content"].startswith("line2\nline3\nline4\n[已显示")
    assert data["total_lines"] == 10
    assert data["next_offset"] == 5
    assert data["truncated"]
    assert data["truncated_by"] == "lines"


@pytest.mark.asyncio
async def test_workspace_read_file_offset_out_of_range(tmp_path):
    entry, wm = _make_entry(tmp_path)
    _write_workspace_file(wm, "notes.txt", TEXT)
    result = await entry.handler({"file_path": "notes.txt", "offset": 99}, _ctx())
    assert "超出文件行数" in json_loads(result.content)["error"]


@pytest.mark.asyncio
async def test_workspace_read_file_default_char_cap(tmp_path):
    entry, wm = _make_entry(tmp_path)
    big = "\n".join(f"x" * 100 for _ in range(300))  # 300 行 × 100 字符
    _write_workspace_file(wm, "big.txt", big)
    result = await entry.handler({"file_path": "big.txt"}, _ctx())
    data = json_loads(result.content)
    assert data["truncated"]
    assert data["truncated_by"] == "chars"
    assert data["total_lines"] == 300
    assert data["next_offset"] > 1
    assert "offset=" in data["content"]


@pytest.mark.asyncio
async def test_workspace_read_file_whole_file_unchanged_when_small(tmp_path):
    entry, wm = _make_entry(tmp_path)
    _write_workspace_file(wm, "small.txt", TEXT)
    result = await entry.handler({"file_path": "small.txt"}, _ctx())
    data = json_loads(result.content)
    assert data["content"] == TEXT
    assert not data["truncated"]
    assert data["next_offset"] == 0


@pytest.mark.asyncio
async def test_workspace_read_file_over_1mb_pages(tmp_path):
    """工作区 >1MB 文件不再拒绝：窗口读 + 翻页。"""
    entry, wm = _make_entry(tmp_path)
    big_path = wm.sandbox_dir(is_group=False, chat_id="c1") / "big.log"
    with big_path.open("w", encoding="utf-8") as f:
        for i in range(60_000):
            f.write(f"row{i:06d}-{'x' * 40}\n")
    assert big_path.stat().st_size > 1024 * 1024
    result = await entry.handler(
        {"file_path": "big.log", "offset": 50_000, "limit": 2}, _ctx()
    )
    data = json_loads(result.content)
    assert data["success"]
    assert data["content"].startswith("row049999-")
    assert data["total_lines"] == 60_001
    assert data["next_offset"] == 50_002
    assert data["truncated"]  # limit=2 行截断，需翻页续读


@pytest.mark.asyncio
async def test_workspace_read_file_binary_rejected(tmp_path):
    entry, wm = _make_entry(tmp_path)
    _write_workspace_file(wm, "blob.bin", "x")
    target = wm.sandbox_dir(is_group=False, chat_id="c1") / "blob.bin"
    target.write_bytes(b"\x00\x01\x02\xff" * 100)
    result = await entry.handler({"file_path": "blob.bin"}, _ctx())
    assert "二进制" in json_loads(result.content)["error"]


@pytest.mark.asyncio
async def test_workspace_read_file_invalid_offset_rejected(tmp_path):
    entry, wm = _make_entry(tmp_path)
    _write_workspace_file(wm, "notes.txt", TEXT)
    result = await entry.handler({"file_path": "notes.txt", "offset": 0}, _ctx())
    assert "offset 必须 >= 1" in json_loads(result.content)["error"]


# ── media 服务层（附件） ──


@pytest.mark.asyncio
async def test_media_read_file_line_paging(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=TEXT.encode(),
    )
    page = await service.read_file(
        chat_id="g1", media_uri=record.media_uri, offset=3, limit=2
    )
    assert page.content.startswith("line3\nline4\n[已显示")
    assert page.total_lines == 10
    assert page.next_offset == 5
    assert page.truncated_by == "lines"
    next_page = await service.read_file(
        chat_id="g1", media_uri=record.media_uri, offset=page.next_offset
    )
    assert next_page.content.startswith("line5")


@pytest.mark.asyncio
async def test_media_read_file_offset_out_of_range(tmp_path):
    service = MediaService(http_client=AsyncMock(), storage_dir=tmp_path)
    await service.open()
    record = await service.store.save(
        chat_id="g1",
        message_id="m1",
        sender_id="u1",
        resource_type="file",
        source_url="https://example.test/notes.txt",
        mime_type="text/plain",
        filename="notes.txt",
        data=TEXT.encode(),
    )
    result = await service.read_file(
        chat_id="g1", media_uri=record.media_uri, offset=99
    )
    assert result.error == "OFFSET_OUT_OF_RANGE"


def json_loads(s):
    import json

    return json.loads(s)
