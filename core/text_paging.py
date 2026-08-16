"""文本分页读取 —— read_file（工作区 + 媒体附件）共用语义。

行级 offset/limit 导航 + 字符级 max_chars 上限（参考 OpenClaw read 工具）：
- offset 为 1-based 起始行号，越界抛 OffsetOutOfRangeError。
- 只保留完整行（逐行累加，超 max_chars 即停），保证 next_offset 续读不重不漏。
- 例外：单行本身超过 max_chars 时返回该行前缀并标记 last_line_partial——
  半行内容无法按行续读，调用方提示模型调大 max_chars 或跳过该行。
- max_chars 在函数内部钳制到 [1, MAX_PAGE_CHARS]；limit 必须 >= 1。

读取来源有两种，行模型一致：
- paginate_text：内存全文分页（输入须为已归一化文本，无 \\r\\n）。
- read_text_page：磁盘窗口读，不整读文件（先二进制快扫统计行数，
  再流式读取目标行段）。文本行经 universal newlines 归一（CRLF/CR → LF，
  与旧 read_text 行为一致）；已读取区域按 strict UTF-8 校验，非法字节抛
  BinaryFileError——未读取区域不校验（窗口读的固有权衡，换取任意大小
  文件可翻页且不整读进内存）。超长单行（> max_chars）只缓冲前缀。
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_PAGE_CHARS = 20_000  # read_file 默认每页字符上限
MAX_PAGE_CHARS = 1024 * 1024  # 单页输出字符硬顶（防 tool result 撑爆上下文）
_READ_CHUNK = 65536  # 跳行/丢弃超长行时分块大小（字符）


class OffsetOutOfRangeError(ValueError):
    """offset 超出文件行数。"""


class BinaryFileError(ValueError):
    """文件不是 UTF-8 文本（二进制或非 UTF-8 编码）。"""


class TruncateReason(Enum):
    """截断原因：wire 格式为 .value（"" | "lines" | "chars"）。"""

    NONE = ""
    LINES = "lines"
    CHARS = "chars"


@dataclass(frozen=True)
class TextPage:
    content: str
    start_line: int  # 本页起始行（1-based）
    total_lines: int
    output_lines: int
    next_offset: int  # 续读起点（1-based）；0 = 已读完全部
    truncated_by: TruncateReason
    last_line_partial: bool
    max_chars: int

    @property
    def truncated(self) -> bool:
        return self.truncated_by is not TruncateReason.NONE


def page_to_dict(page: TextPage) -> dict:
    """TextPage → wire dict（元数据簇的唯一序列化出口）。"""
    return {
        "start_line": page.start_line,
        "total_lines": page.total_lines,
        "output_lines": page.output_lines,
        "next_offset": page.next_offset,
        "truncated": page.truncated,
        "truncated_by": page.truncated_by.value,
        "last_line_partial": page.last_line_partial,
    }


def _clamp_chars(max_chars: int | None) -> int:
    if max_chars is None:
        return DEFAULT_PAGE_CHARS
    return min(max(1, int(max_chars)), MAX_PAGE_CHARS)


def _resolve_range(
    total: int, offset: int | None, limit: int | None
) -> tuple[int, int]:
    """解析 offset/limit 为 0-based 行区间 [start, end)。"""
    if offset is not None:
        offset = int(offset)
        if offset < 1:
            raise ValueError("offset 必须为 >= 1 的整数")
        start = offset - 1
    else:
        start = 0
    if start >= total:
        raise OffsetOutOfRangeError(f"offset {start + 1} 超出文件行数（共 {total} 行）")
    end = total
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit 必须为 >= 1 的整数")
        end = min(start + limit, total)
    return start, end


def _build_page(
    output: list[str],
    start: int,
    end: int,
    total: int,
    max_chars: int,
    reason: TruncateReason,
    partial: bool,
) -> TextPage:
    """由收集结果构造 TextPage（next_offset 规则集中于此）。"""
    if reason is TruncateReason.NONE and end < total:
        reason = TruncateReason.LINES
    if partial:
        next_offset = start + 2  # 跳过被截断的半行（剩余内容不可按行续读）
    elif reason is not TruncateReason.NONE:
        next_offset = start + len(output) + 1
    else:
        next_offset = 0
    if next_offset > total:
        next_offset = 0
    return TextPage(
        content="\n".join(output),
        start_line=start + 1,
        total_lines=total,
        output_lines=len(output),
        next_offset=next_offset,
        truncated_by=reason,
        last_line_partial=partial,
        max_chars=max_chars,
    )


def _collect_page_rows(rows, *, limit_rows: int, max_chars: int):
    """对行迭代器应用预算规则（唯一一份字符预算逻辑）。

    rows 最多消费 limit_rows 行。返回 (output, reason, partial)。
    """
    output: list[str] = []
    used = 0
    reason = TruncateReason.NONE
    partial = False
    for i, row in enumerate(rows):
        if i >= limit_rows:
            break
        cost = len(row) + (1 if output else 0)  # 行 + 前导换行
        if used == 0 and len(row) > max_chars:
            # 首个内容行超上限：返回前缀，剩余部分无法按行续读
            prefix = max_chars - (1 if output else 0)
            output.append(row[: max(0, prefix)])
            partial = True
            reason = TruncateReason.CHARS
            break
        if used > 0 and used + cost > max_chars:
            reason = TruncateReason.CHARS
            break
        output.append(row)
        used += cost
    return output, reason, partial


def paginate_text(
    text: str,
    *,
    offset: int | None = None,
    limit: int | None = None,
    max_chars: int = DEFAULT_PAGE_CHARS,
) -> TextPage:
    """对内存全文按 offset/limit/max_chars 切一页（输入须已归一化，无 \\r\\n）。"""
    max_chars = _clamp_chars(max_chars)
    lines = text.split("\n")
    total = len(lines)
    start, end = _resolve_range(total, offset, limit)
    output, reason, partial = _collect_page_rows(
        lines[start:end], limit_rows=end - start, max_chars=max_chars
    )
    return _build_page(output, start, end, total, max_chars, reason, partial)


def _count_newlines(path: Path) -> int:
    """二进制分块快扫行分隔符数量（\\n 与 \\r，与 universal newlines 对齐）。

    块间 1 字节重叠消除 \\r\\n 跨块边界被拆分的计数误差。
    """
    count = 0
    carry = b""
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            chunk = carry + chunk
            count += chunk.count(b"\n") + chunk.count(b"\r") - chunk.count(b"\r\n")
            carry = chunk[-1:]
    return count


def _ends_with_newline(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) in (b"\n", b"\r")
    except OSError:
        return False


def _discard_line(f) -> bool:
    """分块丢弃当前行剩余部分（超长行不整体缓冲）。返回 False 表示 EOF。"""
    while True:
        chunk = f.readline(_READ_CHUNK)
        if chunk == "":
            return False
        if chunk.endswith("\n"):
            return True


def _window_rows(path: Path, start: int, end: int, max_chars: int):
    """流式产出 [start, end) 行的生成器：strict UTF-8 逐行校验。

    - 跳行阶段分块丢弃，不缓冲超长行。
    - 读取阶段 readline(max_chars+1) 限长，超长行只缓冲前缀。
    - universal newlines 归一（\\r\\n/\\r → \\n），与旧 read_text 行为一致。
    - 行内非法 UTF-8 抛 BinaryFileError（已读取区域严格校验）。
    - 末尾补 split 语义的空行（文件以换行结尾或空文件时）。
    """
    last_newline = _ends_with_newline(path)
    empty = path.stat().st_size == 0
    with path.open(encoding="utf-8", errors="strict") as f:
        for _ in range(start):
            if not _discard_line(f):
                break
        eof = False
        for _ in range(start, end):
            if eof:
                break
            line = f.readline(max_chars + 1)
            if line == "":
                eof = True
                if not (last_newline or empty):
                    break
                yield ""  # split("\n") 语义的末尾空行
                continue
            row = line[:-1] if line.endswith("\n") else line
            yield row
            if len(line) > max_chars and not line.endswith("\n"):
                _discard_line(f)  # 超长行：吞掉剩余部分，避免下轮误读


def read_text_page(
    path: Path,
    *,
    offset: int | None = None,
    limit: int | None = None,
    max_chars: int = DEFAULT_PAGE_CHARS,
) -> TextPage:
    """磁盘窗口读一页：不整读文件，与 paginate_text 行模型一致。

    行模型与 universal newlines 归一后的 split("\n") 相同：文件以换行结尾时
    末尾有一个空行；空文件视为 1 行空行。非 UTF-8（含二进制）抛 BinaryFileError。
    """
    max_chars = _clamp_chars(max_chars)
    total = _count_newlines(path) + 1
    start, end = _resolve_range(total, offset, limit)
    try:
        output, reason, partial = _collect_page_rows(
            _window_rows(path, start, end, max_chars),
            limit_rows=end - start,
            max_chars=max_chars,
        )
    except UnicodeDecodeError:
        raise BinaryFileError(
            "文件不是 UTF-8 文本（疑似二进制或非 UTF-8 编码）"
        ) from None
    return _build_page(output, start, end, total, max_chars, reason, partial)


def build_pagination_hint(page: TextPage) -> str:
    """截断时附加在 content 末尾的续读提示（OpenClaw 风格）。"""
    if not page.truncated:
        return ""
    end_line = page.start_line + page.output_lines - 1
    if page.last_line_partial:
        if page.next_offset == 0:
            return (
                f"\n[第 {page.start_line} 行超过单页上限（{page.max_chars} 字符），"
                f"仅显示前缀，该行是最后一行且内容不完整。"
                f"可调大 max_chars 重读本页。]"
            )
        return (
            f"\n[第 {page.start_line} 行超过单页上限（{page.max_chars} 字符），"
            f"仅显示前缀，该行内容不完整。"
            f"可调大 max_chars 重读本页，或 offset={page.next_offset} 跳过该行继续。]"
        )
    if page.truncated_by is TruncateReason.CHARS:
        return (
            f"\n[已显示第 {page.start_line}-{end_line} 行，共 {page.total_lines} 行"
            f"（{page.max_chars} 字符上限）。使用 offset={page.next_offset} 继续读取。]"
        )
    return (
        f"\n[已显示第 {page.start_line}-{end_line} 行，共 {page.total_lines} 行。"
        f"使用 offset={page.next_offset} 继续读取。]"
    )
