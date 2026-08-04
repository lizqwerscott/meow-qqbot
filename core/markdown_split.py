"""Markdown 文本拆分 — 发送前的安全切块与流式转发安全切点。

同一份围栏/表格状态机供两处使用（曾各抄一份，改动必须同步，现已归位）：

1. `split_markdown` — 非流式发送前的字节级拆块（QQ 单条消息上限兜底）。
2. `markdown_safe_cut` / `trailing_structure` / `pending_starts_incomplete` —
   流式 block 转发时找不切断代码围栏/markdown 表格的切点。

判定谓词（is_fence_line / is_closing_fence_line / is_table_separator /
is_table_row）是两份状态机的公共基础，改动只需在本模块内同步。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# QQ 单条消息的 markdown 安全字节上限（_split_markdown 的默认块大小）。
MARKDOWN_SAFE_CHUNK_BYTE_LIMIT = 3600


def utf8len(text: str) -> int:
    return len(text.encode("utf-8"))


# ── 结构判定谓词（两种状态机共用） ──


def is_fence_line(line: str) -> str | None:
    m = re.match(r"^(\s*)(`{3,}|~{3,})", line)
    return m.group(2) if m else None


def is_closing_fence_line(line: str, marker: str) -> bool:
    marker_char = marker[0]
    m = re.match(r"^\s*(" + re.escape(marker_char) + r"{3,})\s*$", line)
    return bool(m and len(m.group(1)) >= len(marker))


def is_table_separator(line: str) -> bool:
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return False
    cells = [c.strip() for c in line[1:-1].split("|")]
    return len(cells) >= 2 and all(re.match(r"^:?-+:?$", c) for c in cells)


def is_table_row(line: str) -> bool:
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return False
    return len([c.strip() for c in line[1:-1].split("|")]) >= 2


# ── 非流式拆块（split_markdown 内部工具） ──


def _append_or_flush(
    chunks: list[str], text: str, max_bytes: int, spacer: str = "\n\n"
):
    if not text:
        return
    if chunks:
        last = chunks[-1]
        cand = last + spacer + text
        if utf8len(cand) <= max_bytes:
            chunks[-1] = cand
            return
    chunks.append(text)


def _chunk_text(t: str, max_bytes: int, chunks: list[str]):
    if utf8len(t) <= max_bytes:
        _append_or_flush(chunks, t, max_bytes)
        return
    for para in t.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if utf8len(para) <= max_bytes:
            _append_or_flush(chunks, para, max_bytes)
        else:
            for sentence in re.split(r"(?<=[。！？!?\n])", para):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if utf8len(sentence) <= max_bytes:
                    _append_or_flush(chunks, sentence, max_bytes, spacer="\n")
                else:
                    buf = ""
                    for char in sentence:
                        cand = buf + char
                        if utf8len(cand) <= max_bytes:
                            buf = cand
                        else:
                            chunks.append(buf)
                            buf = char
                    if buf:
                        chunks.append(buf)


def _flush_text(chunks: list[str], text_lines: list[str], max_bytes: int):
    if not text_lines:
        return
    t = "\n".join(text_lines)
    text_lines.clear()
    _chunk_text(t, max_bytes, chunks)


def _flush_table(chunks: list[str], table_lines: list[str], max_bytes: int):
    if len(table_lines) < 3:
        return
    full = "\n".join(table_lines)
    if utf8len(full) <= max_bytes:
        chunks.append(full)
        table_lines.clear()
        return
    header = table_lines[0:2]
    rows = table_lines[2:]
    out_lines = list(header)
    for row in rows:
        cand = "\n".join(out_lines + [row])
        if utf8len(cand) <= max_bytes:
            out_lines.append(row)
        else:
            if len(out_lines) > 2:
                chunks.append("\n".join(out_lines))
            out_lines = list(header) + [row]
    if len(out_lines) > 2:
        chunks.append("\n".join(out_lines))
    table_lines.clear()


def split_markdown(
    text: str, max_bytes: int = MARKDOWN_SAFE_CHUNK_BYTE_LIMIT
) -> list[str]:
    """按行状态机拆 markdown 文本为不超 max_bytes 的块。

    代码围栏整体保留（超长围栏按行扩容、逐段闭合），表格按「表头+分隔行」
    分页续传，普通文本按段落/句子/字符逐级切分。围栏内文本不参与表格识别。
    """
    if not text:
        return []
    if utf8len(text) <= max_bytes:
        return [text]

    chunks: list[str] = []
    text_lines: list[str] = []
    fence_body: list[str] = []
    active_fence: tuple[str, str] | None = None  # (open_line, marker)

    pending_header: str | None = None
    pending_header_cells: list[str] | None = None
    table_lines: list[str] = []
    in_table = False

    def _ct(t: str):
        _chunk_text(t, max_bytes, chunks)

    def _flush_text_buf():
        nonlocal text_lines
        if active_fence:
            return
        _flush_text(chunks, text_lines, max_bytes)

    def _flush_fence_and_close():
        nonlocal fence_body, active_fence
        if not active_fence:
            return
        open_line, marker = active_fence
        close = marker
        if not fence_body:
            chunks.append(f"{open_line}\n{close}")
        else:
            body = "\n".join(fence_body)
            full = f"{open_line}\n{body}\n{close}"
            if utf8len(full) <= max_bytes:
                chunks.append(full)
            else:
                lines = list(fence_body)
                cur = [open_line]
                for line in lines:
                    cand = "\n".join(cur + [line, close])
                    if utf8len(cand) <= max_bytes:
                        cur.append(line)
                    else:
                        chunks.append("\n".join(cur + [close]))
                        cur = [open_line, line]
                if len(cur) > 1:
                    chunks.append("\n".join(cur + [close]))
        fence_body.clear()
        active_fence = None

    def _flush_table_lines():
        nonlocal table_lines, in_table, pending_header, pending_header_cells
        if in_table or (pending_header and table_lines):
            _flush_table(chunks, table_lines, max_bytes)
            in_table = False
            pending_header = None
            pending_header_cells = None
        elif pending_header:
            text_lines.append(pending_header)
            pending_header = None
            pending_header_cells = None

    lines = text.split("\n")

    for line in lines:
        marker = is_fence_line(line)
        if marker:
            if active_fence is None:
                _flush_table_lines()
                _flush_text_buf()
                active_fence = (line, marker)
            elif is_closing_fence_line(line, active_fence[1]):
                _flush_fence_and_close()
            continue

        if active_fence:
            fence_body.append(line)
            continue

        if in_table and is_table_row(line):
            table_lines.append(line)
            continue

        if is_table_separator(line):
            if pending_header is not None:
                _flush_text_buf()
                table_lines = [pending_header, line]
                in_table = True
                pending_header = None
                pending_header_cells = None
            else:
                text_lines.append(line)
            continue

        if is_table_row(line):
            if pending_header is not None:
                text_lines.append(pending_header)
                pending_header = None
                pending_header_cells = None
                text_lines.append(line)
            elif in_table:
                table_lines.append(line)
            else:
                pending_header = line
                pending_header_cells = [
                    c.strip() for c in line.strip()[1:-1].split("|")
                ]
            continue

        _flush_table_lines()
        text_lines.append(line)

    _flush_table_lines()
    if active_fence:
        _flush_fence_and_close()
    _flush_text_buf()

    return chunks


# ── 流式转发安全切点（tool_loop block 投递） ──


@dataclass(frozen=True)
class TrailingState:
    """已发文本的末尾结构状态（表格内 / 围栏内 + 围栏 marker）。

    trailing_structure 产出、markdown_safe_cut 消费，捆绑传递避免三参数拆包。
    """

    in_table: bool = False
    in_fence: bool = False
    fence_marker: str | None = None


def trailing_structure(text: str) -> TrailingState:
    """扫描已发文本末尾状态：是否处于表格内 / 代码围栏内（含围栏 marker）。

    供 markdown_safe_cut 作为 initial 状态，使跨块的表格/围栏延续正确。
    """
    in_fence = False
    fence_marker: str | None = None
    in_table = False
    for line in text.rstrip("\n").split("\n"):
        marker = is_fence_line(line)
        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif is_closing_fence_line(line, fence_marker):
                in_fence = False
                fence_marker = None
        elif in_fence:
            pass
        elif is_table_separator(line):
            in_table = True
        elif is_table_row(line):
            if not in_table:
                in_table = True  # 表头行后（未遇分隔行）也视为表内延续
        else:
            in_table = False
    return TrailingState(
        in_table=in_table, in_fence=in_fence, fence_marker=fence_marker
    )


def markdown_safe_cut(
    text: str,
    limit: int,
    initial: TrailingState | None = None,
) -> int:
    """在 text 的 limit 字符内找安全的块切点：不切断代码围栏与 markdown 表格。

    返回切点索引（行尾，含换行）；整个文本都处于表格/围栏内时返回 0，
    调用方按原样切（超长表格由 split_markdown 兜底拆块）。
    状态机与 split_markdown 保持一致：表内/围栏体内不可切，
    普通行行尾即安全切点（表格或围栏整体留在块内，宁小勿断）。
    initial 由调用方传入已发文本的末尾状态（trailing_structure 产出），
    使跨块的表格/围栏延续正确（pending 开头是数据行时行尾可切）。
    """
    if limit <= 0:
        return 0
    safe = 0
    pos = 0
    in_fence = initial.in_fence if initial else False
    fence_marker = initial.fence_marker if initial and initial.in_fence else None
    in_table = initial.in_table if initial else False
    pending_header = False
    for line in text.split("\n"):
        if pos >= limit:
            break
        marker = is_fence_line(line)
        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif is_closing_fence_line(line, fence_marker):
                in_fence = False
                fence_marker = None
                safe = pos + len(line) + 1  # 围栏完整结束后可切（行尾）
        elif in_fence:
            pass  # 围栏体内不可切
        elif is_table_separator(line):
            in_table = True
            pending_header = False
        elif is_table_row(line):
            if in_table:
                # 表内数据行：行尾可切（表格延续，行保持完整；表头在块首或上一块）
                safe = pos + len(line) + 1
            else:
                pending_header = True  # 表头候选：等分隔行确认（表头行尾不可切）
        elif line.strip().startswith("|"):
            # 以 | 开头但未闭合：流式生成中的半截表格行/表头 → 结构内，不可切
            pass
        else:
            # 普通行：表格/围栏结束，行尾即安全切点。半截行（流式生成中，
            # 无换行结尾）不更新 safe——否则切点 = len+1 超出文本长度，
            # 调用方不切，半截链接/单词会整段发出（QQ 半截链接的根因）。
            in_table = False
            pending_header = False
            end = pos + len(line) + 1
            if end <= len(text):
                safe = end
        pos += len(line) + 1
    return safe


def pending_starts_incomplete(text: str, sent_prefix: str) -> bool:
    """pending 是否以未完成结构开头（表格行 / 上一块切在围栏体内）。

    空闲 flush 时命中则跳过本次发送（等流继续，收尾补发兜底），
    避免把半截表格或代码块发出。
    """
    first = text.split("\n", 1)[0].strip()
    if first and (is_table_separator(first) or is_table_row(first)):
        return True  # 表头/分隔行已在上一块或尚未生成
    sent_lines = [ln for ln in sent_prefix.rstrip().split("\n") if ln.strip()]
    if sent_lines and is_fence_line(sent_lines[-1]):
        return True  # 已发部分以围栏开始行结尾 → pending 是围栏体
    return False
