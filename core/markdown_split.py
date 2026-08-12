"""Markdown 文本拆分 — 发送前的安全切块与流式转发安全切点。

同一份围栏/表格/列表状态机供两处使用（曾各抄一份，改动必须同步，现已归位）：

1. `split_markdown` — 非流式发送前的字节级拆块（QQ 单条消息上限兜底）。
2. `markdown_safe_cut` / `trailing_structure` / `pending_starts_incomplete` —
   流式 block 转发时找不切断代码围栏/markdown 表格/列表项的切点。

判定谓词（is_fence_line / is_table_* / is_list_marker / is_url_line /
is_annotation_continuation）是两份状态机的公共基础，改动只需在本模块内同步。
列表项 = 标记行（`1.` / `-` / `*` 等）+ 续行（URL / 「——」注解 / 缩进行）：
切点只落在项边界、空行与「——」注解行尾，绝不拆散条目。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# QQ 单条消息的 markdown 安全字节上限（_split_markdown 的默认块大小）。
MARKDOWN_SAFE_CHUNK_BYTE_LIMIT = 3600


def utf8len(text: str) -> int:
    return len(text.encode("utf-8"))


def utf8_prefix(text: str, max_bytes: int) -> int:
    """text 的不超过 max_bytes 字节的最长前缀长度（UTF-8 安全）。

    逐字符累加字节数，不把多字节字符（中文/emoji）从中间切开。
    供超限单行（无换行、无安全切点）的硬切使用。
    """
    n = 0
    for i, ch in enumerate(text):
        n += utf8len(ch)
        if n > max_bytes:
            return i
    return len(text)


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


_LIST_MARKER_RE = re.compile(r"^\s{0,4}(?:[-*+]|\d{1,3}[.)])(?:\s+|$)")


def is_list_marker(line: str) -> bool:
    """有序/无序列表标记行（可缩进 0-4 空格）：`- item` / `* item` /
    `+ item` / `1. item` / `1) item`。

    要求标记后跟空白或行尾，避免误匹配 `*强调文本*`、`192.168.x`（数字点
    后紧跟数字无空白）等普通文本。
    """
    return bool(_LIST_MARKER_RE.match(line))


def is_annotation_continuation(line: str) -> bool:
    """「——」开头的注解/续行（猫猫回复里列表项的点评行）。

    视为上一列表项的续行：切块时不允许把注解与其标题/链接拆到两个气泡。
    仅匹配双 em-dash「——」：单「—」可能是对话/散文破折号开头，误粘会
    拖累延迟（正确性由收尾补发兜底）。
    """
    return line.lstrip().startswith("——")


_URL_LINE_RE = re.compile(
    r"^\s{0,4}"
    r"(?:"
    r"(?:https?://|www\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/\S*)?"
    r"|(?:(?:\d{1,3}\.){3}\d{1,3}(?:/\S*)?)"
    r")\s*$"
)


def is_url_line(line: str) -> bool:
    """裸 URL 行（无 markdown 链接语法）：`reddit.com/r/emacs/...`、
    `https://www.reddit.com/...`、IPv4 地址。

    视为上一行的续行（列表条目的标题/链接对）：草稿/无标记格式下条目没有
    `1.` 前缀可锚定，靠「标题 + URL + 「——」注解」的行模式保持整体。
    """
    return bool(_URL_LINE_RE.match(line))


# 由 _LIST_MARKER_RE 构建的拆块切分正则（非流式 split_markdown 用），
# 与谓词共用同一份模式，避免第三份复制漂移。re.MULTILINE 使 ^/$ 按行匹配。
# 唯一差异：缩进用 [ \t] 而非 \s——否则 lookahead 会跨空行匹配，把
# 列表项间的空行分隔符吞掉（拼接无法还原原文）。
_LIST_MARKER_SPLIT_RE = re.compile(
    r"\n(?=" + _LIST_MARKER_RE.pattern.replace(r"^\s{0,4}", r"^[ \t]{0,4}") + ")",
    re.MULTILINE,
)


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
            # 段落超限：先按列表项边界拆（每项保持完整），再按句子逐级切。
            # 无标记的普通段落保持原行为（直接按句子拆）。
            pieces = re.split(_LIST_MARKER_SPLIT_RE, para)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if utf8len(piece) <= max_bytes:
                    _append_or_flush(chunks, piece, max_bytes, spacer="\n")
                else:
                    for sentence in re.split(r"(?<=[。！？!?\n])", piece):
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
    """已发文本的末尾结构状态（表格内 / 围栏内 / 列表项内 + 围栏 marker）。

    trailing_structure 产出、markdown_safe_cut 消费，捆绑传递避免三参数拆包。
    """

    in_table: bool = False
    in_fence: bool = False
    fence_marker: str | None = None
    in_list_item: bool = False


def trailing_structure(text: str) -> TrailingState:
    """扫描已发文本末尾状态：是否处于表格内 / 代码围栏内（含围栏 marker）/
    列表项内。

    供 markdown_safe_cut 作为 initial 状态，使跨块的表格/围栏/列表延续正确。
    """
    in_fence = False
    fence_marker: str | None = None
    in_table = False
    in_list_item = False
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
            in_list_item = False
        elif is_table_row(line):
            if not in_table:
                in_table = True  # 表头行后（未遇分隔行）也视为表内延续
            in_list_item = False
        elif is_list_marker(line):
            in_table = False
            in_list_item = True  # 列表项开始（含其后续行）
        elif is_url_line(line):
            in_table = False
            in_list_item = True  # 裸 URL：上一行条目的续行（无标记格式）
        elif is_annotation_continuation(line):
            # 「——」注解行是条目收尾：其后是新条目/段落（边界语义）
            in_table = False
            in_list_item = False
        elif line.strip():
            # 非空续行（缩进注释等）：保持列表项状态；普通段落行
            # 前若无空行也保守视为续行（宁小勿断）
            in_table = False
        else:
            in_table = False
            in_list_item = False
    return TrailingState(
        in_table=in_table,
        in_fence=in_fence,
        fence_marker=fence_marker,
        in_list_item=in_list_item,
    )


def markdown_safe_cut(
    text: str,
    limit: int,
    initial: TrailingState | None = None,
) -> int:
    """在 text 的 limit 字符内找安全的块切点：不切断代码围栏、markdown 表格
    与列表项（标记行 + URL/「——」注解续行）。

    返回切点索引（行尾，含换行）；整个文本都处于表格/围栏/列表项内时返回 0，
    调用方跳过本次发送（等结构闭合，收尾补发兜底）或按行尾硬切（超过单条
    字节上限时）。状态机与 split_markdown 保持一致：结构体内不可切，
    普通行行尾即安全切点（表格/围栏/列表项整体留在块内，宁小勿断）。
    initial 由调用方传入已发文本的末尾状态（trailing_structure 产出），
    使跨块的表格/围栏/列表延续正确（pending 开头是续行时行尾可切）。
    """
    if limit <= 0:
        return 0
    safe = 0
    pos = 0
    in_fence = initial.in_fence if initial else False
    fence_marker = initial.fence_marker if initial and initial.in_fence else None
    in_table = initial.in_table if initial else False
    in_list_item = initial.in_list_item if initial else False
    pending_header = False
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if pos >= limit:
            break
        # 下一行是「——」注解 → 本行是列表项的一部分，行尾不可切
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
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
            in_list_item = False
        elif is_table_row(line):
            if in_table:
                # 表内数据行：行尾可切（表格延续，行保持完整；表头在块首或上一块）
                safe = pos + len(line) + 1
            else:
                pending_header = True  # 表头候选：等分隔行确认（表头行尾不可切）
            in_list_item = False
        elif line.strip().startswith("|"):
            # 以 | 开头但未闭合：流式生成中的半截表格行/表头 → 结构内，不可切
            pass
        elif in_list_item:
            if is_list_marker(line):
                # 新列表项开始 → 上一项已完整结束，边界处可切（含行尾换行）
                safe = pos
            elif not line.strip() and pos + len(line) < len(text):
                # 完整空行（有换行结尾）结束列表项：行尾可切。流式生成中的
                # 半截空白行（如缩进写到一半的 "  "，无换行）必须按续行处理——
                # 否则 safe = pos+len+1 越出文本末尾，整段照发造成中线切块。
                # （trailing_structure 扫描的总是完整行，故无此守卫——
                # 两份状态机的空行规则在「扫描完整行 vs 流式增量」下等价。）
                safe = pos + len(line) + 1
                in_list_item = False
            elif is_annotation_continuation(line):
                # 「——」注解行是条目收尾（边界语义，同 trailing_structure）：
                # 其后为新条目/空行，完整注解行（有换行结尾）行尾可切；
                # 半截注解行（模型写到一半）不可切
                end = pos + len(line) + 1
                if end <= len(text):
                    safe = end
                in_list_item = False
            # 其余为续行（URL/缩进）：不可切，避免拆散列表项
        else:
            # 普通行：表格/围栏结束，行尾即安全切点。半截行（流式生成中，
            # 无换行结尾）不更新 safe——否则切点 = len+1 超出文本长度，
            # 调用方不切，半截链接/单词会整段发出（QQ 半截链接的根因）。
            # 下一行是「——」注解或 URL 时本行行尾也不切（条目整体同气泡）。
            in_table = False
            pending_header = False
            if is_list_marker(line):
                in_list_item = True
                safe = pos  # 标记行起点 = 上一行行尾（条目边界）
            elif is_url_line(line):
                pass  # 裸 URL：标题的续行，行尾不设切点
            elif is_annotation_continuation(line):
                # 「——」注解行（无标记格式的条目收尾）：完整行行尾即条目边界
                end = pos + len(line) + 1
                if end <= len(text):
                    safe = end
            elif is_annotation_continuation(next_line) or is_url_line(next_line):
                pass  # 本行是条目标题/内容（下一行是 URL/注解）——不可切
            else:
                # 普通行：行尾可切，但若它是 pending 的最后一行（其后尚无
                # 任何内容），暂不设切点——它可能是列表条目标题，下一行可能
                # 是 URL/注解（等后续行确认，宁慢勿断；收尾补发兜底）。
                end = pos + len(line) + 1
                if end < len(text):
                    safe = end
        pos += len(line) + 1
    return safe


def pending_starts_incomplete(text: str, sent_prefix: str) -> bool:
    """pending 是否以未完成结构开头（表格行 / 注解或 URL 续行 / 围栏体）。

    空闲 flush 时命中则跳过本次发送（等流继续，收尾补发兜底），
    避免把半截表格、代码块或列表项续行发出。
    """
    first = text.split("\n", 1)[0].strip()
    if first and (is_table_separator(first) or is_table_row(first)):
        return True  # 表头/分隔行已在上一块或尚未生成
    if first and (is_annotation_continuation(first) or is_url_line(first)):
        return True  # 「——」注解/裸 URL 是上一项的续行：等完整项生成
    sent_lines = [ln for ln in sent_prefix.rstrip().split("\n") if ln.strip()]
    if sent_lines and is_fence_line(sent_lines[-1]):
        return True  # 已发部分以围栏开始行结尾 → pending 是围栏体
    return False
