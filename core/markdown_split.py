"""Markdown 文本拆分 — 发送前的安全切块与流式转发安全切点。

同一份围栏/表格/列表状态机供两处使用（曾各抄一份，改动必须同步，现已归位）：

1. `split_markdown` — 非流式发送前的字节级拆块（QQ 单条消息上限兜底）。
2. `markdown_safe_cut` / `trailing_structure` / `pending_starts_incomplete` —
   流式 block 转发时找不切断代码围栏/markdown 表格/列表项的切点。

判定谓词（is_fence_line / is_table_* / is_list_marker / is_url_line /
is_annotation_continuation / is_heading_line / is_continuation_line）是两份
状态机的公共基础，改动只需在本模块内同步。
列表项 = 标记行（`1.` / `-` / `*` 等）+ 续行（URL / 「——」注解 / 缩进行）：
切点只落在项边界、空行与「——」注解行尾，绝不拆散条目。
标题行（ATX `#` 或「短行 + 前一行不是续行」，见 is_heading_line）绑定其
后内容（含标题后首个空行）：切点/块尾不落在标题行尾——标题与其列表同块，
不孤悬块尾。
超过单条字节上限（MARKDOWN_SAFE_CHUNK_BYTE_LIMIT）的列表项按行尾 /
UTF-8 安全字节边界硬切——QQ 单条消息上限约束下的既定行为（宁断勿丢）。
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


# 标题行判定：ATX 标题，或「短行 + 前一行不是续行」的启发式。
_ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_HEADING_MAX_LEN = 24  # 标题行字符上限（短行启发式）

# 续行判定：URL / 表格行 / 围栏行 / 半截 | 行——这些行不结束一个块。
# 供标题行判定（is_heading_line 的行内排除 + after_boundary 上下文）与
# 拆块后置处理（_de_orphan_headings）共用，避免第三份复制漂移。


def is_continuation_line(line: str) -> bool:
    """续行判定：裸 URL / 表格行 / 表格分隔行 / 围栏行 / 半截 `|` 行。

    这些行是前一结构的续行或结构行，不结束一个块——其后的行不构成
    「新块」上下文（标题行判定的 after_boundary 不成立），拆块时也不会
    被当作可移动的标题（is_heading_line 行内排除）。
    """
    if (
        is_url_line(line)
        or is_table_row(line)
        or is_table_separator(line)
        or is_fence_line(line)
    ):
        return True
    return line.lstrip().startswith("|")


def is_heading_line(line: str, after_boundary: bool = False) -> bool:
    """标题行判定：ATX 标题（`# ` 开头，`## 每日推荐`）无条件成立；
    普通短行（≤ _HEADING_MAX_LEN 字符、不以句末标点结尾）还需前一行
    不是续行（after_boundary=True：空行 / 「——」注解 / 列表标记行 / 普通行 /
    文本起点）——`🧠 机器之心（今日 3 篇）`、`📄 arXiv cs.AI（今日更新）`、
    `🔧 实用求助/讨论：` 都落在「短行 + 前一行不是续行」上。

    前一行必须是「非续行」而非严格块边界：无标记格式下模型在普通行（开场白）
    后直接写段落标题，只认空行/注解会漏判（标题照旧孤悬块尾）。
    「——」注解行与续行（URL / 表格行 / 围栏行 / 半截 `|` 行，is_continuation_line）
    行内直接排除：注解是条目的收尾不是标题（防止短注解被当标题移走造成
    注解孤儿——与流式状态机先判注解的顺序一致），条目标题跟在 URL 后
    不构成标题。只短不行：无标记列表里条目也短（`ForestBench：…`），全判成
    标题会把切点整体后移甚至失去安全切点；但状态机在标题行后把标志置 False，
    短条目链上切点仍保持（隔行可切，宁多勿跑）；句末标点收尾的短行是
    内容不是标题（`明白了！`）。
    标题行绑定其后内容：拆块/切点不落在标题行尾——标题与其列表同气泡，
    避免「标题孤悬块尾、列表在下一块」的视觉断裂（tmp/new_chat.txt 事故）。
    """
    if is_annotation_continuation(line) or is_continuation_line(line):
        return False
    if _ATX_HEADING_RE.match(line):
        return True
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_LEN:
        return False
    if stripped.endswith(("。", "！", "？", "!", "?")):
        return False
    return after_boundary


def is_annotation_continuation(line: str) -> bool:
    """「—」/「——」开头的注解/续行（猫猫回复里列表项的点评行）。

    视为上一列表项的续行：切块时不允许把注解与其标题/链接拆到两个气泡。
    单/双 em-dash 都匹配：单「—」注解不被保护会造成注解孤儿（视觉断裂），
    而散文/对话破折号行被误粘只拖累延迟——正确性由收尾补发兜底，
    误拆比误粘更伤体验。
    """
    return line.lstrip().startswith("—")


_URL_LINE_RE = re.compile(
    r"^\s{0,4}"
    r"(?:"
    r"(?:https?://|www\.)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?:/\S*)?"
    # IPv4 仅带端口/路径时视为 URL——裸 IP 行（如 192.168.100.203）是
    # 普通文本（服务器地址枚举），不应被当作不可切续行粘住
    r"|(?:(?:\d{1,3}\.){3}\d{1,3}(?:(?::\d{1,5})|(?:/\S+)))"
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
                # 只去换行不去缩进：列表项可能带缩进（嵌套列表），
                # strip() 会把缩进吃掉（项被降级）
                piece = piece.strip("\n")
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
            if len(table_lines) < 3:
                # 表头+分隔行而无数据行（2 行）不构成表格：回归文本行，
                # 不丢弃（宁断勿丢——否则这两行内容凭空消失）
                text_lines.extend(table_lines)
            else:
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

    return _de_orphan_headings(chunks, max_bytes)


def _de_orphan_headings(chunks: list[str], max_bytes: int) -> list[str]:
    """标题不孤悬块尾：块以标题行结尾（且存在后续块）时，把标题行移到
    下一块开头——标题与其列表同块。

    块内上下文：前一行不是续行（空行 / 「——」注解 / 列表标记行 / 普通行，
    用 is_continuation_line 判定）或该块只有标题一行（单行块视同边界）；
    复用 is_heading_line 短行启发式，与流式状态机的标题判定一致。注解 /
    URL / 表格 / 围栏行不会被误移（is_heading_line 行内排除）。

    字节上限不破：下一块接近 max_bytes 放不下标题时，先把其尾部行顺延到
    再下一块（级联，保持顺序）腾出空间——任何块都不超过 max_bytes
    （标题 ≤ 97B，顺延必然腾得出；极端单行块整体顺延后标题单独成块）。
    """
    out: list[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        lines = chunk.split("\n")
        if i + 1 < len(chunks) and lines and lines[-1].strip():
            last = lines[-1]
            prev = lines[-2].strip() if len(lines) >= 2 else ""
            prev_boundary = not prev or bool(prev and not is_continuation_line(prev))
            if is_heading_line(last, prev_boundary):
                lines.pop()
                nxt_lines = chunks[i + 1].split("\n")
                carried: list[str] = []
                # 标题移入下一块；放不下则顺延其尾部行（逆序收集、正序拼接）
                while nxt_lines and (
                    utf8len(last) + 1 + utf8len("\n".join(nxt_lines)) > max_bytes
                ):
                    carried.insert(0, nxt_lines.pop())
                if lines:
                    # 补回被弹出的标题行后的换行：annotation\nheading → annotation\n
                    out.append("\n".join(lines) + "\n")
                chunks[i + 1] = last + "\n" + "\n".join(nxt_lines)
                if carried:
                    chunks.insert(i + 2, "\n".join(carried))
                i += 1
                continue
        out.append(chunk)
        i += 1
    return out


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
    prev_boundary = True  # pending 首行：前一行不是续行（已发文本末行或文本起点）
    heading_pending = False  # 刚处理过标题行：紧随的空行仍绑定标题（不设切点）
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if pos >= limit:
            break
        this_is_heading = False
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
            prev_boundary = True  # 围栏开/闭行都是块边界
        elif in_fence:
            prev_boundary = False  # 围栏体内：其后行不是块边界
        elif is_table_separator(line):
            in_table = True
            pending_header = False
            in_list_item = False
            prev_boundary = True
        elif is_table_row(line):
            if in_table:
                # 表内数据行：行尾可切（表格延续，行保持完整；表头在块首或上一块）
                safe = pos + len(line) + 1
            else:
                pending_header = True  # 表头候选：等分隔行确认（表头行尾不可切）
            in_list_item = False
            prev_boundary = False
        elif line.strip().startswith("|"):
            # 以 | 开头但未闭合：流式生成中的半截表格行/表头 → 结构内，不可切
            prev_boundary = False
        elif in_list_item:
            if is_list_marker(line):
                # 新列表项开始 → 上一项已完整结束，边界处可切（含行尾换行）
                safe = pos
                prev_boundary = True
            elif not line.strip() and pos + len(line) < len(text):
                # 完整空行（有换行结尾）结束列表项：行尾可切。流式生成中的
                # 半截空白行（如缩进写到一半的 "  "，无换行）必须按续行处理——
                # 否则 safe = pos+len+1 越出文本末尾，整段照发造成中线切块。
                # （trailing_structure 扫描的总是完整行，故无此守卫——
                # 两份状态机的空行规则在「扫描完整行 vs 流式增量」下等价。）
                safe = pos + len(line) + 1
                in_list_item = False
                prev_boundary = True
            elif is_annotation_continuation(line):
                # 「——」注解行是条目收尾（边界语义，同 trailing_structure）：
                # 其后为新条目/空行，完整注解行（有换行结尾）行尾可切；
                # 半截注解行（模型写到一半）不可切
                end = pos + len(line) + 1
                if end <= len(text):
                    safe = end
                in_list_item = False
                prev_boundary = True
            else:
                prev_boundary = False  # 其余为续行（URL/缩进）：非块边界
        else:
            # 普通行：表格/围栏结束，行尾即安全切点。半截行（流式生成中，
            # 无换行结尾）不更新 safe——否则切点 = len+1 超出文本长度，
            # 调用方不切，半截链接/单词会整段发出（QQ 半截链接的根因）。
            # 下一行是「——」注解或 URL 时本行行尾也不切（条目整体同气泡）。
            # 标题行（ATX 或 短行+前一行是块边界）行尾也不切：标题与其
            # 列表同气泡（标题不孤悬块尾，tmp/new_chat.txt 的 arXiv 标题事故）。
            in_table = False
            pending_header = False
            if is_list_marker(line):
                in_list_item = True
                # 标记行起点 = 上一行行尾（条目边界）。标题仍在绑定时
                # （标题+空行+首个标记行）不设切点——否则标题在标记前孤悬
                if not heading_pending:
                    safe = pos
                prev_boundary = True
            elif is_continuation_line(line):
                # 续行（此处等价于裸 URL——围栏/表格/半截 | 行已在上方分支
                # 拦截；共用谓词避免第三份复制漂移）：行尾不设切点
                prev_boundary = False
            elif is_annotation_continuation(line):
                # 「——」注解行（无标记格式的条目收尾）：完整行行尾即条目边界
                end = pos + len(line) + 1
                if end <= len(text):
                    safe = end
                prev_boundary = True
            elif is_heading_line(line, prev_boundary):
                # 标题行绑定其后内容（含紧随的空行）：行尾不设切点
                # （等列表内容确认，宁慢勿断）
                prev_boundary = False
                this_is_heading = True
            elif is_annotation_continuation(next_line) or is_url_line(next_line):
                prev_boundary = False  # 本行是条目标题/内容（下一行是 URL/注解）
            elif not line.strip():
                # 空行：块边界，行尾可切（其后行按新块看待）。标题后第一个
                # 空行除外——CommonMark 风格「标题 + 空行 + 列表」里空行被
                # 标题绑定（标题跨空行粘住列表，不设切点，宁慢勿断）
                if not heading_pending:
                    end = pos + len(line) + 1
                    if end < len(text):
                        safe = end
                    prev_boundary = True
            else:
                # 普通行：行尾可切，但若它是 pending 的最后一行（其后尚无
                # 任何内容），暂不设切点——它可能是列表条目标题，下一行可能
                # 是 URL/注解（等后续行确认，宁慢勿断；收尾补发兜底）。
                # 普通行后置 True：下一短行仍按标题候选（无标记格式的段落
                # 标题常跟在开场白等普通行后）；标题行置 False 打断链条，
                # 短条目列表的切点不整体消失（隔行可切）。
                end = pos + len(line) + 1
                if end < len(text):
                    safe = end
                prev_boundary = True
        pos += len(line) + 1
        # 标题绑定只存活到第一个非空行：标题行置 True，空行继承绑定
        # （连续空行也绑定），其余非空行消费绑定
        if this_is_heading:
            heading_pending = True
        elif line.strip():
            heading_pending = False
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
