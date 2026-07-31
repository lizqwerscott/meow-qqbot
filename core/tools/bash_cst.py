"""tree-sitter-bash 语法分析 — CST 级命令切段（对齐 openclaw command-explainer）。

价值：shlex 只是引号感知 tokenizer，不知道 `$(...)`、反引号、复合命令、
管道优先级等 bash 语法。tree-sitter-bash 产出具体语法树（CST），本模块
从中提取：

- **顶层段**：按链操作符（&& || ; | &）切分的命令单元，段内保留原始 token
  （含重定向 `> file`，与 shlex 行为一致），执行语义不变
- **command_substitution / process_substitution**：`$(...)`、反引号、
  `<(...)` 内部的命令文本（供递归分析）
- **复合命令**：for/while/if/case 等标记 is_compound（shell=False 下无
  对应可执行文件，fail-closed 走审批/拒绝）
- **尾随重定向**：tree-sitter 会把 `a && b > f` 的重定向提升为包裹整个
  链的 redirected_statement——递归展开内部链，重定向挂到最后一个段

解析失败（语法错误）返回 None → 调用方 fail-closed 拒绝，绝不回退到
放宽的解析。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import tree_sitter_bash
from tree_sitter import Language, Node, Parser

_log = logging.getLogger(__name__)

# 链操作符节点（CST 中为独立 token 节点）
_CHAIN_TOKENS = frozenset({"&&", "||", ";", "|", "&"})

# 简单命令单元节点（pipeline 由专用分支处理，不在集合内）
_COMMAND_NODES = frozenset(
    {
        "command",
        "concatenation",
    }
)

# 复合命令节点（shell=False 下无独立可执行文件，fail-closed）
_COMPOUND_NODES = frozenset(
    {
        "for_statement",
        "while_statement",
        "if_statement",
        "case_statement",
        "function_definition",
        "do_group",
        "subshell",
        "c_style_for_statement",
        "declaration_command",
        "unset_command",
        "variable_assignment",
        "test_command",
        "compound_command",
        "heredoc_redirect",
    }
)

# 重定向节点（附加到所属命令段末尾，保持执行语义）
_REDIRECT_NODES = frozenset({"file_redirect", "heredoc_redirect", "redirect"})

# 内部命令容器节点（redirected_statement 包裹链时递归展开）
_REDIRECT_INNER_NODES = frozenset(
    {"list", "pipeline", "command", "concatenation", "redirected_statement"}
)

_language: Optional[Language] = None
_parser: Optional[Parser] = None


def _get_parser() -> Optional[Parser]:
    global _language, _parser
    if _parser is not None:
        return _parser
    try:
        _language = Language(tree_sitter_bash.language())
        _parser = Parser(_language)
    except Exception as e:
        _log.error("tree-sitter-bash 初始化失败: %s", e)
        _parser = None
    return _parser


@dataclass
class CstSegment:
    """CST 提取出的顶层命令单元。"""

    op: str  # 与上一段的连接符（首段 ''）
    text: str  # 段原始文本（含重定向），供 shlex 提取 argv
    substitutions: List[str] = field(
        default_factory=list
    )  # $(...) / `...` / <(...) 内部文本
    is_compound: bool = False  # 复合命令（for/if/...）


def _extract_substitutions(node: Node, out: List[str], depth: int = 0) -> None:
    """收集节点内所有 command_substitution / process_substitution 的内部命令文本。"""
    if depth > 8:
        return
    if node.type in ("command_substitution", "process_substitution"):
        text = node.text.decode("utf-8", errors="replace")
        inner = text
        if text.startswith("$(") and text.endswith(")"):
            inner = text[2:-1]
        elif text.startswith("`") and text.endswith("`"):
            inner = text[1:-1]
        elif text.startswith("<(") and text.endswith(")"):
            inner = text[2:-1]
        if inner.strip():
            out.append(inner)
        return
    for child in node.children:
        _extract_substitutions(child, out, depth + 1)


def _collect_chain(node: Node, segments: List[CstSegment], cur_op: str) -> str:
    """递归收集顶层段；返回下一段的连接操作符。

    list / pipeline 语义一致（子节点含操作符 token），合并处理。
    """
    if node.type in _CHAIN_TOKENS:
        return node.type
    if node.type in ("list", "pipeline"):
        op = cur_op
        for child in node.children:
            op = _collect_chain(child, segments, op)
        return op
    if node.type == "redirected_statement":
        # 分离内部命令容器与重定向。tree-sitter 会把 `a && b > f` 的重定向
        # 提升为包裹整个链的 redirected_statement：展开内部链，重定向挂到
        # 最后一个段，避免整链塌缩成单段导致后续命令逃过 allowlist。
        inner = None
        redirect_parts: List[str] = []
        for child in node.children:
            if child.type in _REDIRECT_INNER_NODES:
                inner = child
            elif child.type in _REDIRECT_NODES:
                redirect_parts.append(child.text.decode("utf-8", errors="replace"))
        op = _collect_chain(inner, segments, cur_op) if inner is not None else cur_op
        if redirect_parts and segments:
            segments[-1].text = (
                segments[-1].text + " " + " ".join(redirect_parts)
            ).strip()
        return op
    if node.type in _COMPOUND_NODES:
        text = node.text.decode("utf-8", errors="replace").strip()
        if text:
            subs: List[str] = []
            _extract_substitutions(node, subs)
            segments.append(
                CstSegment(op=cur_op, text=text, substitutions=subs, is_compound=True)
            )
        return ""
    if node.type in _COMMAND_NODES:
        text = node.text.decode("utf-8", errors="replace").strip()
        if text:
            subs = []
            _extract_substitutions(node, subs)
            segments.append(CstSegment(op=cur_op, text=text, substitutions=subs))
        return ""
    # 其他节点：递归找命令
    op = cur_op
    for child in node.children:
        op = _collect_chain(child, segments, op)
    return op


def parse_shell_command(command: str) -> Optional[List[CstSegment]]:
    """tree-sitter 解析命令 → 顶层段列表。

    Returns:
        CstSegment 列表；解析失败或语法错误返回 None（调用方 fail-closed）。
    """
    parser = _get_parser()
    if parser is None:
        return None
    try:
        tree = parser.parse(command.encode("utf-8"))
    except Exception as e:
        _log.warning("tree-sitter 解析异常: %s", e)
        return None
    root = tree.root_node
    if root.has_error:
        _log.warning("shell 命令语法错误，fail-closed 拒绝: %s", command[:80])
        return None
    segments: List[CstSegment] = []
    _collect_chain(root, segments, "")
    return segments
