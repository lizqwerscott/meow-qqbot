from dataclasses import dataclass, field
from pathlib import Path

from core.text_paging import TextPage, build_pagination_hint


@dataclass(frozen=True)
class MediaRecord:
    media_id: str
    media_uri: str
    chat_id: str
    message_id: str
    sender_id: str
    resource_type: str
    mime_type: str
    size: int
    sha256: str
    local_path: Path
    created_at: float
    expires_at: float
    filename: str = ""
    summary: str = ""
    summary_model: str = ""
    summary_version: str = ""
    file_summary: str = ""
    file_summary_model: str = ""
    file_summary_version: str = ""


@dataclass(frozen=True)
class ImageInspection:
    media_uri: str
    analysis: str = ""
    cached: bool = False
    error: str = ""
    message: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        if self.error:
            return {"error": self.error, "message": self.message}
        d = {
            "media_uri": self.media_uri,
            "analysis": self.analysis,
            "cached": self.cached,
        }
        if self.note:
            d["note"] = self.note
        return d


@dataclass(frozen=True)
class FileInspection:
    media_uri: str
    content: str = ""
    truncated: bool = False
    error: str = ""
    message: str = ""
    # 分页元数据（行级 offset/limit + 字符级 max_chars，语义见 core/text_paging.py）
    start_line: int = 1
    total_lines: int = 0
    output_lines: int = 0
    next_offset: int = 0  # 续读起点（1-based）；0 = 已读完全部
    truncated_by: str = ""  # "" | "lines" | "chars"
    last_line_partial: bool = False

    @classmethod
    def from_page(cls, media_uri: str, page: TextPage) -> "FileInspection":
        """由 TextPage 构造（元数据簇单一来源，避免逐字段拷贝）。"""
        return cls(
            media_uri=media_uri,
            content=page.content + build_pagination_hint(page),
            truncated=page.truncated,
            start_line=page.start_line,
            total_lines=page.total_lines,
            output_lines=page.output_lines,
            next_offset=page.next_offset,
            truncated_by=page.truncated_by.value,
            last_line_partial=page.last_line_partial,
        )

    def as_dict(self) -> dict:
        if self.error:
            return {"error": self.error, "message": self.message}
        return {
            "media_uri": self.media_uri,
            "content": self.content,
            "truncated": self.truncated,
            "start_line": self.start_line,
            "total_lines": self.total_lines,
            "output_lines": self.output_lines,
            "next_offset": self.next_offset,
            "truncated_by": self.truncated_by,
            "last_line_partial": self.last_line_partial,
        }


@dataclass(frozen=True)
class PdfInspection:
    media_uri: str
    analysis: str = ""
    pages: int = 0
    cached: bool = False
    error: str = ""
    message: str = ""

    def as_dict(self) -> dict:
        if self.error:
            return {"error": self.error, "message": self.message}
        return {
            "media_uri": self.media_uri,
            "analysis": self.analysis,
            "pages": self.pages,
            "cached": self.cached,
        }


@dataclass(frozen=True)
class VoiceTranscription:
    media_uri: str
    transcript: str = ""
    cached: bool = False
    error: str = ""
    message: str = ""

    def as_dict(self) -> dict:
        if self.error:
            return {"error": self.error, "message": self.message}
        return {
            "media_uri": self.media_uri,
            "transcript": self.transcript,
            "cached": self.cached,
        }


@dataclass(frozen=True)
class MediaTurnContext:
    current_blocks: tuple[str, ...] = field(default_factory=tuple)
    replied_blocks: tuple[str, ...] = field(default_factory=tuple)
    recent_block: str = ""

    def as_text(self) -> str:
        blocks = [*self.current_blocks, *self.replied_blocks]
        if self.recent_block:
            blocks.append(self.recent_block)
        return "\n\n".join(blocks)
