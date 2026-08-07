from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass(frozen=True)
class ImageInspection:
    media_uri: str
    analysis: str = ""
    cached: bool = False
    error: str = ""
    message: str = ""

    def as_dict(self) -> dict:
        if self.error:
            return {"error": self.error, "message": self.message}
        return {
            "media_uri": self.media_uri,
            "analysis": self.analysis,
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
