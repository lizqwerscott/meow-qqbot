import json

from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps


def create_media_entries(deps: ToolDeps) -> list[ToolEntry]:
    params = {
        "type": "object",
        "properties": {
            "media_uri": {
                "type": "string",
                "description": "当前会话中的 media://inbound/... 图片引用",
            },
            "question": {
                "type": "string",
                "description": "需要从图片中确认的具体问题",
            },
        },
        "required": ["media_uri", "question"],
        "additionalProperties": False,
    }

    async def _image(args: dict, ctx: ToolContext) -> ToolResult:
        service = deps.media_service
        if service is None:
            return ToolResult(content=json.dumps({"error": "MEDIA_NOT_AVAILABLE"}))
        result = await service.inspect_image(
            chat_id=ctx.chat_id,
            media_uri=str(args.get("media_uri", "")),
            question=str(args.get("question", "")),
        )
        return ToolResult(content=json.dumps(result.as_dict(), ensure_ascii=False))

    entries = []
    if service := deps.media_service:
        if service.image_tools_enabled:
            entries.append(
                ToolEntry(
                    name="image",
                    section="media",
                    description=(
                        "查看当前会话中的一张图片。仅在摘要不足、需要识别细节、文字、关系，"
                        "或用户明确要求深入分析时调用。media_uri 必须来自媒体上下文。"
                    ),
                    parameters=params,
                    handler=_image,
                )
            )

        if service.pdf_tools_enabled:

            async def _pdf(args: dict, ctx: ToolContext) -> ToolResult:
                result = await service.inspect_pdf(
                    chat_id=ctx.chat_id,
                    media_uri=str(args.get("media_uri", "")),
                    prompt=str(args.get("prompt", "分析这份 PDF 文档。")),
                )
                return ToolResult(
                    content=json.dumps(result.as_dict(), ensure_ascii=False)
                )

            entries.append(
                ToolEntry(
                    name="pdf",
                    section="media",
                    description=(
                        "分析当前会话中的 PDF 文档。" "media_uri 必须来自媒体上下文。"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "media_uri": {
                                "type": "string",
                                "description": "当前会话中的 media://inbound/... PDF 引用",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "希望从 PDF 中分析的问题或要求",
                            },
                        },
                        "required": ["media_uri"],
                        "additionalProperties": False,
                    },
                    handler=_pdf,
                )
            )

    return entries
