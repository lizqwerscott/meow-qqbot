import json
import logging

from qqbot_agent_sdk.constants import MEDIA_TYPE_VOICE

from core.tools._types import ToolEntry, ToolResult, ToolContext
from core.tools.impl import _DEPS

_log = logging.getLogger(__name__)


async def _synthesize_speech(args: dict, ctx: ToolContext) -> ToolResult:
    tts_service = _DEPS.get("tts_service")
    media_uploader = _DEPS.get("media_uploader")
    bot_engine = _DEPS.get("bot_engine")

    if not tts_service or not media_uploader or not bot_engine:
        return ToolResult(content=json.dumps(
            {"error": "TTS 语音服务或媒体上传器未就绪"},
            ensure_ascii=False,
        ))

    text = (args.get("text") or "").strip()
    if not text:
        return ToolResult(content=json.dumps(
            {"error": "请提供要合成的文本"}, ensure_ascii=False,
        ))

    instructions = (args.get("instructions") or "").strip()
    voice_mode = (args.get("voice_mode") or "preset").strip()

    try:
        audio_path = await tts_service.synthesize(
            text=text,
            instructions=instructions or None,
            voice_mode=voice_mode,
        )
    except Exception as e:
        return ToolResult(content=json.dumps(
            {"error": f"语音合成失败: {e}"}, ensure_ascii=False,
        ))

    if not audio_path:
        return ToolResult(content=json.dumps(
            {"error": "语音合成无输出"}, ensure_ascii=False,
        ))

    effective_chat_id = ctx.delivery_channel or ctx.chat_id
    is_background = bool(ctx.delivery_channel)
    effective_reply_to = None if is_background else ctx.reply_to
    chat_type = "group" if ctx.is_group else "c2c"

    try:
        file_info = await media_uploader.upload(
            chat_type=chat_type,
            chat_id=effective_chat_id,
            source=str(audio_path),
            file_type=MEDIA_TYPE_VOICE,
        )
    except Exception as e:
        return ToolResult(content=json.dumps(
            {"error": f"语音上传失败: {e}"}, ensure_ascii=False,
        ))

    try:
        if effective_reply_to:
            await bot_engine.send_reply(
                chat_id=effective_chat_id, is_group=ctx.is_group,
                message_id=effective_reply_to, media_file_info=file_info,
            )
        else:
            await bot_engine.send_proactive(
                chat_id=effective_chat_id, is_group=ctx.is_group,
                media_file_info=file_info,
            )
    except Exception as e:
        return ToolResult(content=json.dumps(
            {"error": f"发送语音失败: {e}"}, ensure_ascii=False,
        ))

    return ToolResult(content=json.dumps({
        "success": True,
        "message": "语音已发送到聊天中",
    }, ensure_ascii=False))


TTS_PARAMS = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": (
                "要转为语音的文字内容。请使用短句配合标点控制停顿韵律（句号长停顿、逗号短中断、省略号犹豫），"
                "过长的文本会导致生成超时，应在 30 秒可读完的范围内。"
                "如需方言，使用地道方言词汇。"
                "可在文中加 [laughing]、[sigh] 等非语言标签增强表现力。"
                "注意：text 至少 3-5 个字，太短会生成断裂的音频。"
            ),
        },
        "instructions": {
            "type": "string",
            "description": (
                "说话风格/语气描述（可选）。\n"
                "根据 voice_mode 行为不同：\n"
                "- preset（默认）：只调整情绪、语速和表达方式，不要写性别/年龄/身份。"
                "例如「语速稍快，语气热情」「温柔地慢慢说，略带笑意」\n"
                "- creative：自由设计音色。按三维结构：身份基底 + 音色质感 + 情绪表现力。"
                "例如「热情洋溢的中年男性播音员，声音低沉富有磁性」"
            ),
        },
        "voice_mode": {
            "type": "string",
            "enum": ["preset", "creative"],
            "description": (
                "语音生成模式。\n"
                "- preset（默认）：使用管理员预设的固定音色，instructions 仅调整情绪/语速/表达方式，不改变音色身份\n"
                "- creative：自由创造新声音，instructions 可完整指定身份+音色+情绪\n"
                "注意：preset 模式不要尝试改变音色身份，creative 模式会丢弃预设音色"
            ),
        },
    },
    "required": ["text"],
}


def _register_all(register):
    register(ToolEntry(
        name="synthesize_speech",
        section="tts",
        description=(
            "将文字转换为语音并发送到聊天中。\n\n"
            "在以下场景应主动调用：\n"
            "1. 用户要求你说句话、念一段文字、或问你能不能说话时\n"
            "2. 你觉得用语音回复比纯文字更有表现力时（如问候、祝贺、撒娇、吐槽等）\n"
            "3. 用户明确要求用某种语气说话（如'用热情的语气说'、'温柔地说'）\n"
            "4. 回复内容较短且有感情色彩，适合语音表达"
        ),
        parameters=TTS_PARAMS,
        handler=_synthesize_speech,
    ))
