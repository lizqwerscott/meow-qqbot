import json
import logging

from qqbot_agent_sdk.constants import MEDIA_TYPE_VOICE

from core.ai.tts_service import TtsService
from core.tools._types import ToolContext, ToolEntry, ToolResult
from core.tools.deps import ToolDeps

_log = logging.getLogger(__name__)


def create_tts_entries(deps: ToolDeps) -> list[ToolEntry]:
    tts_service = deps.tts_service.value
    backend_config = (
        tts_service.tool_config
        if isinstance(tts_service, TtsService)
        else TtsService.default_tool_config()
    )

    tool_rules = [backend_config.text_rules]
    if backend_config.supports_instructions:
        tool_rules.append(backend_config.instructions_rules)
    tool_rules.append(backend_config.voice_mode_rules)
    tool_rules_prompt = "\n".join(f"- {rule}" for rule in tool_rules)

    tag_examples = (
        f"\n\n{backend_config.tag_examples}" if backend_config.tag_examples else ""
    )

    tts_properties = {
        "text": {
            "type": "string",
            "description": (
                "要转为语音的文字内容。限制不超过 500 字；至少 3-5 个字，"
                "过短可能生成断裂音频。\n"
                f"当前后端正文规则：{backend_config.text_rules}"
            ),
        },
    }
    if backend_config.supports_instructions:
        tts_properties["instructions"] = {
            "type": "string",
            "description": (
                "说话风格/语气描述（可选）。\n"
                f"当前后端 instructions 规则：{backend_config.instructions_rules}"
            ),
        }
    tts_properties["voice_mode"] = {
        "type": "string",
        "enum": list(backend_config.voice_modes),
        "description": (
            "语音生成模式。\n"
            f"当前后端 voice_mode 规则：{backend_config.voice_mode_rules}"
        ),
    }

    tts_params = {
        "type": "object",
        "properties": tts_properties,
        "required": ["text"],
    }

    async def _synthesize_speech(args: dict, ctx: ToolContext) -> ToolResult:
        tts_service = deps.tts_service.value
        media_uploader = deps.media_uploader.value
        bot_engine = deps.bot_engine.value

        if not tts_service or not media_uploader or not bot_engine:
            return ToolResult(
                content=json.dumps(
                    {"error": "TTS 语音服务或媒体上传器未就绪"},
                    ensure_ascii=False,
                )
            )

        text = (args.get("text") or "").strip()
        if not text:
            return ToolResult(
                content=json.dumps(
                    {"error": "请提供要合成的文本"},
                    ensure_ascii=False,
                )
            )

        instructions = (args.get("instructions") or "").strip()
        voice_mode = (args.get("voice_mode") or "preset").strip()

        if instructions and not tts_service.tool_config.supports_instructions:
            # Models can emit an obsolete field despite the current JSON schema.
            instructions = ""

        MAX_TEXT_LENGTH = 500
        if len(text) > MAX_TEXT_LENGTH:
            return ToolResult(
                content=json.dumps(
                    {
                        "error": f"文本过长（{len(text)} 字），超过 {MAX_TEXT_LENGTH} 字限制，请精简后重试"
                    },
                    ensure_ascii=False,
                )
            )

        try:
            audio_bytes = await tts_service.synthesize(
                text=text,
                instructions=instructions or None,
                voice_mode=voice_mode,
            )
        except Exception as e:
            return ToolResult(
                content=json.dumps(
                    {"error": f"语音合成失败: {e}"},
                    ensure_ascii=False,
                )
            )

        if not audio_bytes:
            return ToolResult(
                content=json.dumps(
                    {"error": "语音合成无输出"},
                    ensure_ascii=False,
                )
            )

        temp_path = tts_service.save_temp_audio(audio_bytes)

        effective_chat_id = ctx.delivery_channel or ctx.chat_id
        effective_reply_to = None if ctx.delivery_channel else ctx.reply_to
        chat_type = "group" if ctx.is_group else "c2c"

        try:
            file_info = await media_uploader.upload(
                chat_type=chat_type,
                chat_id=effective_chat_id,
                source=temp_path,
                file_type=MEDIA_TYPE_VOICE,
                file_name="tts.wav",
            )
        except Exception as e:
            return ToolResult(
                content=json.dumps(
                    {"error": f"语音上传失败: {e}"},
                    ensure_ascii=False,
                )
            )

        try:
            if effective_reply_to:
                await bot_engine.send_reply(
                    chat_id=effective_chat_id,
                    is_group=ctx.is_group,
                    message_id=effective_reply_to,
                    media_file_info=file_info,
                )
            else:
                await bot_engine.send_proactive(
                    chat_id=effective_chat_id,
                    is_group=ctx.is_group,
                    media_file_info=file_info,
                )
        except Exception as e:
            return ToolResult(
                content=json.dumps(
                    {"error": f"发送语音失败: {e}"},
                    ensure_ascii=False,
                )
            )

        return ToolResult(
            content=json.dumps(
                {
                    "success": True,
                    "message": "语音已发送到聊天中",
                },
                ensure_ascii=False,
            )
        )

    return [
        ToolEntry(
            name="synthesize_speech",
            section="tts",
            description=(
                "将文字转换为语音并发送到聊天中。\n\n"
                "在以下场景应主动调用：\n"
                "1. 用户要求你说句话、念一段文字、或问你能不能说话时\n"
                "2. 你觉得用语音回复比纯文字更有表现力时（如问候、祝贺、撒娇、吐槽等）\n"
                "3. 用户明确要求用某种语气说话（如'用热情的语气说'、'温柔地说'）\n"
                "4. 回复内容较短且有感情色彩，适合语音表达\n\n"
                "当前后端模型规则：\n"
                f"{tool_rules_prompt}"
                f"{tag_examples}"
            ),
            parameters=tts_params,
            handler=_synthesize_speech,
        ),
    ]
