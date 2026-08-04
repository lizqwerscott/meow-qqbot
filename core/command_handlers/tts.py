import logging
from typing import Any, Dict, List

from core.ai.tts_service import TtsService
from core.command_handlers.base import command, make_reply
from core.engine.client import BotEngine
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="tts", aliases=["语音", "朗读"], description="将文字转为语音")
class TtsCommand:
    def __init__(self, bot_engine: BotEngine, tts_service: TtsService):
        self.bot_engine = bot_engine
        self.tts_service = tts_service

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if not self.tts_service:
            return make_reply(input_message, "TTS 语音服务未启用")

        text = args.strip()
        if not text:
            return make_reply(
                input_message, "用法：/tts <文字>\n例如：/tts 大家好，我是猫猫"
            )

        audio_bytes = await self.tts_service.synthesize(text)
        if not audio_bytes:
            return make_reply(input_message, "语音合成失败，请稍后重试")

        temp_path = self.tts_service.save_temp_audio(audio_bytes)
        chat_type = "group" if input_message.is_group else "c2c"

        try:
            from qqbot_agent_sdk.constants import MEDIA_TYPE_VOICE

            uploader = self.bot_engine.media_uploader
            if not uploader:
                return make_reply(input_message, "媒体上传器未就绪")

            file_info = await uploader.upload(
                chat_type=chat_type,
                chat_id=input_message.chat_id,
                source=temp_path,
                file_type=MEDIA_TYPE_VOICE,
                file_name="tts.wav",
            )

            await self.bot_engine.send_reply(
                chat_id=input_message.chat_id,
                content="",
                message_id=input_message.id,
                is_group=input_message.is_group,
                media_file_info=file_info,
            )
            return []

        except Exception as e:
            _log.error("TTS 命令发送失败: %s", e, exc_info=True)
            return make_reply(input_message, f"语音发送失败: {e}")
