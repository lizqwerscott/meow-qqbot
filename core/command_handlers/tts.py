import logging
from typing import Any, Dict, List

from core.ai.tts_service import TtsService
from core.command_handlers.base import command, make_reply
from core.engine.client import BotEngine
from core.engine.delivery_ledger import DeliveryController, DeliveryReceipt
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(name="tts", aliases=["语音", "朗读"], description="将文字转为语音")
class TtsCommand:
    def __init__(
        self,
        bot_engine: BotEngine,
        tts_service: TtsService,
        delivery_controller: DeliveryController | None = None,
    ):
        self.bot_engine = bot_engine
        self.tts_service = tts_service
        self.delivery_controller = delivery_controller

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

            async def _send_media(**kwargs):
                return await self.bot_engine.send_reply(
                    chat_id=kwargs["chat_id"],
                    content="",
                    message_id=kwargs["message_id"],
                    is_group=kwargs["is_group"],
                    media_file_info=file_info,
                )

            if self.delivery_controller is not None:
                receipt = await self.delivery_controller.deliver_text(
                    delivery_id=(
                        f"command:{input_message.chat_id}:{input_message.id}:tts"
                    ),
                    chat_id=input_message.chat_id,
                    content="",
                    callback=_send_media,
                    message_id=input_message.id,
                    is_group=input_message.is_group,
                    reason="command_media",
                    timeline_delivery_kind=None,
                )
            else:
                receipt = await _send_media(
                    chat_id=input_message.chat_id,
                    message_id=input_message.id,
                    is_group=input_message.is_group,
                )
            if isinstance(receipt, DeliveryReceipt) and receipt.status not in {
                "accepted",
                "partial",
            }:
                return make_reply(input_message, "语音发送未确认，请稍后重试")
            return []

        except Exception as e:
            _log.error("TTS 命令发送失败: %s", e, exc_info=True)
            return make_reply(input_message, f"语音发送失败: {e}")
