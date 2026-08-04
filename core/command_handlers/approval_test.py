import asyncio
import logging
import time
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage

_log = logging.getLogger(__name__)


@command(
    name="测试审批",
    aliases=["test-approval", "审批测试", "approval"],
    permission="admin",
    description="测试审批流程和自定义键盘按钮",
)
class ApprovalTestCommand:
    def __init__(self, api_client=None, bot_engine=None):
        self.api = api_client
        self.bot_engine = bot_engine

    async def execute(
        self, input_message: InputMessage, args: str
    ) -> List[Dict[str, Any]]:
        if self.api is None:
            return make_reply(input_message, "❌ api_client 未注入")

        from qqbot_agent_sdk import ApprovalRequest, ApprovalSender, InlineKeyboard
        from qqbot_agent_sdk.dto import (
            KeyboardButton,
            KeyboardButtonAction,
            KeyboardButtonRenderData,
            KeyboardContent,
            KeyboardRow,
        )

        chat_type = "group" if input_message.is_group else "c2c"
        chat_id = input_message.chat_id

        # ── 1. 发送审批消息 ──
        session_key = f"approval_test_{int(time.time())}"
        approval = ApprovalRequest(
            session_key=session_key,
            title="🔧 测试审批请求",
            description="这是一个测试审批流程，请点击下方按钮回复",
            command_preview="echo 'approval test'",
            cwd="/tmp",
            severity="info",
            timeout_sec=60,
        )

        # 注册到审批管理器（以便按钮回调能识别 session_key）
        if self.bot_engine and self.bot_engine.approval_manager:
            future = asyncio.get_running_loop().create_future()
            self.bot_engine.approval_manager._pending[session_key] = future

        approval_sender = ApprovalSender(self.api, log_tag="ApprovalTest")
        await approval_sender.send(
            chat_type=chat_type,
            chat_id=chat_id,
            req=approval,
            msg_id=input_message.id,
        )

        # ── 2. 发送自定义键盘 ──
        keyboard = InlineKeyboard(
            content=KeyboardContent(
                rows=[
                    KeyboardRow(
                        buttons=[
                            KeyboardButton(
                                id="btn_a",
                                render_data=KeyboardButtonRenderData(
                                    label="选项 A",
                                    visited_label="已选 A",
                                    style=1,
                                ),
                                action=KeyboardButtonAction(
                                    type=1, data="test_keyboard:A"
                                ),
                            ),
                            KeyboardButton(
                                id="btn_b",
                                render_data=KeyboardButtonRenderData(
                                    label="选项 B",
                                    visited_label="已选 B",
                                    style=0,
                                ),
                                action=KeyboardButtonAction(
                                    type=1, data="test_keyboard:B"
                                ),
                            ),
                        ]
                    ),
                    KeyboardRow(
                        buttons=[
                            KeyboardButton(
                                id="btn_c",
                                render_data=KeyboardButtonRenderData(
                                    label="选项 C",
                                    visited_label="已选 C",
                                    style=1,
                                ),
                                action=KeyboardButtonAction(
                                    type=1, data="test_keyboard:C"
                                ),
                            ),
                        ]
                    ),
                ]
            )
        )

        if self.bot_engine:
            await self.bot_engine.send_reply(
                chat_id,
                "📋 请选择一个选项：",
                message_id=input_message.id,
                is_group=input_message.is_group,
                keyboard=keyboard,
            )

        return make_reply(
            input_message, "✅ 已发送审批消息和自定义键盘测试，请在聊天中查看并点击按钮"
        )
