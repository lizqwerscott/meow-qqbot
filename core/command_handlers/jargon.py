"""俚语管理命令：猫猫学词 / 猫猫词典 / 猫猫删除词"""

import logging
from typing import Any, Dict, List

from core.command_handlers.base import command, make_reply
from core.message import InputMessage
from core.learners.orchestrator import LearningOrchestrator

_log = logging.getLogger(__name__)


@command(name="学词", aliases=["learn", "学个词"], permission="admin", description="添加社群俚语。用法：猫猫学词 <词> = <定义> [例:<例句>]")
class JargonLearnCommand:
    def __init__(self, learning_orchestrator: LearningOrchestrator):
        self.learners = learning_orchestrator

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        if not self.learners:
            return make_reply(input_message, "学习系统未启用。")
        if not args.strip():
            return make_reply(input_message, "用法：猫猫学词 <词> = <定义> [例:<例句>]")

        term = ""
        definition = ""
        example = ""

        if "=" in args:
            term, rest = args.split("=", 1)
            term = term.strip()
            if "例:" in rest:
                definition, example = rest.split("例:", 1)
                definition = definition.strip()
                example = example.strip()
            else:
                definition = rest.strip()
        else:
            return make_reply(input_message, "格式错误，需要等号分隔。\n正确用法：猫猫学词 <词> = <定义> [例:<例句>]")

        if not term or not definition:
            return make_reply(input_message, "俚语词汇和含义都不能为空。")

        examples = [example] if example else []
        entry = await self.learners.add_jargon(
            term=term,
            definition=definition,
            examples=examples,
            added_by=input_message.sender_id,
            chat_id=input_message.chat_id,
        )

        return make_reply(
            input_message,
            f"已学习俚语「{entry.term}」: {entry.definition}",
        )


@command(name="词典", aliases=["dict", "俚语列表"], permission="default", description="查看已学习的社群俚语词典。可用：猫猫词典 [搜索词]")
class JargonListCommand:
    def __init__(self, learning_orchestrator: LearningOrchestrator):
        self.learners = learning_orchestrator

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        if not self.learners:
            return make_reply(input_message, "学习系统未启用。")

        query = args.strip()
        if query:
            entries = self.learners.search_jargon(query)
            if not entries:
                return make_reply(input_message, f"未找到包含「{query}」的俚语。")
        else:
            entries = self.learners.get_jargon_entries()

        if not entries:
            return make_reply(input_message, "词典中还没有俚语，可以让管理员用「猫猫学词」添加。")

        if query:
            lines = [f"搜索「{query}」结果（共 {len(entries)} 条）："]
        else:
            lines = [f"社群俚语词典（共 {len(entries)} 条）："]

        for e in entries:
            source = "📝" if e.source == "manual" else "🤖"
            level_tag = f" Lv{e.inference_level}" if e.source == "auto" else ""
            def_text = e.definition or "(含义待推理)"
            lines.append(f"  {source} {e.term}{level_tag} — {def_text}")
            if e.examples:
                ex = e.examples[0][:60]
                lines.append(f"    例: {ex}")

        reply = "\n".join(lines)
        if len(reply) > 2000:
            reply = reply[:2000] + "\n…(过长已截断)"

        return make_reply(input_message, reply)


@command(name="删除词", aliases=["forget", "遗忘"], permission="admin", description="删除已学习的俚语。用法：猫猫删除词 <词>")
class JargonForgetCommand:
    def __init__(self, learning_orchestrator: LearningOrchestrator):
        self.learners = learning_orchestrator

    async def execute(self, input_message: InputMessage, args: str) -> List[Dict[str, Any]]:
        if not self.learners:
            return make_reply(input_message, "学习系统未启用。")

        term = args.strip()
        if not term:
            return make_reply(input_message, "用法：猫猫删除词 <词>")

        success = await self.learners.delete_jargon(term)
        if success:
            return make_reply(input_message, f"已删除俚语「{term}」。")
        else:
            return make_reply(input_message, f"未找到俚语「{term}」。")
