import pytest

from core.managers.template_manager import TemplateManager


@pytest.mark.parametrize("prompt_kind", ["private", "group"])
def test_chat_prompt_uses_adaptive_reply_format(prompt_kind):
    manager = TemplateManager(character_card="")

    if prompt_kind == "private":
        prompt = manager.get_private_chat_prompt("测试用户")
    else:
        prompt = manager.get_group_chat_prompt()

    assert "不改变人物卡设定的语气、长度、情绪表达和互动方式" in prompt
    assert "不需要结构化呈现的单一问答" in prompt
    assert "不要给一句话或短段落套标题" in prompt
    assert "消息以 markdown 发送" not in prompt


def test_task_prompt_uses_adaptive_reply_format():
    prompt = TemplateManager(character_card="").get_task_chat_prompt()

    assert "任务规则优先" in prompt
    assert "不要给短结果套标题" in prompt
    assert "按基础 markdown 输出" not in prompt
