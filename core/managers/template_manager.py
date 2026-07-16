import os
from typing import Any, Dict, Optional

import jinja2


class TemplateManager:
    """模板管理器类，负责加载和渲染提示模板"""

    _PRIVATE_CHAT_TEMPLATE = "prompts/private_chat.j2"
    _GROUP_CHAT_TEMPLATE = "prompts/group_chat.j2"
    _TASK_CHAT_TEMPLATE = "prompts/task_chat.j2"
    _HEARTBEAT_CHAT_TEMPLATE = "prompts/heartbeat_chat.j2"

    def __init__(self, config: Dict[str, Any]):
        """
        初始化模板管理器

        Args:
            config: 配置文件字典
        """
        self.template_loader = jinja2.FileSystemLoader(searchpath=".")
        self.template_env = jinja2.Environment(loader=self.template_loader)

        # 加载角色卡
        card_path = config.get("character_card", "characters/default.md")
        self.character_card = ""
        if card_path and os.path.exists(card_path):
            try:
                with open(card_path, "r", encoding="utf-8") as f:
                    self.character_card = f.read().strip()
            except Exception as e:
                print(f"读取角色卡文件失败: {e}")

    def render_prompt_template(
        self, template_path: str, context: Dict[str, Any]
    ) -> str:
        """
        渲染提示模板

        Args:
            template_path: 模板文件路径
            context: 模板上下文变量

        Returns:
            渲染后的提示文本
        """
        try:
            if not os.path.exists(template_path):
                print(f"提示模板文件不存在: {template_path}")
                return ""

            template = self.template_env.get_template(template_path)
            return template.render(**context)
        except Exception as e:
            print(f"渲染提示模板失败: {e}")
            return ""

    def get_private_chat_prompt(
        self,
        user_name: str,
        *,
        has_emojis: bool = False,
        has_users: bool = False,
        memory_system_desc: str = "",
        skill_system_intro: str = "",
    ) -> str:
        """
        获取私聊系统提示（纯静态，不含动态工具/记忆说明）

        Args:
            user_name: 用户昵称
            has_emojis: 是否有表情
            has_users: 是否有群友（私聊永远 False）
            memory_system_desc: 记忆系统说明
            skill_system_intro: 技能系统原则介绍

        Returns:
            私聊系统提示文本
        """

        context = {
            "user_name": user_name,
            "character_card": self.character_card,
            "has_emojis": has_emojis,
            "has_users": has_users,
            "memory_system_desc": memory_system_desc,
            "skill_system_intro": skill_system_intro,
        }

        prompt = self.render_prompt_template(self._PRIVATE_CHAT_TEMPLATE, context)
        if not prompt:
            prompt = f"你是一个贴心的 AI 助手，正在与用户「{user_name}」进行一对一的私密对话。"

        return prompt

    def get_group_chat_prompt(
        self,
        group_name: Optional[str] = None,
        *,
        has_emojis: bool = False,
        has_users: bool = False,
        memory_system_desc: str = "",
        skill_system_intro: str = "",
    ) -> str:
        """
        获取群聊系统提示（纯静态，不含动态工具/记忆说明）

        Args:
            group_name: 群组名称（可选）
            has_emojis: 是否有表情
            has_users: 是否有群友
            memory_system_desc: 记忆系统说明
            skill_system_intro: 技能系统原则介绍

        Returns:
            群聊系统提示文本
        """

        context = {
            "group_name": group_name or "当前群组",
            "character_card": self.character_card,
            "has_emojis": has_emojis,
            "has_users": has_users,
            "memory_system_desc": memory_system_desc,
            "skill_system_intro": skill_system_intro,
        }

        prompt = self.render_prompt_template(self._GROUP_CHAT_TEMPLATE, context)
        if not prompt:
            prompt = "你是一个友好的QQ群机器人助手，正在与多个用户进行群聊对话。"
            if group_name:
                prompt += f" 当前群组名称是「{group_name}」。"

        return prompt

    def get_task_chat_prompt(
        self,
        *,
        current_time: str = "",
    ) -> str:
        """
        获取后台任务系统提示（使用 task_chat.j2 模板）

        Args:
            current_time: 当前时间字符串

        Returns:
            后台任务系统提示文本
        """
        context = {
            "current_time": current_time,
        }
        prompt = self.render_prompt_template(self._TASK_CHAT_TEMPLATE, context)
        if not prompt:
            prompt = "你是一个后台任务执行助手。请根据指令完成任务。"
        return prompt

    def get_heartbeat_prompt(
        self,
        *,
        current_time: str = "",
    ) -> str:
        """
        获取心跳检查系统提示（使用 heartbeat_chat.j2 模板）

        Args:
            current_time: 当前时间字符串

        Returns:
            心跳系统提示文本
        """
        context = {
            "current_time": current_time,
        }
        prompt = self.render_prompt_template(self._HEARTBEAT_CHAT_TEMPLATE, context)
        if not prompt:
            prompt = "你是一个心跳检查器。检查是否有需要关注的事项。如果没有，回复 HEARTBEAT_OK。"
        return prompt

    def get_template_paths(self) -> Dict[str, str]:
        """获取所有模板路径"""
        return {
            "private_chat": self._PRIVATE_CHAT_TEMPLATE,
            "group_chat": self._GROUP_CHAT_TEMPLATE,
            "task_chat": self._TASK_CHAT_TEMPLATE,
        }
