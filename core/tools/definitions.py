"""工具（Function Calling）JSON 定义，纯数据，零外部依赖。"""

EMOJI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_emoji",
            "description": "搜索表情图片。输入一个或多个标签，用空格分开。系统会匹配其中任意标签，按匹配数量排序返回。输入多个标签可以得到更精准的搜索结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于搜索的标签，多个标签用空格分隔，例如：开心 撒娇 猫娘。标签越具体搜索越精准。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_emoji",
            "description": "发送一个指定的表情图片到聊天中。需要提供通过 search_emoji 获取到的表情 hash。一条回复最多发送 1 个表情。",
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji_hash": {
                        "type": "string",
                        "description": "表情的唯一标识 hash（完整 hash 或前 12 位短 hash），通过 search_emoji 获取",
                    },
                    "reason": {
                        "type": "string",
                        "description": "发送这个表情的原因或想表达的情绪，仅用于记录",
                    },
                },
                "required": ["emoji_hash", "reason"],
            },
        },
    },
]

SEARCH_USER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_user",
            "description": "根据昵称或昵称的一部分模糊搜索群里的用户。输入昵称的一部分（如'小'）即可找到所有匹配的人。返回用户的ID和昵称，获取到用户ID后你可以在回复中使用 <qqbot-at-user id=\"xxx\" /> 来@该用户。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如用户名、昵称或ID的一部分",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

SEARCH_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "搜索记忆系统，可查询人物画像、过往经历、具体事实、"
                "用户偏好等任何信息。如果不指定 person_name，则搜索"
                "当前对话用户的记忆；如果指定 person_name，则搜索对应群友的记忆。"
                "person_name 支持模糊搜索，输入昵称的一部分也能匹配到。"
                "当需要了解某人的背景、确认某件事、查找说过的话时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题，例如 '他喜欢什么'、'上次提到的新显卡'、'生日是什么时候'",
                    },
                    "person_name": {
                        "type": "string",
                        "description": "要搜索的人名或昵称（可选）。支持模糊搜索，输入昵称的一部分（如'小'）也能匹配。不填则搜索当前对话用户。私聊中不可用。",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hybrid", "agentic"],
                        "description": "检索方法。hybrid（默认）适合大多数情况；agentic 适合需要深度挖掘的复杂查询，会进行多轮检索。",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

SEARCH_RELATION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_relation",
            "description": (
                "搜索两个人之间的关系记忆。当你需要了解两个用户之间的关联、"
                "共同经历、相互评价、关系背景时使用。"
                "系统会同时搜索两个人的相关记忆以及当前用户的记载。"
                "人名支持模糊搜索，输入昵称的一部分即可匹配。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_a": {
                        "type": "string",
                        "description": "第一个人的人名或昵称，支持模糊搜索（部分匹配即可）。如果搜'我'或'自己'则代表当前说话者。",
                    },
                    "person_b": {
                        "type": "string",
                        "description": "第二个人的人名或昵称，支持模糊搜索（部分匹配即可）。如果搜'我'或'自己'则代表当前说话者。",
                    },
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题（可选）。例如'他们什么关系'、'一起做过什么'",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hybrid", "agentic"],
                        "description": "检索方法。hybrid（默认）适合大多数情况；agentic 适合需要深度挖掘的复杂查询。",
                    },
                },
                "required": ["person_a", "person_b"],
            },
        },
    },
]

MARK_IMPORTANT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "mark_important",
            "description": (
                "记录重要信息至长期记忆。主动判断，不需要用户每次都说'记好了'。"
                "在以下场景应主动调用：\n"
                "1. 用户明确要求'记住这个'、'记好了'\n"
                "2. 用户在解释自己的背景、喜好、习惯、个人信息\n"
                "3. 用户在描述关于自己或他人的重要事实或关系\n"
                "4. 用户分享值得长期记住的知识或信息\n"
                "5. 当前讨论出现对理解用户有重要帮助的上下文"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


RESCAN_SKILLS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "rescan_skills",
            "description": "重新扫描 skills 目录，刷新可用技能列表。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


VIEW_SKILL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "view_skill",
            "description": "查看并加载某个技能的完整指导说明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要查看的技能名称，从 <available_skills> 中获取",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
]

EXECUTE_SKILL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "execute_skill",
            "description": "执行技能附带的脚本（如自动化分析、代码生成等）。参数以 JSON 形式传入。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "技能名称",
                    },
                    "script_name": {
                        "type": "string",
                        "description": "脚本名称（无需后缀，如 'release'、'extract'）",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "传递给脚本的参数（JSON 对象）",
                        "additionalProperties": True,
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "执行超时时间（秒），默认 30，最大 120",
                    },
                },
                "required": ["skill_name", "script_name"],
            },
        },
    },
]

EXECUTE_COMMAND_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "执行任意 bash 命令（受黑名单限制）。可用于运行 git 操作、python 脚本、文件查看等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 bash 命令",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "执行超时时间（秒），默认 30，最大 120",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "工作目录（可选，默认项目根目录）",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


SKILL_TOOLS = [
    *RESCAN_SKILLS_TOOL,
    *VIEW_SKILL_TOOL,
    *EXECUTE_SKILL_TOOL,
    *EXECUTE_COMMAND_TOOL,
]


def tool_names() -> set[str]:
    """返回所有已注册工具的名称集合。"""
    return {
        "search_emoji", "send_emoji", "search_user",
        "search_memory", "mark_important", "search_relation",
        "rescan_skills", "view_skill", "execute_skill", "execute_command",
    }
