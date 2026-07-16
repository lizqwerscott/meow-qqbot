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
                "properties": {
                    "profile_data": {
                        "type": "string",
                        "description": (
                            "需要记住的关于用户的结构化信息，JSON 对象格式。"
                            "例如 {\"name\": \"小明\", \"likes\": \"打篮球\", \"job\": \"程序员\"}。"
                            "这些信息会写入长期记忆，下次查询时将作为该用户画像返回。"
                            "如果不需要记录画像则不传。"
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "需要记住的重要事件或事实的一句话摘要，"
                            "将作为该用户的一条经历存入长期记忆。"
                            "如果不需要记录则不传。"
                        ),
                    },
                },
            },
        },
    },
]


LEARNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "define_jargon",
            "description": "学习社群俚语/黑话。当你听到某个生疏的词汇反复出现，或者用户询问某个俚语的含义时，主动调用此工具学习并记录该俚语。",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "俚语词汇本身，如 'YBB'、'暴龙'",
                    },
                    "definition": {
                        "type": "string",
                        "description": "俚语的含义解释",
                    },
                    "example": {
                        "type": "string",
                        "description": "一个使用该俚语的例句（可选）",
                    },
                },
                "required": ["term", "definition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_behavior_effect",
            "description": "报告你刚才的回复风格是否获得了良好效果，用于学习优化未来行为。当你发现某种语气或策略让用户积极性明显提高，或用户给了负面反馈时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_summary": {
                        "type": "string",
                        "description": "场景简要描述，如'用户抱怨工作压力大'",
                    },
                    "action_taken": {
                        "type": "string",
                        "description": "你采取的行为策略，如'先用幽默缓和气氛，再给实用建议'",
                    },
                    "effect": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "效果评估：positive=用户反应积极，negative=用户反应差，neutral=无明显变化",
                    },
                },
                "required": ["scene_summary", "action_taken", "effect"],
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


FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作区内的文件内容。路径相对于当前聊天的 files/ 目录。不支持路径穿越(..)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件到工作区。路径相对于当前聊天的 files/ 目录，父目录自动创建。已存在的文件会被覆盖。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "编辑工作区内的文件，进行精确的字符串替换。路径相对于当前聊天的 files/ 目录。比使用 sed 更可靠，推荐用于文本修改。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件相对路径，例如 'note.txt' 或 'dir/file.md'",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "要被替换的旧文本，必须完全匹配（包括空格和换行）",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "替换后的新文本",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "是否替换所有匹配。false 时仅替换第一个，存在多处匹配时会报错",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出工作区内的文件和子目录。路径相对于当前聊天的 files/ 目录。支持 glob 模式过滤文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要列出的目录路径。留空或 '.' 表示当前工作区根目录，例如 '.'、'subdir'",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "glob 过滤模式，例如 '*.py' 只显示 Python 文件，'*.*' 只显示有后缀的文件，'**/*.md' 递归显示所有 md 文件。不填则显示全部",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "搜索工作区内的文件内容，支持正则表达式。使用 ripgrep 引擎，比 grep 更快更智能。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "要搜索的正则表达式模式，例如 'TODO'、'def \\w+'、'error.*timeout'",
                    },
                    "glob": {
                        "type": "string",
                        "description": "文件过滤 glob，例如 '*.py' 只搜索 Python 文件。不填则搜索工作区内所有文件",
                    },
                    "path": {
                        "type": "string",
                        "description": "搜索范围路径，相对于当前聊天的 files/ 目录。不填则搜索整个工作区",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


HEARTBEAT_RESPOND_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat_respond",
            "description": "回应心跳检查。notify=false 表示本次心跳无需要关注的事项；notify=true 时附带提醒内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "notify": {
                        "type": "boolean",
                        "description": "是否发送通知。false=无需关注，true=需要提醒",
                    },
                    "notification_text": {
                        "type": "string",
                        "description": "提醒文本，不超过 300 字。仅在 notify=true 时需要",
                    },
                },
                "required": ["notify"],
            },
        },
    },
]


TASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "创建一个一次性后台任务。AI 可以在后台独立执行较长时间的工作"
                "（如生成报告、批量查询、执行脚本等），执行期间不阻塞当前对话，"
                "完成后可以通过 /tasks show 查看结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "后台任务要执行的指令或工作描述。越详细越好，AI 会根据这个指令独立完成任务。",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_cron_job",
            "description": (
                "创建一个定时任务或一次性提醒任务。"
                "三种载荷类型（通过 payload_type 指定）：\n"
                "- message（默认）: AI 智能体执行 prompt\n"
                "- command: 在服务器上执行 shell 命令（须提供 command 参数，prompt 可选）\n"
                "- system_event: 系统事件通知（仅记录日志投递，prompt 可选）\n"
                "所有时间均为北京时间 (CST/UTC+8)。\n"
                "两种模式二选一：\n"
                "1. 周期性任务：设置 cron_expression（标准 5 字段 cron 表达式：分 时 日 月 周，"
                "使用北京时间 CST/UTC+8 计算）\n"
                "2. 一次性任务：设置 at（ISO 8601 格式时间，使用北京时间 CST/UTC+8），"
                "到时间执行一次后自动删除"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "任务的名字，方便管理和查找，如'早安提醒'、'新年提醒'",
                    },
                    "cron_expression": {
                        "type": "string",
                        "description": "周期性 cron 表达式（北京时间 CST/UTC+8，与 at 二选一）。例如：'0 8 * * *' 表示北京时间每天早上8点，'*/30 * * * *' 每30分钟",
                    },
                    "at": {
                        "type": "string",
                        "description": "一次性执行时间，ISO 8601 格式（北京时间 CST/UTC+8，与 cron_expression 二选一）。例如：'2027-01-01T08:00:00+08:00' 表示北京时间2027年1月1日早上8点。如果省略时区偏移，默认视为北京时间",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "执行时 AI 要执行的指令。payload_type=message 时必填。例如：'对大家说早上好，今天的天气是...'。payload_type=command 或 system_event 时可选。",
                    },
                    "session_mode": {
                        "type": "string",
                        "enum": ["isolated", "current", "custom", "main"],
                        "description": "任务执行所在的 session 模式。默认为 isolated。\n"
                        "- isolated: 每次执行使用全新隔离 session（默认）\n"
                        "- current: 在创建时绑定的当前对话中执行，共享聊天上下文\n"
                        "- custom: 持久化命名 session，跨多次执行保留上下文\n"
                        "- main: 专用系统提醒通道 cron:main",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "custom 模式下使用的命名 session ID。例如 'daily-standup' 会使用 cron:daily-standup session，跨多次执行积累上下文。仅在 session_mode=custom 时有效。",
                    },
                    "payload_type": {
                        "type": "string",
                        "enum": ["message", "command", "system_event"],
                        "description": "任务载荷类型。默认为 message。\n"
                        "- message: AI 智能体轮次（默认），执行 prompt\n"
                        "- command: 在服务器上执行 shell 命令，使用 command 参数\n"
                        "- system_event: 系统事件通知，仅记录日志并投递通知",
                    },
                    "command": {
                        "type": "string",
                        "description": "shell 命令，仅在 payload_type=command 时有效且必填。受安全黑名单限制（禁止 rm/sudo/systemctl 等危险命令）。",
                    },
                    "model": {
                        "type": "string",
                        "description": "AI 模型覆盖，仅对 message 载荷有效。例如 'deepseek-v4-pro'。不设置则使用系统默认模型。",
                    },
                    "thinking": {
                        "type": "string",
                        "enum": ["off", "low", "medium", "high"],
                        "description": "AI 思考级别覆盖，仅对 message 载荷有效。不设置则使用系统默认。",
                    },
                },
                "oneOf": [
                    {"required": ["cron_expression"]},
                    {"required": ["at"]},
                ],
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_task",
            "description": "取消一个正在运行或等待中的后台任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "要取消的任务 ID（完整 ID 或前 12 位短 ID）",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
]


def tool_names() -> set[str]:
    """返回所有已注册工具的名称集合。"""
    return {
        "search_emoji", "send_emoji", "search_user",
        "search_memory", "mark_important", "search_relation",
        "rescan_skills", "view_skill", "execute_skill", "execute_command",
        "define_jargon", "report_behavior_effect",
        "create_task", "create_cron_job", "cancel_task",
        "read_file", "write_file", "edit_file",
        "list_files", "search_files",
        "heartbeat_respond",
    }
