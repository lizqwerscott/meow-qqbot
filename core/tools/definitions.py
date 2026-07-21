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


APPLY_PATCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "批量修改工作区文件。一个 patch 可以同时新建、删除、更新、移动多个文件。"
                "输入格式使用 *** Begin Patch 和 *** End Patch 包裹：\n\n"
                "支持的操作：\n"
                "- *** Add File: {path}\\n+文件内容行（每行以 + 开头）\n"
                "- *** Delete File: {path}\n"
                "- *** Update File: {path}\\n@@ 上下文行（可选）\\n-删除行\\n+新增行\\n 不变行\\n*** End of File\n"
                "- *** Move to: {newPath}（紧跟在 Update File 行之后）"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "patch 完整内容，包含 *** Begin Patch 和 *** End Patch 包围",
                    },
                },
                "required": ["input"],
            },
        },
    },
]


FILE_TOOLS = [
    *APPLY_PATCH_TOOL,
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

ANNOUNCE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "announce",
            "description": "向父会话报告当前子智能体的进度或中间结果。父 AI 在下一轮对话时会看到这条消息。适合汇报阶段性进展、发现的异常或请求帮助。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "要报告给父会话的消息内容，建议简洁明了",
                    },
                },
                "required": ["message"],
            },
        },
    },
]


LIST_TASKS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "列出后台任务执行记录。返回任务的 ID、类型、状态、创建时间、指令摘要、结果/错误摘要等信息。可选按状态过滤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "按状态过滤。可选: pending, running, success, failed, cancelled, timeout。不填则返回所有状态",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回数量上限，默认 20",
                    },
                },
            },
        },
    },
]


LIST_CRON_JOBS_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "list_cron_jobs",
            "description": "列出所有定时/一次性任务。返回任务的 ID、名称、调度表达式、启用状态、下次执行时间等信息。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


TASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_cron_job",
            "description": (
                "创建一个定时任务或一次性提醒任务。"
                "三种载荷类型（通过 payload_type 指定）：\n"
                "- message（默认）: AI 智能体执行 prompt\n"
                "- command: 在服务器上执行 shell 命令（须提供 command 参数，prompt 可选）\n"
                "- system_event: 系统事件通知（文本将被注入 AI 上下文，prompt 可选）\n"
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
                        "description": "AI 要执行的指令。仅在 payload_type=message 时有效且必填。与 command 互斥，请根据 payload_type 只传其一。例如：'对大家说早上好，今天的天气是...'",
                    },
                    "session_mode": {
                        "type": "string",
                        "enum": ["isolated", "custom", "main"],
                        "description": "任务执行所在的 session 模式。默认为 isolated。\n"
                        "- isolated: 每次执行使用全新隔离 session（默认）\n"
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
                        "- system_event: 系统事件通知，文本将被注入 AI 上下文",
                    },
                    "command": {
                        "type": "string",
                        "description": "shell 命令。仅在 payload_type=command 时有效且必填。与 prompt 互斥，请根据 payload_type 只传其一。受安全黑名单限制（禁止 rm/sudo/systemctl 等危险命令）。",
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
                    "enable_notify": {
                        "type": "boolean",
                        "description": "是否投递执行结果到频道。默认为 true，设为 false 则静默执行不通知。",
                    },
                    "tools_allow": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "（可选）指定该定时任务可用的工具列表。仅在 payload_type=message 时有效。"
                            "不设置则使用默认工具集（search_user + 记忆读写 + 文件操作）。"
                            "设置为 ['*'] 可使用全部 cron 允许的工具。传入 null 则恢复默认工具集。\n\n"
                            "可用工具包括：\n"
                            "execute_command — 执行 bash 命令（受安全黑名单限制）\n"
                            "view_skill — 查看技能说明文档\n"
                            "execute_skill — 执行技能脚本\n"
                            "rescan_skills — 重新扫描技能列表\n"
                            "以及默认工具集内的所有工具。\n\n"
                            "不在此范围内的工具（如 emoji、TTS、子智能体、心跳回应、"
                            "学习工具、任务管理工具）不可用于定时任务，传入会被拒绝。\n\n"
                            "示例：['execute_command', 'search_memory', 'read_file']"
                        ),
                    },
                },
                "allOf": [
                    {
                        "oneOf": [
                            {"required": ["cron_expression"]},
                            {"required": ["at"]},
                        ],
                    },
                    {
                        "oneOf": [
                            {
                                "properties": {"payload_type": {"const": "message"}},
                                "required": ["prompt"],
                            },
                            {
                                "properties": {"payload_type": {"const": "command"}},
                                "required": ["command"],
                            },
                            {
                                "properties": {"payload_type": {"const": "system_event"}},
                            },
                        ],
                    },
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
    *LIST_TASKS_TOOL,
    *LIST_CRON_JOBS_TOOL,
    {
        "type": "function",
        "function": {
            "name": "update_cron_job",
            "description": (
                "修改一个已有定时/一次性任务的参数。只修改提供的字段，未提供的字段保持不变。"
                "所有时间均为北京时间 (CST/UTC+8)。\n"
                "可通过 cron_expression、at、prompt、session_mode、session_id、"
                "payload_type、command、model、thinking 参数修改对应设置。\n"
                "如果要修改执行时间：传入 cron_expression 会覆盖为周期性任务，传入 at 会覆盖为一次性任务。\n"
                "job_id 支持完整 ID 或任务名称前缀模糊匹配。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "要修改的任务 ID（完整 ID 或名称前缀模糊匹配）",
                    },
                    "name": {
                        "type": "string",
                        "description": "新的任务名字",
                    },
                    "cron_expression": {
                        "type": "string",
                        "description": "新的周期性 cron 表达式（北京时间 CST/UTC+8）。传入此参数会覆盖 at 变为周期性任务。例如：'0 8 * * *' 每天早上8点",
                    },
                    "at": {
                        "type": "string",
                        "description": "新的一次性执行时间，ISO 8601 格式（北京时间 CST/UTC+8）。传入此参数会覆盖 cron_expression 变为一次性任务。例如：'2027-01-01T08:00:00+08:00'",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "新的 AI 执行指令",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用。false 表示暂停，true 表示启用",
                    },
                    "session_mode": {
                        "type": "string",
                        "enum": ["isolated", "custom", "main"],
                        "description": "新的 session 模式",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "custom 模式下新的命名 session ID",
                    },
                    "payload_type": {
                        "type": "string",
                        "enum": ["message", "command", "system_event"],
                        "description": "新的载荷类型",
                    },
                    "command": {
                        "type": "string",
                        "description": "新的 shell 命令（payload_type=command 时）",
                    },
                    "model": {
                        "type": "string",
                        "description": "新的 AI 模型覆盖",
                    },
                    "thinking": {
                        "type": "string",
                        "enum": ["off", "low", "medium", "high"],
                        "description": "新的 AI 思考级别",
                    },
                    "enable_notify": {
                        "type": "boolean",
                        "description": "是否投递执行结果到频道。true 表示投递，false 表示静默执行。",
                    },
                    "tools_allow": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": (
                            "新的工具权限配置。不传则不修改。"
                            "传入 null 可重置为默认工具集。"
                            "可用值参考 create_cron_job 的 tools_allow 说明。"
                            "传入 ['*'] 可使用全部 cron 允许的工具。"
                        ),
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_cron_job",
            "description": "删除一个定时/一次性任务。job_id 支持完整 ID 或任务名称前缀模糊匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "要删除的任务 ID（完整 ID 或名称前缀模糊匹配）",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_cron_job",
            "description": "启用一个已暂停的定时任务。job_id 支持完整 ID 或任务名称前缀模糊匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "要启用的任务 ID（完整 ID 或名称前缀模糊匹配）",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_cron_job",
            "description": "暂停一个定时任务（不会删除，后续可用 enable_cron_job 恢复）。job_id 支持完整 ID 或任务名称前缀模糊匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "要暂停的任务 ID（完整 ID 或名称前缀模糊匹配）",
                    },
                },
                "required": ["job_id"],
            },
        },
    },
]


SUB_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "spawn_subagent",
            "description": (
                "创建一个子智能体在后台独立执行任务，不阻塞当前对话。"
                "子智能体有独立的会话和工具，执行完成后结果会通过系统事件通知你。"
                "适合执行耗时的研究、批量查询、文件处理等任务。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "子智能体要执行的详细任务指令。越详细越好，AI 会根据指令独立完成任务。",
                    },
                    "context": {
                        "type": "string",
                        "enum": ["isolated"],
                        "description": "上下文模式。isolated（默认）使用全新隔离上下文。",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subagents",
            "description": "列出或取消子智能体。action=list（默认）列出当前会话的所有子智能体状态；action=cancel 按 subagent_id 取消指定的子智能体。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "cancel"],
                        "description": "操作类型。list（默认）列出子智能体，cancel 取消子智能体。",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed", "timeout", "cancelled"],
                        "description": "仅 action=list 时有效。按状态过滤（可选）。不传则返回全部。",
                    },
                    "subagent_id": {
                        "type": "string",
                        "description": "仅 action=cancel 时必填。要取消的子智能体 ID。",
                    },
                },
            },
        },
    },
]


TTS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "synthesize_speech",
            "description": (
                "将文字转换为语音并发送到聊天中。\n\n"
                "在以下场景应主动调用：\n"
                "1. 用户要求你说句话、念一段文字、或问你能不能说话时\n"
                "2. 你觉得用语音回复比纯文字更有表现力时（如问候、祝贺、撒娇、吐槽等）\n"
                "3. 用户明确要求用某种语气说话（如'用热情的语气说'、'温柔地说'）\n"
                "4. 回复内容较短且有感情色彩，适合语音表达\n\n"
                "【语音模式 (voice_mode)】\n"
                "- preset（预设模式，默认）：使用管理员预设的固定音色。instructions 只用来调整情绪、语速和表达方式，"
                "不要写性别/年龄/身份等改变音色的描述\n"
                "- creative（创造模式）：自由创造全新音色。instructions 可以完整描述身份基底+音色质感+情绪表现力，"
                "例如「热情洋溢的中年男性播音员，声音低沉富有磁性」\n\n"
                "【instructions 编写指南】\n"
                "voice_mode=preset 时：只写情绪/语速/表达方式，不要试图改变音色身份。"
                "示例：'语速稍快，语气热情'、'温柔地慢慢说'\n"
                "voice_mode=creative 时：按三维结构自由设计——身份基底 + 音色质感 + 情绪表现力。"
                "示例：'热情洋溢的中年男性播音员，声音低沉富有磁性，带着节奏感呼喊'\n\n"
                "【方言提示】\n"
                "如需方言效果（粤语、四川话、东北话等），text 必须使用地道方言词汇书写，"
                "不能写标准语。例如粤语写「唔該」而非「谢谢」。\n"
                "voice_mode=preset 时 instructions 中注明方言名即可，如「广东话，语气随意」。\n\n"
                "【非语言标签】\n"
                "可在 text 中插入以下标签让语音更自然（用小写，不要堆叠多个）：\n"
                "[laughing] 笑声、[sigh] 叹气、[Uhm] 犹豫吞吞吐吐、[Shh] 示意安静\n\n"
                "【标点与停顿】\n"
                "标点符号是语音的韵律信号，直接影响自然度：\n"
                "- 句号（。）和问号（？）→ 较长的句末停顿\n"
                "- 逗号（，）→ 较短的中断\n"
                "- 省略号（……）→ 犹豫、拖沓效果\n"
                "- 需要更强停顿的地方，把长句拆成短句\n\n"
                "【性能注意】\n"
                "text 不宜过长。超过 30 秒朗读量的文本会导致推理时间过长。\n"
                "text 也不宜过短，最少 3-5 个字，太短会生成断裂的音频。\n"
                "如果需要朗读较长的内容，请分成多次 synthesize_speech 调用。"
            ),
            "parameters": {
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
        "create_cron_job", "cancel_task",
        "list_tasks",
        "list_cron_jobs", "update_cron_job", "delete_cron_job",
        "enable_cron_job", "disable_cron_job",
        "read_file", "write_file", "edit_file",
        "list_files", "search_files",
        "apply_patch",
        "heartbeat_respond",
        "announce",
        "spawn_subagent", "subagents",
        "synthesize_speech",
    }


# ── 工具名称 → 定义列表映射（用于后台任务动态工具解析） ──
# ⚠️ 警告：以下引用组合列表（FILE_TOOLS、TASK_TOOLS、LEARNER_TOOLS、EMOJI_TOOLS、SUB_AGENT_TOOLS）
# 时使用了硬编码索引。如果上述列表的结构或顺序发生变化，以下索引必须同步更新。
# FILE_TOOLS 布局: [0]=apply_patch, [1]=read_file, [2]=write_file, [3]=edit_file, [4]=list_files, [5]=search_files
# TASK_TOOLS 布局: [0]=create_cron_job, [1]=cancel_task, [2]=list_tasks, [3]=list_cron_jobs, [4]=update_cron_job,
#                   [5]=delete_cron_job, [6]=enable_cron_job, [7]=disable_cron_job
# LEARNER_TOOLS 布局: [0]=define_jargon, [1]=report_behavior_effect
# EMOJI_TOOLS 布局: [0]=search_emoji, [1]=send_emoji
# SUB_AGENT_TOOLS 布局: [0]=spawn_subagent, [1]=subagents

TOOL_DEFINITION_MAP: dict[str, list[dict]] = {
    "search_emoji": [EMOJI_TOOLS[0]],
    "send_emoji": [EMOJI_TOOLS[1]],
    "search_user": SEARCH_USER_TOOL,
    "search_memory": SEARCH_MEMORY_TOOL,
    "search_relation": SEARCH_RELATION_TOOL,
    "mark_important": MARK_IMPORTANT_TOOL,
    "define_jargon": [LEARNER_TOOLS[0]],
    "report_behavior_effect": [LEARNER_TOOLS[1]],
    "rescan_skills": RESCAN_SKILLS_TOOL,
    "view_skill": VIEW_SKILL_TOOL,
    "execute_skill": EXECUTE_SKILL_TOOL,
    "execute_command": EXECUTE_COMMAND_TOOL,
    "apply_patch": APPLY_PATCH_TOOL,
    "read_file": [FILE_TOOLS[1]],
    "write_file": [FILE_TOOLS[2]],
    "edit_file": [FILE_TOOLS[3]],
    "list_files": [FILE_TOOLS[4]],
    "search_files": [FILE_TOOLS[5]],
    "heartbeat_respond": HEARTBEAT_RESPOND_TOOL,
    "announce": ANNOUNCE_TOOL,
    "list_tasks": LIST_TASKS_TOOL,
    "list_cron_jobs": LIST_CRON_JOBS_TOOL,
    "create_cron_job": [TASK_TOOLS[0]],
    "cancel_task": [TASK_TOOLS[1]],
    "update_cron_job": [TASK_TOOLS[4]],
    "delete_cron_job": [TASK_TOOLS[5]],
    "enable_cron_job": [TASK_TOOLS[6]],
    "disable_cron_job": [TASK_TOOLS[7]],
    "spawn_subagent": [SUB_AGENT_TOOLS[0]],
    "subagents": [SUB_AGENT_TOOLS[1]],
    "synthesize_speech": TTS_TOOLS,
}

TOOL_SHORT_DESCRIPTIONS: dict[str, str] = {
    "announce": "向父会话报告进度或中间结果",
    "search_user": "按昵称模糊搜索群用户信息",
    "search_memory": "搜索长期记忆（可查群友画像、经历、事实）",
    "search_relation": "搜索两人之间的关系记忆",
    "mark_important": "记录重要信息至长期记忆",
    "read_file": "读取工作区文件",
    "write_file": "写入文件到工作区（新建或覆盖）",
    "edit_file": "编辑工作区文件（精确字符串替换）",
    "list_files": "列出工作区文件和目录",
    "search_files": "在工作区中搜索文件内容（正则表达式）",
    "apply_patch": "批量新建/更新/删除/移动文件",
    "execute_command": "执行 bash 命令（受安全黑名单限制）",
    "view_skill": "查看并加载完整的技能说明文档",
    "execute_skill": "执行技能附带的脚本（如自动化分析、代码生成）",
    "rescan_skills": "重新扫描刷新可用技能列表",
}

# Cron 定时任务可用的工具白名单
# 不在列表中的工具（emoji、TTS、心跳、子智能体、学习工具、任务管理等）不可用于定时任务
CRON_ALLOWED_TOOL_NAMES: frozenset = frozenset({
    # 核心
    "announce", "search_user",
    # 记忆
    "search_memory", "search_relation", "mark_important",
    # 文件
    "read_file", "write_file", "edit_file",
    "list_files", "search_files", "apply_patch",
    # 命令
    "execute_command",
    # 技能
    "view_skill", "execute_skill", "rescan_skills",
})
