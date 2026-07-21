"""声明式工具目录 — 工具分组、Profile 定义、功能映射"""

SECTIONS: dict[str, set[str]] = {
    "emoji":     {"search_emoji", "send_emoji"},
    "memory":    {"memory", "mark_important"},
    "user":      {"search_user"},
    "skill":     {"view_skill", "execute_skill", "rescan_skills"},
    "exec":      {"exec", "process"},
    "file":      {"read_file", "write_file", "edit_file", "apply_patch"},
    "cron":      {"cron"},
    "task":      {"task"},
    "tts":       {"synthesize_speech"},
    "sub_agent": {"spawn_subagent", "subagents"},
    "heartbeat": {"heartbeat_respond"},
    "learner":   {"define_jargon", "report_behavior_effect"},
}

_ALL_TOOLS: set[str] = set()
for _s in SECTIONS.values():
    _ALL_TOOLS.update(_s)

# normal profile = 所有工具，但排除仅 heartbeat 专用的工具
_NORMAL_EXCLUDE = SECTIONS["heartbeat"]

PROFILES: dict[str, set[str]] = {
    "normal": _ALL_TOOLS - _NORMAL_EXCLUDE,
    "heartbeat": {
        "heartbeat_respond",
        "memory", "mark_important",
        "exec", "process",
        "read_file", "write_file", "edit_file", "apply_patch",
        "task",
    },
    "task": {
        "announce", "search_user",
        "memory", "mark_important",
        "read_file", "write_file", "edit_file", "apply_patch",
        "exec", "process",
        "view_skill", "execute_skill", "rescan_skills",
    },
}

# ChatContext 标志位 → section 名称映射
FEATURE_SECTION_MAP: dict[str, str] = {
    "has_emojis":      "emoji",
    "has_hindsight":   "memory",
    "has_skills":      "skill",
    "has_workspace":   "file",
    "has_tts":         "tts",
    "has_sub_agents":  "sub_agent",
    "has_learners":    "learner",
}

CRON_ALLOWED: set[str] = {
    "announce", "search_user",
    "memory", "mark_important",
    "read_file", "write_file", "edit_file", "apply_patch",
    "exec", "process",
    "view_skill", "execute_skill", "rescan_skills",
}
