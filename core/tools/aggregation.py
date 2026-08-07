"""Tool aggregation — explicit factory calls for all tool modules.

Replaces import-time _bootstrap() with a single explicit function call.
"""

from core.tools._types import ToolEntry
from core.tools.deps import ToolDeps


def create_all_tool_entries(deps: ToolDeps) -> list[ToolEntry]:
    from core.tools.impl.emoji import create_emoji_entries
    from core.tools.impl.exec_process import create_exec_process_entries
    from core.tools.impl.file import create_file_entries
    from core.tools.impl.heartbeat import create_heartbeat_entries
    from core.tools.impl.learner import create_learner_entries
    from core.tools.impl.media import create_media_entries
    from core.tools.impl.memory import create_memory_entries
    from core.tools.impl.message import create_message_entries
    from core.tools.impl.search import create_search_entries
    from core.tools.impl.skill import create_skill_entries
    from core.tools.impl.sub_agent import create_sub_agent_entries
    from core.tools.impl.task import create_task_entries
    from core.tools.impl.tts import create_tts_entries
    from core.tools.impl.user import create_user_entries
    from core.tools.impl.web import create_web_entries

    entries = []
    entries.extend(create_emoji_entries(deps))
    entries.extend(create_user_entries(deps))
    entries.extend(create_memory_entries(deps))
    entries.extend(create_learner_entries(deps))
    entries.extend(create_skill_entries(deps))
    entries.extend(create_task_entries(deps))
    entries.extend(create_file_entries(deps))
    entries.extend(create_search_entries(deps))
    entries.extend(create_web_entries(deps))
    entries.extend(create_heartbeat_entries(deps))
    entries.extend(create_sub_agent_entries(deps))
    entries.extend(create_tts_entries(deps))
    entries.extend(create_exec_process_entries(deps))
    entries.extend(create_message_entries(deps))
    entries.extend(create_media_entries(deps))
    return entries
