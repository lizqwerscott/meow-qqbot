"""EngineContext — 服务依赖上下文的类型化容器。

将 AgentEngine / PromptBuilder / ToolLoop 的构造参数按职责分组，
消除 20+ 个独立参数的上帝构造函数。
"""

from dataclasses import dataclass, field
from typing import Any, Tuple


@dataclass
class AIContext:
    ai_service: Any
    model_registry: Any = None
    rule_router: Any = None
    multimodal_service: Any = None
    max_tool_rounds: int = -1


@dataclass
class PromptContext:
    template_manager: Any
    nickname_manager: Any = None
    emoji_manager: Any = None
    skill_managers: Any = None
    learning_orchestrator: Any = None


@dataclass
class MemoryContext:
    hindsight_memory: Any = None
    search_top_k: int = 3


@dataclass
class MgmtContext:
    context_manager: Any
    permission_manager: Any = None
    cost_tracker: Any = None
    workspace_manager: Any = None
    archive_manager: Any = None
    system_events: Any = None


@dataclass
class BgContext:
    task_manager: Any = None
    cron_job_manager: Any = None


@dataclass
class SubContext:
    sub_agent_manager: Any = None


@dataclass
class SysContext:
    bot_id: str = ""
    admin_ids: Tuple[str, ...] = ()


@dataclass
class EngineContext:
    ai: AIContext
    prompt: PromptContext
    memory: MemoryContext
    mgmt: MgmtContext
    bg: BgContext = field(default_factory=BgContext)
    sub: SubContext = field(default_factory=SubContext)
    sys: SysContext = field(default_factory=SysContext)


