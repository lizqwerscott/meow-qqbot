"""EngineContext — 服务依赖上下文的类型化容器。

将 AgentEngine / PromptBuilder / ToolLoop 的构造参数按职责分组，
消除 20+ 个独立参数的上帝构造函数。
"""

from dataclasses import dataclass, field
from typing import Any, Tuple

from core.engine.engagement_config import EngagementConfig


@dataclass
class AIContext:
    ai_service: Any
    model_registry: Any = None
    rule_router: Any = None
    multimodal_service: Any = None
    max_tool_rounds: int = -1
    # 流式回复：stream_reply 开启后，文本轮走 chat_completion_stream，
    # 以 block 模式投递（对齐 openclaw qqbot 插件）：累积到 stream_block_chars
    # 或距上次发送空闲 stream_block_idle_ms 才发一块，避免逐句连发刷屏。
    stream_reply: bool = False
    stream_block_chars: int = 800
    stream_block_idle_ms: int = 1000
    engagement_config: EngagementConfig = field(default_factory=EngagementConfig)


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
    task_state_store: Any = None


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
