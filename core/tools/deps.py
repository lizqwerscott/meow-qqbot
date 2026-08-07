"""ToolDeps — aggregated dependencies for the tool system.

New files in impl/ receive a ToolDeps instance via factory functions.
Dependencies that may change after construction are wrapped in Ref.
"""

from dataclasses import dataclass, field
from typing import Any

from core.tools.ref import Ref


@dataclass
class ToolDeps:
    # ── Static deps (set once, never change) ──
    emoji_manager: Any = None
    nickname_manager: Any = None
    skill_managers: Any = None
    hindsight: Any = None
    learning_orchestrator: Any = None
    permission_manager: Any = None
    workspace_manager: Any = None
    sub_agent_manager: Any = None
    system_events: Any = None
    web: Any = None
    search_top_k: int = 5
    admin_ids: list = field(default_factory=list)
    bot_id: str = ""
    media_service: Any = None

    # ── Mutable deps (Ref containers, updated after construction) ──
    media_uploader: Ref = field(default_factory=Ref)  # Ref[MediaUploader]
    bot_engine: Ref = field(default_factory=Ref)  # Ref[BotEngine]
    api_client: Ref = field(default_factory=Ref)  # Ref[QQApiClient]
    tts_service: Ref = field(default_factory=Ref)  # Ref[TtsService]
    process_registry: Ref = field(default_factory=Ref)  # Ref[ProcessRegistry]
    approval_manager: Ref = field(default_factory=Ref)  # Ref[ApprovalManager]
    exec_reviewer: Ref = field(
        default_factory=Ref
    )  # Ref[ExecAutoReviewer] (mode=auto 用)
    task_manager: Ref = field(default_factory=Ref)  # Ref[TaskManager]
    cron_job_manager: Ref = field(default_factory=Ref)  # Ref[CronJobManager]
    background_task_runner: Ref = field(
        default_factory=Ref
    )  # Ref[BackgroundTaskRunner]
