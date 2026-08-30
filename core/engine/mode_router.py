"""Pure, deterministic Chat/Agent mode routing.

The router deliberately selects a capability profile only.  It does not create
or mutate work plans, call a model, execute a tool, or send a reply.  Command
messages are handled by ``Router`` before they reach this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from core.engine.prompt_snapshot import PromptMode
from core.managers.session_manager import InboundIntent, ModeRoutingMetadata
from core.message import InputMessage


class ModeRouteSource(StrEnum):
    USER = "user"
    AMBIENT = "ambient"
    PROACTIVE = "proactive"
    BACKGROUND = "background"


class ModeReasonCode(StrEnum):
    WORK_PLAN_FOLLOW_UP = "work_plan_follow_up"
    DISCUSSION_ONLY = "discussion_only"
    EXPLICIT_WORK = "explicit_work"
    AMBIENT_CHAT = "ambient_chat"
    PROACTIVE_CHAT = "proactive_chat"
    DEFAULT_CHAT = "default_chat"


@dataclass(frozen=True)
class ActiveWorkPlanHint:
    """A pre-validated candidate; semantic attachment remains a planner concern."""

    work_plan_id: str
    chat_id: str
    owner_id: str
    revision: int
    scheduler_revision: int | None = None
    is_eligible: bool = True

    def matches(self, message: InputMessage, scheduler_revision: int) -> bool:
        return (
            self.is_eligible
            and bool(self.work_plan_id)
            and self.chat_id == message.chat_id
            and self.owner_id == message.sender_id
            and (
                self.scheduler_revision
                if self.scheduler_revision is not None
                else self.revision
            )
            == scheduler_revision
        )


@dataclass(frozen=True)
class ModeRouteInput:
    """The frozen, runtime-owned inputs used for a single route decision."""

    message: InputMessage
    source: ModeRouteSource = ModeRouteSource.USER
    intent: InboundIntent = InboundIntent.PRIVATE_CONVERSATION
    role: str = "default"
    scheduler_revision: int = 0
    active_work_plan: ActiveWorkPlanHint | None = None


@dataclass(frozen=True)
class ModeDecision:
    """An auditable, capability-only route decision."""

    mode: PromptMode
    confidence: float
    reason: str
    reason_code: ModeReasonCode
    intent: InboundIntent
    capability_profile: str
    policy_version: str
    scheduler_revision: int
    work_plan_hint: str | None = None

    def to_metadata(self) -> ModeRoutingMetadata:
        """Freeze the decision fields that must survive ingress and queueing."""
        return ModeRoutingMetadata(
            mode=self.mode.value,
            capability_profile=self.capability_profile,
            reason_code=self.reason_code.value,
            policy_version=self.policy_version,
            scheduler_revision=self.scheduler_revision,
            work_plan_hint=self.work_plan_hint,
        )


class ModeRouter:
    """Route non-command messages by stable safety rules and no model calls."""

    POLICY_VERSION = "mode-router/v1"
    _DISCUSSION_ONLY = re.compile(
        r"(?:只解释|只讲|先给(?:个)?方案|先别执行|不要(?:执行|修改|改文件)|不(?:要)?(?:执行|修改|改文件)|仅(?:讨论|解释|分析)|just\s+(?:explain|discuss)|do\s+not\s+(?:execute|modify))",
        re.IGNORECASE,
    )
    _EXPLANATION_QUESTION = re.compile(
        r"(?:为什么|怎么回事|什么意思|是什么意思|原因(?:是|为)?什么|why\b|what\s+does)",
        re.IGNORECASE,
    )
    _WORK_ACTION = re.compile(
        r"(?:读取|查看|创建|新建|修改|删除|修复|修|重构|实现|运行|执行|测试|部署|安装|启动|停止|重启|写(?:一个|个|.*脚本)|read|create|modify|delete|fix|refactor|implement|run|test|deploy|install|start|stop|restart)",
        re.IGNORECASE,
    )
    _WORK_TARGET = re.compile(
        r"(?:文件|代码|脚本|命令|测试|服务|依赖|定时任务|cron|task|git|workspace|仓库|项目|[\\/][\w.\-/]+|\.[a-zA-Z0-9]{1,8}\b)",
        re.IGNORECASE,
    )
    _NEW_ARTIFACT = re.compile(
        r"(?:写|创建|新建).{0,24}(?:脚本|定时任务|项目|文件|工具|程序)",
        re.IGNORECASE,
    )

    def __init__(self, *, policy_version: str = POLICY_VERSION):
        if not policy_version:
            raise ValueError("policy_version is required")
        self.policy_version = policy_version

    def route(self, request: ModeRouteInput) -> ModeDecision:
        """Return the first matching fixed-priority decision."""
        message = request.message
        work_plan = request.active_work_plan
        if work_plan is not None and work_plan.matches(
            message, request.scheduler_revision
        ):
            return self._decision(
                request,
                mode=PromptMode.AGENT,
                confidence=0.95,
                reason="validated active WorkPlan follow-up candidate",
                reason_code=ModeReasonCode.WORK_PLAN_FOLLOW_UP,
                capability_profile="agent_full",
                work_plan_hint=work_plan.work_plan_id,
            )

        text = (message.content or "").strip()
        if request.source is ModeRouteSource.AMBIENT:
            return self._decision(
                request,
                mode=PromptMode.CHAT,
                confidence=1.0,
                reason="ambient turns always use the low-risk Chat profile",
                reason_code=ModeReasonCode.AMBIENT_CHAT,
                capability_profile="group_ambient",
            )
        if request.source is ModeRouteSource.PROACTIVE:
            return self._decision(
                request,
                mode=PromptMode.CHAT,
                confidence=1.0,
                reason="proactive turns always use the low-risk Chat profile",
                reason_code=ModeReasonCode.PROACTIVE_CHAT,
                capability_profile="group_proactive",
            )

        if self._DISCUSSION_ONLY.search(text):
            return self._decision(
                request,
                mode=PromptMode.CHAT,
                confidence=0.95,
                reason="the request explicitly limits this turn to discussion",
                reason_code=ModeReasonCode.DISCUSSION_ONLY,
                capability_profile=self._chat_profile(request),
            )

        if (
            self._EXPLANATION_QUESTION.search(text)
            and len(self._WORK_ACTION.findall(text)) <= 1
            and not re.search(
                r"(?:再|然后|并且|之后|then\b).{0,16}" + self._WORK_ACTION.pattern,
                text,
                re.IGNORECASE,
            )
        ):
            return self._decision(
                request,
                mode=PromptMode.CHAT,
                confidence=0.85,
                reason="the message asks for an explanation rather than execution",
                reason_code=ModeReasonCode.DEFAULT_CHAT,
                capability_profile=self._chat_profile(request),
            )

        if self._is_explicit_work_request(message, text):
            return self._decision(
                request,
                mode=PromptMode.AGENT,
                confidence=0.9,
                reason="an explicit work action and target require Agent capabilities",
                reason_code=ModeReasonCode.EXPLICIT_WORK,
                capability_profile="agent_full",
            )

        return self._decision(
            request,
            mode=PromptMode.CHAT,
            confidence=0.7,
            reason="no deterministic work request was found",
            reason_code=ModeReasonCode.DEFAULT_CHAT,
            capability_profile=self._chat_profile(request),
        )

    def _decision(
        self,
        request: ModeRouteInput,
        *,
        mode: PromptMode,
        confidence: float,
        reason: str,
        reason_code: ModeReasonCode,
        capability_profile: str,
        work_plan_hint: str | None = None,
    ) -> ModeDecision:
        return ModeDecision(
            mode=mode,
            confidence=confidence,
            reason=reason,
            reason_code=reason_code,
            intent=request.intent,
            capability_profile=capability_profile,
            policy_version=self.policy_version,
            scheduler_revision=request.scheduler_revision,
            work_plan_hint=work_plan_hint,
        )

    def _is_explicit_work_request(self, message: InputMessage, text: str) -> bool:
        if not text or not self._WORK_ACTION.search(text):
            return False
        if message.is_group and not self._is_explicit_group_wake(message):
            return False
        return bool(self._WORK_TARGET.search(text) or self._NEW_ARTIFACT.search(text))

    @staticmethod
    def _is_explicit_group_wake(message: InputMessage) -> bool:
        return message.is_at_mention or message.content.lstrip().startswith("猫猫")

    @staticmethod
    def _chat_profile(request: ModeRouteInput) -> str:
        if not request.message.is_group:
            return "private_chat"
        if ModeRouter._is_explicit_group_wake(request.message):
            return "group_explicit"
        return "group_reply"
