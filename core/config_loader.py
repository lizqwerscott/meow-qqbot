"""ConfigLoader — 配置加载封装层。

使用 Pydantic 运行时校验，捕获 config.toml 的拼写错误和类型错误。

模型相关配置（providers / groups / cooldown / multimodal / routing）默认
从独立的 `config/models.toml` 加载，主 `config/config.toml` 里保留的非模型
配置。两者按 section 合并：分离文件中的模型 section 优先，主配置里仍残留的
同名 section 会被合并（便于平滑迁移）。
"""

import os
import tomllib
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError


class ConfigError(Exception):
    """配置加载相关的错误。"""


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appid: SecretStr
    secret: SecretStr
    bot_id: str = ""
    max_tool_rounds: int = -1
    character_card: str = "characters/default.md"
    providers: dict = {}
    groups: dict = {}
    models: dict = {}
    cooldown: dict = {}
    multimodal: dict = {}
    context_management: dict = {}
    archive: dict = {}
    cost_tracking: dict = {}
    hindsight: dict = {}
    tasks: dict = {}
    workspace: dict = {}
    routing: dict = {}
    learners: dict = {}
    sub_agents: dict = {}
    tts: dict = {}
    webui: dict = {}
    heartbeat: dict = {}
    web_search: dict = {}
    web_fetch: dict = {}
    approval: dict = {}  # [approval]：审批卡转发目标等


class ConfigLoader:
    # 模型相关 section 可拆分到独立文件，避免主配置过于冗长
    MODEL_SECTIONS = ("providers", "groups", "cooldown", "multimodal", "routing")

    def __init__(
        self,
        path: str = "config/config.toml",
        models_path: Optional[str] = None,
    ):
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except FileNotFoundError as exc:
            raise ConfigError(
                f"配置文件不存在: {path}\n"
                "请复制 config.toml 并填写 QQ bot 凭证与 API key 后重试"
            ) from exc

        # 独立的模型配置文件默认与主配置同目录（config/models.toml 或 tmp_path 下同款）
        models_path = models_path or os.path.join(
            os.path.dirname(path) or ".", "models.toml"
        )
        # 加载独立的模型配置文件，并合并模型相关 section
        raw = self._merge_models_file(raw, models_path)

        try:
            self._cfg = AppConfig(**raw)
        except ValidationError as exc:
            lines = ["config.toml 校验失败:"]
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"])
                lines.append(f"  [{loc}] {err['msg']} (got: {err.get('input', '?')})")
            raise ConfigError("\n".join(lines)) from exc
        except Exception as exc:
            raise ConfigError(f"config.toml 加载失败: {exc}") from exc

    def _merge_models_file(
        self, raw: dict, models_path: str
    ) -> dict:
        """将 config/models.toml 中的模型 section 合并进主配置。

        models_path 可通过环境变量 MQ_MODELS_CONFIG 覆盖；文件不存在时不报错
        （保持向下兼容——只有 .toml 的单文件老配置也能用）。
        合并规则：分离文件里的 section 优先主配置，provider/groups 内部做深度合并。
        """
        if os.environ.get("MQ_MODELS_CONFIG"):
            models_path = os.environ["MQ_MODELS_CONFIG"]
        try:
            with open(models_path, "rb") as f:
                model_cfg = tomllib.load(f)
        except FileNotFoundError:
            return raw

        # 分离文件中未声明的模型 section 从主配置移除（避免两处同时定义造成困惑）
        for sec in self.MODEL_SECTIONS:
            if sec not in model_cfg:
                raw.pop(sec, None)

        for sec in model_cfg:
            if sec not in self.MODEL_SECTIONS:
                continue
            incoming = model_cfg[sec]
            existing = raw.get(sec)
            if isinstance(incoming, dict) and isinstance(existing, dict):
                merged = dict(existing)
                for key, value in incoming.items():
                    if (
                        isinstance(value, dict)
                        and isinstance(merged.get(key), dict)
                    ):
                        merged[key] = {
                            **merged[key],
                            **value,
                        }
                    else:
                        merged[key] = value
                raw[sec] = merged
            else:
                raw[sec] = incoming
        return raw

    # ── 顶层标量 ──

    @property
    def bot_id(self) -> str:
        return self._cfg.bot_id

    @property
    def appid(self) -> str:
        return self._cfg.appid.get_secret_value()

    @property
    def secret(self) -> str:
        return self._cfg.secret.get_secret_value()

    @property
    def max_tool_rounds(self) -> int:
        return self._cfg.max_tool_rounds

    @property
    def character_card(self) -> str:
        return self._cfg.character_card

    # ── 模型 ──

    @property
    def providers(self) -> dict:
        return self._cfg.providers

    @property
    def groups(self) -> dict:
        return self._cfg.groups

    @property
    def models(self) -> dict:
        return self._cfg.models

    @property
    def cooldown(self) -> dict:
        return self._cfg.cooldown

    # ── 功能 section ──

    @property
    def multimodal(self) -> dict:
        return self._cfg.multimodal

    @property
    def context_management(self) -> dict:
        return self._cfg.context_management

    @property
    def archive(self) -> dict:
        return self._cfg.archive

    @property
    def cost_tracking(self) -> dict:
        return self._cfg.cost_tracking

    @property
    def hindsight(self) -> dict:
        return self._cfg.hindsight

    @property
    def tasks(self) -> dict:
        return self._cfg.tasks

    @property
    def workspace(self) -> dict:
        return self._cfg.workspace

    @property
    def routing(self) -> dict:
        return self._cfg.routing

    @property
    def learners(self) -> dict:
        return self._cfg.learners

    @property
    def sub_agents(self) -> dict:
        return self._cfg.sub_agents

    @property
    def tts(self) -> dict:
        return self._cfg.tts

    @property
    def webui(self) -> dict:
        return self._cfg.webui

    @property
    def heartbeat(self) -> dict:
        return self._cfg.heartbeat

    @property
    def web_search(self) -> dict:
        return self._cfg.web_search

    @property
    def web_fetch(self) -> dict:
        return self._cfg.web_fetch
