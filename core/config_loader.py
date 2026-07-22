"""ConfigLoader — 配置加载封装层。

使用 Pydantic 运行时校验，捕获 config.toml 的拼写错误和类型错误。
"""

import sys
import tomllib

from pydantic import BaseModel, ConfigDict, SecretStr


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    appid: SecretStr
    secret: SecretStr
    bot_id: str = ""
    max_tool_rounds: int = -1
    character_card: str = "characters/default.md"
    providers: dict = {}
    groups: dict = {}
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


class ConfigLoader:
    def __init__(self, path: str = "config/config.toml"):
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        except FileNotFoundError:
            print(f"错误: 配置文件不存在: {path}")
            print("请复制 config.toml 并填写 QQ bot 凭证与 API key 后重试")
            sys.exit(1)

        try:
            self._cfg = AppConfig(**raw)
        except Exception as exc:
            from pydantic import ValidationError
            if isinstance(exc, ValidationError):
                print("config.toml 校验失败:")
                for err in exc.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    print(f"  [{loc}] {err['msg']} (got: {err.get('input', '?')})")
            else:
                print(f"config.toml 加载失败: {exc}")
            sys.exit(1)

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
