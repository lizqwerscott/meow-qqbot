"""ConfigLoader — 配置加载封装层。

纯 TOML 加载 + 属性化 section 访问，不含验证逻辑（M5 可选增加 Pydantic）。
"""

import sys
import tomllib


_MISSING = object()


class ConfigLoader:
    def __init__(self, path: str = "config/config.toml"):
        try:
            with open(path, "rb") as f:
                self._raw = tomllib.load(f)
        except FileNotFoundError:
            print(f"错误: 配置文件不存在: {path}")
            print("请复制 config.toml 并填写 QQ bot 凭证与 API key 后重试")
            sys.exit(1)

    def _require(self, key: str) -> str:
        v = self._raw.get(key, _MISSING)
        if v is _MISSING or not v:
            raise KeyError(f"config.toml 缺少必要字段: [{key}]")
        return v

    # ── 顶层标量 ──

    @property
    def bot_id(self) -> str:
        return self._require("bot_id")

    @property
    def appid(self) -> str:
        return self._require("appid")

    @property
    def secret(self) -> str:
        return self._require("secret")

    @property
    def max_tool_rounds(self) -> int:
        return self._raw.get("max_tool_rounds", -1)

    @property
    def character_card(self) -> str:
        return self._raw.get("character_card", "characters/default.md")

    # ── 模型 ──

    @property
    def providers(self) -> dict:
        return self._raw.get("providers", {})

    @property
    def groups(self) -> dict:
        return self._raw.get("groups", {})

    @property
    def cooldown(self) -> dict:
        return self._raw.get("cooldown", {})

    # ── 功能 section ──

    @property
    def multimodal(self) -> dict:
        return self._raw.get("multimodal", {})

    @property
    def context_management(self) -> dict:
        return self._raw.get("context_management", {})

    @property
    def archive(self) -> dict:
        return self._raw.get("archive", {})

    @property
    def cost_tracking(self) -> dict:
        return self._raw.get("cost_tracking", {})

    @property
    def hindsight(self) -> dict:
        return self._raw.get("hindsight", {})

    @property
    def tasks(self) -> dict:
        return self._raw.get("tasks", {})

    @property
    def workspace(self) -> dict:
        return self._raw.get("workspace", {})

    @property
    def routing(self) -> dict:
        return self._raw.get("routing", {})

    @property
    def learners(self) -> dict:
        return self._raw.get("learners", {})

    @property
    def sub_agents(self) -> dict:
        return self._raw.get("sub_agents", {})

    @property
    def tts(self) -> dict:
        return self._raw.get("tts", {})

    @property
    def webui(self) -> dict:
        return self._raw.get("webui", {})

    @property
    def heartbeat(self) -> dict:
        return self._raw.get("heartbeat", {})
