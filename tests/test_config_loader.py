import tomllib
from pathlib import Path

import pytest

from core.config_loader import ConfigError, ConfigLoader


def _write_toml(path: str, content: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _minimal_config() -> str:
    return """appid = "test_appid"
secret = "test_secret"
bot_id = "bot_001"
max_tool_rounds = 5
character_card = "characters/default.md"
providers = {}
groups = {}
"""


# ── 缺失文件 ──


def test_missing_file(tmp_path):
    path = str(tmp_path / "config.toml")
    with pytest.raises(ConfigError, match="配置文件不存在"):
        ConfigLoader(path)


# ── 有效配置 ──


def test_valid_config(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, _minimal_config())
    cfg = ConfigLoader(path)
    assert cfg.bot_id == "bot_001"
    assert cfg.max_tool_rounds == 5
    assert cfg.character_card == "characters/default.md"
    assert cfg.providers == {}
    assert cfg.groups == {}
    assert cfg.media == {}


def test_appid_secret_access(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, _minimal_config())
    cfg = ConfigLoader(path)
    assert cfg.appid == "test_appid"
    assert cfg.secret == "test_secret"


# ── 缺失必需字段 ──


def test_missing_appid(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, """secret = "x"\n""")
    with pytest.raises(ConfigError, match="appid"):
        ConfigLoader(path)


def test_missing_secret(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, """appid = "x"\n""")
    with pytest.raises(ConfigError, match="secret"):
        ConfigLoader(path)


# ── 额外键（extra="forbid"）──


def test_extra_key_raises(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, _minimal_config() + '\nunknown_key = "whatever"\n')
    with pytest.raises(ConfigError, match="unknown_key"):
        ConfigLoader(path)


def test_extra_nested_key_raises(tmp_path):
    path = str(tmp_path / "config.toml")
    # 在顶层声明未知 section 会触发 extra="forbid"
    content = _minimal_config().replace("providers = {}", "")
    content += '\n[unknown_section]\nfoo = "bar"\n'
    _write_toml(path, content)
    with pytest.raises(ConfigError, match="unknown_section"):
        ConfigLoader(path)


# ── 无效类型 ──


def test_invalid_type_for_int_field(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, _minimal_config().replace("max_tool_rounds = 5", 'max_tool_rounds = "not_a_number"'))
    with pytest.raises(ConfigError, match="max_tool_rounds"):
        ConfigLoader(path)


# ── 无效 TOML 语法 ──


def test_invalid_toml_syntax(tmp_path):
    path = str(tmp_path / "config.toml")
    p = Path(path)
    p.write_text("this is not valid toml {{{", encoding="utf-8")
    with pytest.raises((ConfigError, tomllib.TOMLDecodeError)):
        ConfigLoader(path)


# ── 空配置 ──


def test_empty_config(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, "")
    with pytest.raises(ConfigError):
        ConfigLoader(path)


# ── 可选字段默认值 ──


def test_optional_fields_default(tmp_path):
    path = str(tmp_path / "config.toml")
    _write_toml(path, _minimal_config())
    cfg = ConfigLoader(path)
    assert cfg.cooldown == {}
    assert cfg.multimodal == {}
    assert cfg.context_management == {}
    assert cfg.tasks == {}
    assert cfg.routing == {}
    assert cfg.learners == {}
    assert cfg.tts == {}
    assert cfg.webui == {}
    assert cfg.heartbeat == {}
