"""共享安全逻辑 — 命令解析检查、环境变量过滤。

从 SkillManagers 提取，供 exec 工具和后台任务执行器共用。

安全模型（对齐 OpenClaw）：**无命令黑名单**——exec 防线是逐段 allowlist
（真实路径解析）+ safe-bin + 审批；命令是否危险由 allowlist 覆盖率决定，
不在准入层硬编码命令名。
"""

import logging
import os
import shlex
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


def parse_command_safe(raw_command: str) -> Optional[List[str]]:
    """安全地将命令字符串解析为 args 列表。返回 None 表示解析失败。"""
    try:
        parts = shlex.split(raw_command)
    except ValueError:
        return None
    if not parts:
        return None
    return parts


_BLOCKED_ENV_PREFIXES = frozenset(
    {
        "LD_",
        "DYLD_",
        "BASH_FUNC_",
        "GIT_CONFIG_",
        "NPM_CONFIG_",
    }
)

_SAFE_DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"

_BLOCKED_ENV_KEYS = frozenset(
    {
        "SHELLOPTS",
        "BASHOPTS",
        # shell 启动文件重定向投毒面（对齐 openclaw blockedEverywhereKeys）
        "BASH_ENV",
        "ENV",
        "KSH_ENV",
        "SHELL",
        "ZDOTDIR",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "NODE_OPTIONS",
        "RUBYOPT",
        "GEM_PATH",
        "PERLLIB",
        "PERL5LIB",
        "IFS",
    }
)


def sanitize_env(
    base_env: Optional[Dict[str, str]] = None,
    *,
    path: Optional[str] = _SAFE_DEFAULT_PATH,
) -> Dict[str, str]:
    """返回安全的子进程环境变量副本（继承当前进程，移除危险项）。

    Args:
        base_env: 基础环境变量（默认 os.environ）。
        path: PATH 值。
            - 默认 _SAFE_DEFAULT_PATH：强制安全默认 PATH（原行为，防止父进程
              被污染后指向恶意二进制）
            - None：保留 base_env 中的 PATH 不变（由调用方自行决定 PATH 来源，
              如 core/tools/shell_env.build_exec_env 的 login shell 探测）
            - 其他字符串：使用该值

    过滤目标：
    - LD_* / DYLD_* — 动态库劫持
    - BASH_FUNC_* — shellshock 类攻击
    - PYTHONPATH / NODE_OPTIONS 等语言运行时注入
    - 其他已知的危险环境变量
    """
    env = dict(base_env if base_env is not None else os.environ)
    removed: List[str] = []
    for key in list(env):
        if key in _BLOCKED_ENV_KEYS:
            removed.append(key)
            del env[key]
        else:
            for prefix in _BLOCKED_ENV_PREFIXES:
                if key.upper().startswith(prefix):
                    removed.append(key)
                    del env[key]
                    break
    if path is not None:
        env["PATH"] = path
    if removed:
        _log.debug("环境变量过滤: 移除了 %d 个危险项: %s", len(removed), removed)
    return env
