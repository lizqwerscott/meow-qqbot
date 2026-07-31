"""共享安全逻辑 — 命令黑名单、解析检查、重定向检测、环境变量过滤。

从 SkillManagers 提取，供 exec 工具和后台任务执行器共用。
"""

import logging
import os
import re
import shlex
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

DENIED_COMMANDS: frozenset = frozenset(
    {
        "rm",
        "chmod",
        "chown",
        "sudo",
        "su",
        "doas",
        "dd",
        "mkfs",
        "fdisk",
        "parted",
        "mkswap",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "init",
        "systemctl",
        "useradd",
        "usermod",
        "groupadd",
        "userdel",
        "groupdel",
        "setuid",
        "setgid",
        "chattr",
        "lsattr",
        "tcpdump",
        "nmap",
        "tshark",
        "pkill",
        "killall",
        "kill",
        "passwd",
        "service",
        "grub-install",
        "grub-mkconfig",
        "modprobe",
        "insmod",
        "rmmod",
        "iptables",
        "ufw",
        "docker",
        "podman",
        "crontab",
        "at",
        "mount",
        "umount",
        "swapon",
        "swapoff",
        "sysctl",
        "unshare",
        "nsenter",
    }
)

DANGEROUS_TARGET_PATTERNS = re.compile(r">(?:/[^/\s]+){1,4}(?:/[^/\s]+)?")


def parse_command_safe(raw_command: str) -> Optional[List[str]]:
    """安全地将命令字符串解析为 args 列表。返回 None 表示解析失败。"""
    try:
        parts = shlex.split(raw_command)
    except ValueError:
        return None
    if not parts:
        return None
    return parts


def check_command_denied(parts: List[str]) -> Optional[str]:
    """检查命令是否命中黑名单或危险重定向。返回 None 表示通过，否则返回拒绝原因。"""
    cmd_name = os.path.basename(parts[0])
    if cmd_name in DENIED_COMMANDS:
        return f"命令 '{cmd_name}' 被禁止执行"
    for arg in parts[1:]:
        if DANGEROUS_TARGET_PATTERNS.search(arg):
            return f"参数包含危险的重定向目标: {arg[:60]}"
    return None


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
