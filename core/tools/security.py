"""共享安全逻辑 — 命令黑名单、解析检查、重定向检测。

从 SkillManagers 提取，供 exec 工具和后台任务执行器共用。
"""

import logging
import os
import re
import shlex
from typing import List, Optional

_log = logging.getLogger(__name__)

DENIED_COMMANDS: frozenset = frozenset({
    "rm", "chmod", "chown", "sudo", "su", "doas",
    "dd", "mkfs", "fdisk", "parted", "mkswap",
    "shutdown", "reboot", "poweroff", "halt", "init", "systemctl",
    "useradd", "usermod", "groupadd", "userdel", "groupdel",
    "setuid", "setgid", "chattr", "lsattr",
    "tcpdump", "nmap", "tshark",
    "pkill", "killall", "kill", "passwd",
    "service", "grub-install", "grub-mkconfig",
    "modprobe", "insmod", "rmmod",
    "iptables", "ufw",
    "docker", "podman",
    "crontab", "at",
    "mount", "umount",
    "swapon", "swapoff",
    "sysctl",
    "unshare", "nsenter",
})

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
