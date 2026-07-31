"""Login-shell 环境加载（借鉴 openclaw 的 shell-env.ts）。

exec/process/cron 工具均以 shell=False 直接 exec argv，从不经过用户 shell，
因此 ~/.bashrc / ~/.zshrc 中配置的环境变量（PATH 等）默认不生效。
本模块通过 login shell（`$SHELL -l -c '/usr/bin/env -0'`）探测用户
shell 配置产生的环境，在安全过滤后合并进子进程环境，使机器人执行的
命令与用户终端行为一致。

安全约束（对齐 openclaw shell-env.ts）：
- 只信任注册在 /etc/shells 中的 shell，否则回退硬编码常量 /bin/sh
- 探测时固定 HOME 为真实用户主目录、删除 ZDOTDIR/BASH_ENV/ENV/KSH_ENV
  （防 shell 启动文件被环境变量重定向到攻击者文件 → env 投毒）
- 探测结果仍会经过 security.sanitize_env 的完整危险变量过滤
  （LD_* / DYLD_* / BASH_FUNC_* / PYTHONPATH / NODE_OPTIONS / BASH_ENV 等）
- PATH 采用 prepend 合并（login shell 条目在前 + 父进程条目保留去重），
  与 openclaw "request 层禁止 PATH 覆盖" 的安全边界一致：只增不换
- 非 PATH 键仅合并允许列表（_MERGE_ALLOWLIST）：阻止 .zshrc 里 echo
  输出的 "KEY=value" 文本被误当作环境变量注入子进程。这是有意的取舍：
  JAVA_HOME / GOPATH / CARGO_HOME / NVM_DIR 等工具特定变量默认不传播
  （“与终端一致”仅对 PATH 成立）；如需某个键，把它加入 _MERGE_ALLOWLIST
- 探测结果带 TTL 缓存（失败用更短的 TTL，瞬时故障不会长期禁用）；
  失败时静默降级为安全默认 PATH

注意（openclaw 相同的取舍）：探测使用 login 非交互模式（shell -l -c）。
bash 的 ~/.bashrc 仅交互式加载，bash -l 只读 .bash_profile/.profile
（其中通常 source .bashrc）；zsh 则完整加载。HOME 沿用 bot 进程自身值
（bot 以用户身份运行时即用户主目录）。
"""

import asyncio
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

from core.tools.security import sanitize_env

_log = logging.getLogger(__name__)

_DEFAULT_SHELL = "/bin/sh"
_PROBE_TIMEOUT = 15.0
# 探测结果缓存时长（秒）；到期后重探，用户改 .zshrc 后无需重启 bot
_PROBE_TTL = 300.0
# 探测失败时的缓存时长（秒）：瞬时故障不会长时间禁用功能
_PROBE_FAIL_TTL = 60.0

# env 命令用绝对路径，避免用户 shell 函数/alias 覆盖 env 造成探测投毒
_ENV_BIN = "/usr/bin/env"

_PORTABLE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 非 PATH 键允许合并进子进程环境的白名单（locale / 编辑器 / 终端等常见项）
_MERGE_ALLOWLIST = frozenset(
    {
        "HOME",
        "TZ",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LC_TIME",
        "LC_COLLATE",
        "LC_MONETARY",
        "LC_NUMERIC",
        "LC_PAPER",
        "LC_TELEPHONE",
        "EDITOR",
        "VISUAL",
        "PAGER",
        "LESS",
        "LESSOPEN",
        "LESSCLOSE",
        "TERM",
        "COLORTERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "CLICOLOR",
        "CLICOLOR_FORCE",
        "GPG_TTY",
        "SSH_AUTH_SOCK",
    }
)

# exec / search / cron 共用的配置键（单一来源，避免散落字符串字面量）
LOAD_SHELL_ENV_CONFIG_KEY = "load_shell_env"


@dataclass
class _ProbeEntry:
    """探测结果缓存条目：探测时间 + 环境（None 表示失败）。"""

    timestamp: float
    env: Optional[Dict[str, str]]


_etc_shells_cache: Optional[set[str]] = None
_probe_cache: Optional[_ProbeEntry] = None
_probe_lock = asyncio.Lock()


def _read_etc_shells() -> set[str]:
    """读取 /etc/shells，只保留绝对路径条目。失败返回空集合。"""
    global _etc_shells_cache
    if _etc_shells_cache is not None:
        return _etc_shells_cache
    try:
        with open("/etc/shells", encoding="utf-8") as f:
            _etc_shells_cache = {
                line.strip()
                for line in f
                if line.strip()
                and not line.startswith("#")
                and line.strip().startswith("/")
            }
    except OSError:
        _log.warning("无法读取 /etc/shells，login shell 探测降级")
        _etc_shells_cache = set()
    return _etc_shells_cache


def _pick_probe_shell() -> str:
    """选择探测用 shell：优先 $SHELL（须注册在 /etc/shells 且为绝对路径）；
    否则回退硬编码常量 /bin/sh（与 openclaw 一致）。
    注意：/etc/shells 缺失时无法校验 /bin/sh 是否注册，直接使用该已知
    常量（系统自带 shell，不存在从任意路径解析的风险）。"""
    shells = _read_etc_shells()
    if not shells:
        return _DEFAULT_SHELL
    shell = os.environ.get("SHELL", "")
    if (
        shell
        and os.path.isabs(shell)
        and os.path.normpath(shell) == shell
        and shell in shells
    ):
        return shell
    return _DEFAULT_SHELL


def _probe_login_shell_env(shell: str) -> Optional[Dict[str, str]]:
    """同步探测 login shell 环境（应放入线程池调用）。

    使用 `shell -l -c '/usr/bin/env -0'`：login 模式会执行
    /etc/profile、~/.zprofile、~/.zshrc（或 bashrc 链）等启动文件，
    从而拿到用户真实配置的 PATH 与自定义变量。
    """
    probe_env = dict(os.environ)
    probe_env["HOME"] = os.path.expanduser("~")
    # 防启动文件重定向投毒：zsh 读 ZDOTDIR、bash 读 BASH_ENV、
    # sh/ksh 读 ENV/KSH_ENV —— 全部剥离，避免探测加载到被重定向的文件
    probe_env.pop("ZDOTDIR", None)
    probe_env.pop("BASH_ENV", None)
    probe_env.pop("ENV", None)
    probe_env.pop("KSH_ENV", None)

    try:
        proc = subprocess.run(
            [shell, "-l", "-c", f"{_ENV_BIN} -0"],
            env=probe_env,
            capture_output=True,
            timeout=_PROBE_TIMEOUT,
            text=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _log.warning("login shell 环境探测失败 (%s): %s", shell, e)
        return None

    if proc.returncode != 0:
        _log.warning(
            "login shell 环境探测退出码非 0 (%s): %s",
            shell,
            proc.stderr[:200].decode("utf-8", errors="replace"),
        )
        return None

    result: Dict[str, str] = {}
    for part in proc.stdout.split(b"\0"):
        if not part or b"=" not in part:
            continue
        key, _, value = part.partition(b"=")
        k = key.decode("utf-8", errors="replace").strip()
        if not k or not _PORTABLE_KEY_RE.match(k):
            continue
        result[k] = value.decode("utf-8", errors="replace")

    if not result:
        _log.warning("login shell 环境探测结果为空 (%s)", shell)
        return None

    _log.debug("login shell 环境探测成功 (%s): %d 个变量", shell, len(result))
    return result


async def get_login_shell_env() -> Optional[Dict[str, str]]:
    """获取用户 login shell 环境（TTL 缓存 + 线程池探测，防阻塞事件循环）。

    探测成功缓存 _PROBE_TTL 秒；失败缓存 _PROBE_FAIL_TTL 秒（更短，
    瞬时故障不会长期禁用功能）。返回 None 表示探测失败，调用方应降级。
    """
    global _probe_cache

    def _fresh(cache: Optional[_ProbeEntry]) -> bool:
        if cache is None:
            return False
        ttl = _PROBE_FAIL_TTL if cache.env is None else _PROBE_TTL
        return time.monotonic() - cache.timestamp < ttl

    if _fresh(_probe_cache):
        assert _probe_cache is not None
        return _probe_cache.env
    async with _probe_lock:
        if _fresh(_probe_cache):
            assert _probe_cache is not None
            return _probe_cache.env
        shell = _pick_probe_shell()
        env = await asyncio.to_thread(_probe_login_shell_env, shell)
        _probe_cache = _ProbeEntry(timestamp=time.monotonic(), env=env)
        return env


def _merge_path(prepend: str, existing: str) -> str:
    """PATH prepend 合并：login shell 条目在前，父进程条目在后，整体去重。
    对应 openclaw infra/path-prepend.ts 的 mergePathPrepend。"""
    seen = set()
    merged = []
    for entry in (prepend + os.pathsep + existing).split(os.pathsep):
        entry = entry.strip()
        if entry and entry not in seen:
            seen.add(entry)
            merged.append(entry)
    return os.pathsep.join(merged)


async def build_exec_env(enabled: bool = True) -> Dict[str, str]:
    """构建子进程安全环境（exec / process / cron / search 共用）。

    策略（对应 openclaw shell-env.ts + bash-tools.exec.ts）：
    1. enabled=False（开关关闭）：恢复旧行为 —— sanitize_env() 强制
       安全默认 PATH，父进程 PATH 完全不进入子进程（S3：既有防御不削弱）
    2. enabled=True：以过滤后的父进程 env 为基底（PATH 保留），探测
       login shell 环境；探测成功则 login shell 的 PATH 条目 prepend 到
       现有 PATH 前（去重），非 PATH 键仅合并 _MERGE_ALLOWLIST 白名单
    3. 探测失败：保持父进程 PATH（危险变量已过滤）

    信任模型（对齐 openclaw 本地 exec）：宿主机为用户自有机器，父进程
    PATH 与 login shell PATH 均属用户可控输入；本模块不额外过滤 PATH
    条目，只保证危险环境变量不进入子进程。
    """
    if not enabled:
        return sanitize_env()

    base = sanitize_env(path=None)

    shell_env = await get_login_shell_env()
    if not shell_env:
        return base

    # 探测结果同样过一遍完整过滤（防 zshrc 里写了危险变量）
    safe_shell = sanitize_env(shell_env, path=None)

    shell_path = (safe_shell.get("PATH") or "").strip()
    if shell_path:
        base_path = (base.get("PATH") or "").strip()
        base["PATH"] = _merge_path(shell_path, base_path)
        _log.debug(
            "exec 环境 PATH prepend login shell 条目: %s",
            shell_path[:200],
        )

    for key, value in safe_shell.items():
        if key in _MERGE_ALLOWLIST and key not in base and value:
            base[key] = value

    return base


async def build_exec_env_for(perm) -> Dict[str, str]:
    """读取安全配置开关并构建子进程环境（exec / search 工具共用入口）。

    Args:
        perm: PermissionManager 实例；为 None 时视为开关开启。
    """
    enabled = True
    if perm is not None:
        enabled = bool(perm.get_security_config(LOAD_SHELL_ENV_CONFIG_KEY, True))
    return await build_exec_env(enabled=enabled)
