"""exec 工具 env 覆盖策略（对齐 OpenClaw host-env-security 的安全边界）。

背景：模型可经 `exec(command=..., env={KEY: value})` 注入子进程环境变量，
从而不再需要包 `bash -c 'export K=V && ...'`（后者触发 strictInlineEval 门禁）。
这是模型**主动覆盖**环境的场景，边界必须比"继承父进程环境"更严——危险环境变量
可在不经过 allowlist 的情况下劫持二进制解析路径 / 语言运行时 / 构建系统 / 凭据通道。

与 OpenClaw 的对齐与差异：
- OpenClaw（`infra/host-env-security-policy.json` + `bash-tools.exec.ts`）把策略
  拆成 `blockedEverywhereKeys`、`blockedOverrideOnlyKeys` 两类，且 `blockPathOverrides
  : true`（PATH 覆盖直接 Security Violation）。本模块只关心**覆盖**场景，故把两类
  "覆盖时禁止"的键合并成一组（override 场景下 everywhere 与 override-only 行为一致），
  语义终点对齐；不区分"仅继承时放行"的键——那部分由 `shell_env.build_exec_env`
  的继承白名单（`_MERGE_ALLOWLIST`）另管。
- 边界比 OpenClaw 略严：OpenClaw 的模型覆盖前缀只有 4 个（CARGO_REGISTRIES_ /
  GIT_CONFIG_ / NPM_CONFIG_ / TF_VAR_），多数键按名精确匹配；本模块补了 LD_ /
  DYLD_ / BASH_FUNC_ / CARGO_TARGET_ 前缀（对齐继承侧 sanitize_env 的前缀）。偏严
  符合 "deny over claim"（宁可误拒不可误放）原则。

注意：本模块是有意添加的**枚举黑名单**，与 exec 命令层"无命令黑名单"哲学不同——
命令层由 allowlist + 审批兜底，而 env 注入无法靠命令审批兜底（环境变量不触发命令
审批），因此这里是正当的安全例外，独立于命令黑名单原则维护、集中在此一处便于审查。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

_PORTABLE_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 模型通过 env 参数覆盖时禁止的键（对齐 openclaw blockedEverywhereKeys +
# blockedOverrideOnlyKeys 中"覆盖时禁止"的核心子集）。语义终点：防止通过环境变量
# 劫持二进制解析路径 / 语言运行时 / 初始化文件 / 构建系统 / 凭证 / 代理通道。
ENV_OVERRIDE_BLOCKED_KEYS = frozenset(
    k.upper()
    for k in {
        "PATH",  # 覆盖 PATH 可重定向到任意恶意二进制（对齐 openclaw blockPathOverrides）
        # 语言运行时 / 初始化文件注入
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "NODE_OPTIONS",
        "NODE_PATH",
        "NODE_EXTRA_CA_CERTS",
        "NODE_TLS_REJECT_UNAUTHORIZED",
        "RUBYOPT",
        "RUBYLIB",
        "PERL5LIB",
        "PERL5OPT",
        "GEM_PATH",
        "GEM_HOME",
        "LUA_PATH",
        "LUA_CPATH",
        "CLASSPATH",
        "JAVA_OPTS",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "MAVEN_OPTS",
        "GRADLE_OPTS",
        "SBT_OPTS",
        "ANT_OPTS",
        # shell 环境
        "SHELL",
        "SHELLOPTS",
        "BASHOPTS",
        "BASH_ENV",
        "ENV",
        "KSH_ENV",
        "ZDOTDIR",
        "IFS",
        "PS4",
        "FPATH",
        "PROMPT_COMMAND",
        # git
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_NAMESPACE",
        "GIT_CONFIG",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_EDITOR",
        "GIT_EXTERNAL_DIFF",
        "GIT_SEQUENCE_EDITOR",
        "GIT_SSH_COMMAND",
        "GIT_SSL_NO_VERIFY",
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
        # 编译器/构建系统
        "CC",
        "CXX",
        "CPP",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_TOOLCHAIN_FILE",
        "MAKE",
        "MAKEFLAGS",
        "MFLAGS",
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTFLAGS",
        "CFLAGS",
        "CPPFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "GOFLAGS",
        "GOPROXY",
        "GOPATH",
        "GONOSUMDB",
        "GONOSUMCHECK",
        "GOPRIVATE",
        # 凭证 / 敏感通道
        "SSH_AUTH_SOCK",
        "SSH_ASKPASS",
        "SUDO_ASKPASS",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "NOPROXY",
        "HTTP_PROXY_HTTP",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_AUTH_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "KUBECONFIG",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITLAB_TOKEN",
        "NPM_TOKEN",
        "NODE_AUTH_TOKEN",
        "DATABASE_URL",
        "REDIS_URL",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CERT_PATH",
        "DOCKER_TLS_VERIFY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "OPENSSL_CONF",
        "OPENSSL_ENGINES",
    }
)

ENV_OVERRIDE_BLOCKED_PREFIXES = (
    "LD_",
    "DYLD_",
    "BASH_FUNC_",
    "GIT_CONFIG_",
    "NPM_CONFIG_",
    "CARGO_REGISTRIES_",
    "TF_VAR_",
    "CARGO_TARGET_",
)


def validate_env_override(
    env_args: Optional[Dict[str, object]],
) -> "tuple[Dict[str, str], List[str]]":
    """校验并归一 exec 的 env 覆盖参数（对齐 openclaw sanitizeHostExecEnv）。

    Returns:
        (validated_overrides, errors)：validated_overrides 为通过校验、可注入的
        ``KEY -> str`` 覆盖子集；errors 非空时调用方必须拒绝执行（硬拒语义）。
    """
    overrides: Dict[str, str] = {}
    errors: List[str] = []
    if not env_args:
        return overrides, errors
    if not isinstance(env_args, dict):
        return overrides, ["env 参数必须是 {KEY: value} 字典"]
    for key, value in env_args.items():
        if not isinstance(key, str) or not _PORTABLE_ENV_KEY_RE.match(key):
            errors.append(f"非法环境变量键名: {key!r}")
            continue
        upper = key.upper()
        if upper in ENV_OVERRIDE_BLOCKED_KEYS:
            errors.append(f"禁止覆盖环境变量: {key}")
            continue
        if any(upper.startswith(p) for p in ENV_OVERRIDE_BLOCKED_PREFIXES):
            errors.append(f"禁止覆盖环境变量前缀: {key}")
            continue
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            errors.append(f"环境变量 {key} 的值必须是字符串或数字")
            continue
        overrides[key] = str(value)
    return overrides, errors
