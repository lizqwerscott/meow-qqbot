"""Auto-reviewer — 移植 OpenClaw 的 mode=auto 原生审查器。

allowlist miss 时先用轻量 LLM 做一次低风险判定（只读命令放行一次），
reviewer 拒绝或不确定才转人工。对齐 openclaw 的两条约束：
- reviewer 判定结果不落 allowlist（allow-always 不因 reviewer 产生）
- inline-eval 命令一律跳过 reviewer，直接转人工
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

_log = logging.getLogger(__name__)

# 可注入的审查函数签名：输入 plan dict，输出 "allow" | "ask"
ReviewFn = Callable[[dict], Awaitable[str]]

REVIEW_PROMPT = """你是命令执行安全审查器。判断以下命令是否属于"只读/低风险"操作。

命令: {command}
工作目录: {cwd}
解析出的可执行文件: {resolved_path}
角色: {role}

只读/低风险示例：查看状态（git status、ls、cat、df、ps、env 等）、
查询（curl GET、ping）、无副作用的脚本。
高风险示例：写入/删除/修改文件或系统状态（rm、mv、> 重定向、
install、docker、git push、网络写操作等）、任何无法确认副作用的命令。

只输出一个词：allow（放行一次）或 ask（转人工审批）。
"""


class ExecAutoReviewer:
    def __init__(self, review_fn: Optional[ReviewFn] = None):
        self._review_fn = review_fn
        self._calls = 0

    @property
    def available(self) -> bool:
        return self._review_fn is not None

    @property
    def calls(self) -> int:
        return self._calls

    async def review(self, plan: dict) -> str:
        """返回 "allow"（放行一次）或 "ask"（转人工）。"""
        if self._review_fn is None:
            _log.warning("auto-reviewer 未注入 review_fn，降级为人工审批")
            return "ask"
        self._calls += 1
        try:
            return await self._review_fn(plan)
        except Exception as e:
            _log.warning("auto-reviewer 调用失败，降级为人工审批: %s", e)
            return "ask"
