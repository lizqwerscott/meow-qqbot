"""Base — 学习系统的公共类型与工具函数。"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── 学习系统文本清洗 ──

_REPLY_PREFIX = re.compile(r"^\[正在回复[^\]]*\]\n?")
_EMOJI_MARKER = re.compile(r"\[表情:[^\]]*\]")
_EMOTION_MARKER = re.compile(r"\[情绪:[^\]]*\]")
_AT_MENTION = re.compile(r"@\S+")
_JSON_LIKE = re.compile(r"^\s*[\{\[].*[\}\]]\s*$", re.DOTALL)


def sanitize_for_learners(text: str) -> str:
    """去掉 QQ 结构标记，返回纯用户文本供学习系统使用。

    清理项：
    - [正在回复 ...] 回复上下文前缀
    - [表情: ...] 表情描述
    - [情绪: ...] 情绪标签
    - @昵称 提及
    - JSON 风格的卡片消息内容
    """
    if not text:
        return text

    text = _REPLY_PREFIX.sub("", text)
    text = _EMOJI_MARKER.sub("", text)
    text = _EMOTION_MARKER.sub("", text)
    text = _AT_MENTION.sub("", text)
    text = text.strip()

    if not text:
        return ""
    if _JSON_LIKE.match(text):
        return ""

    return text


# ── 余弦相似度 ──


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """两个向量的余弦相似度。"""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"向量维度不匹配: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── 指数衰减 ──


def exponential_decay(
    confidence: float, decay_rate: float, delta_hours: float
) -> float:
    """指数衰减公式: confidence *= exp(-λ * Δt_hours)"""
    return confidence * math.exp(-decay_rate * delta_hours / 24.0)


# ── 学习配置 ──


@dataclass
class LearnerConfig:
    enabled: bool = True
    data_dir: str = "data/learners/"

    # JargonMiner
    inference_thresholds: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    cross_group_min: int = 2
    max_jargon_per_room: int = 500


def config_from_dict(d: Optional[dict]) -> LearnerConfig:
    """从 config.yaml 的 dict 构造 LearnerConfig。"""
    if not d:
        return LearnerConfig(enabled=False)
    return LearnerConfig(
        enabled=d.get("enabled", True),
        data_dir=d.get("data_dir", "data/learners/"),
        inference_thresholds=d.get("jargon_miner", {}).get(
            "inference_thresholds", [1, 3, 5, 10]
        ),
        cross_group_min=d.get("jargon_miner", {}).get("cross_group_min", 2),
        max_jargon_per_room=d.get("jargon_miner", {}).get("max_jargon_per_room", 500),
    )
