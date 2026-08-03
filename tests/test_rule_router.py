"""RuleRouter 打分对齐测试（对照 libs/ClawRouter 原版）。

覆盖：
- 短寒暄 → simple（原版: short token → 负分压向 SIMPLE）
- 贴分档边界 → ambiguous → medium（原版: confidence < 阈值 → 默认 MEDIUM）
- 长任务句 → complex/reasoning（不受 short_message 负分影响）
"""

import pytest

from core.rule_router import AMBIGUOUS_BAND, SHORT_MESSAGE_THRESHOLD, RuleRouter


@pytest.fixture
def router():
    return RuleRouter()


# (消息, 期望 tier 集合) — 允许取集合避免脆弱断言
TIER_CASES = [
    # 短寒暄 → simple
    ("晚安猫猫", {"simple"}),
    ("猫猫 快来说句晚安", {"simple"}),
    ("你好呀", {"simple"}),
    # 短工具意图消息 → simple（对齐原版：SIMPLE 档不关工具，工具可用性独立于 tier）
    ("用语音说一句", {"simple"}),
    ("猫猫 看看你备忘录里面有什么东西", {"simple"}),
    ("帮我查一下今天的天气", {"simple"}),
    ("搜一下表情包", {"simple"}),
    # 对齐原版 agentic-light（conf 0.70）：1 个工具词不升档，仍 simple
    ("猫猫 用工具执行一下 ls", {"simple"}),
    # 贴边界 → ambiguous → medium（对齐原版 ambiguousDefaultTier=MEDIUM）
    ("写一个二分查找算法", {"medium"}),  # 4.07 贴 complex 下边界 4.0
    # 短推理句：中文推理词被 short 负分抵消 → simple（对齐原版 SIMPLE）
    ("为什么天空是蓝色的", {"simple"}),
    # 长任务句 → 高级档
    (
        "请用 Python 写一个完整的二分查找算法实现，包含测试用例和复杂度分析，注释要详细",
        {"complex", "reasoning"},
    ),
    (
        "帮我分析一下这个问题的原因，然后给出三种解决方案并比较优劣，最后写一个实验验证方案",
        {"complex", "reasoning"},
    ),
]


@pytest.mark.parametrize("text,expected_tiers", TIER_CASES)
def test_tier_classification(router, text, expected_tiers):
    assert router.classify(text) in expected_tiers, (
        f"{text!r} → {router.classify(text)}, 期望 {expected_tiers}"
    )


def test_short_message_penalty(router):
    """短消息吃 -2.0 负分（对齐原版 tokenCount: short → -1.0）。"""
    d = router.score("晚安")
    assert d["short_message"]["score"] == -2.0
    assert d["token_count"]["score"] >= 0  # 长消息奖励维度仍为正


def test_ambiguous_band_constant():
    """模糊带宽度与边界间距成比例（原版 0.0706/0.3 ≈ 23.5% × 2.0 ≈ 0.47 → 0.5）。"""
    assert 0.3 < AMBIGUOUS_BAND <= 0.6
    assert SHORT_MESSAGE_THRESHOLD >= 20  # 原版 50 tokens，中文取 30 字


def test_ambiguous_boundary_behavior(router):
    """总分落在分档边界 band 内的消息不硬判，回退默认档 medium。

    对齐原版：sigmoid 置信度 < 阈值 → ambiguous → 默认 MEDIUM。
    用真实 classify() 验证 complex 下边界（4.0）band 内的消息降档。
    """
    text = "写一个二分查找算法"
    total = router.score(text)["total"]
    assert 4.0 <= total < 4.0 + AMBIGUOUS_BAND  # complex 下边界 band 内
    assert router.classify(text) == "medium"

    # 对照：远离边界的消息不受影响，仍落 simple
    text2 = "帮我查一下今天的天气"
    total2 = router.score(text2)["total"]
    assert total2 < 2.0 - AMBIGUOUS_BAND  # simple 档 band 外
    assert router.classify(text2) == "simple"


def test_classify_with_detail_consistency(router):
    d = router.classify_with_detail("晚安猫猫")
    assert d["tier"] == router.classify("晚安猫猫")
    assert "total" in d and "dimensions" in d
    assert len(d["dimensions"]) >= 16
